# --------------------------------------------------------
# Source code for Topo2D
# @Time    : 2024/07/16
# @Author  : Han Li
# bryce18373631@gmail.com
# --------------------------------------------------------

import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
import torch.nn.functional as F
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[5] - pc_range[2] # z
    patch_w = pc_range[3] - pc_range[0] # x
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0] # x
    new_pts[...,1:2] = pts[..., 1:2] - pc_range[2] # z
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1] * (pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[..., 1:2] * (pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

@BBOX_ASSIGNERS.register_module()
class Topo2DAssigner3DLane(BaseAssigner):

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pts_cost=dict(type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0),
                 pc_range=None, 
                 org_img_size=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pts_cost = build_match_cost(pts_cost)
        self.pc_range = pc_range
        self.org_img_size = org_img_size

    def assign(self,
               bbox_pred,
               cls_pred,
               pts_pred,
               gt_bboxes, 
               gt_labels,
               gt_pts,
               gt_vis,
               gt_bboxes_ignore=None,
               eps=1e-7):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        num_vec, num_orders, num_pts, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        pts_cost_ordered = self.pts_cost(pts_pred, normalized_gt_pts, gt_vis)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)
        cost = cls_cost + pts_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index
