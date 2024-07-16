# --------------------------------------------------------
# Source code for Topo2D
# @Time    : 2024/07/16
# @Author  : Han Li
# bryce18373631@gmail.com
# --------------------------------------------------------

import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[...,0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[...,0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    new_pts[...,2:3] = pts[...,2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h, patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts

def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[...,0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[...,0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return bboxes

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

@BBOX_CODERS.register_module()
class Topo2DCoder3D(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds, 
                      lclc_preds=None, lcte_preds=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        # [NOTE] maintain the order of queries
        # import pdb; pdb.set_trace()
        # scores, indexs = cls_scores.max(-1)
        # labels = indexs + 1
        # bbox_index = torch.arange(max_num).to(scores.device)
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        if lclc_preds is not None:
            lclc_preds = lclc_preds.squeeze(-1).sigmoid()
            lclc_preds = lclc_preds[bbox_index][:, bbox_index]

        if lcte_preds is not None:
            lcte_preds = lcte_preds.squeeze(-1).sigmoid()
            lcte_preds = lcte_preds[bbox_index]
       
        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range) 
        if pts_preds.shape[-1] == 2:
            final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) #num_q,num_p,2
        else:
            final_pts_preds = denormalize_3d_pts(pts_preds, self.pc_range) #num_q,num_p,3
        final_scores = scores 
        final_preds = labels 

        predictions_dict = {
            'bboxes': final_box_preds,
            'scores': final_scores,
            'labels': final_preds,
            'pts': final_pts_preds,
            'topo_lclc': lclc_preds,
            'topo_lcte': lcte_preds,
        }

        return predictions_dict

    def decode(self, preds_dicts, lclc_preds_list=None, lcte_preds_list=None):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        if lclc_preds_list is None:
            lclc_preds_list = [None for _ in range(batch_size)]
        if lcte_preds_list is None:
            lcte_preds_list = [None for _ in range(batch_size)]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(
                all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i], 
                lclc_preds_list[i], lcte_preds_list[i]))
        return predictions_list

def denormalize_2d_pts_xz(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1] * (pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[..., 1:2] * (pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

@BBOX_CODERS.register_module()
class Topo2DCoder3DLane(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 org_img_size,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=0.,
                 num_classes=10):
        self.pc_range = pc_range
        self.org_img_size = org_img_size
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, vis_scores, bbox_preds, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num # 50

        cls_scores = cls_scores.sigmoid() # 50, 1
        scores, indexs = cls_scores.view(-1).topk(max_num) # 50
        labels = indexs % self.num_classes + 1 # 50, [0, 19] -> [1, 20]
        bbox_index = indexs // self.num_classes # 50
        # [NOTE] maintain the order of queries
        # import pdb; pdb.set_trace()
        # scores, indexs = cls_scores.max(-1)
        # labels = indexs + 1
        # bbox_index = torch.arange(max_num).to(scores.device)

        vis_scores = vis_scores.sigmoid() > 0.5 # 50, 20
        vis_scores = vis_scores[bbox_index] # 50, 20
        bbox_preds = bbox_preds[bbox_index] # 50, 4
        pts_preds = pts_preds[bbox_index] # 50, 20, 2

        # refine vises to ensure consistent lane
        flag_l = vis_scores.cumsum(dim=1)
        flag_r = vis_scores.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        vis_scores = (flag_l > 0) & (flag_r > 0)

        final_box_preds = bbox_preds
        final_pts_preds = denormalize_2d_pts_xz(pts_preds, self.pc_range)
        final_scores = scores 
        final_preds = labels
        final_vis = vis_scores 

        # use score threshold(default 0.)
        thresh_mask = final_scores > self.score_threshold
        tmp_score = self.score_threshold
        while thresh_mask.sum() == 0:
            tmp_score *= 0.9
            if tmp_score < 0.01:
                thresh_mask = final_scores > -1
                break
            thresh_mask = final_scores >= tmp_score

        boxes3d = final_box_preds[thresh_mask]
        scores = final_scores[thresh_mask]
        pts = final_pts_preds[thresh_mask]
        labels = final_preds[thresh_mask]
        vis = final_vis[thresh_mask]

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels,
            'pts': pts,
            'vis': vis,
        }

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # last layer [1, 50, 1]
        all_vis_scores = preds_dicts['all_vis_scores'][-1] # last layer [1, 50, 20]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # last layer [1, 50, 4]
        all_pts_preds = preds_dicts['all_pts_preds'][-1] # last layer [1, 50, 20, 2]

        batch_size = all_cls_scores.size()[0] # 1 when test
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], 
                    all_vis_scores[i], all_bbox_preds[i], all_pts_preds[i]))
        return predictions_list

def denormalize_2d_bbox_img(bboxes, img_size):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * img_size[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * img_size[1]
    return bboxes

def denormalize_2d_pts_img(pts, img_size):
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1]* img_size[0]
    new_pts[...,1:2] = pts[..., 1:2] * img_size[1]
    return new_pts

@BBOX_CODERS.register_module()
class Topo2DCoder2D(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 org_img_size,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.org_img_size = org_img_size
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, vis_scores, bbox_preds, pts_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num # 50

        cls_scores = cls_scores.sigmoid() # 50, 1
        scores, indexs = cls_scores.view(-1).topk(max_num) # 50
        labels = indexs % self.num_classes # 50
        bbox_index = indexs // self.num_classes # 50
        # [NOTE] maintain the order of queries
        # import pdb; pdb.set_trace()
        # scores, indexs = cls_scores.max(-1)
        # labels = indexs + 1
        # bbox_index = torch.arange(max_num).to(scores.device)

        # if not output_vis, this sigmoid(act on all-zero tensor) is useless
        vis_scores = vis_scores.sigmoid() > 0.5 # 50, 20
        vis_scores = vis_scores.view(cls_scores.shape[0], -1)
        vis_scores = vis_scores[bbox_index] # 50, 20
        bbox_preds = bbox_preds[bbox_index] # 50, 4
        pts_preds = pts_preds[bbox_index] # 50, 20, 2

        final_box_preds = denormalize_2d_bbox_img(bbox_preds, self.org_img_size) 
        final_pts_preds = denormalize_2d_pts_img(pts_preds, self.org_img_size)
        final_scores = scores 
        final_preds = labels
        final_vis = vis_scores 

        # use score threshold
        # import pdb; pdb.set_trace()
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        # use post center range
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :4] >=
                    self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <=
                     self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            vis = final_vis[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'pts': pts,
                'vis': vis,
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1] # last layer [1, 50, 1]
        all_vis_scores = preds_dicts['all_vis_scores'][-1] # last layer [1, 50, 20]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1] # last layer [1, 50, 4]
        all_pts_preds = preds_dicts['all_pts_preds'][-1] # last layer [1, 50, 20, 2]

        batch_size = all_cls_scores.size()[0] # 1 when test
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], 
                    all_vis_scores[i], all_bbox_preds[i], all_pts_preds[i]))
        return predictions_list