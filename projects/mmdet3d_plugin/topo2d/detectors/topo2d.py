# --------------------------------------------------------
# Source code for Topo2D
# @Time    : 2024/07/16
# @Author  : Han Li
# bryce18373631@gmail.com
# --------------------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import builder
from mmcv.cnn import Linear, Conv2d
from ..modules.query_generator import QueryGenerator
from ..modules.sparse_int import SparseInsDecoderMask
from ..modules.ms2one import build_ms2one
import os
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

@DETECTORS.register_module()
class Topo2D(MVXTwoStageDetector):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 modality='vision',
                 lidar_encoder=None,
                 only_2d=True, # only train 2d decoder
                 intrins_feat_scale=0.1, # scale intrins
                 pc_range=None, # normalize pts
                 position_range=None,
                 query_generator=None, # pred depth
                 pts_bbox_head_3d=None, # register
                 lclc_head=None,
                 lcte_head=None,
                 te_head=None,
                 fusion_method='query', # query, ref, roi
                 pos_embed_method=None, # None, uniform, ipm, pred, anchor
                 num_vec=50,
                 num_pts_per_vec=20,
                 vis_attn_map=False,
                 mask_head=False,
                 sparse_ins_decoder=None,
                 ms2one=None,
                 feat_size=None,
                 keep_assign=False,
                #  reg_sigmoid=True,
                 nms_2d_proposal=False,
                 topk_2d_proposal=0,
                 proposal_cfg_score_thresh=0., 
                 proposal_cfg_nms_thresh=20,
                 learn_3d_query=False,
                 learn_3d_pe=False,
                 pe_sample=False,
                 ref_pts_detach=False,
                 lane_topo=True,
                 traffic=True,
                 ):
        # import pdb; pdb.set_trace()
        super(Topo2D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        # self.reg_sigmoid = reg_sigmoid

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.modality = modality
        if self.modality == 'fusion' and lidar_encoder is not None :
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

        self.only_2d = only_2d
        self.intrins_feat_scale = intrins_feat_scale
        self.pc_range = pc_range
        self.position_range = position_range
        self.fusion_method = fusion_method
        self.pos_embed_method = pos_embed_method
        self.num_pts_per_vec = num_pts_per_vec
        self.num_vec = num_vec
        self.vis_attn_map = vis_attn_map
        self.feat_size = feat_size
        self.keep_assign = keep_assign
        self.nms_2d_proposal = nms_2d_proposal
        self.topk_2d_proposal = topk_2d_proposal
        self.proposal_cfg_score_thresh = proposal_cfg_score_thresh
        self.proposal_cfg_nms_thresh = proposal_cfg_nms_thresh
        self.ref_pts_detach = ref_pts_detach
        self.lane_topo = lane_topo
        if self.lane_topo:
            self.topo_ll_head = builder.build_head(lclc_head)
        self.traffic = traffic
        if self.traffic:
            self.topo_lt_head = builder.build_head(lcte_head)
            self.te_head = builder.build_head(te_head)
        self.multiScale = False
        if ms2one is not None:
            self.multiScale = True
            self.ms2one = build_ms2one(ms2one)
        if not self.only_2d:
            # self.out_fov_reference_points = nn.Embedding(self.num_vec * self.num_pts_per_vec, 3)
            # nn.init.uniform_(self.out_fov_reference_points.weight.data, 0, 1)
            self.query_generator = QueryGenerator(**query_generator)
            self.embed_dims = 256
            self.learn_3d_query = learn_3d_query
            if self.learn_3d_query:
                self.learn_query_embed = nn.Embedding(self.num_vec * self.num_pts_per_vec, self.embed_dims)
            self.learn_3d_pe = learn_3d_pe
            if self.learn_3d_pe:
                self.learn_pe_embed = nn.Embedding(self.num_vec * self.num_pts_per_vec, self.embed_dims)
            self.pe_sample = pe_sample
            if self.pos_embed_method == 'pred':
                self.query_embedding = nn.Sequential(
                    nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims),
                )
            elif self.pos_embed_method == 'anchor':
                self.query_embedding_petr = nn.Sequential( # 64 * 3 -> 1024
                    nn.Linear(self.embed_dims * 3 // 4, self.embed_dims * 4),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims * 4, self.embed_dims),
                )
            self.num_fc = 2
            self.fc_dim = 256
            # self.num_pts_per_vec = 20
            # self.num_vec = 50
            # fcs_query = []
            # for i in range(self.num_fc):
            #     in_dim = self.fc_dim * self.num_pts_per_vec if i == 0 else self.fc_dim
            #     fcs_query.append(Linear(in_dim, self.fc_dim))
            #     fcs_query.append(nn.ReLU(inplace=True))
            # self.fcs_query = nn.Sequential(*fcs_query)
            # fcs_pos = []
            # for i in range(self.num_fc):
            #     in_dim = self.fc_dim * self.num_pts_per_vec if i == 0 else self.fc_dim
            #     fcs_pos.append(Linear(in_dim, self.fc_dim))
            #     fcs_pos.append(nn.ReLU(inplace=True))
            # self.fcs_pos = nn.Sequential(*fcs_pos)
            self.pts_bbox_head_3d = builder.build_head(pts_bbox_head_3d)
            self.mask_head = mask_head
            if self.mask_head:
                self.sparse_int = SparseInsDecoderMask(cfg=sparse_ins_decoder)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # if img.dim() == 5 and img.size(0) == 1:
            #     img.squeeze_()
            # elif img.dim() == 5 and img.size(0) > 1:
            #     B, N, C, H, W = img.size()
            #     img = img.reshape(B * N, C, H, W)
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        if torch.is_tensor(img):
            img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        else:
            front_view_img = [torch.stack(bs[:1], dim=0) for bs in img]
            front_view_img = torch.stack(front_view_img, dim=0)
            front_view_img_feats = self.extract_img_feat(front_view_img, img_metas, len_queue=len_queue)
            other_view_img = [torch.stack(bs[1:], dim=0) for bs in img]
            other_view_img = torch.stack(other_view_img, dim=0)
            other_view_img_feats = self.extract_img_feat(other_view_img, img_metas, len_queue=len_queue)

            img_feats = []
            b, cam = len(img), len(img[0])
            for lvl in range(len(front_view_img_feats)):
                _front_feat = front_view_img_feats[lvl] # bs, 1, c, h, _w
                _other_feat = other_view_img_feats[lvl] # bs, 6, c, _h, w
                h, w = _front_feat.shape[-2], _other_feat.shape[-1]
                _feat = _front_feat.new_zeros(b, cam, self.embed_dims, h, w)
                _feat[:, :1, :, :, :_front_feat.shape[-1]] = _front_feat
                _feat[:, 1:, :, :_other_feat.shape[-2], :] = _other_feat
                img_feats.append(_feat)
        
        return img_feats

    def roi_feature(self, pts_feats, proposal_list):
        feats_list = []
        for img_feat, line_pts in zip(pts_feats[0], proposal_list):
            # img_feat: 1, C, H, W
            # line_pts: num_vec, num_pts, 2
            img_line_pts = line_pts.clone()
            img_line_pts[..., 1] = 1 - img_line_pts[..., 1]
            img_line_pts = 2 * img_line_pts - 1
            sampled_feats = F.grid_sample(img_feat, img_line_pts[None])  # 1, C, num_vec, num_pts
            sampled_feats = sampled_feats.permute(0, 2, 3, 1)  # 1, num_vec, num_pts, C
            feats_list.append(sampled_feats)
        bbox_feats = torch.cat(feats_list, axis=0) # 16, 50, 20, 256
        return bbox_feats
    
    def normalize_ref_pts(self, reference_points, pc_range):
        reference_points[..., 0:1] = (reference_points[..., 0:1] - pc_range[0]) / (
                pc_range[3] - pc_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - pc_range[1]) / (
                pc_range[4] - pc_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - pc_range[2]) / (
                pc_range[5] - pc_range[2])
        # BUG: not in-place function
        # reference_points.clamp(min=0, max=1)
        reference_points = reference_points.clamp(min=0, max=1)
        return reference_points
    
    def pos2posemb3d(self, pos, num_pos_feats=128, temperature=10000):
        import math
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
        return posemb

    def pos2posemb2d(self, pos, num_pos_feats=128, temperature=10000):
        import math
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        # pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    # def get_proposal(self, outs, topk_num):
    #     proposal_list = outs['all_pts_preds'][-1] # [12, 50, 20, 2]
    #     query_embeds = outs['query'][-1] # [12, 1000, 256]
    #     query_embeds = query_embeds.view(*proposal_list.shape[:3], -1) # [12, 50, 20, 256]
    #     cls_scores = outs['all_cls_scores'][-1] # [12, 50, 1]
    #     cls_scores = cls_scores.sigmoid() # 12, 50, 1
    #     cls_scores = cls_scores.squeeze(-1) # 12, 50
    #     scores, indexs = cls_scores.topk(topk_num)
    #     ret_proposal = []
    #     ret_query = []
    #     for i in range(len(indexs)):
    #         ret_proposal.append(proposal_list[i][indexs[i]])
    #         ret_query.append(query_embeds[i][indexs[i]])
    #     return torch.stack(ret_proposal, dim=0), torch.stack(ret_query, dim=0)

    def get_proposal(self, outs, topk_num, score_thresh=0.1):
        # proposal_list = outs['all_pts_preds'][-1] # [12, 50, 20, 2]
        # query_embeds = outs['query'][-1] # [12, 1000, 256]
        # query_embeds = query_embeds.view(*proposal_list.shape[:3], -1) # [12, 50, 20, 256]
        cls_scores = outs['all_cls_scores'][-1] # [12, 50, 1]
        cls_scores = cls_scores.sigmoid() # 12, 50, 1
        num_classes = cls_scores.shape[-1] # cls
        cls_scores = cls_scores.flatten(1) # B, N * cls
        
        indexs_batched = []
        for i in range(len(cls_scores)):
            scores = cls_scores[i] # 50 * 20
            order = scores.argsort(descending=True)
            keep = []
            for o in order:
                if scores[o] < score_thresh:
                    break
                else:
                    keep.append(o)
            if len(keep):
                indexs = torch.tensor(keep)[:topk_num]
            else:
                indexs = torch.tensor(order)[:topk_num]
            indexs = indexs // num_classes
            indexs_batched.append(indexs)
        return indexs_batched

        # scores, indexs = cls_scores.topk(topk_num)
        # indexs = indexs // num_classes
        # return indexs # B, topk

        # ret_proposal = []
        # ret_query = []
        # for i in range(len(indexs)):
        #     ret_proposal.append(proposal_list[i][indexs[i]])
        #     ret_query.append(query_embeds[i][indexs[i]])
        # return torch.stack(ret_proposal, dim=0), torch.stack(ret_query, dim=0)

    @torch.no_grad()
    def nms_2d(self, 
               proposals, 
               scores, 
               vises=None,
               score_thresh=0.1, 
               nms_thresh=50, 
               keep_num=10, 
               normalize=False
            ):
        new_pts = proposals.clone()
        if normalize:
            img_size = self.query_generator.img_size
            new_pts[..., 0:1] = new_pts[..., 0:1] * img_size[0]
            new_pts[..., 1:2] = new_pts[..., 1:2] * img_size[1]

        # proposals: [N, 20, 2], scores: [N], vises: [N, 20]
        order = scores.argsort(descending=True)
        keep = []
        while order.shape[0] > 0:
            i = order[0]
            if scores[i] < score_thresh:
                break
            else:
                keep.append(i)

            x1 = new_pts[i][:, 0] # [l]
            y1 = new_pts[i][:, 1] # [l]
            x1_r = torch.flip(x1, dims=[0]) # [l]
            y1_r = torch.flip(y1, dims=[0]) # [l]
            # [TODO] set vis pred as all one
            # vis1 = torch.ones_like(x1)  # [l]
            vis1 = vises[i]
            
            x2s = new_pts[order[1:]][:, :, 0] # [n, l]
            y2s = new_pts[order[1:]][:, :, 1] # [n, l]
            # vis2s = torch.ones_like(x2s)    # [n, l]
            vis2s = vises[order[1:]]

            matched = vis1 * vis2s  # [n, l]
            lengths = matched.sum(dim=1)    # [n]
            
            dis = ((x1 - x2s) ** 2 + (y1 - y2s) ** 2) ** 0.5  # [n, l]
            dis = (dis.sum(dim=1) + 1e-6) / (lengths + 1e-6)  # [n]
            dis_r = ((x1_r - x2s) ** 2 + (y1_r - y2s) ** 2) ** 0.5  # [n, l]
            dis_r = (dis_r.sum(dim=1) + 1e-6) / (lengths + 1e-6)  # [n]

            inds = torch.where(torch.logical_and(dis > nms_thresh, dis_r > nms_thresh))[0]  # [n']
            order = order[inds + 1]   # [n']
        if len(keep):
            return torch.tensor(keep)[:keep_num]
        else:
            return torch.tensor(order)[:keep_num]

    def get_proposal_nms(self, 
                         outs, 
                         topk_num, 
                        #  score_thresh=0.1,
                         score_thresh=0.,
                         nms_thresh=50,
                         ):
        proposal_list = outs['all_pts_preds'][-1] # [12, 50, 20, 2]
        
        cls_scores = outs['all_cls_scores'][-1] # [12, 50, 1]
        cls_scores = cls_scores.sigmoid() # 12, 50, 1
        cls_scores, _ = cls_scores.max(-1)
        
        vis_scores = outs['all_vis_scores'][-1] # [12, 50 * 20, 1]
        vis_scores = vis_scores.sigmoid() > 0.5
        vis_scores = vis_scores.view(*proposal_list.shape[:-1]) # [12, 50, 20]
        
        indexs_batched = []
        for i in range(len(cls_scores)):
            proposals_for_nms = proposal_list[i] # 50, 20, 2
            scores_for_nms = cls_scores[i] # 50
            vis_for_nms = vis_scores[i] # 50, 20
            indexs = self.nms_2d(proposals_for_nms, 
                                 scores_for_nms, 
                                 vises=vis_for_nms,
                                 score_thresh=score_thresh, 
                                 nms_thresh=nms_thresh, 
                                 keep_num=topk_num, 
                                 normalize=True,
                                )
            indexs_batched.append(indexs)
        return indexs_batched

    def forward_pts_train_3d(self, outs, pts_feats, gt_camera_extrinsic,
                            gt_camera_intrinsic, gt_homography_matrix,
                            gt_project_matrix, img_metas):
        # NOTE: deform detr encoder
        pts_feats = outs['mlvl_feats_encoder']
        
        B, N, C, H, W = pts_feats[0].shape
        proposal_list = outs['all_pts_preds'][-1] # [12, 50, 20, 2]
        if self.ref_pts_detach:
            proposal_list = proposal_list.clone().detach()
        query_embeds = outs['query'][-1] # [12, 1000, 256]
        query_embeds = query_embeds.view(*proposal_list.shape[:3], -1) # [12, 50, 20, 256]
        if N > 1: # multi view feat
            # [12, topk, 20, 2] and [12, topk, 20, 256]
            # proposal_list, query_embeds = self.get_proposal(outs, topk=10)
            pts_feats[0] = pts_feats[0].view(B * N, 1, C, H, W)
            gt_camera_extrinsic = gt_camera_extrinsic.view(B * N, 1, 
                                        *gt_camera_extrinsic.shape[-2:]) # 4, 4
            gt_camera_intrinsic = gt_camera_intrinsic.view(B * N, 1, 
                                        *gt_camera_intrinsic.shape[-2:])[:, :, :3, :3] # 3, 3
            gt_homography_matrix = gt_homography_matrix.view(B * N, 1, 
                                        *gt_homography_matrix.shape[-2:])[:, :, :3, :3] # 3, 3
            gt_project_matrix = gt_project_matrix.view(B * N, 1, 
                                        *gt_project_matrix.shape[-2:])[:, :, :3, :] # 3, 4
        # else:
        #     proposal_list = outs['all_pts_preds'][-1] # [12, 50, 20, 2]
        #     query_embeds = outs['query'][-1] # [12, 1000, 256]
        #     query_embeds = query_embeds.view(*proposal_list.shape[:3], -1) # [12, 50, 20, 256]
        # backbone feature [16, 50, 20, 256]
        bbox_feats = self.roi_feature(pts_feats, proposal_list)

        # extra_feats: gt_camera_intrinsic
        viewpad = torch.zeros_like(gt_camera_extrinsic) # 16, 1, 4, 4
        viewpad[:, :, :3, :3] = gt_camera_intrinsic
        viewpad[:, :, 3, 3] = 1. # eye matric
        extra_feats = viewpad.reshape(-1, 1, 1, 16) # 16, 1, 1, 16
        extra_feats = extra_feats.repeat(1, bbox_feats.shape[1], bbox_feats.shape[2], 1) # 16, 50, 20, 16
        extra_feats = extra_feats * self.intrins_feat_scale # scale 0.1

        # reference_points: 16, 50, 20, 3
        # pos_embeds: 16, 50, 256
        reference_points, pos_embeds = None, None
        if self.pos_embed_method == 'uniform':
            pos_embeds = self.pos2posemb2d(proposal_list, num_pos_feats=self.embed_dims//2)
            pos_embeds = self.query_embedding_2d(pos_embeds)
            pos_embeds = pos_embeds.flatten(2)  # bs, num_vec, num_pts * c
            pos_embeds = self.fcs_pos(pos_embeds) # B, Q, C
            pass
        elif self.pos_embed_method == 'ipm' or self.pos_embed_method == 'pred':
            # import pdb; pdb.set_trace()
            reference_points, _ = self.query_generator(bbox_feats, extra_feats, 
                                    img_metas, proposal_list, self.pos_embed_method,
                                    gt_homography_matrix, gt_project_matrix)
            # NOTE: use query_embeds or bbox_feats
            # reference_points, _ = self.query_generator(query_embeds, extra_feats, 
            #                         img_metas, proposal_list, self.pos_embed_method,
            #                         gt_homography_matrix, gt_project_matrix)
            reference_points = self.normalize_ref_pts(reference_points, self.position_range)
            pos_embeds = self.pos2posemb3d(reference_points, num_pos_feats=self.embed_dims//2)
            pos_embeds = self.query_embedding(pos_embeds)
            # pos_embeds = pos_embeds.flatten(2)  # bs, num_vec, num_pts * c
            # pos_embeds = self.fcs_pos(pos_embeds) # B, Q, C
            pass
        elif self.pos_embed_method == 'anchor':
            # [NOTE] to be consistent with petr, position_range is used to normalize ref pts,
            # also, inverse_sigmoid is used to transformer coordinate to logit
            # import pdb; pdb.set_trace()
            reference_points, _ = self.query_generator(bbox_feats, extra_feats, 
                                    img_metas, proposal_list, self.pos_embed_method,
                                    gt_homography_matrix, gt_project_matrix,
                                    self.pts_bbox_head_3d.position_range)
            reference_points = reference_points.flatten(3) # 16, 50, 20, 64 * 3
            # import pdb; pdb.set_trace()
            if self.pe_sample:
                reference_points = reference_points.permute(0, 3, 1, 2).contiguous() # B, C, H, W
                pos_embeds = self.pts_bbox_head_3d.position_encoder(reference_points)
                reference_points = reference_points.permute(0, 2, 3, 1).contiguous() # B, vec, pts, C
                pos_embeds = pos_embeds.permute(0, 2, 3, 1).contiguous() # B, vec, pts, C
            else:
                pos_embeds = self.query_embedding_petr(reference_points)
            # pos_embeds = pos_embeds.flatten(2)  # bs, num_vec, num_pts * c
            # pos_embeds = self.fcs_pos(pos_embeds) # B, Q, C
            pass

        # if self.learn_3d_pe:
        #     out_fov_query_embeds = self.learn_pe_embed.weight.to(pos_embeds.dtype)
        # else:
        #     out_fov_reference_points = self.out_fov_reference_points.weight
        #     out_fov_query_embeds = self.pos2posemb3d(out_fov_reference_points)
        #     out_fov_query_embeds = self.query_embedding(out_fov_query_embeds)
        # out_fov_query_embeds = out_fov_query_embeds.view(1, *pos_embeds.shape[1:])
        # out_fov_query_embeds = out_fov_query_embeds.repeat(B * N, 1, 1, 1)
        vis_scores = outs['all_vis_scores'][-1].sigmoid() # bs, num_vec * num_pts, 1
        vis_scores = vis_scores.view(*pos_embeds.shape[:3], 1)
        # pos_embeds = torch.where(
        #     vis_scores > 0.5,
        #     pos_embeds,
        #     out_fov_query_embeds,
        # ) # bs, num_vec, num_pts, c

        # bbox_feats = bbox_feats.flatten(2)  # bs, num_vec, num_pts * c
        # bbox_feats = self.fcs_pos(bbox_feats) # B, Q, C
        # roi_feat = roi_feat.flatten(2)  # bs, num_vec, num_pts * c
        # roi_feat = self.fcs_pos(roi_feat) # B, Q, C

        # query_embeds = outs['query'][-1].view(bbox_feats.shape)
        # query_embeds = query_embeds.flatten(2)  # bs, num_vec, num_pts * c
        # query_embeds = self.fcs_query(query_embeds) # B, Q, C
        query_embeds_seg = query_embeds.mean(2) # # bs, num_vec, c

        # if self.learn_3d_query:
        #     out_fov_3d_query = self.learn_query_embed.weight.to(query_embeds.dtype)
        #     out_fov_3d_query = out_fov_3d_query.view(1, *query_embeds.shape[1:])
        #     out_fov_3d_query = out_fov_3d_query.repeat(B * N, 1, 1, 1)
        #     query_embeds = torch.where(
        #         vis_scores > 0.5,
        #         query_embeds,
        #         out_fov_3d_query,
        #     ) # bs, num_vec, num_pts, c

        if self.fusion_method == 'query':
            pass
        # elif self.fusion_method == 'roi':
        #     query_embeds = query_embeds + bbox_feats
        # elif self.fusion_method == 'extra':
        #     query_embeds = query_embeds + roi_feat
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        outputs = {}
        if self.mask_head:
            # outputs_mask, outputs_scores = self.sparse_int(pts_feats[0], query_embeds)
            outputs_mask, outputs_scores = self.sparse_int(pts_feats[0], query_embeds_seg)
            outputs['pred_masks'] = outputs_mask
            outputs['pred_scores'] = outputs_scores
        else:
            outputs_mask = pts_feats[0].new_zeros(*proposal_list.shape[:2], H, W) # B * N, Q, H, W
        # if N > 1:
        #     pts_feats[0] = pts_feats[0].view(B, N, C, H, W)
        #     gt_camera_extrinsic = gt_camera_extrinsic.view(B, N, *gt_camera_extrinsic.shape[-2:])
        #     gt_camera_intrinsic = gt_camera_intrinsic.view(B, N, *gt_camera_intrinsic.shape[-2:])
        #     gt_homography_matrix = gt_homography_matrix.view(B, N, *gt_homography_matrix.shape[-2:])
        #     gt_project_matrix = gt_project_matrix.view(B, N, *gt_project_matrix.shape[-2:])
        #     query_embeds = query_embeds.view(B, -1, query_embeds.shape[-1]) # B, N * topk, 256
        #     pos_embeds = pos_embeds.view(B, -1, pos_embeds.shape[-1]) # B, N * topk, 256
        
        # B = 1, N > 1
        pts_feats[0] = pts_feats[0].view(B, N, C, H, W)
        gt_camera_extrinsic = gt_camera_extrinsic.view(B, N, *gt_camera_extrinsic.shape[-2:])
        gt_camera_intrinsic = gt_camera_intrinsic.view(B, N, *gt_camera_intrinsic.shape[-2:])
        gt_homography_matrix = gt_homography_matrix.view(B, N, *gt_homography_matrix.shape[-2:])
        gt_project_matrix = gt_project_matrix.view(B, N, *gt_project_matrix.shape[-2:])    
        # import pdb; pdb.set_trace()
        if self.proposal_cfg_score_thresh > 0:
            proposal_list = proposal_list.reshape(B, -1, *proposal_list.shape[2:])
            query_embeds = query_embeds.reshape(B, -1, *query_embeds.shape[2:])
            reference_points = reference_points.reshape(B, -1, *reference_points.shape[2:])
            pos_embeds = pos_embeds.reshape(B, -1, *pos_embeds.shape[2:])
            vis_scores = vis_scores.reshape(B, -1, *vis_scores.shape[2:])
            outputs_mask = outputs_mask.reshape(B, -1, *outputs_mask.shape[2:])

            cls_scores = outs['all_cls_scores'][-1] # B * N, vec, 1
            cls_scores = cls_scores.view(B, N * self.num_vec, -1)
            cls_scores = cls_scores.sigmoid()
            num_classes = cls_scores.shape[-1]
            cls_scores = cls_scores.flatten(1) # B, N * self.num_vec * cls

            indexes_batched = []
            for i in range(len(cls_scores)):
                scores = cls_scores[i]
                order = scores.argsort(descending=True)
                keep = []
                for o in order:
                    if scores[o] < self.proposal_cfg_score_thresh:
                        break
                    else:
                        keep.append(o)
                if len(keep):
                    indexs = torch.tensor(keep)[:self.topk_2d_proposal]
                else:
                    indexs = torch.tensor(order)[:self.topk_2d_proposal]
                indexs = indexs // num_classes
                indexes_batched.append(indexs)

            def pad_tensor(t, N):
                # B, N, ...
                if t.shape[1] < N:
                    padding = t.new_zeros([t.shape[0], N - t.shape[1], *t.shape[2:]])
                    t = torch.cat((t, padding), dim=1)
                return t

            proposal_list = pad_tensor(proposal_list[:, indexs], self.topk_2d_proposal)
            query_embeds = pad_tensor(query_embeds[:, indexs], self.topk_2d_proposal)
            reference_points = pad_tensor(reference_points[:, indexs], self.topk_2d_proposal)
            pos_embeds = pad_tensor(pos_embeds[:, indexs], self.topk_2d_proposal)
            vis_scores = pad_tensor(vis_scores[:, indexs], self.topk_2d_proposal)
            outputs_mask = pad_tensor(outputs_mask[:, indexs], self.topk_2d_proposal)
        else:
            query_embeds = query_embeds.reshape(B, N * self.topk_2d_proposal, *query_embeds.shape[2:]) # B, N * topk, pts, 256
            reference_points = reference_points.reshape(B, N * self.topk_2d_proposal, *reference_points.shape[2:]) # B, N * topk, pts, 64*3
            pos_embeds = pos_embeds.reshape(B, N * self.topk_2d_proposal, *pos_embeds.shape[2:]) # B, N * topk, pts, 256
            vis_scores = vis_scores.reshape(B, N * self.topk_2d_proposal, *vis_scores.shape[2:]) # B, N * topk, pts, 1

        outputs_3d = self.pts_bbox_head_3d(pts_feats, # img feat
                                    proposal_list, # 2d lane pts
                                    query_embeds.flatten(1, 2), # 2d decoder query
                                    reference_points.flatten(1, 2), # pred ref pts
                                    pos_embeds.flatten(1, 2), # ref pts pos embed
                                    vis_scores.flatten(1, 2), # ref pts vis score
                                    gt_homography_matrix,
                                    gt_project_matrix,
                                    img_metas, 
                                    # outputs['pred_masks'],
                                    outputs_mask,
                                    )
        outputs.update(outputs_3d)
        return outputs

    def forward_pts_train(self,
                          pts_feats,
                          img_metas=None,
                          gt_3dlanes=None, 
                          gt_2dlanes=None,
                          gt_2dboxes=None, 
                          gt_labels=None,
                          gt_labels_3d=None,
                          seg_idx_label=None,
                          gt_camera_extrinsic=None,
                          gt_camera_intrinsic=None,
                          gt_homography_matrix=None,
                          gt_project_matrix=None,
                          gt_topology_lclc=None,
                          gt_te=None, 
                          gt_te_labels=None, 
                          gt_topology_lcte=None,
                          lidar_feat=None,
                        #   gt_bboxes_3d=None,
                        #   gt_labels_3d=None,
                        #   gt_bboxes_ignore=None,   
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # import pdb; pdb.set_trace()
        outs = self.pts_bbox_head(pts_feats, lidar_feat, img_metas, prev_bev)
        outs_dec_2d = outs['query'].clone()
        loss_inputs = [gt_3dlanes, gt_2dlanes, gt_2dboxes, gt_labels, outs]        
        losses, _, _, _ = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        outs_3d = self.forward_pts_train_3d(outs, pts_feats, gt_camera_extrinsic,
                        gt_camera_intrinsic, gt_homography_matrix,
                        gt_project_matrix, img_metas)
        loss, pos_inds_list, pos_assigned_gt_inds, num_total_pos = self.pts_bbox_head_3d.loss(
            gt_3dlanes, outs_3d, img_metas=img_metas)
        losses.update(loss)
        lc_assign_results = dict(
            pos_inds=pos_inds_list[-1],
            pos_assigned_gt_inds=pos_assigned_gt_inds[-1],
        )

        # import pdb; pdb.set_trace()
        self.num_lanes_one2one = self.pts_bbox_head_3d.num_lanes_one2one
        outs_3d = dict(
            outs_dec = outs_3d['outs_dec'][:, :, :self.num_lanes_one2one * self.num_pts_per_vec],
            all_cls_scores = outs_3d['all_cls_scores'][:, :, :self.num_lanes_one2one],
            all_vis_scores = outs_3d['all_vis_scores'][:, :, :self.num_lanes_one2one],
            all_bbox_preds = outs_3d['all_bbox_preds'][:, :, :self.num_lanes_one2one],
            all_pts_preds = outs_3d['all_pts_preds'][:, :, :self.num_lanes_one2one],
        )

        if self.lane_topo:
            lclc_preds_list = self.topo_ll_head(
                outs_3d['outs_dec'], # o1_feats
                outs_3d['outs_dec'], # o2_feats
                outs_3d['all_pts_preds'], # o1_pos
                outs_3d['all_pts_preds'], # o2_pos
                outs_dec_2d,
                outs_dec_2d,
            )
            loss = self.topo_ll_head.loss(
                lclc_preds_list, 
                outs_3d['all_pts_preds'],
                outs_3d['all_pts_preds'],
                [lc_assign_results],
                [lc_assign_results],
                gt_topology_lclc,
            )
            losses.update({
                f'topology_lclc_{key}': val for key, val in loss.items()
            })
        
        if self.traffic:
            pv_feat = []
            for img_feat in pts_feats:
                B, N, output_dim, ouput_H, output_W = img_feat.shape
                pv_feat.append(img_feat[:, 0, ...])
            te_img_metas = [{
                'batch_input_shape': (img_metas[b]['img_shape'][0][0], img_metas[b]['img_shape'][0][1]),
                'img_shape': img_metas[b]['img_shape'][0],
                'scale_factor': img_metas[b]['scale_factor'],
            } for b in range(B)]
            all_te_cls_scores_list, all_te_preds_list, te_outs_dec_list = self.te_head(pv_feat, te_img_metas)
            te_loss_dict, te_assign_results = self.te_head.loss(
                all_te_cls_scores_list, all_te_preds_list, gt_te, gt_te_labels, te_img_metas)
            losses.update({
                f'te_{key}': val for key, val in te_loss_dict.items()
            })
        
            lcte_preds_list = self.topo_lt_head(
                outs_3d['outs_dec'],
                te_outs_dec_list,
                gt_project_matrix,
                outs_dec_2d,
            )
            loss = self.topo_lt_head.loss(
                lcte_preds_list,
                [lc_assign_results],
                te_assign_results,
                gt_topology_lcte,
            )
            losses.update({
                f'topology_lcte_{key}': val for key, val in loss.items()
            })

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # import pdb; pdb.set_trace()
        # kwargs['img_metas'].update({
        #     # 'gt_camera_extrinsic': kwargs['gt_camera_extrinsic'], # 16, 1, 4, 4
        #     # 'gt_camera_intrinsic': kwargs['gt_camera_intrinsic'], # 16, 1, 3, 3
        #     'gt_project_matrix': kwargs['gt_project_matrix'], # 16, 1, 3, 4
        #     'gt_homography_matrix': kwargs['gt_homography_matrix'], # 16, 1, 3, 3
        # })
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, None, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes
    
    @auto_fp16(apply_to=('points'), out_fp32=True)
    def extract_lidar_feat(self,points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        
        return lidar_feat

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                    #   gt_labels=None,
                    #   gt_bboxes=None,
                    #   gt_bboxes_ignore=None,
                      img=None,
                    #   proposals=None,
                    #   img_depth=None,
                    #   img_mask=None,
                      gt_3dlanes=None,
                      gt_2dlanes=None,
                      gt_2dboxes=None,
                      gt_labels=None,
                      seg_idx_label=None,
                      gt_camera_extrinsic=None,
                      gt_camera_intrinsic=None,
                      gt_homography_matrix=None,
                      gt_project_matrix=None,
                      gt_topology_lclc=None,
                      gt_te=None, 
                      gt_te_labels=None, 
                      gt_topology_lcte=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        lidar_feat = None
        if self.modality == 'fusion':
            lidar_feat = self.extract_lidar_feat(points)
        
        ''' single frame
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue>1 else None
        
        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        '''
        
        # import pdb; pdb.set_trace()
        # len(img_feats) = 1
        # img_feats[0].shape = torch.Size([4, 6, 256, 15, 25])
        # bs, cam, C, H, W (480 * 800 -> 15 * 25)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.multiScale:
            img_feats = self.ms2one(img_feats)
            img_feats = [img_feats]
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, gt_3dlanes, 
                                            gt_2dlanes, gt_2dboxes, gt_labels,
                                            gt_labels_3d, seg_idx_label,
                                            gt_camera_extrinsic, gt_camera_intrinsic,
                                            gt_homography_matrix, gt_project_matrix,
                                            gt_topology_lclc, gt_te, gt_te_labels,
                                            gt_topology_lcte)


        losses.update(losses_pts)
        return losses

    def forward_test(self, 
                    points=None,
                    img_metas=None,
                    img=None,
                    **kwargs,
                    # gt_3dlanes=None,
                    # gt_2dlanes=None,
                    # gt_2dboxes=None,
                    # gt_labels=None,
                    # gt_camera_extrinsic=None,
                    # gt_camera_intrinsic=None,
                    ):
        # import pdb; pdb.set_trace()
        '''
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0
        '''

        bbox_results = self.simple_test(img_metas, img, points, prev_bev=None, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        # self.prev_frame_info['prev_pos'] = tmp_pos
        # self.prev_frame_info['prev_angle'] = tmp_angle
        # self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, vis,
                    topo_lclc=None, topo_lcte=None, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to('cpu'),
            vis_3d=vis.to('cpu'),
        )

        # if topo is not None:
        #     result_dict['lclc_topo'] = topo.to('cpu')
        if topo_lclc is not None:
            result_dict['lclc_topo'] = topo_lclc.to('cpu')
        if topo_lcte is not None:
            result_dict['lcte_topo'] = topo_lcte.to('cpu')

        if attrs is not None:
            result_dict['attrs_3d'] = attrs.cpu()

        return result_dict
    
    def simple_test_pts(self, x, lidar_feat, img_metas, 
                    prev_bev=None, rescale=False, **kwargs):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)
        outs_dec_2d = outs['query'].clone()

        # import pdb; pdb.set_trace()
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        
        # len(bbox_list) == 1
        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts, vis)
            for bboxes, scores, labels, pts, vis in bbox_list
        ]
            
        if not self.only_2d:
            outs_3d = self.forward_pts_train_3d(outs, x, kwargs['gt_camera_extrinsic'],
                            kwargs['gt_camera_intrinsic'], kwargs['gt_homography_matrix'],
                            kwargs['gt_project_matrix'], img_metas)
            
            self.num_lanes_one2one = self.pts_bbox_head_3d.num_lanes_one2one
            outs_3d = dict(
                outs_dec = outs_3d['outs_dec'][:, :, :self.num_lanes_one2one * self.num_pts_per_vec],
                all_cls_scores = outs_3d['all_cls_scores'][:, :, :self.num_lanes_one2one],
                all_vis_scores = outs_3d['all_vis_scores'][:, :, :self.num_lanes_one2one],
                all_bbox_preds = outs_3d['all_bbox_preds'][:, :, :self.num_lanes_one2one],
                all_pts_preds = outs_3d['all_pts_preds'][:, :, :self.num_lanes_one2one],
            )

            bbox_list = self.pts_bbox_head_3d.get_bboxes(
                outs_3d, img_metas, rescale=rescale)
            bbox_results = [
                self.pred2result(bboxes, scores, labels, pts, vis)
                for bboxes, scores, labels, pts, vis in bbox_list
            ]

            if self.lane_topo:
                # bs, lc, lc, 1
                lclc_preds_list = self.topo_ll_head(
                    outs_3d['outs_dec'], # o1_feats
                    outs_3d['outs_dec'], # o2_feats
                    outs_3d['all_pts_preds'], # o1_pos
                    outs_3d['all_pts_preds'], # o2_pos
                    outs_dec_2d,
                    outs_dec_2d,
                )
                bbox_list = self.pts_bbox_head_3d.get_bboxes_topo(
                    outs_3d, lclc_preds_list, img_metas, rescale=rescale)
                bbox_results = [
                    self.pred2result(bboxes, scores, labels, pts, vis, topo_lclc)
                    for bboxes, scores, labels, pts, vis, topo_lclc in bbox_list
                ]

            if self.traffic:
                pv_feat = []
                for img_feat in x:
                    B, N, output_dim, ouput_H, output_W = img_feat.shape
                    pv_feat.append(img_feat[:, 0, ...])
                te_img_metas = [{
                    'batch_input_shape': (img_metas[b]['img_shape'][0][0], img_metas[b]['img_shape'][0][1]),
                    'img_shape': img_metas[b]['img_shape'][0],
                    'scale_factor': img_metas[b]['scale_factor'],
                } for b in range(B)]
                all_te_cls_scores_list, all_te_preds_list, te_outs_dec_list = self.te_head(pv_feat, te_img_metas)
                # len(pred_te) = bs = 1
                # pred_te[0] = (box, score), box = (x1, y1, x2, y2, label)
                pred_te = self.te_head.get_bboxes(
                    all_te_cls_scores_list, all_te_preds_list, te_img_metas, rescale=rescale)

                # import mmcv
                # from ...datasets.openlane_v2 import render_corner_rectangle
                # image = mmcv.imread(img_metas[0]['img_paths'][0])
                # for bbox, score in zip(pred_te[0][0][:, :4], pred_te[0][1]):
                #     if score < 0.2:
                #         continue
                #     b = bbox.astype(np.int32)
                #     image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 3, 1)
                # for bbox, score in zip(kwargs['gt_te'][0], kwargs['gt_te_labels'][0]):
                #     bbox = bbox.cpu().numpy()
                #     score = score.cpu().numpy()
                #     b = bbox.astype(np.int32)
                #     image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3, 1)
                # cv2.imwrite('./test.jpg', image)
                
                # bs, lc, te, 1
                lcte_preds_list = self.topo_lt_head(
                    outs_3d['outs_dec'],
                    te_outs_dec_list,
                    kwargs['gt_project_matrix'],
                    outs_dec_2d,
                )
                bbox_list = self.pts_bbox_head_3d.get_bboxes_topo_traffic(
                    outs_3d, lclc_preds_list, lcte_preds_list, img_metas, rescale=rescale)
                bbox_results = [
                    self.pred2result(bboxes, scores, labels, pts, vis, topo_lclc, topo_lcte)
                    for bboxes, scores, labels, pts, vis, topo_lclc, topo_lcte in bbox_list
                ]
                # te
                for idx, re in enumerate(bbox_results):
                    re['pred_te'] = pred_te[idx] # .cpu().numpy()
            return None, bbox_results, None
        return None, bbox_results, None
    
    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality =='fusion':
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.multiScale:
            img_feats = self.ms2one(img_feats)
            img_feats = [img_feats]

        # vis 3d
        bbox_list = [dict() for i in range(len(img_metas))] # bs
        # vis 2d
        # bbox_list = [dict() for i in range(len(img_metas[0]['img_paths']))] # cam num
        # import pdb; pdb.set_trace()
        _, bbox_pts, attn_map = self.simple_test_pts(img_feats, lidar_feat, 
                        img_metas, prev_bev, rescale=rescale, **kwargs)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        # import pdb; pdb.set_trace()
        # if img_metas[0]['sample_idx'] == '315966073049927216':
        #     self.vis_pred_result(dataset='openlanev2', mode='3d', 
        #                         bbox_list=bbox_list, img_metas=img_metas,
        #                         **kwargs,)

        # if self.vis_attn_map:
        #     self.visualize_attention_map(attn_map, img)
        # if self.vis_attn_map:
        #     self.visualize_attention_map(attn_map, img_metas)
        return bbox_list
