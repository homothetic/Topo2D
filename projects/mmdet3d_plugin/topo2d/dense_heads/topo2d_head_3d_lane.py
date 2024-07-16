# --------------------------------------------------------
# Source code for Topo2D
# @Time    : 2024/07/16
# @Author  : Han Li
# bryce18373631@gmail.com
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
import numpy as np
from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmcv.utils import TORCH_VERSION, digit_version
import copy
import math
from mmdet.models.utils import NormedLinear

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
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


def normalize_2d_pts(pts, pc_range, pred_bev=False):
    patch_h = (pc_range[4] - pc_range[1]) if pred_bev else (pc_range[5] - pc_range[2]) # y or z
    patch_w = pc_range[3] - pc_range[0] # x
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0] # x
    new_pts[...,1:2] = (pts[..., 1:2] - pc_range[1]) if pred_bev else (pts[..., 1:2] - pc_range[2]) # y or z
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

# def denormalize_2d_pts(pts, pc_range):
#     new_pts = pts.clone()
#     new_pts[...,0:1] = (pts[..., 0:1] * (pc_range[3] -
#                             pc_range[0]) + pc_range[0])
#     new_pts[...,1:2] = (pts[..., 1:2] * (pc_range[5] -
#                             pc_range[2]) + pc_range[2])
#     return new_pts

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

@HEADS.register_module()
class Topo2DHead3DLane(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0),
                 loss_vis=dict(
                        type="CrossEntropyLoss", 
                        use_sigmoid=True, 
                        loss_weight=2.0, 
                        class_weight=1.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 shared_head_params=False,
                 row_column_attn=False,
                 feat_sample_2d_lane=False,
                 feat_size=None,
                 pred_bev=False,
                 use_2d_query=True,
                 ref_pts=False,
                 with_fpe=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        # import pdb; pdb.set_trace()
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.gt_lane_start = 5
        self.gt_lane_len = 200
        self.gt_interval = 5
        self.row_column_attn = row_column_attn
        self.pred_bev = pred_bev
        self.use_2d_query = use_2d_query
        self.ref_pts = ref_pts

        if train_cfg:
            assert 'assigner' in train_cfg
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight']
            assert loss_bbox['loss_weight'] == assigner['reg_cost']['weight']
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.shared_head_params = shared_head_params
        self.feat_sample_2d_lane = feat_sample_2d_lane
        self.feat_size = feat_size
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6 # transformer decoder layer num
        self.normedlinear = normedlinear
        super(Topo2DHead3DLane, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox) # useless
        self.loss_iou = build_loss(loss_iou) # useless
        self.loss_vis = build_loss(loss_vis)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir) # useless

        # import pdb; pdb.set_trace()
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.with_fpe = with_fpe
        if self.with_fpe:
            self.fpe = SELayer(self.embed_dims)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        vis_branch = []
        for _ in range(self.num_reg_fcs):
            vis_branch.append(Linear(self.embed_dims, self.embed_dims))
            vis_branch.append(nn.LayerNorm(self.embed_dims))
            vis_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            vis_branch.append(NormedLinear(self.embed_dims, 1))
        else:
            vis_branch.append(Linear(self.embed_dims, 1))
        fc_vis = nn.Sequential(*vis_branch)

        reg_x_branch = []
        for _ in range(self.num_reg_fcs):
            reg_x_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_x_branch.append(nn.ReLU())
        reg_x_branch.append(Linear(self.embed_dims, 1))
        reg_x_branch = nn.Sequential(*reg_x_branch)

        reg_z_branch = []
        for _ in range(self.num_reg_fcs):
            reg_z_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_z_branch.append(nn.ReLU())
        reg_z_branch.append(Linear(self.embed_dims, 1))
        reg_z_branch = nn.Sequential(*reg_z_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        if self.ref_pts:
            self.shared_head_params = False

        if self.shared_head_params:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
            self.vis_branches = nn.ModuleList(
                [fc_vis for _ in range(self.num_pred)])
            self.reg_x_branches = nn.ModuleList(
                [reg_x_branch for _ in range(self.num_pred)])
            self.reg_z_branches = nn.ModuleList(
                [reg_z_branch for _ in range(self.num_pred)])
        else:
            self.cls_branches = _get_clones(fc_cls, self.num_pred)
            self.vis_branches = _get_clones(fc_vis, self.num_pred)
            self.reg_x_branches = _get_clones(reg_x_branch, self.num_pred)
            self.reg_z_branches = _get_clones(reg_z_branch, self.num_pred)

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        # self.reference_points = nn.Embedding(self.num_query, 3)
        # self.query_embedding = nn.Sequential(
        #     nn.Linear(self.embed_dims*3//2, self.embed_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dims, self.embed_dims),
        # )

        if self.row_column_attn:
            self.num_fc, self.fc_dim = 2, 256
            if self.feat_size is None:
                self.row_num, self.column_num = 15, 25
            else:
                self.row_num, self.column_num = self.feat_size
            fcs_row = []
            for i in range(self.num_fc):
                in_dim = self.fc_dim * self.column_num if i == 0 else self.fc_dim
                fcs_row.append(Linear(in_dim, self.fc_dim))
                fcs_row.append(nn.ReLU(inplace=True))
            self.fcs_row = nn.Sequential(*fcs_row)
            fcs_column = []
            for i in range(self.num_fc):
                in_dim = self.fc_dim * self.row_num if i == 0 else self.fc_dim
                fcs_column.append(Linear(in_dim, self.fc_dim))
                fcs_column.append(nn.ReLU(inplace=True))
            self.fcs_column = nn.Sequential(*fcs_column)
        # if self.ref_pts:
        #     self.reference_points = nn.Sequential(
        #         nn.Linear(self.embed_dims, self.embed_dims),
        #         nn.ReLU(True),
        #         nn.Linear(self.embed_dims, self.embed_dims),
        #         nn.ReLU(True),
        #         nn.Linear(self.embed_dims, 2)
        #     )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        # nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_vis.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.vis_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, 
                          img_feats, 
                          img_metas, 
                          gt_project_matrix, 
                          masks=None,
                          ):
        eps = 1e-5
        # img_metas['ori_shape'][0] = [1280, 1280, ..., 1280] # h
        # img_metas['ori_shape'][1] = [1920, 1920, ..., 1920] # w
        # pad_h, pad_w = img_metas['ori_shape'][0][0], img_metas['ori_shape'][1][0]
        B, N, C, H, W = img_feats[self.position_level].shape
        if N == 1:
            pad_h, pad_w = img_metas[0]['ori_shape'][0], img_metas[0]['ori_shape'][1]
        else:
            pad_h, pad_w = img_metas[0]['pad_shape'][0][0], img_metas[0]['pad_shape'][0][1]
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        # [TODO] check here, self.position_range[3] is x or y axis
        # nuscene forward x, but waymo forward y
        # [NOTE] waymo and nuscene are the same 
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            # bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            bin_size = (self.position_range[4] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            # bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            bin_size = (self.position_range[4] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        # TODO: check coordinate transform
        lidar2imgs = gt_project_matrix # B, N, 3, 4
        lidar2imgs = torch.cat((lidar2imgs, 
                                gt_project_matrix.new_zeros(B, N, 1, 4)),
                                dim=2)
        lidar2imgs[:, :, 3, 3] = 1.
        img2lidars = torch.inverse(lidar2imgs)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        # NOTE: sigmoid -> logit(raw outputs)
        coords3d = inverse_sigmoid(coords3d) 
        coords_position_embeding = self.position_encoder(coords3d)

        if self.row_column_attn:
            # import pdb; pdb.set_trace()
            cpe_row = coords_position_embeding.clone()
            cpe_row = cpe_row.permute(0, 2, 3, 1).contiguous().view(B*N, H, -1)
            cpe_row = self.fcs_row(cpe_row) # B*N, H, C
            cpe_row = cpe_row.view(B*N, H, 1, -1).repeat(1, 1, W, 1)
            cpe_row = cpe_row.permute(0, 3, 1, 2)

            cpw_column = coords_position_embeding.clone()
            cpw_column = cpw_column.permute(0, 3, 2, 1).contiguous().view(B*N, W, -1)
            cpw_column = self.fcs_column(cpw_column)
            cpw_column = cpw_column.view(B*N, 1, W, -1).repeat(1, H, 1, 1)
            cpw_column = cpw_column.permute(0, 3, 1, 2)
            
            # coords_position_embeding = coords_position_embeding + cpe_row
            # coords_position_embeding = coords_position_embeding + cpw_column
            coords_position_embeding = cpe_row + cpw_column
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is Topo2DHead3DLane:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def roi_feature(self, pts_feats, pos_embed, proposal_list, window_size=1):
        '''
        TODO: only support window size == 1
        Params:
            pts_feats: bs, n, c, h, w
            pos_embed: bs, n, c, h, w
            proposal_list: bs, vecs, pts, 2

        Return:
            x_roi: bs, n, c, vecs, pts * ws
            pos_embed_roi: bs, n, c, vecs, pts * ws
        '''
        # import pdb; pdb.set_trace()
        x_roi, pos_embed_roi = [], []
        for pf, pe, pl in zip(pts_feats, pos_embed, proposal_list):
            # pf: 1, C, H, W
            # pe: 1, C, H, W
            # pl: vec, pts, 2
            img_line_pts = pl.clone()
            img_line_pts[..., 1] = 1 - img_line_pts[..., 1]
            img_line_pts = 2 * img_line_pts - 1
            
            # padding_mode: zeros
            sampled_feats = F.grid_sample(pf, img_line_pts[None])  # 1, C, num_vec, num_pts
            x_roi.append(sampled_feats)
            sampled_embeds = F.grid_sample(pe, img_line_pts[None])  # 1, C, num_vec, num_pts
            pos_embed_roi.append(sampled_embeds)

        x_roi = torch.stack(x_roi, axis=0)
        pos_embed_roi = torch.stack(pos_embed_roi, axis=0)
        return x_roi, pos_embed_roi

    def forward(self, 
                mlvl_feats, 
                proposal_list, # 2d lane pts [16, 50, 20, 2]
                target, # 2d decoder query [16, 50, 256]
                reference_points=None, # query pos embed [16, 50, 20, 3]
                query_embeds=None, # query pos embed [16, 50, 256]
                vis_scores=None, # ref pts vis score
                gt_homography_matrix=None,
                gt_project_matrix=None,
                img_metas=None, 
                pred_masks=None,
                ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # import pdb; pdb.set_trace()
        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        # input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        # masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        # for img_id in range(batch_size):
        #     for cam_id in range(num_cams):
        #         img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
        #         masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        # masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)
        masks = x.new_zeros((batch_size, num_cams, *x.shape[-2:])).to(torch.bool) # B, cam, H, W
        if num_cams > 1:
            input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0] # img camera
            masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
            for img_id in range(batch_size):
                for cam_id in range(num_cams):
                    img_h, img_w, _ = img_metas[img_id]['ori_shape'][cam_id]
                    masks[img_id, cam_id, :img_h, :img_w] = 0
            masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, 
                                                                 img_metas, 
                                                                 gt_project_matrix,
                                                                 masks)
            if self.with_fpe:
                # import pdb; pdb.set_trace()
                coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                # [BUG] self.positional_encoding requires dim 3
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                # [BUG] self.positional_encoding requires dim 3
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        # [NOTE] uniform instead of sigmoid
        # reference_points = self.reference_points.weight
        # query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        # reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()
        # if self.ref_pts:
        #     reference_points = self.reference_points(target + query_embeds)
        #     reference_points = reference_points.sigmoid()

        if self.feat_sample_2d_lane:
            # bs, n, c, vec, pts
            # x_roi, pos_embed_roi = self.roi_feature(x, pos_embed, proposal_list)
            # masks = x_roi.new_zeros((batch_size, num_cams, *x_roi.shape[-2:])).to(torch.bool)
            # sample query pe from fmap pe
            # import pdb; pdb.set_trace()
            x_roi, pos_embed_roi = x, pos_embed
            _, query_embeds_sample = self.roi_feature(x, pos_embed, proposal_list)
            query_embeds_sample = query_embeds_sample.flatten(-2) # bs, n, c, vec * pts
            query_embeds_sample = query_embeds_sample.permute(0, 1, 3, 2).contiguous() # bs, n, vec * pts, c
            query_embeds_sample = query_embeds_sample.flatten(1, 2) # bs, n * vec * pts, c
            query_embeds = torch.where(
                vis_scores > 0.5,
                query_embeds_sample,
                query_embeds,
            )
        else:
            # bs, n, c, h, w
            x_roi, pos_embed_roi = x, pos_embed
        # outs_dec, attn_map_for_vis_list, _ = self.transformer(x, masks, target, query_embeds, pos_embed)
        # import pdb; pdb.set_trace()
        if not self.use_2d_query:
            # import pdb; pdb.set_trace()
            # [NOTE] PETR only has query pos embedding
            target = torch.zeros_like(query_embeds)
        outs_dec, attn_map_for_vis_list, _ = self.transformer(x_roi, 
                                                              masks, 
                                                              target, 
                                                              query_embeds, 
                                                              pos_embed_roi,
                                                              pred_masks,
                                                              )
        outs_dec = torch.nan_to_num(outs_dec) # (layer, B, Q, C)
        outputs_classes = []
        outputs_coords = []
        outputs_vis = []
        outputs_pts_coords = []
        for lvl in range(outs_dec.shape[0]):
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]
                                            # .view(batch_size * num_cams, self.num_query, self.num_pts_per_vec, -1)
                                            .view(batch_size, self.num_query, self.num_pts_per_vec, -1)
                                            .mean(2)) # B, Q, class
            vis = self.vis_branches[lvl](outs_dec[lvl]) # B, Q, num_pts_per_vec
            # vis = vis.view(batch_size * num_cams, self.num_query, self.num_pts_per_vec)
            vis = vis.view(batch_size, self.num_query, self.num_pts_per_vec)
            outputs_coord = torch.zeros_like(vis)[:, :, :4] # B, Q, 4

            outputs_pts_x = self.reg_x_branches[lvl](outs_dec[lvl]) # B, Q, 20
            outputs_pts_z = self.reg_z_branches[lvl](outs_dec[lvl]) # B, Q, 20
            # outputs_pts_x = outputs_pts_x.view(batch_size * num_cams, self.num_query, self.num_pts_per_vec)
            # outputs_pts_z = outputs_pts_z.view(batch_size * num_cams, self.num_query, self.num_pts_per_vec)
            outputs_pts_x = outputs_pts_x.view(batch_size, self.num_query, self.num_pts_per_vec)
            outputs_pts_z = outputs_pts_z.view(batch_size, self.num_query, self.num_pts_per_vec)
            outputs_pts_coord = torch.stack((outputs_pts_x, outputs_pts_z), dim=3) # B, Q, 20, 2
            # outputs_pts_coord = outputs_pts_coord.sigmoid() # NOTE: sigmoid here
            if self.ref_pts and lvl > 0:
                reference = inverse_sigmoid(reference)
                outputs_pts_coord = outputs_pts_coord + reference
            # [BUG]
            outputs_pts_coord = outputs_pts_coord.sigmoid()
            reference = outputs_pts_coord.clone().detach()

            outputs_classes.append(outputs_class)
            outputs_vis.append(vis)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        # all_cls_scores = torch.stack(outputs_classes)
        # all_bbox_preds = torch.stack(outputs_coords)

        # [NOTE] do not denormalize here
        # all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        # all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        # all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # outs = {
        #     'all_cls_scores': all_cls_scores,
        #     'all_bbox_preds': all_bbox_preds,
        #     'enc_cls_scores': None,
        #     'enc_bbox_preds': None, 
        # }
        # return outs
    
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outputs_vis = torch.stack(outputs_vis)
        outs = {
            'all_cls_scores': outputs_classes, # [24, 50, 20] * 6
            'all_vis_scores': outputs_vis, # [24, 50, 20] * 6
            'all_bbox_preds': outputs_coords, # [24, 50, 4] * 6
            'all_pts_preds': outputs_pts_coords, # [24, 50, 20, 2] * 6
            'attn_map': attn_map_for_vis_list, # [num_layers, bs, num_query, n * h * w]
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_vis, 
                           gt_bboxes_ignore=None,
                           pos_inds_list_2d=None, 
                           pos_assigned_gt_inds_2d=None,
                           ):
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]
        if self.pred_bev:
            assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                                gt_bboxes, gt_labels, gt_shifts_pts, gt_bboxes_ignore)
        else:
            assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                                gt_bboxes, gt_labels, gt_shifts_pts, gt_vis,
                                                gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
        if pos_inds_list_2d is not None:
            pos_inds = pos_inds_list_2d
            # neg_inds = torch.arange(self.num_query).to(pos_inds.device) # int64
            neg_inds = sampling_result.neg_inds # useless
            pos_assigned_gt_inds = pos_assigned_gt_inds_2d
        else:
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds] = 1.0

        # pts target
        # assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]
        pts_targets = torch.zeros_like(pts_pred)
        pts_weights = torch.zeros_like(pts_targets)
        # pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        # pts_weights[pos_inds] = (gt_vis[sampling_result.pos_assigned_gt_inds] > 0.5).to(torch.float32)[..., None]
        pts_targets[pos_inds] = gt_shifts_pts[pos_assigned_gt_inds, assigned_shift, :, :]
        pts_weights[pos_inds] = (gt_vis[pos_assigned_gt_inds] > 0.5).to(torch.float32)[..., None]

        # vis target [NOTE: attention here]
        vis_targets = pts_pred.new_ones((pts_pred.size(0), pts_pred.size(1)))
        vis_weights = torch.zeros_like(vis_targets)
        # vis_targets[pos_inds] = (gt_vis[sampling_result.pos_assigned_gt_inds] <= 0.5).to(torch.float32)
        vis_targets[pos_inds] = (gt_vis[pos_assigned_gt_inds] <= 0.5).to(torch.float32)
        vis_weights[pos_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights, vis_targets, vis_weights,
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_vis_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]

        if self.pos_inds_list is not None:
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
            pts_targets_list, pts_weights_list,vis_targets, vis_weights,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
                gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_vis_list, gt_bboxes_ignore_list,
                self.pos_inds_list, self.pos_assigned_gt_inds)
        else:
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, 
            pts_targets_list, pts_weights_list,vis_targets, vis_weights,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
                gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_vis_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list, vis_targets, vis_weights,
                num_total_pos, num_total_neg, pos_inds_list, pos_assigned_gt_inds)

    def loss_single(self,
                    cls_scores,
                    vis_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_vis_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        vis_scores_list = [vis_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                           gt_vis_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, vis_targets_list, vis_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, pos_assigned_gt_inds) = cls_reg_targets
        
        # [TODO] carefully check loss single part

        labels = torch.cat(labels_list, 0) # [800] int64
        label_weights = torch.cat(label_weights_list, 0) # float32
        # bbox_targets = torch.cat(bbox_targets_list, 0)
        # bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0) # 800, 20, 2
        pts_weights = torch.cat(pts_weights_list, 0)
        vis_targets = torch.cat(vis_targets_list, 0) # 800, 20
        vis_weights = torch.cat(vis_weights_list, 0)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels) # 800, 20
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # visable classification loss
        vis_scores = vis_scores.reshape(-1, 1) # 16 * 50 * 20, 1
        vis_targets = vis_targets.view(-1).to(labels.dtype) # 16 * 50 * 20, int64
        vis_weights = vis_weights.view(-1)
        loss_vis = self.loss_vis(
            vis_scores, vis_targets, vis_weights, avg_factor=num_total_pos * self.num_pts_per_vec)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        # normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        # normal x & y or x & z
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range, self.pred_bev)
        loss_pts = self.loss_pts(
            pts_preds, normalized_pts_targets, pts_weights, avg_factor=num_total_pos)

        loss_bbox = torch.zeros_like(loss_cls)
        loss_dir = torch.zeros_like(loss_cls)
        loss_iou = torch.zeros_like(loss_cls)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_vis = torch.nan_to_num(loss_vis)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_vis, loss_bbox, loss_iou, loss_pts, loss_dir, \
            pos_inds_list, pos_assigned_gt_inds, num_total_pos

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
            #  gt_bboxes_list,
            #  gt_labels_list,
             gt_3dlanes, 
            #  gt_2dlanes, 
            #  gt_2dboxes, 
            #  gt_labels,
             preds_dicts,
             pos_inds_list=None, 
             pos_assigned_gt_inds=None, 
             num_total_pos=None,
             gt_bboxes_ignore=None,
             img_metas=None):
        
        all_cls_scores = preds_dicts['all_cls_scores'] # [B, 50, class] * 6
        all_bbox_preds = preds_dicts['all_bbox_preds'] # [B, 50, 4] * 6
        all_pts_preds  = preds_dicts['all_pts_preds'] # [B, 50, 20, 2] * 6
        all_vis_scores = preds_dicts['all_vis_scores'] # [B, 50, 20] * 6

        if pos_inds_list is not None:
            self.pos_inds_list = pos_inds_list[-1]
            self.pos_assigned_gt_inds = pos_assigned_gt_inds[-1]
            self.num_total_pos = num_total_pos[-1]
        else:
            self.pos_inds_list = pos_inds_list
            self.pos_assigned_gt_inds = pos_assigned_gt_inds
            self.num_total_pos = num_total_pos

        num_dec_layers = len(all_cls_scores)
        device = all_cls_scores.device

        '''
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            shape [layer, batch, instance, 4].

        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
            shape [layer, batch, instance].

        gt_pts_list (list[Tensor]): Ground truth pts for each image
            with shape (num_gts, fixed_num, 2) in [x,y] format.
            shape [layer, batch, instance * 19 * 20 * 2].
        '''

        gt_bboxes_list = []
        gt_labels_list = []
        gt_vis_list = []
        gt_shifts_pts_list = []

        if gt_3dlanes[0].dim() == 2: # N, 605
            for gt_3dlane in gt_3dlanes:
                # import pdb; pdb.set_trace()
                # valid_lanes = torch.where(gt_3dlane[:, 1] > 0)
                # gt_pts = gt_3dlane[valid_lanes] # N, 605
                # x_target = gt_pts[:, self.gt_lane_start : self.gt_lane_start + self.gt_lane_len]
                # z_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len : self.gt_lane_start + self.gt_lane_len * 2]
                # vis_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len * 2 : ]
                
                # # anchor y step
                # if self.num_pts_per_vec == 20:
                #     # 4, 9, ..., 99 (100)
                #     indices = torch.arange(4, 100, 5).to(torch.long).to(gt_pts.device) # 20
                # elif self.num_pts_per_vec == 21:
                #     # 2, 7, ..., 97, 102 (103)
                #     indices = torch.arange(2, 103, 5).to(torch.long).to(gt_pts.device) # 21
                # x_target = x_target.index_select(1, indices)
                # z_target = z_target.index_select(1, indices)
                # vis_target = vis_target.index_select(1, indices)   # [N, 20]
                # labels = gt_pts[:, 1].to(torch.int64) - 1 # [1, 20] -> [0, 19]
                labels = gt_3dlane[:, 0].to(torch.int64) - 1
                x_target = gt_3dlane[:, 1 : 1 + self.num_pts_per_vec]
                z_target = gt_3dlane[:, 1 + self.num_pts_per_vec : 1 + self.num_pts_per_vec * 2]
                vis_target = gt_3dlane[:, 1 + self.num_pts_per_vec * 2:]

                gt_shifts_pts = torch.stack((x_target[:, None, :], z_target[:, None, :]), dim=3)
                gt_shifts_pts_list.append(gt_shifts_pts)
                gt_labels_list.append(labels)
                gt_vis_list.append(vis_target)
                gt_bboxes_list.append(torch.zeros_like(vis_target)[:, :4])
        else: # N, direction, pts, coor
            for gt_3dlane in gt_3dlanes:
                # N, 2, 20, 2
                gt_shifts_pts_list.append(gt_3dlane) # x & y attention
                gt_labels_list.append(torch.zeros_like(gt_3dlane)[:, 0, 0, 0].view(-1).to(torch.int64))
                gt_vis_list.append(torch.ones_like(gt_3dlane)[:, 0, :, 0].squeeze(-1)) # all vis
                gt_bboxes_list.append(torch.zeros_like(gt_3dlane)[:, 0, :4, 0].squeeze(-1))

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_vis_list = [gt_vis_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        
        # import pdb;pdb.set_trace()
        losses_cls, losses_vis, losses_bbox, losses_iou, losses_pts, losses_dir, \
            pos_inds_list, pos_assigned_gt_inds, num_total_pos = multi_apply(
            self.loss_single, all_cls_scores, all_vis_scores, all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list, all_gt_vis_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict['loss_3d_cls'] = losses_cls[-1]
        loss_dict['loss_3d_vis'] = losses_vis[-1]
        # loss_dict['loss_3d_bbox'] = losses_bbox[-1]
        # loss_dict['loss_3d_iou'] = losses_iou[-1]
        loss_dict['loss_3d_pts'] = losses_pts[-1]
        # loss_dict['loss_3d_dir'] = losses_dir[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_vis_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_vis[:-1],
                                           losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_3d_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_3d_vis'] = loss_vis_i
            # loss_dict[f'd{num_dec_layer}.loss_3d_bbox'] = loss_bbox_i
            # loss_dict[f'd{num_dec_layer}.loss_3d_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_3d_pts'] = loss_pts_i
            # loss_dict[f'd{num_dec_layer}.loss_3d_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict, pos_inds_list, pos_assigned_gt_inds, num_total_pos

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # import pdb; pdb.set_trace()
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']
            vis = preds['vis'] if 'vis' in preds.keys() else torch.ones_like(pts)

            ret_list.append([bboxes, scores, labels, pts, vis])

        return ret_list
