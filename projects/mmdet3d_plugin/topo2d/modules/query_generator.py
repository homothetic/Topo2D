# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.losses import accuracy
from mmdet3d.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer
import torch.utils.checkpoint as cp
from mmdet.models.utils.transformer import inverse_sigmoid


@HEADS.register_module()
class QueryGenerator(BaseModule):
    def __init__(self,
                 in_channels=256,
                 extra_encoding=dict(
                     num_layers=2,
                     feat_channels=[512, 256],
                     features=[
                         dict(
                             type='intrinsic',
                             in_channels=16,
                         )]
                 ),
                 pos_embed_method='anchor',
                 pc_range=[-50.0, -25.0, -3.0, 50.0, 25.0, 2.0],
                 num_fcs=1,
                 fc_out_channels=1024,
                 img_size=None,
                 init_cfg=None,
                 **kwargs
                 ):
        super(QueryGenerator, self).__init__(init_cfg=init_cfg)
        # fc for x
        self.in_channels = in_channels
        self.num_fcs = num_fcs
        self.fc_out_channels = fc_out_channels

        self.img_size = img_size
        self.relu = nn.ReLU(inplace=True)
        self.fp16_enabled = False

        self.pc_range = pc_range
        self.pos_embed_method = pos_embed_method
        if self.pos_embed_method == 'pred':
            last_layer_dim = self.build_shared_nn()
            self.shared_out_channels = last_layer_dim
            
            # fc for (x + extra)
            self.extra_encoding = extra_encoding
            last_layer_dim = self.build_extra_encoding()
            self.shared_out_channels = last_layer_dim

            # fc for output
            self.reg_predictor_cfg = dict(type='Linear')
            self.build_predictor()

            if init_cfg is None:
                self.init_cfg = []
                self.init_cfg += [
                    dict(
                        type='Xavier',
                        distribution='uniform',
                        override=[
                            dict(name='shared_fcs'),
                            dict(name='extra_enc')])]
                self.init_cfg += [
                    dict(
                        type='Normal', 
                        std=0.001, 
                        override=dict(name='fc_center'))]


    def build_shared_nn(self):
        self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(self.num_fcs, self.in_channels)
        return last_layer_dim

    def _add_conv_fc_branch(self, num_branch_fcs, in_channels):
        last_layer_dim = in_channels
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = (last_layer_dim if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim
    
    def build_extra_encoding(self):
        in_channels = self.shared_out_channels
        feat_channels = self.extra_encoding['feat_channels']
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels] * self.extra_encoding['num_layers']
        else:
            assert len(feat_channels) == self.extra_encoding['num_layers']

        for encoding in self.extra_encoding['features']:
            in_channels = in_channels + encoding['in_channels']

        module = []
        assert self.extra_encoding['num_layers'] > 0
        for i in range(self.extra_encoding['num_layers']):
            module.append(nn.Linear(in_channels, feat_channels[i]))
            # module.append(nn.LayerNorm(feat_channels[i]))
            module.append(nn.ReLU(inplace=True))
            in_channels = feat_channels[i]
        module = nn.Sequential(*module)
        self.extra_enc = module

        return feat_channels[-1]

    def build_predictor(self):
        out_dim_center = 1 # only depth
        self.fc_center = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.shared_out_channels,
            out_features=out_dim_center)
    
    def denormalize_2d_pts(self, pts):
        new_pts = pts.clone()
        new_pts[...,0:1] = pts[..., 0:1] * self.img_size[0]
        new_pts[...,1:2] = pts[..., 1:2] * self.img_size[1]
        return new_pts
    
    @force_fp32(apply_to=('center_pred', ))
    def center2lidar(self, 
                     center_pred, 
                     proposal_list,
                     gt_homography_matrix, gt_project_matrix):
        # center_pred: 16, 50, 20, 1 -> 16000, 1
        # intrinsic: 16, 1, 4, 4 -> 16000, 4, 4
        # extrinsic: 16, 1, 4, 4 -> 16000, 4, 4
        b, vecs, pts, coor = proposal_list.shape
        # intrinsic = intrinsic.repeat(1, vecs * pts, 1, 1).flatten(0, 1)
        # extrinsic = extrinsic.repeat(1, vecs * pts, 1, 1).flatten(0, 1)
        proposal_list = proposal_list.reshape(-1, coor)
        proposal_list = self.denormalize_2d_pts(proposal_list)
        center_pred = center_pred.reshape(-1, 1)
        # NOTE: denormalize depth
        center_pred = center_pred.sigmoid()
        center_pred = center_pred * max(self.pc_range[3], self.pc_range[4]) # max(range_x, range_y)
        center_pred = torch.cat((proposal_list, center_pred), dim=-1)

        center_img = torch.cat([center_pred[:, :2] * center_pred[:, 2:3], center_pred[:, 2:3]], dim=1)
        center_img_hom = torch.cat([center_img, center_img.new_ones([center_img.shape[0], 1])], dim=1)  # [num_rois, 4]
        # lidar2img = torch.bmm(intrinsic, torch.inverse(extrinsic))
        # img2lidar = torch.inverse(lidar2img) # 16000, 4, 4 and 16000, 4, 1
        lidar2imgs = gt_project_matrix # B, N, 3, 4
        lidar2imgs = torch.cat((lidar2imgs, 
                                gt_project_matrix.new_zeros(b, 1, 1, 4)),
                                dim=2)
        lidar2imgs[:, :, 3, 3] = 1.
        img2lidars = torch.inverse(lidar2imgs)
        img2lidars = img2lidars.repeat(1, vecs * pts, 1, 1).flatten(0, 1)
        center_lidar = torch.bmm(img2lidars, center_img_hom[..., None])[:, :3, 0]
        return center_lidar.view(b, vecs, pts, -1) # 16, 50, 20, 3

    def img2lidar(self, 
                  proposal_list,
                  gt_homography_matrix, gt_project_matrix):
        b, vecs, pts, coor = proposal_list.shape
        proposal_list = proposal_list.reshape(-1, coor)
        proposal_list = self.denormalize_2d_pts(proposal_list)

        center_img_hom = torch.cat([proposal_list, proposal_list.new_ones([proposal_list.shape[0], 1])], dim=1)  # [num_rois, 3]
        lidar2imgs = gt_homography_matrix # B, N, 3, 3
        img2lidars = torch.inverse(lidar2imgs)
        img2lidars = img2lidars.repeat(1, vecs * pts, 1, 1).flatten(0, 1)
        center_lidar = torch.bmm(img2lidars, center_img_hom[..., None])[:, :, 0]
        center_lidar = center_lidar[:, :2] / center_lidar[:, 2:3]
        center_lidar = torch.cat([center_lidar, center_lidar.new_zeros([center_lidar.shape[0], 1])], dim=1)
        return center_lidar.view(b, vecs, pts, -1) # 16, 50, 20, 3
    
    def img2lidarPETR_norm(self, 
                  proposal_list,
                #   img_metas,
                  gt_homography_matrix, 
                  gt_project_matrix,
                  position_range):
        b, vecs, pts, coor = proposal_list.shape
        proposal_list = proposal_list.reshape(-1, coor)
        proposal_list = self.denormalize_2d_pts(proposal_list) # N, 2

        depth_start, depth_num = 1, 64
        index = torch.arange(start=0, end=depth_num, step=1, device=proposal_list.device).float()
        index_1 = index + 1
        # x: 100m
        bin_size = (position_range[3] - depth_start) / (depth_num * (1 + depth_num))
        coords_d_x = depth_start + bin_size * index * index_1 # D 
        # y: 50m
        bin_size = (position_range[4] - depth_start) / (depth_num * (1 + depth_num))
        coords_d_y = depth_start + bin_size * index * index_1 # D 

        N, D = b * vecs * pts, depth_num
        proposal_list = proposal_list.view(N, 1, coor).repeat(1, D, 1) # N, D, 2
        # coords_d = coords_d.view(1, D, 1).repeat(N, 1, 1) # N, D, 1
        coords_d_x = coords_d_x.view(1, D, 1) # 1, D, 1
        coords_d_y = coords_d_y.view(1, D, 1) # 1, D, 1
        coords_d = []
        for cam_idx in range(b):
            if cam_idx % 7 in [0, 3, 4]:
                coords_d.append(coords_d_x)
            else:
                coords_d.append(coords_d_y)
        coords_d = torch.cat(coords_d, dim=0) # N, D, 1
        coords_d = coords_d.repeat(vecs * pts, 1, 1)
        
        eps = 1e-5
        coords = torch.cat((proposal_list * torch.maximum(coords_d, torch.ones_like(coords_d) * eps), 
                            coords_d, torch.ones_like(coords_d)), dim=-1) # N, D, 4
        coords = coords.view(N * D, -1, 1) # N * D, 4, 1

        lidar2imgs = gt_project_matrix # B, 1, 3, 4
        lidar2imgs = torch.cat((lidar2imgs, 
                                gt_project_matrix.new_zeros(b, 1, 1, 4)),
                                dim=2)
        lidar2imgs[:, :, 3, 3] = 1.
        img2lidars = torch.inverse(lidar2imgs)
        img2lidars = img2lidars.repeat(1, vecs * pts * D, 1, 1).flatten(0, 1) # N * D, 4, 4

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # N * D, 3
        coords3d[..., 0:1] = (coords3d[..., 0:1] - position_range[0]) / (position_range[3] - position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - position_range[1]) / (position_range[4] - position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - position_range[2]) / (position_range[5] - position_range[2])
        coords3d = coords3d.view(b, vecs, pts, D, 3)
        coords3d = inverse_sigmoid(coords3d) 
        return coords3d

    def img2lidarPETR(self, 
                  proposal_list,
                #   img_metas,
                  gt_homography_matrix, 
                  gt_project_matrix,
                  position_range):
        b, vecs, pts, coor = proposal_list.shape
        proposal_list = proposal_list.reshape(-1, coor)
        proposal_list = self.denormalize_2d_pts(proposal_list) # N, 2

        depth_start, depth_num = 1, 64
        index = torch.arange(start=0, end=depth_num, step=1, device=proposal_list.device).float()
        index_1 = index + 1
        bin_size = (position_range[4] - depth_start) / (depth_num * (1 + depth_num))
        coords_d = depth_start + bin_size * index * index_1 # D 

        N, D = b * vecs * pts, depth_num
        proposal_list = proposal_list.view(N, 1, coor).repeat(1, D, 1) # N, D, 2
        coords_d = coords_d.view(1, D, 1).repeat(N, 1, 1) # N, D, 1
        
        eps = 1e-5
        coords = torch.cat((proposal_list * torch.maximum(coords_d, torch.ones_like(coords_d) * eps), 
                            coords_d, torch.ones_like(coords_d)), dim=-1) # N, D, 4
        coords = coords.view(N * D, -1, 1) # N * D, 4, 1

        lidar2imgs = gt_project_matrix # B, 1, 3, 4
        lidar2imgs = torch.cat((lidar2imgs, 
                                gt_project_matrix.new_zeros(b, 1, 1, 4)),
                                dim=2)
        lidar2imgs[:, :, 3, 3] = 1.
        img2lidars = torch.inverse(lidar2imgs)
        img2lidars = img2lidars.repeat(1, vecs * pts * D, 1, 1).flatten(0, 1) # N * D, 4, 4

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # N * D, 3
        coords3d[..., 0:1] = (coords3d[..., 0:1] - position_range[0]) / (position_range[3] - position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - position_range[1]) / (position_range[4] - position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - position_range[2]) / (position_range[5] - position_range[2])
        coords3d = coords3d.view(b, vecs, pts, D, 3)
        coords3d = inverse_sigmoid(coords3d) 
        return coords3d

    def forward(self, 
                x, 
                extra_feats, 
                img_metas, 
                proposal_list,
                pos_embed_method='ipm',
                gt_homography_matrix=None, 
                gt_project_matrix=None,
                position_range=None):
        '''
        x: 16, 50, 20, 256
        intrinsics: 16, 1, 4, 4
        extrinsics: 16, 1, 4, 4
        extra_feats: 16, 50, 20, 16
        '''
        if pos_embed_method == 'ipm':
            center_lidar = self.img2lidar(proposal_list, gt_homography_matrix, gt_project_matrix)
            return center_lidar, None
        elif pos_embed_method == 'pred':
            roi_feat = self.get_roi_feat(x, extra_feats)
            # TODO: depth loss
            center_pred = self.fc_center(roi_feat)
            center_lidar = self.center2lidar(center_pred, proposal_list, gt_homography_matrix, gt_project_matrix)
            return center_lidar, roi_feat
        elif pos_embed_method == 'anchor':
            center_lidar = self.img2lidarPETR(proposal_list, gt_homography_matrix, gt_project_matrix, position_range)
            return center_lidar, None
        else:
            raise NotImplementedError

    def get_roi_feat(self, x, extra_feats):
        if self.num_fcs > 0:
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # extra encoding
        enc_feat = [x, extra_feats]
        enc_feat = torch.cat(enc_feat, dim=-1).clamp(min=-5e3, max=5e3)
        x = self.extra_enc(enc_feat)
        return x