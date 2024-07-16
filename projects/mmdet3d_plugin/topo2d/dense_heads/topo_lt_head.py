# --------------------------------------------------------
# Source code for Topo2D
# @Time    : 2024/07/16
# @Author  : Han Li
# bryce18373631@gmail.com
# --------------------------------------------------------

import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
import mmcv
from mmcv.cnn import Linear, bias_init_with_prob, build_activation_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network
from mmcv.runner import auto_fp16, force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from mmdet.models.utils.transformer import inverse_sigmoid


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class TopoLTHead(nn.Module):
    def __init__(self,
                 in_channels_o1,
                 in_channels_o2=None,
                 shared_param=False,
                 loss_rel=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25),
                 add_pos=False,
                 add_2d_query=False,
                 pos_dimension=9,
                 num_pts_per_vec=11,
                 is_detach=False):
        super().__init__()

        self.MLP_o1 = MLP(in_channels_o1, in_channels_o1, 128, 3)
        self.shared_param = shared_param
        if shared_param:
            self.MLP_o2 = self.MLP_o1
        else:
            self.MLP_o2 = MLP(in_channels_o2, in_channels_o2, 128, 3)
        self.classifier = MLP(256, 256, 1, 3)
        self.loss_rel = build_loss(loss_rel)
        self.add_pos = add_pos
        if self.add_pos:
            self.pos_embed = MLP(pos_dimension, 128, 128, 3)
        self.add_2d_query = add_2d_query
        if self.add_2d_query:
            self.MLP_o1_2d = MLP(in_channels_o1, in_channels_o1, 128, 3)
            self.cams_embeds = nn.Parameter(torch.Tensor(7, 128))
            normal_(self.cams_embeds)
        self.is_detach = is_detach
        self.num_pts_per_vec = num_pts_per_vec

    def forward_train(self, o1_feats, o1_assign_results, o2_feats, o2_assign_results, gt_adj):
        rel_pred = self.forward(o1_feats, o2_feats)
        losses = self.loss(rel_pred, gt_adj, o1_assign_results, o2_assign_results)
        return losses

    def get_topology(self, pred_adj_list):
        pred_adj = pred_adj_list.squeeze(-1).sigmoid()
        # pred_adj = pred_adj + 0.3

        # pred_adj_index = pred_adj > 0.5
        # pred_adj[pred_adj_index] = 1.0
        # pred_adj_index_neg = pred_adj <= 0.5
        # pred_adj[pred_adj_index_neg] = 0.0
        return pred_adj.cpu().numpy()

    def forward(self, o1_feats, o2_feats, 
                lidar2img=None, o1_feats_2d=None):
        # feats: [D, B, num_query, num_embedding]
        o1_embeds = o1_feats[-1].clone()
        o2_embeds = o2_feats[-1].clone()
        o1_feat_2d = o1_feats_2d[-1] # .clone()

        if self.is_detach:
            o1_embeds = o1_embeds.detach()
            o2_embeds = o2_embeds.detach()
            o1_feat_2d = o1_feat_2d.detach()

        B, _, dim = o1_embeds.shape
        o1_embeds = o1_embeds.view(B, -1, self.num_pts_per_vec, dim).mean(2) # B, vec, embed
        o1_embeds = self.MLP_o1(o1_embeds)
        o2_embeds = self.MLP_o2(o2_embeds)

        # assert o1_embeds.shape[0] == 1 # bs=1
        if self.add_pos:
            # lidar2img[batch][camera]
            lidar2img = lidar2img[:, 0:1, :3, :3].flatten(2).to(torch.float32).to(o1_embeds.device)
            o1_embeds = self.pos_embed(lidar2img) + o1_embeds
        if self.add_2d_query:
            # import pdb; pdb.set_trace()
            o1_feat_2d = o1_feat_2d.view(B, 7, -1, self.num_pts_per_vec, 256).mean(3) # b, cam, vec, embed
            o1_embeds_2d = self.MLP_o1_2d(o1_feat_2d)
            o1_embeds_2d = o1_embeds_2d + self.cams_embeds.view(1, 7, 1, -1).repeat(B, 1, o1_embeds_2d.shape[2], 1)
            o1_embeds_2d = o1_embeds_2d.flatten(1, 2)
            o1_embeds = o1_embeds + o1_embeds_2d

        num_query_o1 = o1_embeds.size(1)
        num_query_o2 = o2_embeds.size(1)
        o1_tensor = o1_embeds.unsqueeze(2).repeat(1, 1, num_query_o2, 1)
        o2_tensor = o2_embeds.unsqueeze(1).repeat(1, num_query_o1, 1, 1)

        relationship_tensor = torch.cat([o1_tensor, o2_tensor], dim=-1)
        relationship_pred = self.classifier(relationship_tensor)

        return relationship_pred

    def loss(self, rel_preds, o1_assign_results, o2_assign_results, gt_adjs):
        # rel_preds = rel_preds[-1]
        B, num_query_o1, num_query_o2, _ = rel_preds.size()
        o1_assign = o1_assign_results[-1]
        o1_pos_inds = o1_assign['pos_inds']
        o1_pos_assigned_gt_inds = o1_assign['pos_assigned_gt_inds']

        if self.shared_param:
            o2_assign = o1_assign
            o2_pos_inds = o1_pos_inds
            o2_pos_assigned_gt_inds = o1_pos_assigned_gt_inds
        else:
            o2_assign = o2_assign_results[-1]
            o2_pos_inds = o2_assign['pos_inds']
            o2_pos_assigned_gt_inds = o2_assign['pos_assigned_gt_inds']

        # targets = []
        losses_rel = 0
        for i in range(B):
            gt_adj = gt_adjs[i]
            target = torch.zeros_like(rel_preds[i].squeeze(-1), dtype=gt_adj.dtype, device=rel_preds.device)
            xs = o1_pos_inds[i].unsqueeze(-1).repeat(1, o2_pos_inds[i].size(0))
            ys = o2_pos_inds[i].unsqueeze(0).repeat(o1_pos_inds[i].size(0), 1)
            target[xs, ys] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            xs_new = o1_pos_inds[i]
            ys_new = o2_pos_inds[i]
            # target[xs_new, :][:, ys_new] = gt_adj[o1_pos_assigned_gt_inds[i]][:, o2_pos_assigned_gt_inds[i]]
            # targets.append(target)
            # targets = torch.stack(targets, dim=0)

            # import pdb; pdb.set_trace()
            # target = 1 - target[xs_new][:, ys_new].view(-1).long()
            # rel_pred = rel_preds[i][xs_new][:, ys_new].view(-1, 1)
            target = 1 - target[xs_new].view(-1).long()
            rel_pred = rel_preds[i][xs_new].view(-1, 1)
            loss_rel = self.loss_rel(rel_pred, target)

            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_rel = torch.nan_to_num(loss_rel)
            losses_rel += loss_rel

        return dict(loss_rel=losses_rel / B)