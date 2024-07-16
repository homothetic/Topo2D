import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill 

from mmcv.utils import TORCH_VERSION, digit_version


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class MaskBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.hidden_dim
        num_convs = cfg.num_convs
        kernel_dim = cfg.kernel_dim
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


class InstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels, **kwargs):
        super().__init__()
        num_mask = cfg.num_query
        dim = cfg.hidden_dim
        num_classes = cfg.num_classes
        kernel_dim = cfg.kernel_dim
        num_convs = cfg.num_convs
        num_group = cfg.get('num_group', 1)
        sparse_num_group = cfg.get('sparse_num_group', 1)
        self.num_group = num_group
        self.sparse_num_group = sparse_num_group
        self.num_mask = num_mask
        self.inst_convs = _make_stack_3x3_convs(
                            num_convs=num_convs, 
                            in_channels=in_channels, 
                            out_channels=dim)

        self.iam_conv = nn.Conv2d(
            dim * num_group,
            num_group * num_mask * sparse_num_group,
            3, padding=1, groups=num_group * sparse_num_group)
        self.fc = nn.Linear(dim * sparse_num_group, dim)
        # output
        self.mask_kernel = nn.Linear(
            dim, kernel_dim)
        self.cls_score = nn.Linear(
            dim, num_classes)
        self.objectness = nn.Linear(
            dim, 1)
        self.prior_prob = 0.01

    def forward(self, seg_features, is_training=True):
        out = {}
        # SparseInst part
        seg_features = self.inst_convs(seg_features)
        # predict instance activation maps
        iam = self.iam_conv(seg_features.tile(
            (1, self.num_group, 1, 1)))
        if not is_training:
            iam = iam.view(
                iam.shape[0],
                self.num_group,
                self.num_mask * self.sparse_num_group,
                *iam.shape[-2:])
            iam = iam[:, 0, ...]
            num_group = 1
        else:
            num_group = self.num_group

        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        C = seg_features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob_norm_hw = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        # (B x N x HW) @ (B x HW x C) -> B x N x C
        all_inst_features = torch.bmm(
            iam_prob_norm_hw,
            seg_features.view(B, C, -1).permute(0, 2, 1)) #BxNxC

        # concat sparse group features
        inst_features = all_inst_features.reshape(
            B, num_group,
            self.sparse_num_group,
            self.num_mask, -1
        ).permute(0, 1, 3, 2, 4).reshape(
            B, num_group,
            self.num_mask, -1)
        inst_features = F.relu_(
            self.fc(inst_features))

        # avg over sparse group
        iam_prob = iam_prob.view(
            B, num_group,
            self.sparse_num_group,
            self.num_mask,
            iam_prob.shape[-1])
        iam_prob = iam_prob.mean(dim=2).flatten(1, 2)
        inst_features = inst_features.flatten(1, 2)
        out.update(dict(
            iam_prob=iam_prob,
            inst_features=inst_features))
        return out

class SparseInsDecoder(nn.Module):
    def __init__(self, cfg, **kargs) -> None:
        super().__init__()
        in_channels = cfg.encoder.out_dims + 2
        self.output_iam = cfg.decoder.output_iam
        self.scale_factor = cfg.decoder.scale_factor
        self.sparse_decoder_weight = cfg.sparse_decoder_weight
        self.inst_branch = InstanceBranch(cfg.decoder, in_channels)
        # dim, num_convs, kernel_dim, in_channels
        self.mask_branch = MaskBranch(cfg.decoder, in_channels)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features, is_training=True, **kwargs):
        output = {}
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        inst_output = self.inst_branch(
            features, is_training=is_training)
        output.update(inst_output)
        return output

class InstanceBranchMask(nn.Module):
    def __init__(self, cfg, in_channels, **kwargs):
        super().__init__()
        dim = cfg.hidden_dim
        # num_convs = cfg.num_convs
        # num_mask = cfg.num_query
        kernel_dim = cfg.kernel_dim
        # self.num_classes = cfg.num_classes
        
        # self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # self.iam_conv = nn.Conv2d(dim, num_mask, 3, padding=1)

        # outputs
        # self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        # for m in self.inst_convs.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)
        # bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        # for module in [self.iam_conv, self.cls_score]:
        #     init.constant_(module.bias, bias_value)
        # init.normal_(self.iam_conv.weight, std=0.01)
        # init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, inst_features):
        # features = self.inst_convs(features)
        # iam = self.iam_conv(features)
        # iam_prob = iam.sigmoid()

        # B, N = iam_prob.shape[:2]
        # C = features.size(1)
        # # BxNxHxW -> BxNx(HW)
        # iam_prob = iam_prob.view(B, N, -1)
        # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        # iam_prob = iam_prob / normalizer[:, :, None]
        # # aggregate features: BxCxHxW -> Bx(HW)xC
        # inst_features = torch.bmm(iam_prob, 
        #                           features.view(B, C, -1).permute(0, 2, 1)) # B, N, C
        # predict classification & segmentation kernel & objectness
        # pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        # return pred_logits, pred_kernel, pred_scores, iam
        return pred_kernel, pred_scores

def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape # BN, HW
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(1) + (targets * targets).sum(1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise NotImplementedError

def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

class SparseInsDecoderMask(nn.Module):
    def __init__(self, cfg, **kargs) -> None:
        super().__init__()
        in_channels = cfg.encoder.out_dims + 2
        self.scale_factor = cfg.decoder.scale_factor # 32.
        self.inst_branch = InstanceBranchMask(cfg.decoder, in_channels)
        self.mask_branch = MaskBranch(cfg.decoder, in_channels)
        self.loss_mask_dice_weight = cfg.loss_mask_dice_weight
        self.loss_mask_pixel_weight = cfg.loss_mask_pixel_weight
        self.loss_objectness_weight = cfg.loss_objectness_weight

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features, inst_features):
        # import pdb; pdb.set_trace()
        features = features.flatten(0,1) # B, N, C, H, W -> BN, C, H, W
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        # pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        pred_kernel, pred_scores = self.inst_branch(inst_features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1] # B, N, C
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks, scale_factor=self.scale_factor,
            mode='bilinear', align_corners=False)

        return pred_masks, pred_scores # (B, N, H, W), (B, N, 1)

    def loss(self, 
             pred_masks, # 16, 50, 30, 50
             pred_scores, # 16, 50, 1
             seg_idx_label, # B * [N, 480, 800]
             pos_inds_list, # layer, B, N
             pos_assigned_gt_inds, # layer, B, N
             num_total_pos, # layer
            ): 
        loss_dict = {}
        # import pdb; pdb.set_trace()

        # 2d decoder last layer
        pos_inds_list = pos_inds_list[-1]
        pos_assigned_gt_inds = pos_assigned_gt_inds[-1]
        num_total_pos = num_total_pos[-1] # reduce mean

        mask_preds = []
        mask_preds_scores = []
        mask_targets = []
        for bs in range(len(seg_idx_label)):
            mask_pred = pred_masks[bs][pos_inds_list[bs]]
            mask_preds_score = pred_scores[bs][pos_inds_list[bs]]
            mask_target = seg_idx_label[bs][pos_assigned_gt_inds[bs]]
            mask_preds.append(mask_pred)
            mask_preds_scores.append(mask_preds_score)
            mask_targets.append(mask_target)
        mask_preds = torch.cat(mask_preds, dim=0) # BN, 30, 50
        mask_preds_scores = torch.cat(mask_preds_scores, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0) # BN, 480, 800
        mask_targets = F.interpolate(mask_targets[:, None], 
                                         size=mask_preds.shape[-2:], 
                                         mode='bilinear', 
                                         align_corners=False).squeeze(1)

        mask_preds = mask_preds.flatten(1)
        mask_targets = mask_targets.flatten(1)

        with torch.no_grad():
            mask_targets_scores = compute_mask_iou(mask_preds, mask_targets)
        mask_targets_scores = mask_targets_scores.flatten(0)
        mask_preds_scores = mask_preds_scores.flatten(0)

        loss_dice = dice_loss(mask_preds, mask_targets) / num_total_pos
        loss_mask = F.binary_cross_entropy_with_logits(mask_preds, mask_targets, reduction='mean')
        loss_objectness = F.binary_cross_entropy_with_logits(mask_preds_scores, mask_targets_scores, reduction='mean')

        loss_dice = loss_dice * self.loss_mask_dice_weight
        loss_mask = loss_mask * self.loss_mask_pixel_weight
        loss_objectness = loss_objectness * self.loss_objectness_weight

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_dice = torch.nan_to_num(loss_dice)
            loss_mask = torch.nan_to_num(loss_mask)
            loss_objectness = torch.nan_to_num(loss_objectness)

        loss_dict['loss_dice'] = loss_dice
        loss_dict['loss_mask'] = loss_mask
        loss_dict['loss_objectness'] = loss_objectness
        return loss_dict