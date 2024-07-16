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
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version


def normalize_2d_bbox(bboxes, img_size):
    patch_h = img_size[1] # 1280
    patch_w = img_size[0] # 1920
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    # cxcywh_bboxes[...,0:1] = cxcywh_bboxes[...,0:1] - pc_range[0]
    # cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    # normalized_bboxes = bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, img_size):
    patch_h = img_size[1] # 1280
    patch_w = img_size[0] # 1920
    new_pts = pts.clone()
    # new_pts[...,0:1] = pts[...,0:1] - pc_range[0]
    # new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, img_size):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * img_size[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * img_size[1]
    return bboxes

def denormalize_2d_pts(pts, img_size):
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1]* img_size[0]
    new_pts[...,1:2] = pts[..., 1:2] * img_size[1]
    return new_pts

@HEADS.register_module()
class Topo2DHead2D(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 org_img_size=None,
                 reg_sigmoid=True,
                 output_vis=True,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_lanes_one2one=20,
                 k_one2many=5,
                 lambda_one2many=2.0,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 loss_pts=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_vis=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.reg_sigmoid = reg_sigmoid
        self.output_vis = output_vis

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        # self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        # import pdb; pdb.set_trace()
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.org_img_size = org_img_size

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.num_lanes_one2one = num_lanes_one2one
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        
        super(Topo2DHead2D, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        self.loss_vis = build_loss(loss_vis)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        # self.positional_encoding = None
        self._init_layers()

        self.pos_inds_list =  None
        self.pos_assigned_gt_inds =  None
        self.num_total_pos = None

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        if self.output_vis:
            vis_branch = []
            for _ in range(self.num_reg_fcs):
                vis_branch.append(Linear(self.embed_dims, self.embed_dims))
                vis_branch.append(nn.LayerNorm(self.embed_dims))
                vis_branch.append(nn.ReLU(inplace=True))
            vis_branch.append(Linear(self.embed_dims, 1))
            # vis_branch.append(Linear(self.embed_dims, self.num_pts_per_vec))
            fc_vis = nn.Sequential(*vis_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        assert self.with_box_refine
        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        if self.output_vis:
            self.vis_branches = _get_clones(fc_vis, num_pred)

        # if self.with_box_refine:
        #     self.cls_branches = _get_clones(fc_cls, num_pred)
        #     self.reg_branches = _get_clones(reg_branch, num_pred)
        # else:
        #     self.cls_branches = nn.ModuleList(
        #         [fc_cls for _ in range(num_pred)])
        #     self.reg_branches = nn.ModuleList(
        #         [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            # if self.bev_encoder_type == 'BEVFormerEncoder':
            #     self.bev_embedding = nn.Embedding(
            #         self.bev_h * self.bev_w, self.embed_dims)
            # else:
            #     self.bev_embedding = None
            self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)
        if self.output_vis and self.loss_vis.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.vis_branches:
                nn.init.constant_(m[-1].bias, bias_init)
    
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # import pdb;pdb.set_trace()
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        input_img_h, input_img_w, _ = img_metas[0]['img_shape'][0]
        img_masks = mlvl_feats[0].new_zeros((bs, num_cam, input_img_h, input_img_w))
        # img_masks = mlvl_feats[0].new_ones((bs, num_cam, input_img_h, input_img_w))
        # for img_id in range(bs):
        #     for cam_id in range(num_cam):
        #         img_h, img_w, _ = img_metas[img_id]['ori_shape'][cam_id]
        #         img_masks[img_id, cam_id, :img_h, :img_w] = 0
        img_masks = img_masks.flatten(0, 1) # bs * cam

        mlvl_feats_multiview = []
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_feats_multiview.append(feat.flatten(0, 1)) # bs * cam
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        # import pdb; pdb.set_trace()
        self_attn_mask = torch.zeros(
            [self.num_vec * self.num_pts_per_vec, self.num_vec * self.num_pts_per_vec]
        ).bool().to(img_masks.device)
        self_attn_mask[self.num_lanes_one2one * self.num_pts_per_vec:, 
            : self.num_lanes_one2one * self.num_pts_per_vec] = True
        self_attn_mask[: self.num_lanes_one2one * self.num_pts_per_vec, 
            self.num_lanes_one2one * self.num_pts_per_vec:] = True

        outputs = self.transformer(
            mlvl_feats_multiview,
            mlvl_masks,
            object_query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            attn_masks=[self_attn_mask, None],
        )

        # import pdb; pdb.set_trace()
        bev_embed, hs, init_reference, inter_references = outputs
        mlvl_feats_encoder = []
        for feat in bev_embed:
            mlvl_feats_encoder.append(feat.view(bs, num_cam, *feat.shape[1:]))

        hs = hs.permute(0, 2, 1, 3) # 6, 1000, 24, 256 -> 6, 24, 1000, 256(layer, bs*cam, Q, C)
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        outputs_vis = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs*num_cam, self.num_vec, self.num_pts_per_vec, -1)
                                            .mean(2))

            # [BUG] requires_grad is false
            # reference = inter_references[lvl]
            # outputs_coord, outputs_pts_coord = self.transform_box(reference)

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            tmp = self.reg_branches[lvl](hs[lvl])
            if self.reg_sigmoid:
                reference = inverse_sigmoid(reference)
                tmp[..., 0:2] += reference[..., 0:2]
                tmp = tmp.sigmoid()
            else:
                tmp[..., 0:2] += reference[..., 0:2]
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)

            if self.output_vis:
                vis = self.vis_branches[lvl](hs[lvl]) # bs*num_cam, num_vec*num_pts_per_vec, 1
                # vis = self.vis_branches[lvl](hs[lvl]
                #                             .view(bs*num_cam, self.num_vec, self.num_pts_per_vec, -1)
                #                             .mean(2))
            else:
                # TODO: check this, tensor or parameter
                # vis = torch.zeros((bs * num_cam, self.num_vec * self.num_pts_per_vec, 1))
                # [NOTE] all pts are visible
                vis = hs.new_ones((bs * num_cam, self.num_vec * self.num_pts_per_vec, 1))
                # vis = torch.zeros((bs * num_cam, self.num_vec, self.num_pts_per_vec))
                # vis = nn.Parameter(vis, requires_grad=False)

            outputs_vis.append(vis)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outputs_vis = torch.stack(outputs_vis)
        outs = {
            'query': hs, # layer, bs * cam, Q, C
            'mlvl_feats_encoder': mlvl_feats_encoder, # None
            'all_cls_scores': outputs_classes, # [24, 50, 1] * 6
            'all_vis_scores': outputs_vis, # [24, 50, 20] * 6
            'all_bbox_preds': outputs_coords, # [24, 50, 4] * 6
            'all_pts_preds': outputs_pts_coords, # [24, 50, 20, 2] * 6
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None
        }

        return outs
    
    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None,
                           pos_inds_list_2d=None, 
                           pos_assigned_gt_inds_2d=None,
                           ):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
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
        # bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            # assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
            assigned_shift = gt_labels[pos_assigned_gt_inds]
        else:
            # assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
            assigned_shift = order_index[pos_inds, pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        if len(pos_inds):
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            # pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
            pts_targets[pos_inds] = gt_shifts_pts[pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds, sampling_result.pos_assigned_gt_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        if self.pos_inds_list is not None:
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pts_targets_list, pts_weights_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
                gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list,
                self.pos_inds_list, self.pos_assigned_gt_inds)
        else:
            (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, pts_targets_list, pts_weights_list,
            pos_inds_list, neg_inds_list, pos_assigned_gt_inds) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
                gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg, pos_inds_list, pos_assigned_gt_inds)

    def loss_single(self,
                    cls_scores,
                    vis_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        vis_scores_list = [vis_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list, num_total_pos, num_total_neg,
         pos_inds_list, pos_assigned_gt_inds) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        # from IPython import embed; embed()
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # import pdb;pdb.set_trace()
        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        # gt_bboxes: xyxy, denormalize
        # bbox_pred: xywh, normalize
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.org_img_size)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        # loss_bbox = self.loss_bbox(
        #     bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
        #             :4], bbox_weights[isnotnan, :4], avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.org_img_size)

        loss_bbox = torch.zeros_like(loss_cls)
        loss_iou = torch.zeros_like(loss_cls)
        loss_vis = torch.zeros_like(loss_cls)
        if self.output_vis:
            # [TODO]: visable classification loss
            vis_cls_scores = torch.cat(vis_scores_list, 0) # 16, 50, 20
            vis_cls_scores = vis_cls_scores.view(-1, 1) # 16*50, 1
            vis_labels = torch.ones_like(vis_cls_scores.view(-1)).long() # 16*50
            mask = ((normalized_pts_targets > 0) & (normalized_pts_targets < 1)).all(-1).view(-1) # 16*50
            vis_labels[mask] = 0 # gt label is zero[TODO: check this]
            vis_label_weights = torch.zeros_like(vis_labels).float()
            vis_label_weights[(pts_weights != 0).all(-1).view(-1)] = 1 # 16*50
            loss_vis = self.loss_vis(
                vis_cls_scores, vis_labels, vis_label_weights, avg_factor=num_total_pos * self.num_pts_per_vec
            )

            # [TODO] add out of fov mask
            mask = ((normalized_pts_targets <= 0) | (normalized_pts_targets >= 1)).any(-1) # 160, 20
            pts_weights[mask] = 0

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.org_img_size)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        # bboxes = denormalize_2d_bbox(bbox_preds, self.org_img_size)
        # # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(
        #     bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
        #     avg_factor=num_total_pos)

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
             gt_2dlanes, 
             gt_2dboxes, 
             gt_labels,
             preds_dicts,
             pos_inds_list=None, 
             pos_assigned_gt_inds=None, 
             num_total_pos=None,
             gt_bboxes_ignore=None,
             img_metas=None):
        
        # import pdb; pdb.set_trace()

        # pred
        all_cls_scores = preds_dicts['all_cls_scores'][:, :, :self.num_lanes_one2one]
        all_bbox_preds = preds_dicts['all_bbox_preds'][:, :, :self.num_lanes_one2one]
        all_pts_preds  = preds_dicts['all_pts_preds'][:, :, :self.num_lanes_one2one]
        all_vis_scores = preds_dicts['all_vis_scores'][:, :, :self.num_lanes_one2one * self.num_pts_per_vec]

        all_cls_scores_one2many = preds_dicts['all_cls_scores'][:, :, self.num_lanes_one2one:]
        all_bbox_preds_one2many = preds_dicts['all_bbox_preds'][:, :, self.num_lanes_one2one:]
        all_pts_preds_one2many  = preds_dicts['all_pts_preds'][:, :, self.num_lanes_one2one:]
        all_vis_scores_one2many = preds_dicts['all_vis_scores'][:, :, self.num_lanes_one2one * self.num_pts_per_vec:]
        
        # assert openlanev2 dataset
        num_dec_layers, bs, num_vec, num_cls = all_cls_scores.shape
        assert bs == 7
        assert num_cls == 1

        # one2one
        gt_bboxes_list = []
        gt_labels_list = []
        gt_shifts_pts_list = []
        for b in range(len(gt_labels)):
            for c in range(len(gt_labels[b])):
                gt_labels_list.append(gt_labels[b][c])
                gt_bboxes_list.append(gt_2dboxes[b][c])
                gt_shifts_pts_list.append(gt_2dlanes[b][c])
        
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        
        # one2many
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []
        one2many_gt_shifts_pts_list = []

        for gt_bboxes in gt_bboxes_list:
            if len(gt_bboxes):
                one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))
            else:
                one2many_gt_bboxes_list.append(gt_bboxes)
        for gt_labels in gt_labels_list:
            if len(gt_labels):
                one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))
            else:
                one2many_gt_labels_list.append(gt_labels)
        for gt_shifts_pts in gt_shifts_pts_list:
            if len(gt_shifts_pts):
                one2many_gt_shifts_pts_list.append(gt_shifts_pts.repeat(self.k_one2many, 1, 1, 1))
            else:
                one2many_gt_shifts_pts_list.append(gt_shifts_pts)
        all_gt_bboxes_list_one2many = [one2many_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [one2many_gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list_one2many = [one2many_gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list_one2many = all_gt_bboxes_ignore_list

        # loss
        losses_cls, _, _, _, losses_pts, losses_dir, _, _, _ = multi_apply(
            self.loss_single, all_cls_scores, all_vis_scores, 
            all_bbox_preds, all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_shifts_pts_list, all_gt_bboxes_ignore_list)
        
        # import pdb; pdb.set_trace()
        losses_cls_one2many, _, _, _, losses_pts_one2many, losses_dir_one2many, _, _, _ = multi_apply(
            self.loss_single, all_cls_scores_one2many, all_vis_scores_one2many, 
            all_bbox_preds_one2many, all_pts_preds_one2many,
            all_gt_bboxes_list_one2many, all_gt_labels_list_one2many, 
            all_gt_shifts_pts_list_one2many, all_gt_bboxes_ignore_list_one2many)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        loss_dict['loss_cls_h'] = losses_cls_one2many[-1] * self.lambda_one2many
        loss_dict['loss_pts_h'] = losses_pts_one2many[-1] * self.lambda_one2many
        loss_dict['loss_dir_h'] = losses_dir_one2many[-1] * self.lambda_one2many
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1], losses_pts[:-1], losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_cls_h'] = losses_cls_one2many[num_dec_layer] * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_pts_h'] = losses_pts_one2many[num_dec_layer] * self.lambda_one2many
            loss_dict[f'd{num_dec_layer}.loss_dir_h'] = losses_dir_one2many[num_dec_layer] * self.lambda_one2many
            num_dec_layer += 1

        return loss_dict, _, _, _

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        # dummy
        # assert False

        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']
            vis = preds['vis']

            ret_list.append([bboxes, scores, labels, pts, vis])

        return ret_list

