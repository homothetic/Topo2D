import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from typing import List
from mmdet.models.utils.transformer import Transformer
import math
import warnings
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention


@TRANSFORMER.register_module()
class MapTRTransformer2DMlvl(Transformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 only_decoder=True,
                 **kwargs):
        super(MapTRTransformer2DMlvl, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.decoder.embed_dims
        self.only_decoder = only_decoder
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        if self.only_decoder:
            self.encoder = None # only decoder

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # if not self.only_decoder:
        normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid, has shape \
                    (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.only_decoder:
                feat = feat + self.level_embeds[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        if not self.only_decoder:
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
            memory = self.encoder(
                query=feat_flatten,
                key=None,
                value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                **kwargs)
        else:
            memory = feat_flatten

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        # if self.as_two_stage:
        #     output_memory, output_proposals = \
        #         self.gen_encoder_output_proposals(
        #             memory, mask_flatten, spatial_shapes)
        #     enc_outputs_class = cls_branches[self.decoder.num_layers](
        #         output_memory)
        #     enc_outputs_coord_unact = \
        #         reg_branches[
        #             self.decoder.num_layers](output_memory) + output_proposals

        #     topk = self.two_stage_num_proposals
        #     topk_proposals = torch.topk(
        #         enc_outputs_class[..., 0], topk, dim=1)[1]
        #     topk_coords_unact = torch.gather(
        #         enc_outputs_coord_unact, 1,
        #         topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords_unact = topk_coords_unact.detach()
        #     reference_points = topk_coords_unact.sigmoid()
        #     init_reference_out = reference_points
        #     pos_trans_out = self.pos_trans_norm(
        #         self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        #     query_pos, query = torch.split(pos_trans_out, c, dim=2)
        # else:
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        # if self.as_two_stage:
        #     return inter_states, init_reference_out,\
        #         inter_references_out, enc_outputs_class,\
        #         enc_outputs_coord_unact
        multi_level_feats = []
        for level_index in range(len(level_start_index)):
            if level_index == len(level_start_index) - 1:
                _memory = memory[level_start_index[level_index]:]
            else:
                _memory = memory[level_start_index[level_index]:level_start_index[level_index + 1]]
            _memory = _memory.permute(1, 2, 0).contiguous() # bs, c, hw
            _memory = _memory.view(*_memory.shape[:2], *spatial_shapes[level_index])
            multi_level_feats.append(_memory)
        return multi_level_feats, inter_states, init_reference_out, inter_references_out

@TRANSFORMER.register_module()
class MapTRTransformer2D(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 len_can_bus=18,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 reg_sigmoid=True,
                 key_padding=False,
                 **kwargs):
        super(MapTRTransformer2D, self).__init__(**kwargs)
        # if modality == 'fusion':
        #     self.fuser = build_fuser(fuser) # TODO
        self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        # self.encoder = build_transformer_layer_sequence(encoder)
        self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.reg_sigmoid=reg_sigmoid
        self.key_padding = key_padding

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # self.level_embeds = nn.Parameter(torch.Tensor(
        #     self.num_feature_levels, self.embed_dims))
        # self.cams_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) # TODO, this is a hack
        # self.can_bus_mlp = nn.Sequential(
        #     nn.Linear(self.len_can_bus, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        # )
        # if self.can_bus_norm:
        #     self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        # normal_(self.level_embeds)
        # normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        # bev_embed = self.get_bev_features(
        #     mlvl_feats,
        #     lidar_feat,
        #     bev_queries,
        #     bev_h,
        #     bev_w,
        #     grid_length=grid_length,
        #     bev_pos=bev_pos,
        #     prev_bev=prev_bev,
        #     **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        # import pdb; pdb.set_trace()
        # len(mlvl_feats) = 1
        # mlvl_feats[0].shape = torch.Size([4, 6, 256, 15, 25])
        # bev_embed.shape = torch.Size([4, 20000, 256]), bev_h * bev_w = 200 * 100

        bs, cam = mlvl_feats[0].size(0), mlvl_feats[0].size(1) # 4, 6
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1) # 1000 * 256
        query_pos = query_pos.unsqueeze(0).expand(bs * cam, -1, -1) # 24 * 1000 * 256
        query = query.unsqueeze(0).expand(bs * cam, -1, -1) # 24 * 1000 * 256

        '''
        reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
        TODO: for openlane dataset, all 3d points are visible on 2d images, 
        but for nuscene dataset, some 3d points are not visible in 2d, 
        so sigmoid cannot be used for normalization here
        '''
        reference_points = self.reference_points(query_pos) # 24, 1000, 2
        if self.reg_sigmoid:
            reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        
        # bev_embed B, HW, C -> HW, B, C
        # bev_embed = bev_embed.permute(1, 0, 2)
        # img_feat bs, cam, C, H, W -> HW, bs*cam, C
        img_feat = mlvl_feats[0].flatten(3).permute(3, 2, 0, 1) # HW, C, bs, cam
        img_feat = img_feat.flatten(2).permute(0, 2, 1) # HW, bs*cam, C

        query = query.permute(1, 0, 2) # Q, B, C
        query_pos = query_pos.permute(1, 0, 2) # Q, B, C

        '''
        reg_branch: same function as self.reference_points
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=2, bias=True)
        '''
        # import pdb; pdb.set_trace()
        x = mlvl_feats[0]
        masks = x.new_zeros((bs, cam, *x.shape[-2:])).to(torch.bool)
        if self.key_padding:
            img_metas = kwargs['img_metas']
            input_img_h, input_img_w, _ = img_metas[0]['img_shape'][0] # img camera
            masks = x.new_ones((bs, cam, input_img_h, input_img_w))
            for img_id in range(bs):
                for cam_id in range(cam):
                    img_h, img_w, _ = img_metas[img_id]['ori_shape'][cam_id]
                    masks[img_id, cam_id, :img_h, :img_w] = 0
            masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)
        masks = masks.view(bs * cam, -1) # bs * cam, HW
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=img_feat, # bev_embed
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches, # None
            key_padding_mask=masks,
            spatial_shapes=torch.tensor([mlvl_feats[0].shape[3:]], device=query.device), # 15 * 25
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        # bev_embed is None
        return None, inter_states, init_reference_out, inter_references_out

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder2D(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, reg_sigmoid=True, **kwargs):
        super(MapTRDecoder2D, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False
        self.reg_sigmoid=reg_sigmoid

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2)  # BS NUM_QUERY NUM_LEVEL 2
            
            if not self.reg_sigmoid:
                # NOTE: for feature sample
                reference_points_input = torch.clamp(reference_points_input, min=0, max=1)

            # import pdb; pdb.set_trace()
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                if self.reg_sigmoid:
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    new_reference_points[..., :2] = tmp[..., :2] + reference_points[..., :2]

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points
