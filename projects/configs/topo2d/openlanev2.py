plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-50.0, -25.0, -3.0, 50.0, 25.0, 2.0]
position_range = [-50.0, -50.0, -10.0, 50.0, 50.0, 10.0]

img_size = (2048, 1600) # W, H
input_size = (1024, 800) # W, H
downsample_rate = 4
feat_size = (input_size[1] // downsample_rate, input_size[0] // downsample_rate)
scale_factor = 1. # pred mask

img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], 
    std = [58.395, 57.12, 57.375], 
    to_rgb = True)

num_vec = 140 * 4
num_vec_per_image = 20 * 4
topk_vec_per_image = 20 * 4
fixed_ptsnum_per_line = 11

map_classes = ['centerline']
num_map_classes = 1

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2

# for 2d lane detection
sample_method = '2d' # sample in 3d or 2d
reg_sigmoid = True # add sigmoid to ref pts
output_vis = False # vis score and out of fov mask

model = dict(
    type='Topo2D',
    use_grid_mask=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    only_2d=False,
    pc_range=point_cloud_range,
    fusion_method='query',
    pos_embed_method='anchor',
    num_vec=num_vec_per_image,
    num_pts_per_vec=fixed_ptsnum_per_line,
    vis_attn_map=False,
    feat_size=feat_size,
    nms_2d_proposal=False,
    topk_2d_proposal=topk_vec_per_image,
    lane_topo=True,
    traffic=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='Topo2DHead2D',
        num_query=num_vec_per_image,
        num_vec=num_vec_per_image,
        num_lanes_one2one=num_vec_per_image // 4,
        k_one2many=3,
        lambda_one2many=1.0,
        num_pts_per_vec=fixed_ptsnum_per_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        query_embed_type='instance_pts',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        org_img_size=img_size,
        reg_sigmoid=reg_sigmoid,
        output_vis=output_vis,
        transformer=dict(
            type='MapTRTransformer2DMlvl',
            num_feature_levels=4,
            # only decoder
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', 
                        embed_dims=_dim_,
                    ),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder2D',
                num_layers=6,
                return_intermediate=True,
                reg_sigmoid=reg_sigmoid,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=4),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='Topo2DCoder2D',
            post_center_range=[-img_size[0], -img_size[1], 0, 0, 
                               img_size[0], img_size[1], img_size[0] * 2, img_size[1] * 2],
            pc_range=point_cloud_range,
            org_img_size=img_size,
            max_num=num_vec_per_image // 4,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_vis=dict(
            type="CrossEntropyLoss", 
            use_sigmoid=True, 
            loss_weight=2.0, 
            class_weight=1.0
        ),
        loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
            grid_size=[512, 512, 1],
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='Topo2DAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                pts_cost=dict(type='OrderedPtsL1CostInFOV', weight=5, mask=output_vis),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pc_range=point_cloud_range,
                org_img_size=img_size))),
    learn_3d_query=False,
    query_generator=dict(
        in_channels=256,
        num_fcs=1,
        fc_out_channels=1024,
        img_size=img_size,
        extra_encoding=dict(
            num_layers=2,
            feat_channels=[512, 256],
            features=[dict(type='intrinsic', in_channels=16,)]
        ),
    ),
    mask_head=False,
    pts_bbox_head_3d=dict(
        type='Topo2DHead3D',
        num_classes=1,
        in_channels=_dim_,
        num_query=num_vec,
        num_lanes_one2one=num_vec // 4,
        k_one2many=3,
        lambda_one2many=1.0,
        num_pts_per_vec=fixed_ptsnum_per_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        LID=True,
        with_position=True,
        with_multiview=False, # flash attention may don't converge, it must be False
        position_range=position_range,
        shared_head_params=True,
        row_column_attn=False,
        feat_sample_2d_lane=False,
        feat_size=feat_size,
        transformer=dict(
            type='PETRTransformerMlvl',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='Topo2DCoder3D',
            post_center_range=[-60, -30, -60, -30, 60, 30, 60, 30],
            pc_range=point_cloud_range,
            max_num=num_vec // 4,
            score_threshold=1.,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_vis=dict(
            type="CrossEntropyLoss", 
            use_sigmoid=True, 
            loss_weight=0.0, 
            class_weight=1.0
        ),
        loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        train_cfg=dict(
            grid_size=[512, 512, 1],
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='Topo2DAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pc_range=point_cloud_range)),
        test_cfg=None),
    te_head=dict(
        type='TrafficHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            num_feature_levels=4,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        test_cfg=dict(max_per_img=50),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    ),
    lclc_head=dict(
        type='TopoLLHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5),
        loss_ll_l1_weight=1.0,
        add_lane_pred=True,
        lane_pred_dimension=fixed_ptsnum_per_line * 3,
        is_detach=True),
    lcte_head=dict(
        type='TopoLTHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5),
        add_pos=True,
        pos_dimension=9,
        num_pts_per_vec=fixed_ptsnum_per_line,
        is_detach=True),
    )

dataset_type = 'OpenLaneV2SubsetADataset'
data_root = './data/OpenLane-V2'
meta_root = './data/OpenLane-V2'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFilesPad', to_float32=True, padding=False), # ori shape
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CropFrontViewImageOpenLaneV2', crop_h=(356, 1906)), # crop + pad, img shape
    dict(type='CustomLoadRandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundleOpenLaneV2'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels',
            'gt_camera_extrinsic', 'gt_camera_intrinsic',
            'gt_project_matrix', 'gt_homography_matrix',
            'gt_topology_lclc', 'gt_topology_lclc', 
            'gt_te', 'gt_te_labels', 'gt_topology_lcte',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus', 'ori_shape',
        ],
    )
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFilesPad', to_float32=True, padding=False),
    dict(type='CropFrontViewImageOpenLaneV2', crop_h=(356, 1906)),
    dict(type='CustomLoadRandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomDefaultFormatBundleOpenLaneV2'),
    dict(
        type='Collect',
        keys=[
            'img',
            'gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels',
            'gt_camera_extrinsic', 'gt_camera_intrinsic',
            'gt_project_matrix', 'gt_homography_matrix',
            'gt_topology_lclc', 'gt_topology_lclc', 
            'gt_te', 'gt_te_labels', 'gt_topology_lcte',
        ],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths', 
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus', 'ori_shape',
        ],
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        fixed_num=fixed_ptsnum_per_line,
        code_size=3,
        img_size=img_size,
        collection='data_dict_subset_A_train',
        pipeline=train_pipeline,
        sample_interval=2,
        crop_h=(356, 1906),
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        fixed_num=fixed_ptsnum_per_line,
        code_size=3,
        img_size=img_size,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        sample_interval=2,
        crop_h=(356, 1906),
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        meta_root=meta_root,
        fixed_num=fixed_ptsnum_per_line,
        code_size=3,
        img_size=img_size,
        collection='data_dict_subset_A_val',
        pipeline=test_pipeline,
        sample_interval=2,
        crop_h=(356, 1906),
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=3e-4, # bs 8
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
fp16 = dict(loss_scale=512.)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=30, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
load_from = None
resume_from = None
work_dir = './work_dirs/openlanev2'
