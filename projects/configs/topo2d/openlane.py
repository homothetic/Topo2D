plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-10.0, 0.0, -5.0, 10.0, 105.0, 5.0]
position_range = [-30.0, -20.0, -5.0, 30.0, 125.0, 5.0]

img_size = (1920, 1280) # W, H
input_size = (1024, 800) # W, H
downsample_rate = 8
feat_size = (input_size[1] // downsample_rate, input_size[0] // downsample_rate)
scale_factor = 1. # pred mask

img_norm_cfg = dict(
    mean = [123.675, 116.28, 103.53], 
    std = [58.395, 57.12, 57.375], 
    to_rgb = True)

num_vec = 20
topk_vec = 10
fixed_ptsnum_per_line = 21

map_classes = ['lane']
num_map_classes = 20

input_modality = dict(
    use_lidar = False,
    use_camera = True,
    use_radar = False,
    use_map = False,
    use_external = True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2

# for 2d lane detection
sample_method = '3dvis' # sample in 3d or 2d
reg_sigmoid = True # add sigmoid to ref pts
output_vis = False # vis score and out of fov mask

model = dict(
    type='Topo2DLane',
    use_grid_mask=True,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    only_2d=False,
    pc_range=point_cloud_range,
    fusion_method='query',
    pos_embed_method='anchor',
    num_vec=num_vec,
    num_pts_per_vec=fixed_ptsnum_per_line,
    vis_attn_map=False,
    feat_size=feat_size,
    nms_2d_proposal=True,
    topk_2d_proposal=topk_vec,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    ms2one=dict(
        type='DilateNaive',
        inc=_dim_, 
        outc=_dim_, 
        num_scales=4,
        dilations=(1, 2, 5, 9)),
    pts_bbox_head=dict(
        type='Topo2DHead2DLane',
        num_query=num_vec,
        num_vec=num_vec,
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
            type='MapTRTransformer2D',
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
                            num_levels=1),
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
            max_num=num_vec,
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
    learn_3d_query=True,
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
    mask_head=True,
    sparse_ins_decoder=dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_convs=4,
            scale_factor=scale_factor,
        ),
        loss_mask_pixel_weight = 5.0,
        loss_mask_dice_weight = 2.0,
        loss_objectness_weight = 1.0,
    ),
    pts_bbox_head_3d=dict(
        type='Topo2DHead3DLane',
        num_classes=20,
        in_channels=_dim_,
        num_query=topk_vec,
        num_pts_per_vec=fixed_ptsnum_per_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        LID=True,
        with_position=True,
        with_multiview=True, # camera num = 1
        position_range=position_range,
        shared_head_params=True,
        row_column_attn=False,
        feat_sample_2d_lane=False,
        feat_size=feat_size,
        transformer=dict(
            type='PETRTransformer',
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
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='Topo2DCoder3DLane',
            pc_range=point_cloud_range,
            org_img_size=img_size,
            max_num=topk_vec,
            score_threshold=0.5,
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
            loss_weight=2.0, 
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
                type='Topo2DAssigner3DLane',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                pts_cost=dict(type='OrderedPtsL1CostInFOV', weight=5, mask=True),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pc_range=point_cloud_range,
                org_img_size=img_size)),
        test_cfg=None))

dataset_type = 'OpenlaneDataset'
data_root = './data/OpenLane'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=input_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 
                               'gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels',
                               'gt_camera_extrinsic', 'gt_camera_intrinsic',
                               'gt_project_matrix', 'gt_homography_matrix',
                               'seg_idx_label']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=input_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='LaneFormat'),
    dict(type='Collect', keys=['img', 
                               'gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels',
                               'gt_camera_extrinsic', 'gt_camera_intrinsic',
                               'gt_project_matrix', 'gt_homography_matrix',
                               'seg_idx_label']),
]

dataset_config = dict(
    max_lanes = 25,
    input_size = (input_size[1], input_size[0]),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='training.txt',
        dataset_config=dataset_config,
        map_classes=map_classes,
        sample_method=sample_method, 
        output_vis=output_vis,
        flip_2d=False,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='validation.txt',
        dataset_config=dataset_config, 
        map_classes=map_classes,
        sample_method=sample_method,
        output_vis=output_vis,
        flip_2d=False,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        data_list='validation.txt',
        dataset_config=dataset_config, 
        sample_method=sample_method,
        output_vis=output_vis,
        flip_2d=False,
        num_pts_per_gt_vec=fixed_ptsnum_per_line,
        map_classes=map_classes,
        test_mode=True,
        pipeline=test_pipeline),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-4,
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

find_unused_parameters=True
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
load_from = None
resume_from = None
work_dir = './work_dirs/openlane'
