_base_ = '../estimate_datasets/ycbv_pbr.py'

num_classes = 21 # TODO: modify this parameter
keypoint_num = 8

model = dict(
    type='WDRPose',
    keypoint_num=keypoint_num,
    ignore_not_valid_keypoints=False,
    num_classes=num_classes,
    vis_tensorboard=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        style='pytorch',
        num_stages=4,
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
    ),
    head=dict(
        type='FCOS_HEAD',
        keypoint_num=keypoint_num,
        num_classes=num_classes,
        in_channel=256,
        feat_channel=256,
        stacked_convs=4,
        num_levels=1,
        dcn_on_last_conv=True,
    ),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0,
    ),
    loss_keypoint_3d=dict(
        beta=0.02,
        loss_type='smooth_l1',
        loss_weight=1.,
    ),
    loss_keypoint_2d=dict(
        reg_decoded_keypoints=True,
        image_resolution=256,
        loss_type='smooth_l1',
        loss_weight=1.,
        beta=2.,
    ),
    assigner=dict(
        # activate all candidates
        num_pos=-1,
        pos_lambda=1.,
        anchor_sizes=(256),
        with_replacement=False,
        suppress_zero_level=True,
    ),
    anchor_generator=dict(
        ratios=[1.0],
        octave_base_scale=16,
        scales_per_octave=1,
        strides=[16],   
    ),
    coder=dict(
        normalizer=1,
        clip_border=False,
    ),
    test_cfg=dict(
        post_process_pre=1000,
        topk=100,
        score_thr=0.1,
        positive_lambda=1.,
        anchor_sizes=(256),
        solve_pose_space='origin',
    )
)


# data = dict(
#     samples_per_gpu=16,
#     workers_per_gpu=16,
# )

lr = 4e-4
steps = 100000
interval = steps // 10
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='OneCycle',
    max_lr=lr,
    total_steps=steps + 100,
    pct_start=0.05,
    anneal_strategy='linear')

evaluation=dict(interval=steps, 
                metric={
                    'auc':[],
                    'add':[0.05, 0.10, 0.20, 0.50],
                    'rotation':[5],
                    'translation':[1, 2],
                    'depth':[1, 2],
                    'trans_xy':[1, 2]
                    },
                save_best='average/add_10',
                rule='greater',
            )

runner = dict(type='IterBasedRunner', max_iters=steps)
num_gpus = 1
checkpoint_config = dict(interval=interval, by_epoch=False)
log_config=dict(interval=50, 
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardImgLoggerHook', image_format='HWC', interval=100)
                ])
work_dir = 'work_dirs/wdr_ycbv_pbr'