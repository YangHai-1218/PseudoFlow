_base_ = '../unsup_refine_datasets/ycbv.py'


RAFT_encoder_cfg = dict(
    type='RAFTEncoder',
    in_channels=3,
    out_channels=256,
    net_type='Basic',
    norm_cfg=dict(type='IN'),
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            mode='fan_out',
            nonlinearity='relu'),
        dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
    ]
)

RAFT_context_encoder_cfg = dict(
    type='RAFTEncoder',
    in_channels=3,
    out_channels=256,
    net_type='Basic',
    norm_cfg=dict(type='BN'),
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            mode='fan_out',
            nonlinearity='relu'),
        dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
    ]
)

init_cfg = dict(
    type='Pretrained',
    checkpoint='work_dirs/raft_ycbv_pbr/latest.pth',
)



Teacher_RAFT_Refiner_cfg = dict(
    type='RAFTRefinerFlow',
    cxt_channels=128,
    h_channels=128,
    seperate_encoder=False,
    encoder=RAFT_encoder_cfg,
    cxt_encoder=RAFT_context_encoder_cfg,
    decoder=dict(
        type='RAFTDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=24,
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')
    ),
    init_cfg=init_cfg,
)
Student_RAFT_Refiner_cfg = dict(
    type='RAFTRefinerFlow',
    cxt_channels=128,
    h_channels=128,
    seperate_encoder=False,
    encoder=RAFT_encoder_cfg,
    cxt_encoder=RAFT_context_encoder_cfg,
    decoder=dict(
        type='RAFTDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')
    ),
    init_cfg=init_cfg,
)


model = dict(
    type='MVCRaftRefinerFlow',
    student_model=Student_RAFT_Refiner_cfg,
    teacher_model=Teacher_RAFT_Refiner_cfg,
    vis_tensorboard=True,
    max_flow=400.,
    freeze_bn=False, # freeze bn?
    sup_flow_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='RAFTLoss',
            loss_weight=0.2,
            max_flow=400.,
        )
    ),
    selfsup_flow_loss_cfg=dict(
        type='MVCLoss',
        valid_view_num_threshold=2,
        gaussian_weights=False,
        use_threshold_for_gaussian=True,
        var_threshold=2.,
        invalid_flow_num=400.,
        loss_func_cfg=dict(
            type='SequenceLoss',
            gamma=0.8,
            loss_func_cfg=dict(
                type='UnsupRAFTLoss',
                loss_weight=1.0,
            )
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        iters=24,
    ),
)

find_unused_parameters = True
steps = 200000
interval = steps//10
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0004,
    total_steps=steps+100,
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
                    dict(type='TensorboardImgLoggerHook', interval=200, image_format='HWC'),
                ])
custom_hooks = [dict(type='MeanTeacherHook', momentum=0.0001)]
work_dir = 'work_dirs/pfc_real_selfsup'