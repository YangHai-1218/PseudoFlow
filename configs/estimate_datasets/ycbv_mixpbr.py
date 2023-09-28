dataset_root = 'data/ycbv'

keypoint_num = 8
keypoint = 'bbox'
image_scale = 256
CLASS_NAMES= ('master_chef_can', 'cracker_box',
            'sugar_box', 'tomato_soup_can',
            'mustard_bottle', 'tuna_fish_can',
            'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana',
            'pitcher_base', 'bleach_cleanser',
            'bowl', 'mug', 'power_drill', 
            'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick')

normalize_mean = [58.395, 57.12, 57.375]
normalize_std = [103.530, 116.280, 123.675]

symmetry_types = { # 1-base
    'cls_13': {'z':0},
    # 'cls_16': {'x':180, 'y':180, 'z':90},
    # 'cls_19': {'y':180},
    # 'cls_20': {'x':180},
    # 'cls_21': {'x':180, 'y':90, 'z':180}
}

mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46,
                183.08, 231.39, 102.92]
file_client_args = dict(
    backend='disk',
)

train_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='LoadMasks', file_client_args=file_client_args),
    dict(type="BboxJitter",
        scale_limit=0.25,
        shift_limit=0.25,
        iof_threshold=0.,
        jitter_bbox_field='gt_bboxes',
        jittered_bbox_field='ref_bboxes',
        mask_field='gt_masks',
    ),
    dict(type='Crop',
        size_range=(1.1, 1.1), 
        crop_bbox_field='ref_bboxes',
        clip_border=False,
        pad_val=128,
    ),
    dict(type='RandomBackground', background_dir='data/coco', p=0.3, file_client_args=file_client_args),
    dict(type='CosyPoseAug', p=0.8,
        pipelines=[
            dict(type='PillowBlur', p=1., factor_interval=(1, 3)),
            dict(type='PillowSharpness', p=0.3, factor_interval=(0., 50.)),
            dict(type='PillowContrast', p=0.3, factor_interval=(0.2, 50.)),
            dict(type='PillowBrightness', p=0.5, factor_interval=(0.1, 6.0)),
            dict(type='PillowColor', p=0.3, factor_interval=(0., 20.)),
    ]),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', 
        size=(image_scale, image_scale), 
        center=True, 
        pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='HandleSymmetryV2', info_path=dataset_root+'/models/models_info.json'),
    dict(type='ProjectKeypoints', clip_border=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'gt_rotations', 'gt_translations', 'gt_masks', 'gt_bboxes', 
            'gt_keypoints_3d_camera','gt_keypoints_3d', 'gt_keypoints_2d',
            'k', 'labels', ], 
        meta_keys=(
            'img_path', 'ori_shape', 'ori_k',
            'img_shape', 'img_norm_cfg', 
            'ori_gt_rotations', 'ori_gt_translations'),
    ),
]


test_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='Crop', 
        size_range=(1.1, 1.1),
        crop_bbox_field='ref_bboxes', 
        clip_border=False,
        pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'labels', 'k', 'ref_keypoints_3d', 
            'ori_k', 'transform_matrix'
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor', 'geometry_transform_mode'),
    ),
]


data = dict(
    samples_per_gpu=16,
    test_samples_per_gpu=1,
    workers_per_gpu=16,
    train=dict(
        type='ConcatDataset',
        ratios=[2, 1],
        dataset_configs=[
            dict(
                type='SuperviseEstimationDataset',
                data_root=dataset_root + '/train_pbr',
                gt_annots_root=dataset_root + '/train_pbr',
                image_list=dataset_root + '/image_lists/train_pbr.txt',
                keypoints_json=dataset_root + f'/keypoints/{keypoint}.json',
                pipeline=train_pipeline,
                class_names=CLASS_NAMES,
                sample_num=8,
                min_visib_fract=0.1,
                keypoints_num=keypoint_num,
                mesh_symmetry=symmetry_types,
                meshes_eval=dataset_root+'/models_eval',
                mesh_diameter=mesh_diameter,
            ), 
            dict(
                type='SuperviseEstimationDataset',
                data_root=dataset_root + '/train_real',
                gt_annots_root=dataset_root + '/train_real',
                image_list=dataset_root + '/image_lists/train_real.txt',
                keypoints_json=dataset_root + f'/keypoints/{keypoint}.json',
                pipeline=train_pipeline,
                class_names=CLASS_NAMES,
                sample_num=8,
                keypoints_num=keypoint_num,
                mesh_symmetry=symmetry_types,
                meshes_eval=dataset_root+'/models_eval',
                mesh_diameter=mesh_diameter,
            ), 
        ]
    ),
    val=dict(
        type='EstimationDataset',
        data_root=dataset_root + '/test',
        ref_bboxes_root='data/reference_bboxes/beyondcenterv2_mixpbr_ycbv',
        image_list=dataset_root + '/image_lists/test_bop19.txt',
        keypoints_json=dataset_root + f'/keypoints/{keypoint}.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=keypoint_num,
        score_thr=0.,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter
    ),
    test=dict(
        type='EstimationDataset',
        data_root=dataset_root + '/test',
        # ref_bboxes_root='data/reference_bboxes/bop20_fcos_detections/ycbv_mixpbr_test',
        # ref_bboxes_root='data/reference_bboxes/bop22_default_detections_and_segmentations/cosypose_maskrcnn_synt+real/ycbv_test',
        ref_bboxes_root='data/reference_bboxes/beyondcenterv2_mixpbr_ycbv',
        image_list=dataset_root + '/image_lists/test_bop19.txt',
        keypoints_json=dataset_root + f'/keypoints/{keypoint}.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=keypoint_num,
        score_thr=0.,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
    ),
)


model=dict(
    loss_keypoint_3d=dict(
        mesh_diameters=mesh_diameter,
    )
)