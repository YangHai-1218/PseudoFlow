dataset_root = 'data/ycbv'

CLASS_NAMES= ('master_chef_can', 'cracker_box',
            'sugar_box', 'tomato_soup_can',
            'mustard_bottle', 'tuna_fish_can',
            'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana',
            'pitcher_base', 'bleach_cleanser',
            'bowl', 'mug', 'power_drill', 
            'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick')
normalize_mean = [0., 0., 0., ]
normalize_std = [255., 255., 255.]
image_scale = 256
mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46,
                183.08, 231.39, 102.92]
file_client_args = dict(backend='disk')
symmetry_types = { # 1-base
    'cls_13': {'z':0},
    'cls_16': {'x':180, 'y':180, 'z':90},
    'cls_19': {'y':180},
    'cls_20': {'x':180},
    'cls_21': {'x':180, 'y':90, 'z':180}
}

train_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='ComputeBbox', 
        mesh_dir=dataset_root + '/models_eval', 
        clip_border=False,
        pose_field=['rotations', 'translations'], 
        bbox_field='bboxes'),
    dict(type='Crop',
        size_range=(1.1, 1.1), 
        # size_range=(1.0, 1.25), 
        crop_bbox_field='bboxes',
        clip_border=False,
        pad_val=128,
    ),
    dict(type='Resize', 
        img_scale=image_scale, 
        keep_ratio=True),
    dict(type='Pad', 
        size=(image_scale, image_scale),
        center=True, 
        pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='MultiViewPoseJitterV2',
        jitter_pose_num=4,
        # jitter_angle_dis=(0, 7.5),
        jitter_angle_dis=(0, 15),
        jitter_x_dis=(0, 7.5),
        jitter_y_dis=(0, 7.5),
        jitter_z_dis=(0, 15),
        angle_limit=25,
        keep_ori_pose=True,
        # keep_ori_pose=False,
        repeat_jittered_image=False,
        jitter_pose_field=['rotations', 'translations'],
        jittered_pose_field=['rotations', 'translations']),
    dict(type='CosyPoseAug', 
        p=1., image_keys=['augmented_img'],
        pipelines=[
            dict(type='PillowBlur', p=1., factor_interval=(1, 3)),
            dict(type='PillowSharpness', p=0.3, factor_interval=(0., 50.)),
            dict(type='PillowContrast', p=0.3, factor_interval=(0.2, 50.)),
            dict(type='PillowBrightness', p=0.5, factor_interval=(0.1, 6.0)),
            dict(type='PillowColor', p=0.3, factor_interval=(0., 20.)),
    ]),
    # dict(type='RandomHSV', h_ratio=0.2, s_ratio=0.5, v_ratio=0.5, image_keys=['augmented_img']),
    # dict(type='RandomNoise', noise_ratio=0.1, image_keys=['augmented_img']),
    # dict(type='RandomSmooth', max_kernel_size=5., image_keys=['augmented_img']),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True, image_keys=['augmented_img', 'ori_img']),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect',
        keys=('ori_img', 'augmented_img'),
        annot_keys=[
            'rotations', 'translations', 'k', 'labels'],
        meta_keys=(
            'img_path', 'ori_shape', 'ori_k',
            'img_shape', 'img_norm_cfg', 
            'valid_gt_rotations', 'valid_gt_translations',
            'scale_factor', 'transform_matrix'),
    ),
]

test_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='ComputeBbox', mesh_dir=dataset_root + '/models_eval', clip_border=False, filter_invalid=False),
    dict(type='Crop', size_range=(1.1, 1.1), crop_bbox_field='ref_bboxes', clip_border=False, pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations',
            'gt_rotations', 'gt_translations',
            'labels','k', 'ori_k', 'transform_matrix', 
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor',  'keypoints_3d', 'geometry_transform_mode'),
    ),
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    test_samples_per_gpu=4,
    train=dict(
        type='UnsuperviseTrainDataset',
        data_root=dataset_root + '/test',
        ref_annots_root='data/reference_poses/detect_wdr/ycbv_pbr_train',
        image_list='data/ycbv/image_lists/valid_train_real.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=train_pipeline,
        class_names=CLASS_NAMES,
        filter_invalid_pose=True,
        depth_range=(200, 10000),
        sample_num=1,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
    ),
    val=dict(
        type='RefineDataset',
        data_root=dataset_root + '/test',
        ref_annots_root='data/reference_poses/ycbv_posecnn_init/init',
        image_list=dataset_root + '/image_lists/test.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        filter_invalid_pose=True,
        depth_range=(200, 10000),
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter
    ),
    test=dict(
        type='RefineDataset',
        data_root=dataset_root + '/test',
        ref_annots_root='data/reference_poses/detect_wdr/ycbv_pbr',
        image_list=dataset_root + '/image_lists/test_bop19.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        filter_invalid_pose=True,
        depth_range=(200, 10000),
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
    ),
)

# renderer setting
model = dict(
    renderer=dict(
        mesh_dir=dataset_root + '/models_1024',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(.5, .5, .5),
    ),
)