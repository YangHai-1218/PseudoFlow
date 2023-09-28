CLASS_NAMES= ('master_chef_can', 'cracker_box',
            'sugar_box', 'tomato_soup_can',
            'mustard_bottle', 'tuna_fish_can',
            'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana',
            'pitcher_base', 'bleach_cleanser',
            'bowl', 'mug', 'power_drill', 
            'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick')

symmetry_types = { # 1-base
    'cls_13': {'z':0},
    'cls_16': {'x':180, 'y':180, 'z':90},
    'cls_19': {'y':180},
    'cls_20': {'x':180},
    'cls_21': {'x':180, 'y':90, 'z':180}
}

camera_K = [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]
mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46,
                183.08, 231.39, 102.92]

dataset_root = 'data/ycbv'
dataset=dict(
        data_root=dataset_root + '/test',
        gt_annots_root=dataset_root + '/test',
        image_list=dataset_root + '/image_lists/test_bop19.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
    )
evaluation=dict(
        auc=[],
        add=[0.05, 0.1, 0.2, 0.5, 0.99],
        rep=[2, 5, 10, 20],
        # rotation=[5],
        # translation=[1, 2],
        # depth=[1, 2, 3],
        # trans_xy=[1, 2, 3]
)

        