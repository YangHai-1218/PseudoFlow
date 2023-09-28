CLASS_NAMES= ('ape', 'benchvise', 'bowl', 'cam', 'can', 
            'cat','cup', 'driller', 'duck', 'eggbox', 
            'glue', 'holepuncher', 'iron', 'lamp', 'phone')

symmetry_types = { # 1-base
    'cls_10': {'x':180, 'y':180, 'z':180},
    'cls_11': {'z':180},
}

camera_K = [572.4114, 0, 325.2611, 0, 573.57043, 242.04899, 0, 0, 1]
mesh_diameter = [
    102.099, 247.506, 167.355, 172.492, 201.404, 
    154.546, 124.264, 261.472, 108.999, 164.628, 
    175.889, 145.543, 278.078, 282.601, 212.358]

dataset_root = 'data/lm'
dataset=dict(
        data_root=dataset_root + '/test',
        gt_annots_root=dataset_root + '/test',
        image_list='data/lm/image_lists/filtered_total_test_new.txt',
        # image_list=dataset_root + '/image_lists/filtered_total_test.txt',
        keypoints_json=dataset_root + '/keypoints/bbox_13obj.json',
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root +'/models_eval',
        mesh_diameter=mesh_diameter,
)

evaluation={'add':[0.05, 0.10, 0.20, 0.50, 0.99],
            'rep':[2, 5, 10, 20],
            'rotation':[5, 10, 25, 45, 90]
            }

