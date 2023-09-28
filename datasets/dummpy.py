from typing import Optional, Sequence
import mmcv
from pathlib import Path
from os import path as osp
from .builder import DATASETS
from .base_dataset import BaseDataset



@DATASETS.register_module()
class DummpyDataset(BaseDataset):
    def __init__(self, 
                data_root: str, 
                image_list: str, 
                keypoints_json: str, 
                class_names: tuple, 
                ref_annots_root_list: list,
                pipeline: list = None, 
                gt_annots_root: str = None,
                target_label: list = None, 
                label_mapping: dict = None, 
                keypoints_num: int = 8, 
                meshes_eval: str = None, 
                mesh_symmetry: dict =  {}, 
                mesh_diameter: list = [],
                ):
        super().__init__(
            data_root, image_list, keypoints_json, class_names, pipeline, gt_annots_root, target_label, label_mapping, keypoints_num, meshes_eval, mesh_symmetry, mesh_diameter)
        self.ref_annots_root_list = ref_annots_root_list
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.data_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.ref_seq_pose_annots_list = []
        for ref_annots_root in self.ref_annots_root_list:
            self.ref_seq_pose_annots_list.append(self._load_ref_pose_annots(ref_annots_root))
        self.gt_seq_pose_annots = self._load_gt_pose_annots()    
    
    def _load_ref_pose_annots(self, ref_annots_root):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        ref_seq_pose_annots = dict()
        for sequence in sequences:
            ref_pose_json_path = osp.join(ref_annots_root, self.pose_json_tmpl.format(int(sequence)))
            ref_pose_annots = mmcv.load(ref_pose_json_path)
            ref_seq_pose_annots[sequence] = dict(pose=ref_pose_annots)
        return ref_seq_pose_annots

    def _load_gt_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        gt_seq_pose_annots = dict()
        for sequence in sequences:
            gt_pose_json_path = osp.join(self.gt_annots_root, self.pose_json_tmpl.format(int(sequence)))
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            gt_seq_pose_annots[sequence] = dict(pose=gt_pose_annots)
        return gt_seq_pose_annots
    

    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots = self.gt_seq_pose_annots[seq_name]
        ref_seq_annots_list = [a[seq_name] for a in self.ref_seq_pose_annots_list]

        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
        
        # load camera intrisic
        if str(img_id) in gt_seq_annots['camera']:
            camera_annots = gt_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = gt_seq_annots['camera']["{:06}".format(img_id)]

        