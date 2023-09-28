import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Dict, Sequence
import mmcv
from mmcv.runner import BaseModule
import cv2, random, math, time
import numpy as np
from pathlib import Path
from kornia.augmentation import AugmentationSequential
from .builder import REFINERS
from ..encoder import build_encoder
from ..decoder import build_decoder
from ..utils import Renderer, build_augmentation, get_flow_from_delta_pose_and_depth, filter_flow_by_mask, cal_epe
from ..utils.utils import simple_forward_warp, tensor_image_to_cv2, Warp

@REFINERS.register_module()
class BaseRefiner(BaseModule):
    def __init__(self, 
                encoder: Optional[Dict]=None,
                decoder: Optional[Dict]=None,
                seperate_encoder: bool=False,
                renderer: Optional[Dict]=None,
                render_augmentations: Optional[Sequence[Dict]]=None,
                train_cfg: dict={},
                test_cfg: dict={},
                init_cfg: dict={},
                max_flow: int=400,
                ):
        super().__init__(init_cfg)
        self.seperate_encoder = seperate_encoder
        if encoder is not None:
            if self.seperate_encoder:
                self.render_encoder = build_encoder(encoder)
                self.real_encoder = build_encoder(encoder)
            else:
                encoder_model = build_encoder(encoder)
                self.render_encoder = encoder_model
                self.real_encoder = encoder_model
        if decoder is not None:
            self.decoder = build_decoder(decoder)  
        if renderer is not None: 
            self.renderer = Renderer(**renderer)    
        else:
            self.renderer = None
        self.max_flow = max_flow
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_cycle_num = self.train_cfg.get('cycles', 1)
        self.train_grad_clip = self.train_cfg.get('grad_norm', None)
        self.test_cycle_num = self.test_cfg.get('cycles', 1)
        if render_augmentations is not None:
            augmentations = []
            for augmentation in render_augmentations:
                augmentations.append(
                    build_augmentation(augmentation)
                )
            self.render_augmentation = AugmentationSequential(
                *augmentations,
                data_keys=['input'],
                same_on_batch=False,
            )
        else:
            self.render_augmentation = None
        # self.measure_runtime()
    
    def measure_runtime(self):
        self.render_time = 0
        self.step = 0
    
    def update_measure_time(self, name, consume_time, steps=1):
        old_time = getattr(self, name)
        new_time = (old_time * self.step + consume_time*steps)/(self.step+steps)
        setattr(self, name, new_time)


    def to(self, device):
        super().to(device)
        if self.renderer is not None:
            self.renderer.to(device)
        
    def loss(self, data_batch):
        raise NotImplementedError
    
    def forward_single_view(self, data):
        raise NotImplementedError

    def format_data_test(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        ref_rotations, ref_translations = annots['ref_rotations'], annots['ref_translations']
        labels, internel_k = annots['labels'], annots['k']
        ori_k, transform_matrixs = annots['ori_k'], annots['transform_matrix']

        per_img_patch_num = [len(images) for images in real_images]
        real_images = torch.cat(real_images)
        ref_rotations, ref_translations = torch.cat(ref_rotations, dim=0), torch.cat(ref_translations, dim=0)
        labels = torch.cat(labels)
        internel_k = torch.cat(internel_k)
        transform_matrixs = torch.cat(transform_matrixs)
        ori_k = torch.cat([k[None].expand(patch_num, 3, 3) for k, patch_num in zip(ori_k, per_img_patch_num)])
        
        render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)

        img_norm_cfg = meta_infos[0]['img_norm_cfg']
        normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        rendered_images = (rendered_images - normalize_mean)/normalize_std
        output = dict(
            real_images = real_images,
            rendered_images = rendered_images,
            labels = labels,
            ori_k = ori_k,
            transform_matrix = transform_matrixs,
            internel_k = internel_k,
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,  
            per_img_patch_num = per_img_patch_num,
            meta_infos=meta_infos,
        )
        if 'depths' in annots:
            real_depths = torch.cat(annots['depths'], dim=0)
            output.update(real_depths=real_depths)
        if 'gt_rotations' in annots:
            gt_rotations, gt_translaions = annots['gt_rotations'], annots['gt_translations']
            gt_rotations, gt_translaions = torch.cat(gt_rotations, dim=0), torch.cat(gt_translaions, dim=0)
            output.update(
                gt_rotations=gt_rotations,
                gt_translations=gt_translaions,
            )
        if 'gt_masks' in annots:
            gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
            gt_masks = torch.cat(gt_masks, axis=0)
            output.update(gt_masks=gt_masks)
        return output


    def format_data_train_sup(self, data_batch):
        real_images, annots, meta_infos = data_batch['img'], data_batch['annots'], data_batch['img_metas']
        gt_rotations, gt_translations = annots['gt_rotations'], annots['gt_translations']
        ref_rotations, ref_translations = annots['ref_rotations'], annots['ref_translations']
        init_add_error, init_rot_error, init_trans_error = annots['init_add_error'], annots['init_rot_error'], annots['init_trans_error']
        labels, internel_k = annots['labels'], annots['k']
        init_rot_error_std, init_rot_error_mean = torch.std_mean(init_rot_error, unbiased=False)
        init_add_error_std, init_add_error_mean = torch.std_mean(init_add_error, unbiased=False)
        init_trans_error_std, init_trans_error_mean = torch.std_mean(init_trans_error, unbiased=False)

        real_images = torch.cat(real_images)
        ref_rotations, ref_translations = torch.cat(ref_rotations, axis=0), torch.cat(ref_translations, axis=0)
        gt_rotations, gt_translations = torch.cat(gt_rotations, axis=0), torch.cat(gt_translations, axis=0)
        labels, internel_k = torch.cat(labels), torch.cat(internel_k)
        render_outputs = self.renderer(ref_rotations, ref_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        if self.render_augmentation is not None:
            rendered_images = self.render_augmentation(rendered_images)

        img_norm_cfg = meta_infos[0]['img_norm_cfg']
        normalize_mean, normalize_std = img_norm_cfg['mean'], img_norm_cfg['std']
        normalize_mean = torch.Tensor(normalize_mean).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        normalize_std = torch.Tensor(normalize_std).view(1, 3, 1, 1).to(real_images[0].device) / 255.
        rendered_images = (rendered_images - normalize_mean)/normalize_std
        output = dict(
            ref_rotations = ref_rotations,
            ref_translations = ref_translations,
            gt_rotations = gt_rotations,
            gt_translations = gt_translations,
            labels = labels,
            internel_k = internel_k,
            rendered_images = rendered_images,
            real_images = real_images,
            rendered_masks = rendered_masks,
            rendered_depths = rendered_depths,
            init_add_error_mean = init_add_error_mean,
            init_add_error_std = init_add_error_std,
            init_rot_error_mean = init_rot_error_mean,
            init_rot_error_std = init_rot_error_std,
            init_trans_error_mean = init_trans_error_mean,
            init_trans_error_std = init_trans_error_std,
        )
        if 'gt_masks' in annots:
            gt_masks = [mask.to_tensor(dtype=torch.bool, device=gt_rotations[0].device) for mask in annots['gt_masks']]
            gt_masks = torch.cat(gt_masks, axis=0)
            output['gt_masks'] = gt_masks
            return output
        else:
            return output




    
    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad_norm = nn.utils.clip_grad.clip_grad_norm_(params, **self.train_grad_clip)
            return grad_norm

    
    def update_data(self, update_rotations, update_translations, data):
        data['ref_rotations'] = update_rotations
        data['ref_translations'] = update_translations
        labels, internel_k = data['labels'], data['internel_k']
        render_outputs = self.renderer(update_rotations, update_translations, internel_k, labels)
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        rendered_masks = (rendered_depths > 0).to(torch.float32)
        data['rendered_images'] = rendered_images
        data['rendered_depths'] = rendered_depths
        data['rendered_masks'] = rendered_masks
        return data

    def train_multiple_iterations(self, data_batch, optimizer):
        train_cycle_num = getattr(self, 'train_cycle_num')
        iter_loss_dict = dict()
        data = self.format_data_train_sup(data_batch)
        log_vars_list, log_imgs_list = [], []
        for i in range(train_cycle_num):
            loss, log_imgs, log_vars, seq_rotations, seq_translations = self.loss(data, data_batch)
            log_vars_list.append(log_vars)
            log_imgs_list.append(log_imgs)

            if i == train_cycle_num - 1:
                continue
            
            optimizer.zero_grad()
            loss.backward()
            self.clip_grads(self.parameters())
            optimizer.step()

            update_rotations, update_translations = seq_rotations[-1], seq_translations[-1]
            update_rotations = update_rotations.detach()
            update_translations = update_translations.detach()
            data = self.update_data(update_rotations, update_translations, data)

            iter_loss_dict[f'iter_{i}_loss'] = loss.item()
        log_vars = {k:sum([log_vars[k] for log_vars in log_vars_list])/train_cycle_num for k in log_vars_list[0]}
        log_vars.update(iter_loss_dict)
        log_imgs = log_imgs_list[random.choice(list(range(train_cycle_num-1)))]
        return loss, log_imgs, log_vars
    
    def forward_multiple_pass(self, data):
        for i in range(self.test_cycle_num):
            results_dict = self.forward_single_pass(data)
            if i == self.test_cycle_num - 1:
                continue
            update_rotations = results_dict['rotations']
            update_translations = results_dict['translations']
            update_rotations = torch.cat(update_rotations)
            update_translations = torch.cat(update_translations)
            # start = time.time()
            data = self.update_data(update_rotations, update_translations, data)
            # render_time = (time.time() - start)/len(runtime_total)
            # runtime_total = [r_1+render_time for r_1 in runtime_total]
        batch_size = len(data['real_images'])
        # self.runtime_record.extend(runtime_total)
        # self.report_runtime(new_step=len(runtime_total))
        return results_dict


    def vis_flow(self, val):
        if isinstance(val, (list, tuple)):
            # sequence preiction
            # visualize the same sample's prediction across different iterations
            flow_list = []
            for i in range(len(val)):
                flow_i = val[i][0].permute(1, 2, 0).cpu().data.numpy()
                flow_i = mmcv.flow2rgb(flow_i, unknown_thr=self.max_flow-1)
                flow_list.append(flow_i)
            return flow_list
        else:
            assert isinstance(val, torch.Tensor)
            if val.ndim == 4:
                flow = val[0].permute(1, 2, 0).cpu().data.numpy()
                flow = mmcv.flow2rgb(flow, unknown_thr=self.max_flow-1)
                return flow
            elif val.ndim == 5:
                flow_list = []
                for i in range(val.size(1)):
                    flow = val[0, i].permute(1, 2, 0).cpu().data.numpy()
                    flow = mmcv.flow2rgb(flow, unknown_thr=self.max_flow-1)
                    flow_list.append(flow)
                return flow_list
            else:
                raise RuntimeError
    
    def vis_images(self, val:torch.Tensor):
        if val.ndim == 4:
            return val[0].permute(1, 2, 0).cpu().data.numpy()
        elif val.ndim == 5:
            image_list = []
            for i in range(val.size(1)):
                image = val[0, i].permute(1, 2, 0).cpu().data.numpy()
                image_list.append(image)
            return image_list
    
    def vis_masks(self, val:torch.Tensor):
        if val.ndim == 3:
            return val[0, None].permute(1, 2, 0).cpu().data.numpy()
        elif val.ndim == 4:
            mask_list = []
            for i in range(val.size(1)):
                mask = val[0, i][None].permute(1, 2, 0).cpu().data.numpy()
                mask_list.append(mask)
            return mask_list
            
    def add_vis_images(self, **kwargs):
        log_imgs = dict()
        
        for key, val in kwargs.items():
            if 'flow' in key:
                log_imgs[key] = self.vis_flow(val)
            elif 'image' in key:
                log_imgs[key] = self.vis_images(val)
            elif 'mask' in key:
                log_imgs[key] = self.vis_masks(val)
            else:
                raise RuntimeError
        return log_imgs
    
    def train_step(self, data_batch, optimizer, **kwargs):
        if self.train_cycle_num > 1:
            loss, log_imgs, log_vars = self.train_multiple_iterations(data_batch, optimizer)
        else:
            loss, log_imgs, log_vars, _, _ = self.loss(data_batch)
        outputs = dict(
            loss = loss,
            log_vars = log_vars,
            log_imgs = log_imgs,
            num_samples = len(data_batch['img_metas']),
        )
        return outputs
    
    def forward(self, data_batch, return_loss=False):
        data = self.format_data_test(data_batch)
        if self.test_cycle_num > 1:
            return self.forward_multiple_pass(data)
        else:
            return self.forward_single_pass(data, data_batch)
    
    
    def visualize_sequence_flow_and_fw(self, data, sequence_flow):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        per_img_patch_num = data['per_img_patch_num']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, labels = data['internel_k'], data['labels']
        batchsize = len(real_images)
        show_index = self.test_cfg.get('vis_index', None)
        if show_index is None:
            show_index = range(len(sequence_flow))
        flow_list = [f for j, f in enumerate(sequence_flow) if j in show_index]
        real_images_cv2 = tensor_image_to_cv2(real_images)
        fw_batch_images = [
            tensor_image_to_cv2(simple_forward_warp(rendered_images, f, rendered_masks))
            for f in flow_list
        ]
        image_index = 0
        show_image_list_all = []
        for i in range(batchsize):
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            fw_image = [fw_batch_image[i] for fw_batch_image in fw_batch_images]

            flow_image = [
                (mmcv.flow2rgb((flow[i]*rendered_masks[i][None]).permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255).astype(np.uint8)
                for flow in flow_list]
            
            # diff_image = [
            #     np.abs(real_images_cv2[i] - fw_image_j)
            #     for fw_image_j in fw_image
            # ]
            show_image_list = []
            for j in range(len(flow_image)):
                show_image_list.append(flow_image[j])
                show_image_list.append(fw_image[j])
                # show_image_list.append(diff_image[j])

            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence + '_'+str(Path(meta_info['img_path']).stem) + "_flow.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []






    def visualize_and_save(self, data, sequence_flow, pred_rotations, pred_translations, sequence_pose_flow=None):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        per_img_patch_num = data['per_img_patch_num']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, labels = data['internel_k'], data['labels']
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        batchsize = len(real_images)
        image_index = 0
        render_outputs = self.renderer(pred_rotations, pred_translations, internel_k, labels)
        refined_images, refined_fragments = render_outputs['images'], render_outputs['fragments']
        refined_images = refined_images[..., :3].permute(0, 3, 1, 2).contiguous()
        refined_depths = refined_fragments.zbuf
        refined_depths = refined_depths[..., 0]
        refined_masks = (refined_depths > 0).to(torch.float32)

        if 'gt_rotations' in data:
            gt_flow = get_flow_from_delta_pose_and_depth(data['ref_rotations'], data['ref_translations'], data['gt_rotations'], data['gt_translations'], rendered_depths, internel_k, invalid_num=0)
            warp_image_by_gt_flow = simple_forward_warp(rendered_images, gt_flow, rendered_masks)
            warp_image_by_gt_flow = tensor_image_to_cv2(warp_image_by_gt_flow)
        else:
            gt_flow = None

        if sequence_pose_flow is not None:
            # needs flow shape (B, H, W, 2)
            warp_image_by_pose_flow = simple_forward_warp(rendered_images, sequence_pose_flow[-1], rendered_masks)
            # warp_image_by_pose_flow = forward_warp_func(rendered_images, sequence_pose_flow[-1].permute(0, 2, 3, 1))
            warp_image_by_pose_flow = tensor_image_to_cv2(warp_image_by_pose_flow)
        warp_image_by_pred_flow = simple_forward_warp(rendered_images, sequence_flow[-1], rendered_masks)
        warp_image_by_pred_flow = tensor_image_to_cv2(warp_image_by_pred_flow)
        show_image_list_all = []
        for i in range(batchsize):
            show_image_list = [real_images_cv2[i], rendered_images_cv2[i]]
            # flow = np.concatenate([mmcv.flow2rgb(flow[i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255 for flow in sequence_flow], axis=1)
            # pose_flow = np.concatenate([mmcv.flow2rgb(flow[i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)[..., ::-1]*255 for flow in sequence_pose_flow], axis=1)
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            patch_index = i - sum(per_img_patch_num[:image_index])
            # save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + f"_{patch_index:06d}.png")
            # mmcv.mkdir_or_exist(Path(save_path).parent)
            flow = mmcv.flow2rgb((sequence_flow[-1][i]*rendered_masks[i][None]).permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
            flow = (flow[..., ::-1]*255).astype(np.uint8)
            show_image_list.append(flow)
            show_image_list.append(warp_image_by_pred_flow[i])
            if sequence_pose_flow is not None:
                pose_flow = mmcv.flow2rgb(sequence_pose_flow[-1][i].permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
                pose_flow = (pose_flow[..., ::-1]*255).astype(np.uint8)
                show_image_list.append(pose_flow)
                show_image_list.append(warp_image_by_pose_flow[i])

            if gt_flow is not None:
                gt_flow_i = gt_flow[i]
                gt_flow_i = mmcv.flow2rgb(gt_flow_i.permute(1, 2, 0).cpu().data.numpy(), unknown_thr=self.max_flow)
                gt_flow_i = (gt_flow_i[..., ::-1]*255).astype(np.uint8)
                show_image_list.append(gt_flow_i)
                show_image_list.append(warp_image_by_gt_flow[i])

            
            contours, _ = cv2.findContours(rendered_masks[i].cpu().data.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            ref_pose_show = cv2.drawContours(real_images_cv2[i].copy(), contours, -1, [255, 255, 255], 3)
            show_image_list.append(ref_pose_show)
            contours, _ = cv2.findContours(refined_masks[i].cpu().data.numpy().astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            refine_pose_show = cv2.drawContours(real_images_cv2[i].copy(), contours, -1, [255, 255, 255], 3)
            show_image_list.append(refine_pose_show)
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            # cv2.imwrite(save_path.as_posix(), show_image)

            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence + '_'+str(Path(meta_info['img_path']).stem) + ".png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []
    
    def eval_seq_epe(self, sequence_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, render_masks, gt_masks=None):
        sequence_epe = []
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        for i in range(len(sequence_flow)):
            flow_error = torch.sum((gt_flow - sequence_flow[i])**2, dim=1).sqrt()
            flow_error = flow_error * render_masks
            sequence_epe.append(flow_error)
        for i in range(len(sequence_flow[0])):
            epe_list = [s[i] for s in sequence_epe]
            epe_mean = [torch.sum(epe)/render_masks[i].sum() for epe in epe_list]
            print(epe_mean)
            epe = torch.cat(epe_list, dim=1)
            epe = (epe/epe.max()).cpu().data.numpy()
            # np.save(f'debug/flow_{i}.npy', epe)

            epe = (epe*255).astype(np.uint8)
            epe = cv2.applyColorMap(epe[..., None], cv2.COLORMAP_JET)
            cv2.imwrite(f'debug/flow_{i}.png', epe)          

    
    def eval_similarity(self, feat_render, feat_real, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations,gt_masks=None, similarity_mode='cosine', scale=8):
        assert similarity_mode in ['cosine', 'absolute', 'relative']
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        if gt_masks is not None:
            gt_flow = filter_flow_by_mask(gt_flow, gt_masks, self.max_flow)
        valid_mask = torch.sum(gt_flow**2, dim=1).sqrt() < self.max_flow
        warp_func = Warp()
        downsampled_gt_flow = 1/scale * F.interpolate(
                gt_flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True) 
        valid_mask = valid_mask[:, None].to(torch.float32)
        downsampled_valid_mask = F.interpolate(
            valid_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)[:, 0, ...]
        warped_feat = warp_func(feat_real, downsampled_gt_flow)
        if similarity_mode == 'cosine':
            similarity = torch.cosine_similarity(feat_render, warped_feat, dim=1)
        elif similarity_mode == 'absolute':
            similarity = (feat_real * feat_render).sum(dim=1)/math.sqrt(feat_render.size(1))
        else:
            N, C, H, W = feat_real.size()
            similarity = (feat_real * feat_render).sum(dim=1)/torch.sqrt(torch.tensor(C).float())
            corr = torch.matmul(feat_render.view(N, C, -1).permute(0, 2, 1), feat_real.view(N, C, -1)).view(N, H, W, H, W)
            corr = corr / torch.sqrt(torch.tensor(C).float())
            # corr = torch.exp(corr).sum(dim=(-1, -2))
            # similarity = torch.exp(similarity) / corr
            corr = torch.amax(corr, dim=(-1, -2))
            similarity = similarity / corr

        similarity = similarity * downsampled_valid_mask
        similarity_map = similarity
        similarity = similarity.sum(dim=(-1, -2)) /(downsampled_valid_mask.sum(dim=(-1, -2)) + 1e-8)
        return similarity, similarity_map
    
    def vis_similarity(self, similarity, data, scale=8):
        output_dir = self.test_cfg.get('vis_dir')
        meta_infos = data['meta_infos']
        per_img_patch_num = data['per_img_patch_num']
        real_images = data['real_images']
        rendered_images = data['rendered_images']
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        batch_size = len(similarity)
        similarity = F.interpolate(similarity[:, None], scale_factor=(scale, scale), mode='bilinear', align_corners=True)[:, 0]
        similarity = (similarity * 255).clamp(0, 255).byte().cpu().numpy()
        image_index = 0
        show_image_list_all = []
        for i in range(batch_size):
            show_image_list = [real_images_cv2[i], rendered_images_cv2[i]]
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            save_similarity = cv2.applyColorMap(similarity[i], cv2.COLORMAP_JET)
            show_image_list.append(save_similarity)
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + "_sim_2.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []

    def vis_response_map(self, feat_render, feat_real, data):
        output_dir = self.test_cfg.get('vis_dir')
        rendered_depths = data['rendered_depths']
        meta_infos = data['meta_infos']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        internel_k = data['internel_k']
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        per_img_patch_num = data['per_img_patch_num']
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        if data.get('gt_masks', None) is not None:
            gt_flow = filter_flow_by_mask(gt_flow, data['gt_masks'], self.max_flow)
        valid_mask = torch.sum(gt_flow**2, dim=1).sqrt() < self.max_flow
        valid_mask = valid_mask[:, None].to(torch.float32)
        scale = 8
        downsampled_valid_mask = F.interpolate(
            valid_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)[:, 0, ...]
        downsampled_gt_flow = 1/scale * F.interpolate(
                gt_flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True) 
        real_images, rendered_images = data['real_images'], data['rendered_images']
        N, C, H, W = feat_real.size()
        corr = torch.matmul(feat_render.view(N, C, -1).permute(0, 2, 1), feat_real.view(N, C, -1)).view(N, H, W, H, W)
        batch_size = len(real_images)
        real_images_cv2 = tensor_image_to_cv2(real_images)
        rendered_images_cv2 = tensor_image_to_cv2(rendered_images)
        image_index = 0
        show_image_list_all = []
        for i in range(batch_size):
            meta_info = meta_infos[image_index]
            sequence = str(Path(meta_info['img_path']).parents[1].name)
            valid_points_y, valid_points_x = (downsampled_valid_mask[i] > 0.9).nonzero(as_tuple=True)
            num_valid_points = len(valid_points_x)
            if num_valid_points <= 10:
                continue

            point_x, point_y = 16, 16
            # index = 10
            # point_x, point_y = valid_points_x[index], valid_points_y[index]
            render_bbox = torch.tensor([[point_x*8-4, point_y*8-4, point_x*8+4, point_y*8+4]]).cpu().data.numpy()
            show_render_image = mmcv.imshow_bboxes(rendered_images_cv2[i], render_bbox, thickness=1, show=False)

            real_point_x, real_point_y = downsampled_gt_flow[i, 0, point_y, point_x] + point_x , downsampled_gt_flow[i, 1, point_y, point_x] + point_y
            real_bbox = torch.tensor([[real_point_x*8-4, real_point_y*8-4, real_point_x*8+4, real_point_y*8+4]]).cpu().data.numpy()
            show_real_image = mmcv.imshow_bboxes(real_images_cv2[i], real_bbox, thickness=1, show=False)
            show_image_list = [show_real_image, show_render_image]
            similarity = corr[i, point_y, point_x]
            similarity = similarity / similarity.max()
            similarity = F.interpolate(similarity[None, None], scale_factor=(scale, scale), mode='bilinear', align_corners=True)[0, 0]
            similarity = (similarity * 255).clamp(0, 255).byte().cpu().numpy()
            save_similarity = cv2.applyColorMap(similarity, cv2.COLORMAP_JET)
            # fuse heatmap and real image
            show_real_image = (0.6* save_similarity + 0.4*show_real_image).astype(np.uint8)
            show_image_list = [show_real_image, show_render_image]
            show_image = np.concatenate(show_image_list, axis=1)
            show_image_list_all.append(show_image)
            if i >= sum(per_img_patch_num[:image_index+1])-1:
                image_index += 1
                show_image_all = np.concatenate(show_image_list_all, axis=0)
                save_path = Path(output_dir).joinpath(sequence).joinpath(str(Path(meta_info['img_path']).stem) + "_response.png")
                mmcv.mkdir_or_exist(Path(save_path).parent)
                cv2.imwrite(save_path.as_posix(), show_image_all)
                show_image_list_all = []