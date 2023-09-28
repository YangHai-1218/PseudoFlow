import torch
from torch import nn
from .builder import LOSSES, build_loss
from ..utils import (
    get_flow_from_delta_pose_and_depth,
    filter_flow_by_face_index,
    filter_flow_by_mask,
    flow_to_coords,
    Warp)
from scipy.stats import norm 

EPS, INF = 1e-10, 1e10

@LOSSES.register_module()
class MVCLoss(nn.Module):
    def __init__(self, 
                loss_func_cfg:dict,
                valid_view_num_threshold:int=2, 
                gaussian_weights:bool=False, 
                use_threshold_for_gaussian:bool=True,
                var_threshold:float=1., 
                invalid_flow_num:float=400.) -> None:
        super().__init__()
        self.valid_view_num_threshold = valid_view_num_threshold
        self.var_threshold = var_threshold
        self.gaussian_weights = gaussian_weights
        self.use_threshold_for_gaussian = use_threshold_for_gaussian
        self.invalid_flow_num = invalid_flow_num
        self.warp_op = Warp(use_mask=True, return_mask=False)
        self.loss_func = build_loss(loss_func_cfg)

    def vis_sigma(self, threshold, view_flow_var, mask, save_path):
        import numpy as np
        import cv2
        vis_var = view_flow_var 
        vis_var[mask == 0] = 0.
        vis_var = vis_var / threshold
        vis_var[vis_var > 1.] = 1.
        vis_var = cv2.applyColorMap((vis_var.cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
        mask_extended = torch.stack([mask, mask, mask], dim=-1).cpu().numpy()
        vis_var[mask_extended == 0] = 255
        cv2.imwrite(save_path, vis_var)


    
    def multiview_var_mean(self, base_view_flow, other_view_flow, view_flow, base_view_flow_valid_mask):
        other_view_coords = flow_to_coords(other_view_flow)
        warped_other_view_coords = self.warp_op(other_view_coords, view_flow) # (N-1, 2, H, W)
        warped_other_view_flow = warped_other_view_coords - flow_to_coords(torch.zeros_like(base_view_flow[None])) # dumpy coords
        valid_mask_per_view = (view_flow[:, 0] < self.invalid_flow_num) | (view_flow[:, 1] < self.invalid_flow_num) # (N-1, H, W)
        all_view_flow = torch.cat([base_view_flow[None], warped_other_view_flow], dim=0) # (N, 2, H, W)
        valid_mask_per_view = torch.cat([base_view_flow_valid_mask[None], valid_mask_per_view], dim=0) # (N, H, W)
        valid_mask = torch.sum(valid_mask_per_view, dim=0) > self.valid_view_num_threshold
        valid_view_num_per_pixel = torch.sum(valid_mask_per_view, dim=0)
        
        view_flow_mean = torch.sum(all_view_flow*valid_mask_per_view[:, None], dim=0, keepdim=False) \
                    / (valid_view_num_per_pixel + EPS)

        valid_view_flag = valid_mask_per_view[:, valid_mask].to(torch.bool)
        valid_points_view_flow = all_view_flow[:, :, valid_mask]
        valid_points_flow_mean = view_flow_mean[:, valid_mask]
        valid_points_flow_var = torch.sum((valid_points_view_flow - valid_points_flow_mean[None])**2, dim=1)
        valid_points_flow_var[~valid_view_flag] = 0
        valid_points_flow_var = torch.sqrt(torch.sum(valid_points_flow_var, dim=0) / torch.sum(valid_view_flag, dim=0))
        view_flow_var = torch.full_like(valid_mask, fill_value=INF, dtype=torch.float32)
        view_flow_var[valid_mask] = valid_points_flow_var
        
        weights = torch.zeros_like(valid_mask, dtype=torch.float32)
        if self.gaussian_weights:
            if valid_points_flow_var.numel() > 0:
                if self.use_threshold_for_gaussian:
                    mask = valid_points_flow_var < self.var_threshold
                    if mask.sum() > 0:  
                        var_std = torch.std(torch.cat([-valid_points_flow_var[mask], valid_points_flow_var[mask]]))
                        gaussian_dis = norm(loc=0, scale=var_std.cpu().numpy()/2)
                        valid_gaussian_weights = torch.from_numpy(gaussian_dis.pdf(valid_points_flow_var.cpu().numpy())).to(weights.device).to(torch.float32)
                        weights[valid_mask] = valid_gaussian_weights / valid_gaussian_weights.max()
                else:
                    var_std = torch.std(torch.cat([-valid_points_flow_var, valid_points_flow_var]))
                    gaussian_dis = norm(loc=0, scale=var_std.cpu().numpy())
                    valid_gaussian_weights = torch.from_numpy(gaussian_dis.pdf(valid_points_flow_var.cpu().numpy())).to(weights.device).to(torch.float32)
                    weights[valid_mask] = valid_gaussian_weights / valid_gaussian_weights.max()
            if False:
                from matplotlib import pyplot as plt
                import numpy as np
                norm_dis = norm(loc=0, scale=var_std.cpu().numpy())
                n, bins, patches = plt.hist(valid_points_coords_var.cpu().numpy(), bins=30, density=1., )
                # x = np.linspace(norm_dis.ppf(0.01), norm_dis.ppf(0.99), 100)
                plt.plot(bins, norm_dis.pdf(bins), 'r-', )
                norm_dis = norm(loc=0, scale=var_std.cpu().numpy()/2)
                plt.plot(bins, norm_dis.pdf(bins), 'b-', )
                plt.savefig('debug.png')
                plt.close()
        else:
            valid_mask = valid_mask & (view_flow_var < self.var_threshold)
            weights[valid_mask] = 1.
        return view_flow_mean, view_flow_var, weights


    def forward(self,
                mv_teacher_pred_flow:torch.Tensor, 
                sv_student_pred_flow:torch.Tensor,
                mv_rotations:torch.Tensor, 
                mv_translations:torch.Tensor, 
                mv_rendered_depths:torch.Tensor,
                mv_rendered_masks:torch.Tensor,
                mv_rendered_faces_index:torch.Tensor,
                internel_k:torch.Tensor, ):
        image_num, view_num = mv_teacher_pred_flow.size(0), mv_teacher_pred_flow.size(1)
        mv_flow_mean_list, mv_flow_var_list, flow_weights_list = [],  [], []
        for i in range(image_num):
            view_flow = get_flow_from_delta_pose_and_depth(
                mv_rotations[i, 0][None].expand(view_num-1, -1, -1), 
                mv_translations[i, 0][None].expand(view_num-1, -1), 
                mv_rotations[i, 1:], mv_translations[i, 1:], 
                mv_rendered_depths[i, 0][None].expand(view_num-1, -1, -1), 
                internel_k[i][None].expand(view_num-1, -1, -1),
                invalid_num=self.invalid_flow_num,
            )
            view_flow = filter_flow_by_mask(
                view_flow, mv_rendered_depths[i, 1:], invalid_num=self.invalid_flow_num
            )
            # view_flow = filter_flow_by_face_index(
            #     view_flow, mv_rendered_faces_index[i, 0][None].expand(view_num-1, -1, -1),
            #     mv_rendered_faces_index[i, 1:], invalid_num=self.invalid_flow_num
            # )
            mv_flow_mean, mv_flow_var, flow_weights = self.multiview_var_mean(
                base_view_flow=mv_teacher_pred_flow[i, 0],
                other_view_flow=mv_teacher_pred_flow[i, 1:],
                view_flow=view_flow,
                base_view_flow_valid_mask=mv_rendered_masks[i, 0],
            )
            mv_flow_mean_list.append(mv_flow_mean)
            mv_flow_var_list.append(mv_flow_var)
            flow_weights_list.append(flow_weights)
        
        mv_flow_mean = torch.stack(mv_flow_mean_list, dim=0)
        mv_flow_var = torch.stack(mv_flow_var_list, dim=0)  
        flow_weights = torch.stack(flow_weights_list, dim=0)

        loss, seq_loss = self.loss_func(
            sv_student_pred_flow, gt_flow=mv_flow_mean, weights=flow_weights
        )
        return loss, seq_loss, flow_weights, mv_flow_var
        
