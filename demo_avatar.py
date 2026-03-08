import os
import torch
import trimesh
import numpy as np
import argparse
import shutil
import collections
from pathlib import Path
import imageio
import pickle
from PIL import Image
from typing import Dict

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from amr.utils import recursive_to
from amr.configs import get_config
from amr.datasets.avatar_dataset import AvatarInferDataset
from amr.models.hrm import HRM
from amr.utils.camera_utils import *
from amr.models.rendering.gs_varen_renderer import GaussianModel
from amr.utils.renderer import cam_crop_to_full


class HRMInferer:
    def __init__(self, model: HRM):
        self.model = model

    def forward_gs(self, item: Dict, source_smal_params: Dict):
        query_points, varen_data = self.model.renderer.get_query_points(source_smal_params, item['source_rgbs'].device)
        latent_points, image_feats = self.model.forward_latent_points(query_points, item['source_rgbs'])
        gs_attr_list, query_points, varen_data = self.model.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            varen_data=varen_data,
            additional_features={"image_feats": image_feats, "image": item['source_rgbs'][:, 0]},
        )
        return gs_attr_list, query_points, varen_data

    def animate_gs(self, 
                   item: Dict, 
                   animate_smal_params: Dict, 
                   img_info: Dict):
        """
        Args:
            item: Dict: input image and mask
            animate_smal_params: Dict: animate smal params
            img_info: Dict: image height, image width, focal length
        Return: 
            output: Dict: rendered results
        """
        # Acquire Canonical Space Gaussian Attributes
        gs_attr_list, query_points, varen_data = self.forward_gs(item, animate_smal_params)
        Nv = varen_data['pose'].shape[1]
        device = query_points.device
        c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
        intr = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Nv, 1, 1).unsqueeze(0)
        height = img_info['height']
        width = img_info['width']
        focal_length = img_info['focal_length']
        intr[:, :, 0, 0] = float(focal_length)
        intr[:, :, 1, 1] = float(focal_length)
        intr[:, :, 0, 2] = float(width) / 2.0
        intr[:, :, 1, 2] = float(height) / 2.0

        # Animate Gaussian
        output = self.model.renderer.forward_animate_gs(
            gs_attr_list,
            query_points,
            varen_data,
            c2w,
            intr,
            height, width,
            torch.ones(size=(1, Nv, 3), device=device),
            debug=False,
        )
        output['comp_rgb'] = output['comp_rgb'][0].permute(0, 2, 3, 1)
        return output

    def render_novel_view_cano(self, item: Dict, source_smal_params: Dict, num_views: int=36):
        source_smal_params_ = {k: torch.zeros_like(v) for k, v in source_smal_params.items()}
        gs_attr_list, query_points, varen_data = self.forward_gs(item, source_smal_params_)  # Canonical Space Gaussian Attributes
        # Build Normalization Matrix
        points = query_points[0]
        N, scale, center = build_normalization_matrix(points)
        points = (points - center) * scale
        # Build Camera to World Matrix
        camera_info = build_c2w(points, num_views=num_views)
        c2w = camera_info['cond_sup_c2w'].to(N.device)

        N_inv = torch.linalg.inv(N)
        c2w_norm = torch.bmm(N_inv[None, ...].repeat(num_views, 1, 1), c2w)
        # Convert C2W to Gaussian Renderer Camera Definition
        S = torch.tensor([[1., 0., 0., 0.],
                        [0.,-1., 0., 0.],
                        [0., 0.,-1., 0.],
                        [0., 0., 0., 1.]], dtype=torch.float32, device=N.device)[None].repeat(num_views, 1, 1)
        c2w_norm = torch.bmm(c2w_norm, S)

        # Build Intrinsic Matrix
        intr = torch.eye(4, dtype=torch.float32, device=N.device).unsqueeze(0).repeat(num_views, 1, 1)
        fov = camera_info['cond_sup_fovy'][0]
        # Compute focal length from vertical FOV and image size
        fov = fov.to(dtype=torch.float32, device=N.device)
        height = item['source_rgbs'].shape[-2]
        width = item['source_rgbs'].shape[-1]
        height_t = torch.tensor(float(height), dtype=torch.float32, device=N.device)
        width_t = torch.tensor(float(width), dtype=torch.float32, device=N.device)
        half_height = 0.5 * height_t
        half_width = 0.5 * width_t
        tan_half_fovy = torch.tan(fov / 2.0)
        # For general aspect ratio: fx = (W/2) / (aspect * tan(fovy/2)) where aspect = W/H
        aspect = width_t / height_t
        fy = half_height / tan_half_fovy
        fx = half_width / (aspect * tan_half_fovy)
        intr[:, 0, 0] = fx
        intr[:, 1, 1] = fy
        intr[:, 0, 2] = half_width
        intr[:, 1, 2] = half_height

        # Create Gaussian Model
        gs_model = [GaussianModel(
            xyz=(query_points + gs_attr_list[0].offset_xyz).squeeze(),
            opacity=gs_attr_list[0].opacity,
            rotation=gs_attr_list[0].rotation,
            scaling=gs_attr_list[0].scaling,
            shs=gs_attr_list[0].shs,
            use_rgb=gs_attr_list[0].use_rgb,
        ) for _ in range(num_views)]
        # Render Canonical Space Novel View
        output = self.model.renderer.forward_single_batch(
                    gs_model,
                    c2w_norm,
                    intr,
                    height,
                    width,
                    torch.ones(size=(num_views, 3), device=N.device),
                )
        return output


def main(args):
    # Load Horse Reconstruction Model
    model_cfg = str(Path(args.checkpoint).parent.parent / '.hydra/config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    model = HRM.load_from_checkpoint(checkpoint_path=args.checkpoint, cfg=model_cfg, strict=False)
    model = model.to(args.device)
    model = model.eval()

    # Create Inferer
    inferer = HRMInferer(model)

    # Load Varen Params
    animation_params = torch.load(args.animation_params_path, map_location='cpu', weights_only=True)
    for k, v in animation_params.items():  # TODO: Use a elegant way to get the number of views or process keys here
        if "global_orient" in k:
            N = v.shape[0] # if v.shape[0] <= 180 else 180
            break

    def process_params(params: Dict) -> Dict:
        smal_params = {}
        for k, v in params.items():
            if "global_orient" in k:
                smal_params['global_orient'] = matrix_to_axis_angle(
                        rotation_6d_to_matrix(
                            params[k].view(N, -1, 6))).unsqueeze(0).to(args.device) if params[k].shape[-2:] != (3, 3) else matrix_to_axis_angle(params[k]).unsqueeze(0).to(args.device)
            elif "pose" in k:
                smal_params['pose'] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(
                        params[k].view(N, -1, 6))).unsqueeze(0).to(args.device) if params[k].shape[-2:] != (3, 3) else matrix_to_axis_angle(params[k]).unsqueeze(0).to(args.device)
            elif "betas" in k:
                smal_params['betas'] = params[k].mean(dim=0, keepdim=True).to(args.device)
            elif "tail_scale" in k:
                smal_params['tail_scale'] = params[k].mean(dim=0, keepdim=True).to(args.device)
            elif "cam" in k and "cam_t" not in k:
                smal_params['cam'] = params[k].to(args.device)

        for k, v in smal_params.items():
            if k != "betas" and k != "tail_scale":
                smal_params[k] = v[:, :N]
        return smal_params
    
    animation_params = process_params(animation_params)

    out_dir = os.path.join(args.out_folder, os.path.splitext(os.path.basename(args.img_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.img_path, os.path.join(out_dir, str(0).zfill(6) + ".jpg"))
    # Obtain Frame and Segmetation Mask
    frame = np.array(Image.open(args.img_path))  # TODO: Modify Loading Frame Idx from a Video
    track_masks = pickle.load(open(args.track_mask_file, "rb"))[:N]  # [N, H, W]

    # Create AvatarInferDataset
    dataset = AvatarInferDataset(image=[frame], mask=[track_masks[0]], intrs=[None], render_image_res=448)  # TODO: Choose mask according to the frame idx

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for item in dataloader:  # Only one batch
        item = recursive_to(item, args.device)

        with torch.no_grad():
            # Render Canonical Space Novel View
            out_cano = inferer.render_novel_view_cano(item, animation_params)

            # Animate Gaussian
            # Compute translation for Original images
            scaled_focal_length = 5000 / 256 * max(item['image_ori_height'].item(), item['image_ori_width'].item())
            box_center, box_size, img_size = [], [], []  # [N, 2], [N], [N, 2]
            for i in range(N):
                y, x = np.where(track_masks[i] > 0)
                x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
                box_center.append(np.array([(x1 + x2) // 2, (y1 + y2) // 2]))
                box_size.append(max(x2 - x1, y2 - y1))
                img_size.append([item['image_ori_width'].item(), item['image_ori_height'].item()])
            box_center = torch.tensor(box_center)
            box_size = torch.tensor(box_size)
            img_size = torch.tensor(img_size)
            
            pred_cam_t_full = cam_crop_to_full(animation_params['cam'].cpu()[:N], 
                                               box_center, 
                                               box_size, 
                                               img_size,
                                               scaled_focal_length)
            animation_params['trans'] = pred_cam_t_full.unsqueeze(0).to(args.device)
            img_info = {
                "height": 256,
                "width": 256,
                "focal_length": 5000,
            }
            img_info = {
                "height": item['image_ori_height'].item(),
                "width": item['image_ori_width'].item(),
                "focal_length": scaled_focal_length,
            }
            out_animate = inferer.animate_gs(item, animation_params, img_info)

    def save_video(out, video_path):
        comp_rgb = out.get('comp_rgb').detach().cpu().numpy()
        # Save comp_rgb as a video using imageio
        writer = imageio.get_writer(video_path, fps=10)
        for i in range(comp_rgb.shape[0]):
            frame = (comp_rgb[i] * 255.0).clip(0, 255).astype(np.uint8)
            writer.append_data(frame)
        writer.close()

    save_video(out_cano, os.path.join(out_dir, "cano.mp4"))
    save_video(out_animate, os.path.join(out_dir, "animate.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Horse Avatar Reconstruction Demo Code')
    parser.add_argument("--animation_params_path", type=str, required=True, help="Path to animation parameters")
    parser.add_argument("--track_mask_file", type=str, required=True, help="Path to track mask file")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--out_folder", type=str, default="demo_out", help="Path to the output folder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    main(args)
