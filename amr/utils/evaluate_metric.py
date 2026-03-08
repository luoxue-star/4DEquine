
import torch
import numpy as np

import torch.nn as nn
from typing import Dict, List, Union
from pytorch3d.transforms import axis_angle_to_matrix
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.amp import custom_fwd

from kornia.enhance.adjust import adjust_brightness_accumulative as kornia_aba  # type: ignore
from .pck_accuracy import keypoint_pck_accuracy


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1 ** 2).sum(dim=(1, 2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1.float(), X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K.float())
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1).float()
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U.float(), Vh.float()).float()))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R.float(), mu1.float())

    # 7. Error:
    S1_hat = scale * torch.matmul(R.float(), S1.float()).float() + t

    return S1_hat.permute(0, 2, 1)


class Evaluator:
    def __init__(self, smal_model, image_size: int=256, pelvis_ind: int = 7):
        self.pelvis_ind = pelvis_ind
        self.smal_model = smal_model
        self.image_size = image_size

        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        self._image_metrics = (self.lpips_metric, self.psnr_metric, self.ssim_metric)
        for metric in self._image_metrics:
            metric.eval()
            for param in metric.parameters():
                param.requires_grad_(False)

        self._metric_device = self._get_metric_device()

    def _get_metric_device(self) -> torch.device:
        for metric in self._image_metrics:
            params = list(metric.parameters())
            if params:
                return params[0].device
            buffers = list(metric.buffers())
            if buffers:
                return buffers[0].device
        return torch.device('cpu')

    def _move_metrics_to_device(self, device: torch.device):
        if device == self._metric_device:
            return
        for metric in self._image_metrics:
            metric.to(device)
        self._metric_device = device

    def _compute_metric(self, metric: nn.Module, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        preds = preds.float()
        target = target.float()
        metric.reset()
        metric.update(preds, target)
        return metric.compute()

    @custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def _compute_image_metrics(self, rgb: torch.Tensor, rgb_gt: torch.Tensor):
        self._move_metrics_to_device(rgb.device)
        rgb = rgb.detach()
        rgb_gt = rgb_gt.detach()

        br_list = torch.linspace(0.0, 2.0, 100, device=rgb.device, dtype=rgb.dtype)
        pred_mask = (~(rgb == 1.0).all(dim=1, keepdim=True)).expand_as(rgb)

        best_rgb = rgb
        best_psnr = float('-inf')

        for brightness in br_list:
            adjusted = kornia_aba(rgb, brightness)
            candidate = rgb.clone()
            candidate[pred_mask] = adjusted[pred_mask]
            psnr_value = self._compute_metric(self.psnr_metric, candidate, rgb_gt)
            psnr_scalar = float(psnr_value)
            if psnr_scalar > best_psnr:
                best_psnr = psnr_scalar
                best_rgb = candidate

        psnr_value = self._compute_metric(self.psnr_metric, best_rgb, rgb_gt)
        ssim_value = self._compute_metric(self.ssim_metric, best_rgb, rgb_gt)
        lpips_value = self._compute_metric(self.lpips_metric, best_rgb, rgb_gt)

        return psnr_value, ssim_value, lpips_value

    def compute_pck_bbox(self, output: Dict, batch: Dict, pck_threshold: Union[List, None]):
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().cpu().float().numpy()
        gt_keypoints_2d = batch['keypoints_2d'].detach().cpu().float().numpy() 
        conf = gt_keypoints_2d[:, :, -1][:, None, :]
        gt_keypoints_2d = gt_keypoints_2d[:, :, :-1]
        self.pck_threshold_list = []
        if pck_threshold is not None:
            for i in range(len(pck_threshold)):
                self.pck_threshold_list.append(np.array([pck_threshold[i]] * len(pred_keypoints_2d), dtype=np.float32))

        batch_size = pred_keypoints_2d.shape[0]
        pred_keypoints_2d = pred_keypoints_2d[:, None, :, :]
        gt_keypoints_2d = gt_keypoints_2d[:, None, :, :]

        pcks = []
        for pck_threshold in self.pck_threshold_list:
            pcks.append([
                keypoint_pck_accuracy(
                    pred_keypoints_2d[i, 0, :, :][None],
                    gt_keypoints_2d[i, 0, :, :][None],
                    conf[i, 0, :][None] > 0.5,
                    thr=pck_threshold[i],
                    normalize=np.ones((1, 2))  # Already in [-0.5,0.5] range. No need to normalize
                )
                for i in range(batch_size)]
            )
        return np.mean(np.array(pcks), axis=1)
    
    def compute_pck(self, output: Dict, batch: Dict, pck_threshold: Union[List, None]):
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().cpu()
        gt_keypoints_2d = batch['keypoints_2d'].detach().cpu()
        self.pck_threshold_list = []
        
        pred_keypoints_2d = (pred_keypoints_2d + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
        conf = gt_keypoints_2d[:, :, -1]
        gt_keypoints_2d = (gt_keypoints_2d[:, :, :-1] + 0.5) * self.image_size  # * batch['bbox_expand_factor'].detach().cpu().numpy().reshape(-1, 1, 1)
        if pck_threshold is not None:
            for i in range(len(pck_threshold)):
                self.pck_threshold_list.append(torch.tensor([pck_threshold[i]] * len(pred_keypoints_2d), dtype=torch.float32))

        pcks = []
        seg_area = torch.sum(batch['mask'].detach().cpu().reshape(batch['mask'].shape[0], -1), dim=-1).unsqueeze(-1)
        total_visible = torch.sum(conf, dim=-1)
        for th in self.pck_threshold_list:
            dist = torch.norm(pred_keypoints_2d - gt_keypoints_2d, dim=-1)

            hits = (dist / torch.sqrt(seg_area)) < th.unsqueeze(1)
            pck = torch.sum(hits.float() * conf, dim=-1) / total_visible
            pcks.append(pck.float().numpy().tolist())
        return torch.mean(torch.tensor(pcks), dim=1)

    def compute_pa_mpjpe(self, pred_joints, gt_joints):
        S1_hat = compute_similarity_transform(pred_joints, gt_joints)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().float().numpy() * 1000
        return pa_mpjpe.mean()

    def compute_pa_mpvpe(self, gt_vertices: torch.Tensor, pred_vertices: torch.Tensor):
        batch_size = pred_vertices.shape[0]
        S1_hat = compute_similarity_transform(pred_vertices, gt_vertices)
        pa_mpvpe = torch.sqrt(((S1_hat - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().float().numpy() * 1000
        return pa_mpvpe.mean()

    def eval_3d(self, output: Dict, batch: Dict, f_score_threshold: List[float]=[0.005, 0.015]):
        """
        Evaluate current batch
        Args:
            output: model output
            batch: model input
        Returns: evaluate metric
        """
        if batch['has_smal_params']["betas"].sum() == 0:
            return 0., 0.

        pred_keypoints_3d = output["pred_keypoints_3d"].detach()
        pred_keypoints_3d = pred_keypoints_3d[:, None, :, :]
        batch_size = pred_keypoints_3d.shape[0]
        num_samples = pred_keypoints_3d.shape[1]
        gt_keypoints_3d = batch['keypoints_3d'][:, :, :-1].unsqueeze(1).repeat(1, num_samples, 1, 1)
        gt_vertices = self.smal_forward(batch)
        
        # Align predictions and ground truth such that the pelvis location is at the origin
        pred_keypoints_3d -= pred_keypoints_3d[:, :, [self.pelvis_ind]]
        gt_keypoints_3d -= gt_keypoints_3d[:, :, [self.pelvis_ind]]

        pa_mpjpe = self.compute_pa_mpjpe(pred_keypoints_3d.reshape(batch_size * num_samples, -1, 3),
                                         gt_keypoints_3d.reshape(batch_size * num_samples, -1, 3))
        pa_mpvpe = self.compute_pa_mpvpe(gt_vertices, output['pred_vertices'])
        error_accel = self.compute_error_accel(gt_keypoints_3d[:, :, :-1], pred_keypoints_3d[:, :, :-1], batch['keypoints_3d'][:, :, -1])
        return pa_mpjpe, pa_mpvpe, error_accel
    
    def eval_2d(self, output: Dict, batch: Dict, pck_threshold: List[float]=[0.10, 0.15]):
        pck = self.compute_pck_bbox(output, batch, pck_threshold=pck_threshold)
        auc = self.compute_auc(batch, output)
        error_accel = self.compute_error_accel(batch['keypoints_2d'][:, :, :-1], output['pred_keypoints_2d'], batch['keypoints_2d'][:, :, -1])
        return pck.tolist(), auc, error_accel
    
    def compute_auc(self, batch: Dict, output: Dict, threshold_min: int=0.0, threshold_max: int=1.0, steps: int=100):
        thresholds = np.linspace(threshold_min, threshold_max, steps)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)
        pck_curve = []
        for th in thresholds:
            pck_curve.append(self.compute_pck(output, batch, [th]))
        pck_curve = torch.tensor(pck_curve).tolist()
        auc = np.trapz(pck_curve, thresholds)
        auc /= norm_factor
        return auc

    def smal_forward(self, batch: Dict):
        batch_size = batch['img'].shape[0]
        smal_params_source = batch['smal_params']
        device = batch['img'].device

        smal_params = {}
        smal_params['global_orient'] = axis_angle_to_matrix(
            smal_params_source['global_orient'].reshape(batch_size, -1)
        ).unsqueeze(1).to(device)
        smal_params['pose'] = axis_angle_to_matrix(
            smal_params_source['pose'].reshape(batch_size, -1, 3)
        ).to(device)

        for key, value in smal_params_source.items():
            if key in ('global_orient', 'pose'):
                continue
            smal_params[key] = value.to(device)
        with torch.no_grad():
            smal_output = self.smal_model(**smal_params)
        vertices = smal_output.vertices
        return vertices

    def compute_iou(self, batch: Dict, output: Dict):
        gt_mask = batch['mask']
        img_border_mask = batch['img_border_mask']
        pred_mask = output['pred_mask']

        # Do not penalize parts of the segmentation outside the img range
        gt_mask = (gt_mask * img_border_mask) + pred_mask * (1.0 - img_border_mask)
        intersection = torch.sum((pred_mask * gt_mask).reshape(pred_mask.shape[0], -1), dim=-1)
        union = torch.sum(((pred_mask + gt_mask).reshape(pred_mask.shape[0], -1) > 0.0).float(), dim = -1)
        acc_IOU_SCORE = intersection / union
        return acc_IOU_SCORE.mean().item()

    # def compute_psnr(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor):
    #     self._move_metrics_to_device(pred_rgb.device)
    #     with torch.no_grad():
    #         value = self._compute_metric(self.psnr_metric, pred_rgb, gt_rgb)
    #     return value.detach().cpu().item()

    # def compute_ssim(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor):
    #     self._move_metrics_to_device(pred_rgb.device)
    #     with torch.no_grad():
    #         value = self._compute_metric(self.ssim_metric, pred_rgb, gt_rgb)
    #     return value.detach().cpu().item()

    # def compute_lpips(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor):
    #     self._move_metrics_to_device(pred_rgb.device)
    #     with torch.no_grad():
    #         value = self._compute_metric(self.lpips_metric, pred_rgb, gt_rgb)
    #     return value.detach().cpu().item()

    def eval_image(self, output: Dict, batch: Dict):
        pred_rgb = output['comp_rgb']
        gt_rgb = batch['render_image']
        B, N, C, H, W = pred_rgb.shape
        pred_rgb = pred_rgb.reshape(B*N, C, H, W)
        gt_rgb = gt_rgb.reshape(B*N, C, H, W)
        self._move_metrics_to_device(pred_rgb.device)

        with torch.no_grad():
            psnr_value, ssim_value, lpips_value = self._compute_image_metrics(pred_rgb, gt_rgb)

        return (
            psnr_value.detach().cpu().item(),
            ssim_value.detach().cpu().item(),
            lpips_value.detach().cpu().item()
        )
    
    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes 2D acceleration error:
            1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (NxKx2 or 3): Ground truth 2D or 3D joints. N frames, K joints, (x,y) or (x,y,z) coords.
            joints_pred (NxKx2 or 3): Predicted 2D or 3D joints. N frames, K joints, (x,y) or (x,y,z) coords.
            vis (N, K): Visibility mask for each frame.
        Returns:
            error_accel (N-2): A vector of acceleration errors.
        """
        # (N-2)xKx2 or 3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
        normed = torch.linalg.norm(accel_pred - accel_gt, dim=2)
        
        if vis is None:
            new_vis = torch.ones((len(normed), normed.shape[1]), dtype=torch.bool, 
                                device=normed.device)  # (N-2)x14
        else:
            if vis.dim() == 1:
                invis = torch.logical_not(vis)
                invis1 = torch.roll(invis, shifts=-1, dims=0)
                invis2 = torch.roll(invis, shifts=-2, dims=0)
                new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]
                new_vis = torch.logical_not(new_invis)
                new_vis = new_vis.unsqueeze(1).expand(-1, normed.shape[1])  # (N-2)x14
            else:
                invis = torch.logical_not(vis)  # (N,K)
                invis1 = torch.roll(invis, shifts=-1, dims=0)  # (N,K)
                invis2 = torch.roll(invis, shifts=-2, dims=0)  # (N,K)
                new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]  # (N-2,K)
                new_vis = torch.logical_not(new_invis)  # (N-2,K)
        
        masked_normed = normed * new_vis.float()
        
        valid_count = torch.sum(new_vis, dim=1).float()
        valid_count = torch.clamp(valid_count, min=1.0)
        
        frame_errors = torch.sum(masked_normed, dim=1) / valid_count
        
        return frame_errors.mean().item()
