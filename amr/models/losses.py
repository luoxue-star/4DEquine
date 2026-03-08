import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Dict, Optional, Tuple, Union
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_axis_angle
import torch.nn.functional as F


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMAL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMAL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        mask = torch.ones_like(pred_param, device=pred_param.device, dtype=pred_param.dtype)
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param*mask, gt_param*mask))
        return loss_param.sum()


class PosePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(PosePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        self.register_buffer("mean_pose", torch.from_numpy(data_prior["mean_pose"]).float())
        self.register_buffer("precs", torch.from_numpy(np.array(data_prior["pic"])).float())

        use_index = np.ones(105, dtype=bool)
        use_index[:3] = False  # global rotation set False
        self.register_buffer("use_index", torch.from_numpy(use_index).float())

    def forward(self, x, has_gt):
        """
        Args:
            x: (batch_size, 35, 3, 3)
            has_gt: has pose?
        Returns:
            pose prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x = x[~has_gt]
        x = matrix_to_axis_angle(x.reshape(-1, 3, 3))
        delta = x.reshape(-1, 35*3) - self.mean_pose
        loss = torch.tensordot(delta, self.precs, dims=([1], [0])) * self.use_index
        return (loss ** 2).mean()


class ShapePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(ShapePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        model_covs = np.array(data_prior["cluster_cov"])  # shape: (5, 41, 41)
        inverse_covs = np.stack(
            [np.linalg.inv(model_cov + 1e-5 * np.eye(model_cov.shape[0])) for model_cov in model_covs],
            axis=0)
        prec = np.stack([np.linalg.cholesky(inverse_cov) for inverse_cov in inverse_covs], axis=0)

        self.register_buffer("betas_prec", torch.FloatTensor(prec))
        self.register_buffer("mean_betas", torch.FloatTensor(data_prior["cluster_means"]))

    def forward(self, x, category, has_gt):
        """
        Args:
            x: predicted betas (batch_size, 41)
            category: animal category (batch_size,)
            has_gt: has shape?
        Returns:
            shape prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x, category = x[~has_gt], category[~has_gt]
        delta = (x - self.mean_betas[category.long()])  # [batch_size, 41]
        loss = []
        for x0, c0 in zip(delta, category):
            loss.append(torch.tensordot(x0, self.betas_prec[c0], dims=([0], [0])))
        loss = torch.stack(loss, dim=0)
        return (loss ** 2).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.stack((features, features), dim=1)
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SmoothNetLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, denoise, gt, mask=None):
        """
        denoise: (N, T, C)
        gt: (N, T, C)
        mask: (N, T, C)
        """
        denoise = denoise.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        gt = gt.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        accel_gt = gt[:, :, :-2] - 2 * gt[:, :, 1:-1] + gt[:, :, 2:]
        accel_denoise = denoise[:, :, :-2] - 2 * denoise[:, :, 1:-1] + denoise[:, :, 2:]
        
        loss_accel = self.loss_fn(accel_denoise, accel_gt)  # (N, C, T-2)

        if mask is not None:
            mask = mask.permute(0, 2, 1)  # (N, T, K)->(N, K, T)
            mask_accel = mask[:, :, :-2] * mask[:, :, 1:-1] * mask[:, :, 2:]  # (N, K, T-2)
            
            if loss_accel.shape[1] != mask_accel.shape[1]:
                coord_dim = loss_accel.shape[1] // mask_accel.shape[1]
                mask_accel = mask_accel.repeat_interleave(coord_dim, dim=1)
            
            loss_accel = loss_accel * mask_accel
        loss_accel = loss_accel.sum()

        return loss_accel


class PoseBonePriorLoss(nn.Module):
    def __init__(self, path_prior, loss_type='l2'):
        super(PoseBonePriorLoss, self).__init__()
        self.loss_type = loss_type

        data_prior = torch.load(path_prior, weights_only=True)

        self.register_buffer("pose_mean", data_prior["pose_mean"])
        self.register_buffer("pose_cov_inv", data_prior["pose_cov"].inverse())
        self.register_buffer("bone_mean", torch.ones_like(data_prior["bone_mean"]) if loss_type == 'l2' else data_prior["bone_mean"])
        self.register_buffer("bone_cov_inv", data_prior["bone_cov"].inverse())

    def forward(self, pose, bone, has_gt):
        """
        Args:
            pose: (batch_size, 24, 3)
            bone: (batch_size, 24)
        Returns:
            pose bone prior loss
        """
        if self.loss_type == 'l2':
            loss_pose = self.l2_loss(pose, self.pose_mean, has_gt)
            loss_bone = self.l2_loss(bone, self.bone_mean, has_gt)
        elif self.loss_type == 'mahalanobis':
            loss_pose = self.prior_loss(pose, self.pose_mean, self.pose_cov_inv, has_gt)
            loss_bone = self.prior_loss(bone, self.bone_mean, self.bone_cov_inv, has_gt)
        else:
            raise NotImplementedError('Unsupported loss function')
        return loss_pose, loss_bone

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def prior_loss(self, x, mean, cov_in, has_gt):
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x = x[~has_gt]
        # Squared Mahalanobis distance
        xm = x - mean
        dis = xm @ cov_in @ xm.t()
        dis = torch.diag(dis).sum()
        return dis
    
    def l2_loss(self, x, mean, has_gt):
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x = x[~has_gt]
        # L2 distance
        dis = (x - mean) ** 2
        dis = dis.sum()
        return dis


class SilhouetteLoss(nn.Module):
    def __init__(self):
        """
        Silhouette loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(SilhouetteLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_silhouette: torch.Tensor, gt_silhouette: torch.Tensor) -> torch.Tensor:
        """
        Compute silhouette loss.
        Args:
            pred_silhouette (torch.Tensor): Tensor of shape [B, H, W] containing predicted silhouette.
            gt_silhouette (torch.Tensor): Tensor of shape [B, H, W] containing ground truth silhouette.
        Returns:
            torch.Tensor: Silhouette loss.
        """
        h, w = gt_silhouette.shape[-2:]
        loss = self.loss_fn(pred_silhouette, gt_silhouette).sum(dim=(1, 2)) / (h * w)
        return loss.sum()


class LPIPSLoss(nn.Module):
    """
    Compute LPIPS loss between two images.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None, prefetch: bool = True, compile_model: bool = True):
        super().__init__()
        self.compile_model = compile_model
        self.default_device = self._resolve_device(device)
        self.cached_models: Dict[Tuple[str, torch.device], nn.Module] = {}
        if prefetch:
            self.prefetch_models(self.default_device)

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                current_idx = torch.cuda.current_device()
                return torch.device("cuda", current_idx)
            return torch.device("cpu")
        return torch.device(device)

    @staticmethod
    def _device_cache_key(model_name: str, device: torch.device) -> Tuple[str, torch.device]:
        # torch.device is hashable, so we can use it directly.
        return model_name, device

    def _build_model(self, model_name: str, device: torch.device) -> nn.Module:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import lpips

            model = lpips.LPIPS(net=model_name, eval_mode=True, verbose=False)

        model = model.to(device)

        if self.compile_model:
            try:
                model = torch.compile(model)
            except Exception:
                # Fallback to non-compiled model if torch.compile is not supported in the environment.
                pass

        return model

    def _get_model(self, model_name: str, device: torch.device) -> nn.Module:
        cache_key = self._device_cache_key(model_name, device)
        if cache_key not in self.cached_models:
            self.cached_models[cache_key] = self._build_model(model_name, device)
        return self.cached_models[cache_key]

    def prefetch_models(self, device: Optional[torch.device] = None):
        device = self._resolve_device(device)
        for model_name in ['alex', 'vgg']:
            self._get_model(model_name, device)

    def forward(self, x, y, is_training: bool = True):
        """
        Assume images are 0-1 scaled and channel first.
        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
            is_training: whether to use VGG or AlexNet.
        Returns:
            Mean-reduced LPIPS loss across batch.
        """
        model_name = 'vgg' if is_training else 'alex'
        model_device = x.device
        loss_fn = self._get_model(model_name, model_device)
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        y = y.reshape(N*M, C, H, W)
        image_loss = loss_fn(x, y, normalize=True).mean(dim=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(dim=1)
        all_loss = batch_loss.mean()
        return all_loss


class PixelLoss(nn.Module):
    """
    Pixel-wise loss between two images.
    """

    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise NotImplementedError(f'Unknown pixel loss option: {loss_type}')

    @torch.compile
    def forward(self, x, y):
        """
        Assume images are channel first.
        Args:
            x: [N, M, C, H, W]
            y: [N, M, C, H, W]
        Returns:
            Mean-reduced pixel loss across batch.
        """
        N, M, C, H, W = x.shape
        x = x.reshape(N*M, C, H, W)
        y = y.reshape(N*M, C, H, W)
        image_loss = self.loss_fn(x, y).mean(dim=[1, 2, 3])
        batch_loss = image_loss.reshape(N, M).mean(dim=1)
        all_loss = batch_loss.mean()
        return all_loss


class ASAP_Loss(nn.Module):

    def forward(self, scaling, r=1, **params):
        """where r is the radius of the ball between max-axis and min-axis."""
        raise NotImplementedError(
            "ASAP_Loss is not implemented yet in Inference version"
        )


class Heuristic_ASAP_Loss(nn.Module):
    def __init__(self, group_dict, group_body_mapping):
        super(Heuristic_ASAP_Loss, self).__init__()

        self.group_dict = group_dict  # register weights fro different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _ball_loss):

        _loss = 0.0
        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = self.group_body_mapping[key]
            _loss += key_weights * _ball_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, scaling, r=5, **params):
        """where r is the radius of the ball between max-axis and min-axis."""
        "human motion or rotation is very different in each body parts, for example, the head is more stable than the leg and hand, so we use heuristic_ball_loss"

        _scale = scaling

        _scale_min = torch.min(_scale, dim=-1)[0]
        _scale_max = torch.max(_scale, dim=-1)[0]

        scale_ratio = _scale_max / (_scale_min + 1e-6)

        _ball_loss = torch.clamp(scale_ratio, min=r) - r

        return self._heurisitic_loss(_ball_loss)


class ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from 1.8/32 = 0.0562."""

        offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return offset_loss.mean()


class Heuristic_ACAP_Loss(nn.Module):
    """As close as possibel loss"""

    def __init__(self, group_dict, group_body_mapping):
        super(Heuristic_ACAP_Loss, self).__init__()

        self.group_dict = group_dict  # register weights fro different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _offset_loss):

        _loss = 0.0
        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = self.group_body_mapping[key]
            _loss += key_weights * _offset_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from human prior model, 1.8/32 = 0.0562."""
        "human motion or rotation is very different in each body parts, for example, the head is more stable than the leg and hand, so we use heuristic_ball_loss"

        _offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return self._heurisitic_loss(_offset_loss)


# {'Pelvis': 0, 'Spine': 1, 'Spine1': 2, 'Spine2': 3, 
# 'LScapula': 4, 'RScapula': 5, 
# 'LFLeg1': 6, 'LFLeg2': 7, 'LFLeg3': 8, 'LFLeg4': 9, 'LFToe': 10, 
# 'RFLeg1': 11, 'RFLeg2': 12, 'RFLeg3': 13, 'RFLeg4': 14, 'RFToe': 15, 
# 'LBLeg1': 16, 'LBLeg2': 17, 'LBLeg3': 18, 'LBLeg4': 19, 'LBToe': 20, 
# 'RBLeg1': 21, 'RBLeg2': 22, 'RBLeg3': 23, 'RBLeg4': 24, 'RBToe': 25, 
# 'Neck': 26, 'Neck1': 27, 'Neck2': 28, 'Head': 29, 
# 'Tail1': 30, 'Tail2': 31, 'Tail3': 32, 'Tail4': 33, 'Tail5': 34, 
# 'Mouth': 35, 'LEar': 36, 'REar': 37} 
def leg_sideway_error(optimed_pose_with_glob):
    assert optimed_pose_with_glob.shape[1] == 38
    leg_indices_right = np.asarray([6, 7, 8, 9, 10, 16, 17, 18, 19, 20])      # front, back
    leg_indices_left = np.asarray([11, 12, 13, 14, 15, 21, 22, 23, 24, 25])     # front, back
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[2] = -1
    x0_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec
    x0_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec
    loss_pose_legs_side = (x0_legs_left[:, 1]**2).mean() + (x0_legs_right[:, 1]**2).mean()
    return loss_pose_legs_side


def leg_torsion_error(optimed_pose_with_glob):
    leg_indices_right = np.asarray([6, 7, 8, 9, 10, 16, 17, 18, 19, 20])      # front, back
    leg_indices_left = np.asarray([11, 12, 13, 14, 15, 21, 22, 23, 24, 25])     # front, back
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_legs_left = x0_rotmat[:, leg_indices_left, :, :]
    x0_rotmat_legs_right = x0_rotmat[:, leg_indices_right, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec_x[0] = 1      # in x direction
    x_x_legs_left = x0_rotmat_legs_left.reshape((-1, 3, 3))@vec_x
    x_x_legs_right = x0_rotmat_legs_right.reshape((-1, 3, 3))@vec_x
    loss_pose_legs_torsion = (x_x_legs_left[:, 1]**2).mean() + (x_x_legs_right[:, 1]**2).mean()
    return loss_pose_legs_torsion


def tail_sideway_error(optimed_pose_with_glob):
    tail_indices = np.asarray([30, 31, 32, 33, 34])      
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    '''vec[2] = -1    
    x0_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec
    loss_pose_tail_side = (x0_tail[:, 1]**2).mean()'''
    vec[0] = -1    
    x0_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec
    loss_pose_tail_side = (x0_tail[:, 1]**2).mean()
    return loss_pose_tail_side


def tail_torsion_error(optimed_pose_with_glob):
    tail_indices = np.asarray([30, 31, 32, 33, 34])      
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_tail = x0_rotmat[:, tail_indices, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    '''vec_x[0] = 1      # in x direction
    x_x_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec_x
    loss_pose_tail_torsion = (x_x_tail[:, 1]**2).mean()'''
    vec_x[2] = 1      # in y direction
    x_x_tail = x0_rotmat_tail.reshape((-1, 3, 3))@vec_x
    loss_pose_tail_torsion = (x_x_tail[:, 1]**2).mean()
    return loss_pose_tail_torsion


def spine_sideway_error(optimed_pose_with_glob):
    spine_indices = np.asarray([1, 2, 3, 4])  
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_spine = x0_rotmat[:, spine_indices, :, :]
    vec = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec[0] = -1    
    x0_spine = x0_rotmat_spine.reshape((-1, 3, 3))@vec
    loss_pose_spine_side = (x0_spine[:, 1]**2).mean()
    return loss_pose_spine_side


def spine_torsion_error(optimed_pose_with_glob):
    spine_indices = np.asarray([1, 2, 3, 4])      
    x0_rotmat = optimed_pose_with_glob   # (1, 38, 3, 3)
    x0_rotmat_spine = x0_rotmat[:, spine_indices, :, :]
    vec_x = torch.zeros((3, 1)).to(device=optimed_pose_with_glob.device, dtype=optimed_pose_with_glob.dtype)
    vec_x[2] = 1    # vec_x[0] = 1      # in z direction
    x_x_spine = x0_rotmat_spine.reshape((-1, 3, 3))@vec_x
    loss_pose_spine_torsion = (x_x_spine[:, 1]**2).mean()
    return loss_pose_spine_torsion
