import torch
import pickle
import einops
import pytorch_lightning as pl
from torchvision.utils import make_grid
from typing import Any, Dict
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix, axis_angle_to_matrix, matrix_to_rotation_6d
from ..utils.evaluate_metric import Evaluator
from yacs.config import CfgNode
from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, SmoothNetLoss
from .varen import VAREN
from .heads import build_smal_head
from .components.temporal_attention import TemporalAttention
import wandb

log = get_pylogger(__name__)


class AniMerVAREN(pl.LightningModule):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup AniMer-Varen model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)['state_dict']
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
            
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        if cfg.MODEL.get("ONLY_TEMPORAL", False) and not cfg.MODEL.BACKBONE.get("FINETUNE", False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            log.info("Backbone is frozen")

        # Create SMAL head
        self.smal_head = build_smal_head(cfg)
        if cfg.MODEL.SMAL_HEAD.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading smal head weights from {cfg.MODEL.SMAL_HEAD.PRETRAINED_WEIGHTS}')
            state_dict = torch.load(cfg.MODEL.SMAL_HEAD.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)['state_dict']
            state_dict = {k.replace('smal_head.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.smal_head.load_state_dict(state_dict, strict=False)
        if cfg.MODEL.get("ONLY_TEMPORAL", False) and not cfg.MODEL.SMAL_HEAD.get("FINETUNE", False):
            for param in self.smal_head.parameters():
                param.requires_grad = False
            log.info("SMAL head is frozen")

        # Create temporal attention
        cfg_temporal_module = cfg.MODEL.get('TEMPORAL_MODULE', None)
        self.temporal_module = TemporalAttention(**cfg_temporal_module) if cfg_temporal_module is not None else None

        # Create motion attention
        cfg_motion_module = cfg.MODEL.get('MOTION_MODULE', None)
        self.motion_module = TemporalAttention(**cfg_motion_module) if cfg_motion_module is not None else None

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smal_parameter_loss = ParameterLoss()
        self.smooth_net_loss = SmoothNetLoss()

        # Instantiate SMAL model
        self.smal = VAREN(cfg.SMAL.MODEL_PATH, use_muscle_deformations=True)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smal.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

        self.evaluator = Evaluator(smal_model=self.smal)

    def get_parameters(self):
        all_params = list(self.smal_head.parameters())
        all_params += list(self.backbone.parameters())
        all_params += list(self.temporal_module.parameters()) if self.temporal_module is not None else []
        all_params += list(self.motion_module.parameters()) if self.motion_module is not None else []
        return all_params

    def configure_optimizers(self):
        """
        Setup model Optimizers
        Returns:
            torch.optim.Optimizer: Model and discriminator optimizers
        """
        if self.cfg.MODEL.get("ONLY_TEMPORAL", False):
            params = []
            if self.motion_module is not None:
                params.append({'params': [p for p in self.motion_module.parameters() if p.requires_grad],
                        'lr':self.cfg.TRAIN.LR2})
            if self.temporal_module is not None:
                params.append({'params': [p for p in self.temporal_module.parameters() if p.requires_grad], 
                            'lr':self.cfg.TRAIN.LR2})
            if self.cfg.MODEL.SMAL_HEAD.get("FINETUNE", False):
                params.append({'params': [p for p in self.smal_head.parameters() if p.requires_grad]})
            
            if self.cfg.MODEL.BACKBONE.get("FINETUNE", False):
                params.append({'params': [p for p in self.backbone.parameters() if p.requires_grad], 
                               'lr': self.cfg.TRAIN.get("BACKBONE_LR", 1.25e-6)})
            optimizer = torch.optim.AdamW(params, lr=self.cfg.TRAIN.LR, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY, fused=True)

            warmup_factor = 1. / self.cfg.TRAIN.WARMUP_STEPS

            def f(x):
                if x < self.cfg.TRAIN.WARMUP_STEPS:
                    alpha = float(x) / self.cfg.TRAIN.WARMUP_STEPS
                    return warmup_factor * (1 - alpha) + alpha
                else:
                    milestone = sum([x>=i for i in self.cfg.TRAIN.SCHEDULE])
                    return self.cfg.TRAIN.LR_DECAY ** milestone

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, f),
                },
            }
        else:
            param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
            optimizer = torch.optim.AdamW(params=param_groups,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                        fused=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps, eta_min=1.25e-6)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # 'step' or 'epoch'
                    "frequency": self.cfg.GENERAL.TOTAL_STEPS
                },
            }

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']
        batch_size = x.shape[0]
        temporal = x.ndim == 5
        if temporal:
            T = x.shape[1]
            x = x.view(-1, *x.shape[-3:])
        else:
            T = 1

        # Compute conditioning features using the backbone
        conditioning_feats, _ = self.backbone(x[:, :, :, 32:-32])  # [256, 192]
        # conditioning_feats, _ = self.backbone(x[:, :, 32:-32, :])

        # Temporal module
        if self.temporal_module is not None and temporal:
            conditioning_feats = einops.rearrange(conditioning_feats, '(b t) c h w -> (b h w) t c', t=T)
            conditioning_feats = self.temporal_module(conditioning_feats)
            conditioning_feats = einops.rearrange(conditioning_feats, '(b h w) t c -> (b t) c h w', h=16, w=12)

        pred_smal_params, pred_cam, pred_pose = self.smal_head(conditioning_feats)

        # smpl motion module
        if self.motion_module is not None and temporal:
            pred_pose_temporal_input = einops.rearrange(pred_pose, '(b t) c -> b t c', t=T)
            pred_pose_temporal_output = self.motion_module(pred_pose_temporal_input)
            pred_pose_temporal_output = einops.rearrange(pred_pose_temporal_output, 'b t c -> (b t) c')
            if self.cfg.MODEL.get("RESIDUAL", False):
                pred_pose = pred_pose + pred_pose_temporal_output
            else:
                pred_pose = pred_pose_temporal_output

            pred_smal_params['global_orient'] = rotation_6d_to_matrix(pred_pose[:, :6].reshape(batch_size*T, -1, 6))
            pred_smal_params['pose'] = rotation_6d_to_matrix(pred_pose[:, 6:].reshape(batch_size*T, -1, 6))

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}

        # Compute camera translation
        focal_length = batch['focal_length'].view(-1, 2)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)], dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smal_params['global_orient'] = matrix_to_axis_angle(pred_smal_params['global_orient'].reshape(batch_size*T, -1, 3, 3)).view(batch_size*T, -1)
        pred_smal_params['pose'] = matrix_to_axis_angle(pred_smal_params['pose'].reshape(batch_size*T, -1, 3, 3)).view(batch_size*T, -1)
        pred_smal_params['betas'] = pred_smal_params['betas'].reshape(batch_size*T, -1)
        smal_output = self.smal(**pred_smal_params)

        pred_keypoints_3d = smal_output.surface_keypoints
        pred_vertices = smal_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size*T, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size*T, -1, 3)

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size*T, -1, 2)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        temporal = batch['img'].ndim == 5
        pred_smal_params = output['pred_smal_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        pred_vertices = output['pred_vertices']

        batch_size = pred_smal_params['pose'].shape[0]  # batch_size=batch_size*T as the output in forward_step is (batch_size*T, ...)
        device = pred_smal_params['pose'].device
        dtype = pred_smal_params['pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d'].view(batch_size, -1, 3)
        gt_keypoints_3d = batch['keypoints_3d'].view(batch_size, -1, 4)
        gt_smal_params = {'global_orient': batch['global_orient'],
                          'pose': batch['pose'],
                          'betas': batch['betas'],
                          }
        has_smal_params = {'global_orient': batch['has_global_orient'],
                           'pose': batch['has_pose'],
                           'betas': batch['has_betas'],
                           }
        is_axis_angle = {'global_orient': torch.tensor(True, dtype=torch.bool),
                         'pose': torch.tensor(True, dtype=torch.bool),
                         'betas': torch.tensor(False, dtype=torch.bool)}

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
        # if temporal:
        #     with torch.no_grad():
        #         smal_output = self.smal(**{"global_orient": batch['global_orient'].view(batch_size, -1),
        #                                 "pose": batch['pose'].view(batch_size, -1),
        #                                 "betas": batch['betas'].view(batch_size, -1)})
        #     gt_vertices = smal_output.vertices.view(batch_size, -1, 3)
        #     loss_vertices = torch.nn.functional.l1_loss(pred_vertices, gt_vertices, reduction='sum')
        # else:
        #     loss_vertices = torch.tensor(0.0, device=device, dtype=dtype)

        if temporal:
            T = batch['img'].shape[1]
            # (batch_size*T, 38, 3)
            gt_pose_aa = torch.cat([batch['global_orient'].view(batch_size, -1, 3), batch['pose'].view(batch_size, -1, 3)], dim=1)
            # (batch_size*T, 38, 3, 3)
            gt_pose_mt = axis_angle_to_matrix(gt_pose_aa)
            # (batch_size*T, 38, 6)
            gt_pose_6d = matrix_to_rotation_6d(gt_pose_mt)
            gt_pose_6d = gt_pose_6d.view(batch_size//T, T, -1)

            pred_pose_mt = torch.cat([pred_smal_params['global_orient'].view(batch_size, -1, 3, 3), pred_smal_params['pose'].view(batch_size, -1, 3, 3)], dim=1)
            pred_pose_6d = matrix_to_rotation_6d(pred_pose_mt)
            pred_pose_6d = pred_pose_6d.view(batch_size//T, T, -1)
            
            loss_smooth_net = self.smooth_net_loss(pred_pose_6d, gt_pose_6d)
        else:
            loss_smooth_net = torch.tensor(0.0, device=device, dtype=dtype)
            

        # Compute loss on SMAL parameters
        loss_smal_params = {}
        for k, pred in pred_smal_params.items():
            gt = gt_smal_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smal_params[k]
            loss_smal_params[k] = self.smal_parameter_loss(pred.reshape(batch_size, -1),
                                                           gt.reshape(batch_size, -1),
                                                           has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d + \
               sum([loss_smal_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smal_params]) + \
               self.cfg.LOSS_WEIGHTS['SMOOTH_NET'] * loss_smooth_net

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_smooth_net=loss_smooth_net.detach(),
                      )

        for k, v in loss_smal_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    def compute_metric(self, batch: Dict, output: Dict):
        with torch.no_grad():
            # pa_mpjpe, pa_mpvpe = self.evaluator.eval_3d(output, batch)
            pck, auc, error_accel = self.evaluator.eval_2d(output, batch)
        return dict(PCK=pck[1], AUC=auc, ErrorAccel=error_accel)
  
    @pl.utilities.rank_zero.rank_zero_only
    def log_visualizations_to_wandb(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log results to W&B
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].view(-1, *batch['keypoints_2d'].shape[-2:]).shape[0]
        images = batch['img'].view(batch_size, *batch['img'].shape[-3:])
        # Un-normalize images for visualization
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        gt_keypoints_2d = batch['keypoints_2d'].view(batch_size, -1, 3)
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        def _to_float32_numpy(tensor: torch.Tensor) -> Any:
            return tensor.to(torch.float32).cpu().numpy()

        # Create a dictionary to hold all logging data
        log_data = {}

        # Add losses to the log dictionary
        for loss_name, val in losses.items():
            log_data[f"{mode}/{loss_name}"] = val.detach().item()

        # If in validation mode, add metrics to the log dictionary
        if not train and "metric" in output:
            for metric_name, val in output['metric'].items():
                log_data[f"{mode}/{metric_name}"] = val

        num_images_to_log = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        # Create the visualization grid
        predictions_grid = self.mesh_renderer.visualize_tensorboard(
            _to_float32_numpy(pred_vertices[:num_images_to_log]),
            _to_float32_numpy(pred_cam_t[:num_images_to_log]),
            _to_float32_numpy(images[:num_images_to_log]),
            self.cfg.SMAL.get("FOCAL_LENGTH", 5000),
            _to_float32_numpy(pred_keypoints_2d[:num_images_to_log]),
            _to_float32_numpy(gt_keypoints_2d[:num_images_to_log]),
        )
        predictions_grid = [img.float().cpu() for img in predictions_grid]
        predictions_grid = make_grid(predictions_grid, nrow=5, padding=2)
        # Scale to [0, 255] and convert to uint8 to fix wandb warning
        predictions_grid_for_wandb = (predictions_grid * 255).to(torch.uint8)
        # Add the image grid to the log dictionary
        log_data[f'{mode}/predictions'] = wandb.Image(predictions_grid_for_wandb)
        # Log the entire dictionary to W&B
        self.logger.experiment.log(log_data, step=step_count)

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step(self, batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d',
                                                'box_center', 'box_size', 'img_size', 'smal_params',
                                                'smal_params_is_axis_angle', '_trans', 'imgname', 'focal_length'}
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        optimizer.step()
        scheduler = self.lr_schedulers()
        scheduler.step()

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            # self.tensorboard_logging(batch, output, self.global_step, train=True)
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch_size = batch['img'].shape[0]
        temporal = batch['img'].ndim == 5
        T = batch['img'].shape[1] if temporal else 1
        output = self.inference_chunk(batch)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.view(batch_size*T, *v.shape[2:])
            elif isinstance(v, Dict):
                for k2, v2 in v.items():
                    batch[k][k2] = v2.view(batch_size*T, *v2.shape[2:])
            else:
                raise TypeError("The type in batch must be Dict or Tensor")
        loss_keypoints_2d = self.keypoint_2d_loss(output['pred_keypoints_2d'], batch['keypoints_2d'])
        loss_keypoints_3d = self.keypoint_3d_loss(output['pred_keypoints_3d'], batch['keypoints_3d'], pelvis_id=0)
        loss = loss_keypoints_2d * self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] + loss_keypoints_3d * self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D']
        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      )
        metric = self.compute_metric(batch, output)
        output['losses'] = losses
        output['metric'] = metric

        # Log visualizations for the first batch of each validation epoch
        if batch_idx == 0:
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=False)
        self.log('val/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
            batch_size=batch_size, sync_dist=True)
        return output

    @torch.no_grad()
    def inference_chunk(self, batch: Dict):
        T = batch['img'].shape[1]
        if T <= 16:
            return self.forward_step(batch, train=False)
        else:
            pred_keypoints_2d = []
            pred_keypoints_3d = []
            for i in range(0, T - 16, 16):
                output = self.forward_step(batch, train=False)
                if i == 0:  # First 16 frames
                    expected_out = {k: v[:9] for k, v in output.items()}
                elif i == T - 16:  # Last 16 frames
                    expected_out = {k: v[8:] for k, v in output.items()}
                else:  # Middle 16 frames
                    expected_out = {k: v[[8]] for k, v in output.items()}

                pred_keypoints_2d.append(expected_out['pred_keypoints_2d'])
                pred_keypoints_3d.append(expected_out['pred_keypoints_3d'])
            pred_keypoints_2d = torch.cat(pred_keypoints_2d, dim=0)
            pred_keypoints_3d = torch.cat(pred_keypoints_3d, dim=0)
            return {'pred_keypoints_2d': pred_keypoints_2d, 'pred_keypoints_3d': pred_keypoints_3d}
