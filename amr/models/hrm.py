"""
Horse Avatar Reconstruction Model
"""

import torch
import math
import pickle
import einops
import pytorch_lightning as pl
from torchvision.utils import make_grid
from typing import Any, Dict
from yacs.config import CfgNode
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads.sd3_head import TransformerDecoder
from .varen import VAREN
from .rendering.gs_varen_renderer import GS3DRenderer, PointEmbed
from .losses import PixelLoss, LPIPSLoss
from ..utils.evaluate_metric import Evaluator
import wandb
import torch.nn as nn

log = get_pylogger(__name__)


class HRM(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        """
        Setup Horse Avatar Reconstruction Model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        # Create Backbone
        self.backbone = create_backbone(cfg)

        # Create Point Embedding
        self.point_embedding = PointEmbed(dim=cfg.MODEL.POINT_EMBEDDING.get("DIM", 1024))

        # Create Motion Embedding
        input_dim = cfg.MODEL.BACKBONE.get("ENCODER_FEAT_DIM", 1024)
        mid_dim = input_dim // 2
        self.motion_embed_mlp = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, cfg.MODEL.POINT_EMBEDDING.get("DIM", 1024)),
        )
        if cfg.MODEL.DECODER.BLOCK_TYPE == "cross_attention":
            self.motion_embed_mlp[0].weight.requires_grad = False
            self.motion_embed_mlp[0].bias.requires_grad = False
            self.motion_embed_mlp[2].weight.requires_grad = False
            self.motion_embed_mlp[2].bias.requires_grad = False

        # Create Decoder
        self.decoder = TransformerDecoder(cfg.MODEL)  # Motion Embedding + Point Embedding + Image Embedding

        # Create Gaussian Splatting Renderer
        self.renderer = GS3DRenderer(cfg.MODEL)
        self.renderer.varen_model.layer.betas.requires_grad = False
        self.renderer.varen_model.layer.body_pose.requires_grad = False
        self.renderer.varen_model.layer.global_orient.requires_grad = False
        self.renderer.varen_model.layer.transl.requires_grad = False

        # Define Losses
        # TODO: Maybe add regularization loss for offset and scaling
        self.pixel_loss = PixelLoss(loss_type='l1')  # RGB Loss and Silhouette Loss
        self.lpips_loss = LPIPSLoss(prefetch=True)  # LPIPS Loss

        # Create Evaluator
        self.evaluator = Evaluator(smal_model=self.renderer.varen_model.layer, image_size=cfg.DATASETS.RENDER_IMAGE_RES_LOW)
        
        self.automatic_optimization = True

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             if "backbone" not in name:
    #                 print(name)

    def get_parameters(self):
        """
        Get the parameters of the model
        Returns:
            List[Dict]: List of parameter dictionaries
        """
        all_params = []
        all_params.extend(self.backbone.parameters())
        all_params.extend(self.point_embedding.parameters())
        all_params.extend(self.motion_embed_mlp.parameters())
        all_params.extend(self.decoder.parameters())
        all_params.extend(self.renderer.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model Optimizers and Learning Rate Scheduler
        Returns:
            Dict: Dictionary containing model optimizers and learning rate scheduler
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
        optimizer = torch.optim.AdamW(params=param_groups,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                      fused=False)  # Disable fused for compatibility

        warmup_steps = self.cfg.TRAIN.get('WARMUP_STEPS', 3000)
        total_steps = self.cfg.GENERAL.TOTAL_STEPS
        base_lr_value = float(self.cfg.TRAIN.LR)
        initial_lr = self.cfg.TRAIN.get('INITIAL_LR', 1e-10)
        # Scale factor such that base_lr * factor == initial_lr at step 0
        warmup_start_factor = min(max(initial_lr / max(base_lr_value, 1e-30), 0.0), 1.0)

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                s = float(current_step) / float(max(1, warmup_steps))
                return warmup_start_factor + (1.0 - warmup_start_factor) * s
            if total_steps and total_steps > warmup_steps:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        result = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        # Add gradient clipping if configured
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            result["gradient_clip_val"] = self.cfg.TRAIN.GRAD_CLIP_VAL
            result["gradient_clip_algorithm"] = "norm"

        return result

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm, optimizer_idx=None):
        """
        Apply gradient clipping using the configured value/algorithm.
        PyTorch Lightning calls this hook after backward() and before optimizer.step().
        """
        clip_val = self.cfg.TRAIN.get("GRAD_CLIP_VAL", 0)
        if clip_val is None or clip_val <= 0:
            return

        clip_alg = self.cfg.TRAIN.get("GRAD_CLIP_ALGORITHM", "norm")
        # Delegate to Lightning's helper to handle norm/value modes uniformly.
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=clip_alg,
        )

    def forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward the DINOV2 Backbone
        Args:
            image (torch.Tensor): Image tensor of shape [B, 3, H, W]
        Returns:
            torch.Tensor: Features from the backbone
        """
        img_feats = self.backbone(image)  # DINOV2: [B, H/14*W/14, 1024]  DINOV3: [B, H/16*W/16, 1024]
        return img_feats

    def forward_latent_points(self, query_points: torch.Tensor, image: torch.Tensor): 
        """
        Forward the latent points
        Args:
            query_points (torch.Tensor): Query points tensor of shape [B, N, 3]
            image (torch.Tensor): Image tensor of shape [B, 3, H, W]
        Returns:
            torch.Tensor: Latent points tensor of shape [B, N, 1024]
        """
        image_feats = self.forward_backbone(image.squeeze(1))  # [B, H/14*W/14, 1024]
        motion_tokens = image_feats.mean(dim=1, keepdim=True)  # [B, 1, 1024]
        motion_tokens = self.motion_embed_mlp(motion_tokens).squeeze(1)  # [B, 1024]

        tokens = self.forward_transformer(image_feats, 
                                          query_points=query_points,
                                          motion_tokens=motion_tokens)
        return tokens, image_feats

    def forward_transformer(self, image_feats: torch.Tensor, query_points: torch.Tensor, motion_tokens: torch.Tensor):
        """
        Forward the transformer
        Args:
            image_feats (torch.Tensor): Image features tensor of shape [B, H/14*W/14, 1024] for DINOV2
            query_points (torch.Tensor): Query points tensor of shape [B, N, 3]
            motion_tokens (torch.Tensor): Motion tokens tensor of shape [B, 1024*2]
        Returns:
            torch.Tensor: Transformed query points tensor of shape [B, N, 1024]
        """
        B = image_feats.shape[0]
        x = self.point_embedding(query_points)  # [B, N, D]
        x = self.decoder(x,
                        cond=image_feats,  # [B, H/14*W/14, 1024]
                        temb=motion_tokens)  # [B, 1024*2]
        return x

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        query_points, varen_data = self.renderer.get_query_points(batch['smal_params'], device=batch['source_rgbs'].device)
        latent_points, image_feats = self.forward_latent_points(query_points, batch['source_rgbs'])
        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            varen_data=varen_data,
            c2w=batch['c2ws'],
            intrinsic=batch['intrs'],
            height=batch['render_image'].shape[-2],
            width=batch['render_image'].shape[-1],
            background_color=batch['render_bg_colors'],
            train=train,
            additional_features={"image_feats": image_feats, "image": batch['source_rgbs'][:, 0]},  # TODO: Check if this is correct
        )

        N, M = batch['c2ws'].shape[:2]  # c2w: [batch_size, N_view, 4, 4]
        assert (
            render_results["comp_rgb"].shape[0] == N
        ), "Batch size mismatch for render_results"
        assert (
            render_results["comp_rgb"].shape[1] == M
        ), "Number of rendered views should be consistent with render_cameras"

        gs_attrs_list = render_results.pop("gs_attr")

        offset_list = []
        scaling_list = []
        for gs_attrs in gs_attrs_list:
            offset_list.append(gs_attrs.offset_xyz)
            scaling_list.append(gs_attrs.scaling)
        offset_output = torch.stack(offset_list)
        scaling_output = torch.stack(scaling_list)

        return {
            "latent_points": latent_points,
            "offset_output": offset_output,
            "scaling_output": scaling_output,
            **render_results,
        }

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute the loss
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor: Loss
        """
        pred_rgb = output['comp_rgb']
        pred_silhouette = output['comp_mask']
        
        gt_rgb = batch['render_image']
        gt_silhouette = batch['render_mask']

        loss_rgb = self.pixel_loss(pred_rgb, gt_rgb)
        loss_silhouette = self.pixel_loss(pred_silhouette, gt_silhouette)
        loss_lpips = self.lpips_loss(pred_rgb, gt_rgb)
        loss_reg = torch.mean(output['offset_output'] ** 2)
        
        loss = (self.cfg.LOSS_WEIGHTS.RGB * loss_rgb + 
                self.cfg.LOSS_WEIGHTS.SILHOUETTE * loss_silhouette + 
                self.cfg.LOSS_WEIGHTS.LPIPS * loss_lpips +
                self.cfg.LOSS_WEIGHTS.REG * loss_reg)

        losses = dict(loss=loss.detach(),
                      loss_rgb=loss_rgb.detach(),
                      loss_silhouette=loss_silhouette.detach(),
                      loss_lpips=loss_lpips.detach(),
                      loss_reg=loss_reg.detach(),
                      )

        output['losses'] = losses
        return loss

    def compute_metric(self, batch: Dict, output: Dict):
        with torch.no_grad():
            psnr, ssim, lpips = self.evaluator.eval_image(output, batch)
            return dict(PSNR=psnr, SSIM=ssim, LPIPS=lpips)
  
    @pl.utilities.rank_zero.rank_zero_only
    def log_visualizations_to_wandb(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Log visualizations to WandB
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """
        mode = 'train' if train else 'val'
        log_data = dict()
        for loss_name, val in output['losses'].items():
            log_data[f'{mode}/{loss_name}'] = val.detach().item()
        if not train and "metric" in output:
            for metric_name, val in output['metric'].items():
                log_data[f'{mode}/{metric_name}'] = val
        
        gt_rbg = batch['render_image']  # [B, 3, H, W]
        pred_rgb = output['comp_rgb']
        gt_mask = batch['render_mask']
        pred_mask = output['comp_mask']

        pred_grid = []
        for i in range(min(gt_rbg.shape[1], self.cfg.EXTRA.NUM_LOG_IMAGES)):
            pred_grid.append(gt_rbg[0][i])
            pred_grid.append(pred_rgb[0][i])
            pred_grid.append(gt_mask[0][i].repeat(3, 1, 1))
            pred_grid.append(pred_mask[0][i].repeat(3, 1, 1))

        pred_grid = make_grid(pred_grid, nrow=4, padding=0)
        pred_grid_for_wandb = (pred_grid * 255).to(torch.uint8)
        log_data[f'{mode}/predictions'] = wandb.Image(pred_grid_for_wandb)
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

    def training_step(self, batch: Dict) -> torch.Tensor:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'uid',
                                                'source_c2ws', 'source_intrs',
                                                'source_rgbs', 'render_image', 'render_mask',
                                                'c2ws', 'intrs', 'render_full_resolutions', 'render_bg_colors',
                                                'pytorch3d_transpose_R',
                                                'smal_params', 'source_smal_params'}
        Returns:
            torch.Tensor: Loss tensor for automatic optimization.
        """
        batch_size = batch['source_rgbs'].shape[0]
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f'Loss is NaN or Inf: {loss}')

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            # self.tensorboard_logging(batch, output, self.global_step, train=True)
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
                 batch_size=batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch_size = batch['source_rgbs'].shape[0]
        output = self.forward_step(batch, train=False)

        loss = self.compute_loss(batch, output, train=False)
        metric = self.compute_metric(batch, output)
        output['loss'] = loss
        output['metric'] = metric

        # Log visualizations for the first batch of each validation epoch
        if batch_idx == 0:
            self.log_visualizations_to_wandb(batch, output, self.global_step, train=False)
        self.log('val/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False,
            batch_size=batch_size, sync_dist=True)
        return output


