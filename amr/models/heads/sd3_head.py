import logging
from typing import Dict

import torch
import torch.nn as nn
from diffusers.utils import is_torch_version
from diffusers.models.normalization import RMSNorm
from yacs.config import CfgNode

logger = logging.getLogger(__name__)


class TransformerDecoder(nn.Module):
    """Qwen-only transformer decoder for joint point-image conditioning."""

    def __init__(
        self,
        cfg: CfgNode,
    ):
        super().__init__()
        block_type = cfg.DECODER.get('BLOCK_TYPE', 'qwen_mm_cond')
        num_layers = cfg.DECODER.get('NUM_LAYERS', 5)
        num_heads = cfg.DECODER.get('NUM_HEADS', 16)
        inner_dim = cfg.DECODER.get('INNER_DIM', 1024)
        cond_dim = cfg.DECODER.get('COND_DIM', 1024)
        mod_dim = cfg.DECODER.get('MOD_DIM', None)
        gradient_checkpointing = cfg.DECODER.get('GRADIENT_CHECKPOINTING', True)
        eps = cfg.DECODER.get('EPS', 1e-6)

        self.gradient_checkpointing = gradient_checkpointing
        if block_type != "qwen_mm_cond":
            logger.warning(
                "TransformerDecoder only supports qwen_mm_cond; overriding configured block type %s.",
                block_type,
            )

        self.block_type = "qwen_mm_cond"

        from amr.models.components.transformer_dit import QwenMMJointTransformerBlock

        self.layers = nn.ModuleList(
            [
                QwenMMJointTransformerBlock(
                    dim=inner_dim,
                    num_heads=num_heads,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
        self.cond_norm = RMSNorm(cond_dim, eps=eps)

        if self.layers:
            frozen_layer = self.layers[min(4, len(self.layers) - 1)]
            frozen_layer.attn.to_add_out.weight.requires_grad = False
            frozen_layer.attn.to_add_out.bias.requires_grad = False
            frozen_layer.img_mlp.net[0].proj.weight.requires_grad = False
            frozen_layer.img_mlp.net[0].proj.bias.requires_grad = False
            frozen_layer.img_mlp.net[2].weight.requires_grad = False
            frozen_layer.img_mlp.net[2].bias.requires_grad = False

    def assert_runtime_integrity(
        self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        assert x is not None, "Input tensor must be specified"
        assert cond is not None, "Condition must be specified for qwen_mm_cond"
        if mod is not None:
            raise AssertionError("Modulation is not supported for qwen_mm_cond")

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        # mod: [N, D_mod] or None
        self.assert_runtime_integrity(x, cond, mod)

        cond = self.cond_norm(cond)
        cond = self.linear_cond_proj(cond)

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, bool] = (
                    {"use_reentrant": False}
                    if is_torch_version(">=", "1.11.0")
                    else {}
                )

                x, cond = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    cond,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                x, cond = layer(
                    hidden_states=x,
                    encoder_hidden_states=cond,
                    temb=temb,
                )

        x = self.norm(x)

        return x
