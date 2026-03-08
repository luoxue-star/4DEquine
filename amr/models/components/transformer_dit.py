from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

assert hasattr(F, "scaled_dot_product_attention"), print(
    "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
)

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen


class QwenPointImageAttnProcessor2_0:
    """Attention processor aligning points (sample) with image (context) streams using Qwen logic."""

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenPointImageAttnProcessor2_0 requires PyTorch 2.0 or newer."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Points stream
        encoder_hidden_states: torch.FloatTensor = None,  # Image stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError(
                "QwenPointImageAttnProcessor2_0 requires encoder_hidden_states (image stream)."
            )

        seq_image = encoder_hidden_states.shape[1]

        # Points projections (sample branch)
        pts_query = attn.to_q(hidden_states)
        pts_key = attn.to_k(hidden_states)
        pts_value = attn.to_v(hidden_states)

        # Image projections (context branch)
        img_query = attn.add_q_proj(encoder_hidden_states)
        img_key = attn.add_k_proj(encoder_hidden_states)
        img_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape to multi-head format
        pts_query = pts_query.unflatten(-1, (attn.heads, -1))
        pts_key = pts_key.unflatten(-1, (attn.heads, -1))
        pts_value = pts_value.unflatten(-1, (attn.heads, -1))

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        # QK norm if available
        if attn.norm_q is not None:
            pts_query = attn.norm_q(pts_query)
        if attn.norm_k is not None:
            pts_key = attn.norm_k(pts_key)
        if getattr(attn, "norm_added_q", None) is not None:
            img_query = attn.norm_added_q(img_query)
        if getattr(attn, "norm_added_k", None) is not None:
            img_key = attn.norm_added_k(img_key)

        # Optional rotary embeddings (unused but kept for API parity)
        if image_rotary_emb is not None:
            pts_freqs, img_freqs = image_rotary_emb
            pts_query = apply_rotary_emb_qwen(pts_query, pts_freqs, use_real=False)
            pts_key = apply_rotary_emb_qwen(pts_key, pts_freqs, use_real=False)
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)

        # Concatenate streams: [image(context), points(sample)]
        joint_query = torch.cat([img_query, pts_query], dim=1)
        joint_key = torch.cat([img_key, pts_key], dim=1)
        joint_value = torch.cat([img_value, pts_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        img_attn_output = joint_hidden_states[:, :seq_image, :]
        pts_attn_output = joint_hidden_states[:, seq_image:, :]

        pts_attn_output = attn.to_out[0](pts_attn_output)
        if len(attn.to_out) > 1:
            pts_attn_output = attn.to_out[1](pts_attn_output)

        img_attn_output = attn.to_add_out(img_attn_output)

        return pts_attn_output, img_attn_output


class QwenMMJointTransformerBlock(nn.Module):
    """Qwen-inspired dual-stream DiT block for joint point-image conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = "rms_norm",
    ):
        super().__init__()

        if context_pre_only:
            raise ValueError("QwenMMJointTransformerBlock does not support context_pre_only=True")

        attention_head_dim = dim // num_heads
        assert attention_head_dim * num_heads == dim

        self.dim = dim
        self.num_heads = num_heads

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.pts_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pts_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenPointImageAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )

        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.pts_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.pts_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _apply_modulation(self, x: torch.Tensor, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = params.chunk(3, dim=-1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states (torch.Tensor): Input hidden states. Query Points features
            encoder_hidden_states (torch.Tensor): Encoder hidden states. Context features
            temb (torch.Tensor): Motion embedding
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the updated hidden states and encoder hidden states.
        """

        if temb is None:
            raise ValueError("temb must be provided for QwenMMJointTransformerBlock")

        if temb.shape[-1] != self.dim:
            raise ValueError(
                f"temb dimension {temb.shape[-1]} does not match transformer dimension {self.dim}"
            )
        
        pts_mod_params = self.pts_mod(temb)
        img_mod_params = self.img_mod(temb)

        pts_mod1, pts_mod2 = pts_mod_params.chunk(2, dim=-1)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)

        pts_normed = self.pts_norm1(hidden_states)
        pts_modulated, pts_gate1 = self._apply_modulation(pts_normed, pts_mod1)

        img_normed = self.img_norm1(encoder_hidden_states)
        img_modulated, img_gate1 = self._apply_modulation(img_normed, img_mod1)

        # TODO: add image_rotary_emb
        attn_hidden, attn_context = self.attn(
            hidden_states=pts_modulated,
            encoder_hidden_states=img_modulated,
        )

        hidden_states = hidden_states + pts_gate1.unsqueeze(1) * attn_hidden
        encoder_hidden_states = encoder_hidden_states + img_gate1.unsqueeze(1) * attn_context

        pts_normed2 = self.pts_norm2(hidden_states)
        pts_modulated2, pts_gate2 = self._apply_modulation(pts_normed2, pts_mod2)
        pts_ff = self.pts_mlp(pts_modulated2)
        hidden_states = hidden_states + pts_gate2.unsqueeze(1) * pts_ff

        img_normed2 = self.img_norm2(encoder_hidden_states)
        img_modulated2, img_gate2 = self._apply_modulation(img_normed2, img_mod2)
        img_ff = self.img_mlp(img_modulated2)
        encoder_hidden_states = encoder_hidden_states + img_gate2.unsqueeze(1) * img_ff

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp_(min=-65504, max=65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp_(min=-65504, max=65504)

        return hidden_states, encoder_hidden_states
