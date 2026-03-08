# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Type, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import DropPath, get_clones, LayerNorm2d
from .position_encoding import PositionEmbeddingRandom
from .transformer import TwoWayAttentionBlock

class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        fuser,
        position_encoding,
        in_dim=1280,  # in_dim of pix_feats
        num_heads:int=8,
        mlp_dim:int=2048,
        image_embedding_size:Tuple[int, int]=(16, 12)
    ):
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.attention = TwoWayAttentionBlock(embedding_dim=in_dim, 
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              skip_first_layer_pe=True)
        self.pe_layer = PositionEmbeddingRandom(in_dim // 2)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Fuse pix_feats and sparse keypoint embedding
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(point_embedding.device)

        image_pe = self.pe_layer(self.image_embedding_size).unsqueeze(0)
        # queries is the same as query_pe, because here will skip the position embedding
        B, C, H, W = pix_feat.shape
        _, x = self.attention(queries=point_embedding,
                              keys=pix_feat.flatten(2).permute(0, 2, 1).contiguous(),  # B(HW)C
                              query_pe=point_embedding,  
                              key_pe=image_pe.flatten(2).permute(0, 2, 1).contiguous(),) 
        x = self.fuser(x.permute(0, 2, 1).view(B, C, H, W).contiguous())
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": pos}


class PointEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        Encodes keypoints for memory encoder

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 26  # We predict 26 keypoints
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)

    def _embed_points(
        self,
        points: torch.Tensor,
        pad: bool=False,
    ) -> torch.Tensor:
        """
        Embeds keypoints
        points: [batch_size, 26, 2]
        pad: In sam2, pad=boxes is None
        """
        points = (points + 0.5) * self.input_image_size[0]  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        # point_embedding: [batch_size, 26, embed_dim]
        for i in range(self.num_point_embeddings):
            point_embedding[:, i] += self.point_embeddings[i].weight
        return point_embedding

    def _get_batch_size(
        self,
        keypoints_2d: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if keypoints_2d is not None:
            return keypoints_2d.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        keypoints_2d: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds keypoints2d and keypoints3d, returning both sparse embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
        Returns:
          torch.Tensor: sparse embeddings for the keypoints_2d with shape
            BxNx(embed_dim), where N is determined by the number of input keypoints.
        """
        bs = self._get_batch_size(keypoints_2d)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if keypoints_2d is not None:
            point_embeddings = self._embed_points(keypoints_2d)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        return sparse_embeddings
