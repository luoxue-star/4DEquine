import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import copy
import math
import json
import pdb
from collections import defaultdict
from dataclasses import dataclass, field
from yacs.config import CfgNode

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply
from dataclasses import dataclass, fields, is_dataclass
from collections import OrderedDict
from typing import Any, Tuple
from torch import Tensor

from amr.models.rendering.varen import VARENSubdividedMeshModel
from amr.models.rendering.utils.sh_utils import RGB2SH, SH2RGB
from amr.models.rendering.utils.typing import *
from amr.models.rendering.utils.utils import MLP, trunc_exp
from amr.utils import LinerParameterTuner, StaticParameterTuner
from amr.utils.renderer import Renderer as PyrenderMeshRenderer


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    Python dictionary.

    <Tip warning={true}>

    You can't unpack a [`BaseOutput`] directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    first.

    </Tip>
    """

    def __post_init__(self) -> None:
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )

        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name: Any, value: Any) -> None:
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class GaussianAppOutput(BaseOutput):
    """
    Output of the Gaussian Appearance output.

    Attributes:

    """

    offset_xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    use_rgb: bool


def auto_repeat_size(tensor, repeat_num, axis=0):
    repeat_size = [1] * tensor.dim()
    repeat_size[axis] = repeat_num
    return repeat_size


def aabb(xyz):
    return torch.min(xyz, dim=0).values, torch.max(xyz, dim=0).values


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


def generate_rotation_matrix_y(degrees):
    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(w2c.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )
        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
        )


class GaussianModel:

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # rgb activation function
        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, use_rgb=False) -> None:
        """
        Initializes the GSRenderer object.
        Args:
            xyz (Tensor): The xyz coordinates.
            opacity (Tensor): The opacity values.
            rotation (Tensor): The rotation values.
            scaling (Tensor): The scaling values.
            before_activate: if True, the output appearance is needed to process by activation function.
            shs (Tensor): The spherical harmonics coefficients.
            use_rgb (bool, optional): Indicates whether shs represents RGB values. Defaults to False.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]

        self.use_rgb = use_rgb  # shs indicates rgb?

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb
        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)


class GSLayer(nn.Module):
    """W/O Activation Function"""

    def setup_functions(self):

        self.scaling_activation = trunc_exp  # proposed by torch-ngp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
    ):
        super().__init__()
        self.setup_functions()

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
            clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": None,
            "rotation": None,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def forward(self, x, pts, x_fine=None):
        # TODO: Check here: what are x, pts, x_fine?
        assert len(x.shape) == 2
        ret = {}
        for k in self.attr_dict:
            layer = self.out_layers[k]
            v = layer(x)
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(
                        torch.eye(3).type_as(x)[None, :, :].repeat(x.shape[0], 1, 1)
                    )  # constant rotation
                else:
                    # v = torch.nn.functional.normalize(v)
                    v = self.rotation_activation(v)
            elif k == "scaling":
                # v = trunc_exp(v)
                v = self.scaling_activation(v)

                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    # v = torch.sigmoid(v)
                    v = self.opacity_activation(v)
            elif k == "shs":
                if self.use_rgb:
                    # v = torch.sigmoid(v)
                    v = self.rgb_activation(v)

                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                    v = v + pts
                k = "offset_xyz"
            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        return GaussianAppOutput(**ret)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )

        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)

        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )  # B x N x C
        embed = self.norm(embed)
        return embed


class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float = None,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        feedforward=False,
    ):
        super().__init__()
        # TODO check already apply normalization
        # self.norm_q = nn.LayerNorm(inner_dim, eps=eps)
        # self.norm_k = nn.LayerNorm(cond_dim, eps=eps)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

        self.mlp = None
        if feedforward:
            self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=attn_drop,
                bias=attn_bias,
                batch_first=True,
            )
            self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
            self.mlp = nn.Sequential(
                nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(mlp_drop),
                nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
                nn.Dropout(mlp_drop),
            )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = self.cross_attn(
            self.norm_q(x), self.norm_k(cond), cond, need_weights=False
        )[0]
        if self.mlp is not None:
            before_sa = self.norm2(x)
            x = (
                x
                + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
            )
            x = x + self.mlp(self.norm3(x))
        return x


class DecoderCrossAttn(nn.Module):
    def __init__(
        self, query_dim, context_dim, num_heads, mlp=False, decode_with_extra_info=None
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.cross_attn = CrossAttnBlock(
            inner_dim=query_dim,
            cond_dim=context_dim,
            num_heads=num_heads,
            feedforward=mlp,
            eps=1e-5,
        )
        self.decode_with_extra_info = decode_with_extra_info
        if decode_with_extra_info is not None:
            if decode_with_extra_info["type"] == "dinov2p14_feat":
                context_dim = decode_with_extra_info["cond_dim"]
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=context_dim,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                from LHM.models.encoders.dinov2_wrapper import Dinov2Wrapper

                self.encoder = Dinov2Wrapper(
                    model_name="dinov2_vits14_reg", freeze=False, encoder_feat_dim=384
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=384,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_resnet18_feat":
                from LHM.models.encoders.xunet_wrapper import XnetWrapper

                self.encoder = XnetWrapper(
                    model_name="resnet18", freeze=False, encoder_feat_dim=64
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=64,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )

    def resize_image(self, image, multiply):
        B, _, H, W = image.shape
        new_h, new_w = (
            math.ceil(H / multiply) * multiply,
            math.ceil(W / multiply) * multiply,
        )
        image = F.interpolate(
            image, (new_h, new_w), align_corners=True, mode="bilinear"
        )
        return image

    def forward(self, pcl_query, pcl_latent, extra_info=None):
        out = self.cross_attn(pcl_query, pcl_latent)
        if self.decode_with_extra_info is not None:
            out_dict = {}
            out_dict["coarse"] = out
            if self.decode_with_extra_info["type"] == "dinov2p14_feat":
                out = self.cross_attn_color(out, extra_info["image_feats"])
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                img_feat = self.encoder(extra_info["image"])
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_resnet18_feat":
                image = extra_info["image"]
                image = self.resize_image(image, multiply=32)
                img_feat = self.encoder(image)
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
        return out


class GS3DRenderer(nn.Module):
    def __init__(
        self,
        cfg: CfgNode,
    ):
        super().__init__()
        varen_model_path = cfg.RENDERER.get('VAREN_MODEL_PATH', 'data/varen/')
        subdivide_num = cfg.RENDERER.get('SUBDIVIDE_NUM', 1)
        feat_dim = cfg.RENDERER.get('FEAT_DIM', 1024)
        query_dim = cfg.RENDERER.get('QUERY_DIM', 1024)
        use_rgb = cfg.RENDERER.get('USE_RGB', True)
        sh_degree = cfg.RENDERER.get('SH_DEGREE', 3)
        xyz_offset_max_step = cfg.RENDERER.get('XYZ_OFFSET_MAX_STEP', 1.0)
        mlp_network_config = cfg.RENDERER.get('MLP_NETWORK_CONFIG', None)
        shape_param_dim = cfg.RENDERER.get('SHAPE_PARAM_DIM', 39)
        clip_scaling = cfg.RENDERER.get('CLIP_SCALING', [100, 0.01, 0.05, 3000])
        decoder_mlp = cfg.RENDERER.get('DECODER_MLP', False)
        skip_decoder = cfg.RENDERER.get('SKIP_DECODER', True)
        fix_opacity = cfg.RENDERER.get('FIX_OPACITY', False)
        fix_rotation = cfg.RENDERER.get('FIX_ROTATION', False)
        decode_with_extra_info = cfg.RENDERER.get('DECODE_WITH_EXTRA_INFO', None)
        gradient_checkpointing = cfg.RENDERER.get('GRADIENT_CHECKPOINTING', True)
        apply_pose_blendshape = cfg.RENDERER.get('APPLY_POSE_BLENDSHAPE', False)

        self.dropout_factor = cfg.RENDERER.get('DROPOUT_FACTOR', 0.0)
        self.sigma_noise = cfg.RENDERER.get('SIGMA_NOISE', 0.0)

        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder

        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        self.varen_model = VARENSubdividedMeshModel(
            varen_model_path=varen_model_path, 
            subdivide_num=subdivide_num,
            shape_param_dim=shape_param_dim,
            apply_pose_blendshape=apply_pose_blendshape,
        )

        if not self.skip_decoder:
            self.pcl_embed = PointEmbed(dim=query_dim)
            self.decoder_cross_attn = DecoderCrossAttn(
                query_dim=query_dim,
                context_dim=feat_dim,
                num_heads=1,
                mlp=decoder_mlp,
                decode_with_extra_info=decode_with_extra_info,
            )

        self.mlp_network_config = mlp_network_config

        # using to mapping transformer decode feature to regression features. as decode feature is processed by NormLayer.
        if self.mlp_network_config is not None:
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

        self.gs_net = GSLayer(
            in_channels=query_dim,
            use_rgb=use_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            init_scaling=-5.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            use_fine_feat=(
                True
                if decode_with_extra_info is not None
                and decode_with_extra_info["type"] is not None
                else False
            ),
        )

    def hyper_step(self, step):
        self.gs_net.hyper_step(step)

    @torch.no_grad()
    def render_mesh_with_pyrender(
        self,
        varen_data,
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height: int,
        width: int,
        base_image: Optional[Float[Tensor, "B Nv 3 H W"]] = None,
        mesh_base_color: Tuple[float, float, float] = (1.0, 1.0, 0.9),
    ) -> Float[Tensor, "B Nv 3 H W"]:
        """
        Render posed VAREN mesh with Pyrender and composite onto a base image.

        Args:
            varen_data: Dict containing at least keys 'betas', 'pose', 'global_orient', 'trans'.
            c2w: Camera-to-world matrices for each view [B, Nv, 4, 4].
            intrinsic: Intrinsic matrices for each view [B, Nv, 4, 4].
            height: Output image height.
            width: Output image width.
            base_image: Optional base image to composite on [B, Nv, 3, H, W] in [0,1]. If None, uses white.
            mesh_base_color: RGB tuple for mesh color.

        Returns:
            Tensor of composited images [B, Nv, 3, H, W] in [0,1].
        """
        device = c2w.device
        B, Nv = c2w.shape[0], c2w.shape[1]

        # Get posed vertices for each view
        posed_verts, _ = self.varen_model.transform_to_posed_verts(varen_data, device=device)
        # Reshape to [B, Nv, N, 3]
        if posed_verts.dim() == 3 and posed_verts.shape[0] == B * Nv:
            posed_verts = posed_verts.view(B, Nv, posed_verts.shape[1], 3)
        elif posed_verts.dim() == 4:
            # Expected shape [B, Nv, N, 3]
            pass
        else:
            raise ValueError(f"Unexpected posed_verts shape: {posed_verts.shape}")

        # Faces
        faces_np = self.varen_model.face_upsampled.detach().cpu().numpy()

        # Minimal renderer config
        cfg = CfgNode(new_allowed=True)
        cfg.MODEL = CfgNode(new_allowed=True)
        cfg.MODEL.IMAGE_SIZE = max(int(height), int(width))
        cfg.EXTRA = CfgNode(new_allowed=True)
        cfg.EXTRA.FOCAL_LENGTH = float(intrinsic[0, 0, 0, 0].item())

        pyrenderer = PyrenderMeshRenderer(cfg, faces=torch.from_numpy(faces_np))

        # Prepare base image
        if base_image is None:
            base_image = torch.ones((B, Nv, 3, height, width), dtype=torch.float32, device=device)
        else:
            assert (
                base_image.shape[0] == B and base_image.shape[1] == Nv and base_image.shape[2] == 3
            ), "base_image must be [B, Nv, 3, H, W]"

        out_imgs: list[list[Tensor]] = []
        for b in range(B):
            row_imgs: list[Tensor] = []
            for v in range(Nv):
                verts_np = posed_verts[b, v].detach().cpu().numpy()
                cam_t = c2w[b, v, :3, 3].detach().cpu().numpy()
                fx = float(intrinsic[b, v, 0, 0].item())

                base_np = (
                    base_image[b, v].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                )

                # rgba = pyrenderer.render_rgba(
                #     vertices=verts_np,
                #     cam_t=cam_t,
                #     render_res=[int(width), int(height)],
                #     focal_length=fx,
                #     mesh_base_color=mesh_base_color,
                # )
                rgba = pyrenderer.render_rgba_blender(
                    vertices=verts_np,
                    c2w=c2w[b, v].detach().cpu().numpy(),
                    cam_t=cam_t,
                    render_res=[int(width), int(height)],
                    focal_length=fx,
                    mesh_base_color=mesh_base_color,
                )
                rgb = rgba[:, :, :3]
                alpha = rgba[:, :, 3:4]
                comp = rgb * alpha + base_np * (1.0 - alpha)
                comp_t = torch.from_numpy(comp).to(device=device, dtype=torch.float32).permute(2, 0, 1)
                row_imgs.append(comp_t)
            out_imgs.append(torch.stack(row_imgs, dim=0))

        return torch.stack(out_imgs, dim=0)

    def forward_single_view(
        self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        train: bool = False,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # Random Dropout Mask
        if self.dropout_factor > 0.0 and train:
            dropout_mask = torch.rand(gs.opacity.shape[0], device=gs.opacity.device).cuda()
            dropout_mask = dropout_mask < (1 - self.dropout_factor)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1).float()
            shs = None
        else:
            colors_precomp = None
            shs = gs.shs.float()

        # 1. randomly dropout 3DGS points during training
        if self.dropout_factor > 0.0 and train:
            means3D      = means3D[dropout_mask]
            means2D      = means2D[dropout_mask]
            shs          = shs[dropout_mask] if shs is not None else None
            opacity      = opacity[dropout_mask]
            scales       = scales[dropout_mask]
            rotations    = rotations[dropout_mask]
            colors_precomp = colors_precomp[dropout_mask] if colors_precomp is not None else None
        elif (not train):
            # scale oapcity for test stage rendering
            opacity *= 1 - self.dropout_factor

        # 2. add noise to opacity during training
        if train and self.sigma_noise > 0.0:
            epsilon_opacity = torch.randn_like(opacity, device=opacity.device) * self.sigma_noise
            epsilon_opacity = torch.clamp(epsilon_opacity, min=-self.sigma_noise, max=self.sigma_noise)
            opacity = torch.clamp(opacity * (1.0 + epsilon_opacity), min=0.0, max=1.0)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # NOTE that dadong tries to regress rgb not shs
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D.float(),
                means2D=means2D.float(),
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity.float(),
                scales=scales.float(),
                rotations=rotations.float(),
                cov3D_precomp=cov3D_precomp,
            )

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            "comp_mask": rendered_alpha.permute(1, 2, 0),
            "comp_depth": rendered_depth.permute(1, 2, 0),
        }

        # if ret_mask:
        #     mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
        #     raster_settings = GaussianRasterizationSettings(
        #         image_height=int(viewpoint_camera.height),
        #         image_width=int(viewpoint_camera.width),
        #         tanfovx=tanfovx,
        #         tanfovy=tanfovy,
        #         bg=mask_bg_color,
        #         scale_modifier=self.scaling_modifier,
        #         viewmatrix=viewpoint_camera.world_view_transform,
        #         projmatrix=viewpoint_camera.full_proj_transform.float(),
        #         sh_degree=0,
        #         campos=viewpoint_camera.camera_center,
        #         prefiltered=False,
        #         debug=False
        #     )
        #     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        #     with torch.autocast(device_type=self.device.type, dtype=torch.float32):
        #         rendered_mask, radii = rasterizer(
        #             means3D = means3D,
        #             means2D = means2D,
        #             # shs = ,
        #             colors_precomp = torch.ones_like(means3D),
        #             opacities = opacity,
        #             scales = scales,
        #             rotations = rotations,
        #             cov3D_precomp = cov3D_precomp)
        #         ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret

    def animate_gs_model(
        self, gs_attr: GaussianAppOutput, query_points, varen_data, debug=False
    ):
        """
        query_points: [N, 3]
        """
        device = gs_attr.offset_xyz.device

        if debug:
            N = gs_attr.offset_xyz.shape[0]
            gs_attr.xyz = torch.ones_like(gs_attr.offset_xyz) * 0.0

            rotation = matrix_to_quaternion(
                torch.eye(3).float()[None, :, :].repeat(N, 1, 1)
            ).to(
                device
            )  # constant rotation
            opacity = torch.ones((N, 1)).float().to(device)  # constant opacity

            gs_attr.opacity = opacity
            gs_attr.rotation = rotation
            # gs_attr.scaling = torch.ones_like(gs_attr.scaling) * 0.05
            # print(gs_attr.shs.shape)

        # build cano_dependent_pose, we use T-Pose as the canonical pose
        cano_varen_data_keys = [
            "global_orient",
            "pose",
            "trans",
        ]

        merge_varen_data = dict()
        for cano_varen_data_key in cano_varen_data_keys:
            warp_data = varen_data[cano_varen_data_key]
            cano_pose = torch.zeros_like(warp_data[:1])
            merge_pose = torch.cat([warp_data, cano_pose], dim=0)
            merge_varen_data[cano_varen_data_key] = merge_pose

        merge_varen_data["betas"] = varen_data["betas"]
        merge_varen_data["tail_scale"] = varen_data.get("tail_scale", None)
        merge_varen_data["transform_mat_neutral_pose"] = varen_data[
            "transform_mat_neutral_pose"
        ]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = (
                query_points + gs_attr.offset_xyz
            )  # [N, 3]  # canonical space offset.

            # matrix to warp predefined pose to zero-pose
            transform_mat_neutral_pose = merge_varen_data[
                "transform_mat_neutral_pose"
            ]  # [38, 4, 4]
            num_view = merge_varen_data["pose"].shape[0]  # [Nv, 21, 3]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)
            transform_mat_neutral_pose = transform_mat_neutral_pose.unsqueeze(0).repeat(
                num_view, 1, 1, 1
            )

            # print(mean_3d.shape, transform_mat_neutral_pose.shape, query_points.shape, 
            #       varen_data["pose"].shape, varen_data["betas"].shape)
            mean_3d, transform_matrix = (
                self.varen_model.transform_to_posed_verts_from_neutral_pose(
                    mean_3d,
                    merge_varen_data,
                    query_points,
                    transform_mat_neutral_pose=transform_mat_neutral_pose,  # from predefined pose to zero-pose matrix
                    device=device,
                )
            )  # [B, N, 3]

            # rotation appearance from canonical space to view_posed
            num_view, N, _, _ = transform_matrix.shape
            transform_rotation = transform_matrix[:, :, :3, :3]

            rigid_rotation_matrix = torch.nn.functional.normalize(
                matrix_to_quaternion(transform_rotation), dim=-1
            )
            I = matrix_to_quaternion(torch.eye(3)).to(device)

            rotation_neutral_pose = gs_attr.rotation.unsqueeze(0).repeat(num_view, 1, 1)

            # TODO do not move underarm gs
            # QUATERNION MULTIPLY
            rotation_pose_verts = quaternion_multiply(
                rigid_rotation_matrix, rotation_neutral_pose
            )
            # rotation_pose_verts = rotation_neutral_pose

        gs_list = []
        cano_gs_list = []
        for i in range(num_view):
            gs_copy = GaussianModel(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                # rotation=gs_attr.rotation,
                rotation=rotation_pose_verts[i],
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                use_rgb=self.gs_net.use_rgb,
            )  # [N, 3]

            if i == num_view - 1:
                cano_gs_list.append(gs_copy)
            else:
                gs_list.append(gs_copy)

        return gs_list, cano_gs_list

    def forward_gs_attr(self, x, query_points, varen_data, debug=False, x_fine=None):
        """
        x: [N, C] Float[Tensor, "Np Cp"],
        query_points: [N, 3] Float[Tensor, "Np 3"]
        """
        device = x.device
        if self.mlp_network_config is not None:
            # x is processed by LayerNorm
            x = self.mlp_net(x)
            if x_fine is not None:
                x_fine = self.mlp_net(x_fine)

        gs_attr: GaussianAppOutput = self.gs_net(x, query_points, x_fine)

        return gs_attr

    def get_query_points(self, varen_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                positions, _, transform_mat_neutral_pose = (
                    self.varen_model.get_query_points(varen_data, device=device)
                )  # [B, N, 3]
        varen_data["transform_mat_neutral_pose"] = (
            transform_mat_neutral_pose  # [B, 38, 4, 4]
        )
        return positions, varen_data

    def decoder_cross_attn_wrapper(self, pcl_embed, latent_feat, extra_info):
        # if self.training and self.gradient_checkpointing:
        #     def create_custom_forward(module):
        #         def custom_forward(*inputs):
        #             return module(*inputs)
        #         return custom_forward
        #     ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
        #     gs_feats = torch.utils.checkpoint.checkpoint(
        #         create_custom_forward(self.decoder_cross_attn),
        #         pcl_embed.to(dtype=latent_feat.dtype),
        #         latent_feat,
        #         extra_info,
        #         **ckpt_kwargs,
        #     )
        # else:
        gs_feats = self.decoder_cross_attn(
            pcl_embed.to(dtype=latent_feat.dtype), latent_feat, extra_info
        )
        return gs_feats

    def query_latent_feat(
        self,
        positions: Float[Tensor, "*B N1 3"],
        varen_data,
        latent_feat: Float[Tensor, "*B N2 C"],
        extra_info,
    ):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, varen_data = self.get_query_points(varen_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)

            gs_feats = self.decoder_cross_attn_wrapper(
                pcl_embed, latent_feat, extra_info
            )

        return gs_feats, positions, varen_data

    def forward_single_batch(
        self,
        gs_list: list[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
        debug: bool = False,
        train: bool = False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            out_list.append(
                self.forward_single_view(
                    gs_list[v_idx],
                    Camera.from_c2w(c2w, intrinsic, height, width),
                    background_color[v_idx],
                    train=train,
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list

        # debug = True
        if debug:
            import cv2

            cv2.imwrite(
                "fuck.png",
                (out["comp_rgb"].detach().cpu().numpy()[0, ..., ::-1] * 255).astype(
                    np.uint8
                ),
            )

        return out

    def get_single_batch_varen_data(self, varen_data, bidx):
        varen_data_single_batch = {}
        for k, v in varen_data.items():
            varen_data_single_batch[k] = v[
                bidx
            ]  # e.g. pose: [B, N_v, 37, 3] -> [N_v, 37, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset") or (k == "tail_scale"):
                varen_data_single_batch[k] = v[
                    bidx : bidx + 1
                ]  # e.g. betas: [B, 39] -> [1, 39]
        return varen_data_single_batch

    def get_single_view_varen_data(self, varen_data, vidx):
        varen_data_single_view = {}
        for k, v in varen_data.items():
            assert v.shape[0] == 1
            if (
                k == "betas"
                or (k == "joint_offset")
                or (k == "face_offset")
                or (k == "transform_mat_neutral_pose")
                or (k == "tail_scale")
            ):
                varen_data_single_view[k] = v  # e.g. betas: [1, 39] -> [1, 39]
            else:
                varen_data_single_view[k] = v[
                    :, vidx : vidx + 1
                ]  # e.g. body_pose: [1, N_v, 37, 3] -> [1, 1, 37, 3]
        return varen_data_single_view

    def forward_gs(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        varen_data,  # e.g., pose:[B, Nv, 37, 3], betas:[B, 39]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs,
    ):

        batch_size = gs_hidden_features.shape[0]

        # obtain gs_features embedding, cur points position, and also varen params
        query_gs_features, query_points, varen_data = self.query_latent_feat(
            query_points, varen_data, gs_hidden_features, additional_features
        )

        gs_attr_list = []
        for b in range(batch_size):
            if isinstance(query_gs_features, dict):
                gs_attr = self.forward_gs_attr(
                    query_gs_features["coarse"][b],
                    query_points[b],
                    None,
                    debug,
                    x_fine=query_gs_features["fine"][b],
                )
            else:
                gs_attr = self.forward_gs_attr(
                    query_gs_features[b], query_points[b], None, debug
                )
            gs_attr_list.append(gs_attr)

        return gs_attr_list, query_points, varen_data

    def forward_animate_gs(
        self,
        gs_attr_list,
        query_points,
        varen_data,
        c2w,
        intrinsic,
        height,
        width,
        background_color,
        debug=False,
        df_data=None,  # deepfashion-style dataset
        train: bool = False,
    ):
        batch_size = len(gs_attr_list)
        out_list = []
        cano_out_list = []  # inference DO NOT use

        N_view = varen_data["global_orient"].shape[1]

        for b in range(batch_size):
            gs_attr = gs_attr_list[b]
            query_pt = query_points[b]
            # len(animatable_gs_model_list) = num_view
            merge_animatable_gs_model_list, cano_gs_model_list = self.animate_gs_model(
                gs_attr,
                query_pt,
                self.get_single_batch_varen_data(varen_data, b),
                debug=debug,
            )

            animatable_gs_model_list = merge_animatable_gs_model_list[:N_view]

            assert len(animatable_gs_model_list) == c2w.shape[1]

            # gs render animated gs model.
            out_list.append(
                self.forward_single_batch(
                    animatable_gs_model_list,
                    c2w[b],
                    intrinsic[b],
                    height,
                    width,
                    background_color[b] if background_color is not None else None,
                    debug=debug,
                    train=train,
                )
            )

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v

        out["comp_rgb"] = out["comp_rgb"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 3, H, W]
        out["comp_mask"] = out["comp_mask"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        out["comp_depth"] = out["comp_depth"].permute(
            0, 1, 4, 2, 3
        )  # [B, NV, H, W, 3] -> [B, NV, 1, H, W]
        return out

    def forward(
        self,
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        varen_data,  # e.g., pose:[B, Nv, 37, 3], betas:[B, 39]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        train: bool = False,
        **kwargs,
    ):

        # need shape_params of varen_data to get query points and get "transform_mat_neutral_pose"
        # only forward gs params
        gs_attr_list, query_points, varen_data = self.forward_gs(
            gs_hidden_features,
            query_points,
            varen_data=varen_data,
            additional_features=additional_features,
            debug=debug,
        )

        out = self.forward_animate_gs(
            gs_attr_list,
            query_points,
            varen_data,
            c2w,
            intrinsic,
            height,
            width,
            background_color,
            debug,
            df_data=kwargs.get("df_data", None),
            train=train,
        )
        out["gs_attr"] = gs_attr_list

        return out


def test1():
    import cv2

    human_model_path = "./pretrained_models/human_model_files"
    device = "cuda"

    # root_dir = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data"
    # meta_path = "/data1/projects/ExAvatar_RELEASE/avatar/data/Custom/data/data_list.json"
    # dataset = ExAvatarDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(224, 224), source_image_res=384)

    # root_dir = "/data1/datasets1/3d_human_data/humman/humman_compressed"
    # meta_path = "/data1/datasets1/3d_human_data/humman/humman_id_debug_list.json"
    # dataset = HuMManDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=3,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384)

    # from openlrm.datasets.static_human import StaticHumanDataset
    # root_dir = "./train_data/static_human_data"
    # meta_path = "./train_data/static_human_data/data_id_list.json"
    # dataset = StaticHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    # from openlrm.datasets.singleview_human import SingleViewHumanDataset
    # root_dir = "./train_data/single_view"
    # meta_path = "./train_data/single_view/data_list.json"
    # dataset = SingleViewHumanDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=0,
    #                 render_image_res_low=384, render_image_res_high=384,
    #                 render_region_size=(682, 384), source_image_res=384,
    #                 debug=False)

    from accelerate.utils import set_seed

    set_seed(1234)
    from LHM.datasets.video_human import VideoHumanDataset

    root_dir = "./train_data/ClothVideo"
    meta_path = "./train_data/ClothVideo/label/valid_id_with_img_list.json"
    dataset = VideoHumanDataset(
        root_dirs=root_dir,
        meta_path=meta_path,
        sample_side_views=7,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        enlarge_ratio=[0.85, 1.2],
        debug=False,
    )

    data = dataset[0]

    def get_smplx_params(data):
        smplx_params = {}
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
            "betas",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                smplx_params[k] = data[k]
        return smplx_params

    smplx_data = get_smplx_params(data)

    smplx_data_tmp = {}
    for k, v in smplx_data.items():
        smplx_data_tmp[k] = v.unsqueeze(0).to(device)
        print(k, v.shape)
    smplx_data = smplx_data_tmp

    c2ws = data["c2ws"].unsqueeze(0).to(device)
    intrs = data["intrs"].unsqueeze(0).to(device)
    render_images = data["render_image"].numpy()
    render_h = data["render_full_resolutions"][0, 0]
    render_w = data["render_full_resolutions"][0, 1]
    render_bg_colors = data["render_bg_colors"].unsqueeze(0).to(device)
    print("c2ws", c2ws.shape, "intrs", intrs.shape, intrs)

    gs_render = GS3DRenderer(
        human_model_path=human_model_path,
        subdivide_num=2,
        smpl_type="smplx",
        feat_dim=64,
        query_dim=64,
        use_rgb=False,
        sh_degree=3,
        mlp_network_config=None,
        xyz_offset_max_step=1.8 / 32,
        expr_param_dim=10,
        shape_param_dim=10,
        fix_opacity=True,
        fix_rotation=True,
    )
    gs_render.to(device)

    out = gs_render.forward(
        gs_hidden_features=torch.zeros((1, 2048, 64)).float().to(device),
        query_points=None,
        smplx_data=smplx_data,
        c2w=c2ws,
        intrinsic=intrs,
        height=render_h,
        width=render_w,
        background_color=render_bg_colors,
        debug=False,
    )
    os.makedirs("./debug_vis/gs_render", exist_ok=True)
    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(
                        f"./debug_vis/gs_render/{b_idx}_{v_idx}.ply"
                    )
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join(
                    "./debug_vis/gs_render", f"{b_idx}_{v_idx}_{k}.jpg"
                )
                img = (
                    v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() * 255
                ).astype(np.uint8)
                print(img.shape, save_path)
                if "mask" in k:
                    render_img = render_images[v_idx].transpose(1, 2, 0) * 255
                    cv2.imwrite(
                        save_path,
                        np.hstack(
                            [np.tile(img, (1, 1, 3)), render_img.astype(np.uint8)]
                        ),
                    )
                else:
                    cv2.imwrite(save_path, img)


# Simple VAREN debug run using identity cameras and params from data/example_params.json
def test_varen():
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build minimal config
    cfg = CfgNode(new_allowed=True)
    cfg.RENDERER = CfgNode(new_allowed=True)
    cfg.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
    cfg.RENDERER.SUBDIVIDE_NUM = 1
    cfg.RENDERER.FEAT_DIM = 64
    cfg.RENDERER.QUERY_DIM = 64
    cfg.RENDERER.USE_RGB = True
    cfg.RENDERER.SH_DEGREE = 3
    cfg.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
    cfg.RENDERER.MLP_NETWORK_CONFIG = CfgNode()
    cfg.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
    cfg.RENDERER.SHAPE_PARAM_DIM = 39
    cfg.RENDERER.CLIP_SCALING = [100, 0.01, 0.05, 3000]
    cfg.RENDERER.DECODER_MLP = False
    cfg.RENDERER.SKIP_DECODER = True
    cfg.RENDERER.FIX_OPACITY = False
    cfg.RENDERER.FIX_ROTATION = False
    cfg.RENDERER.DECODE_WITH_EXTRA_INFO = None
    cfg.RENDERER.GRADIENT_CHECKPOINTING = False
    cfg.RENDERER.APPLY_POSE_BLENDSHAPE = False

    renderer: GS3DRenderer = GS3DRenderer(cfg).to(device)

    # Load VAREN params
    params_path = os.path.join('data', 'example_params.json')
    assert os.path.exists(params_path), f"Params file not found: {params_path}"
    with open(params_path, 'r') as f:
        params = json.load(f)

    def to_batched_tensor(x, is_betas=False):
        t = torch.tensor(x, dtype=torch.float32, device=device)
        if is_betas:
            if t.dim() == 1:
                t = t.unsqueeze(0)  # [D] -> [1, D]
            return t
        # root/body/trans: expect [1, Nv, ...]
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [3] -> [1, 3]
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [Nv, 3] -> [1, Nv, 3]
        if t.dim() == 3:
            t = t.unsqueeze(0)  # [Nv, J, 3] -> [1, Nv, J, 3]
        return t

    varen_data = {}
    if 'global_orient' in params:
        varen_data['global_orient'] = to_batched_tensor(params['global_orient'])
    if 'pose' in params:
        varen_data['pose'] = to_batched_tensor(torch.tensor(params['pose']).view(-1, 3))
    varen_data['trans'] = to_batched_tensor([0., 0., 30.])
    if 'betas' in params:
        varen_data['betas'] = to_batched_tensor(params['betas'], is_betas=True)

    # Basic validation
    required_keys = ['global_orient', 'pose', 'trans', 'betas']
    missing = [k for k in required_keys if k not in varen_data]
    assert len(missing) == 0, f"Missing required keys in params: {missing}"

    B = varen_data['global_orient'].shape[0]
    Nv = varen_data['global_orient'].shape[1]

    # Query canonical points
    positions, varen_data = renderer.get_query_points(varen_data, device)
    N = positions.shape[1]

    # Zero features for debugging
    query_dim = cfg.RENDERER.QUERY_DIM
    gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

    # Identity cameras and intrinsics
    c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    intrinsic = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    height, width = 512, 512
    # Set focal length and principal point
    intrinsic[:, :, 0, 0] = 5000.0
    intrinsic[:, :, 1, 1] = 5000.0
    intrinsic[:, :, 0, 2] = float(width) * 0.5
    intrinsic[:, :, 1, 2] = float(height) * 0.5
    background_color = torch.ones((B, Nv, 3), dtype=torch.float32, device=device)

    out = renderer.forward(
        gs_hidden_features=gs_hidden_features,
        query_points=positions,
        varen_data=varen_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=height,
        width=width,
        additional_features=None,
        background_color=background_color,
        debug=False,
        df_data=None,
    )

    # Save debug outputs
    save_dir = os.path.join('./debug_vis', 'gs_varen_render')
    os.makedirs(save_dir, exist_ok=True)
    for k, v in out.items():
        if k == 'comp_rgb_bg' or k == 'gs_attr':
            continue
        if k == '3dgs':
            for b_idx in range(len(v)):
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(os.path.join(save_dir, f"{b_idx}_{v_idx}.ply"))
            continue
        # Tensor outputs: [B, NV, C, H, W]
        for b_idx in range(len(v)):
            for v_idx in range(len(v[b_idx])):
                img = v[b_idx, v_idx]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{b_idx}_{v_idx}_{k}.jpg"), img)

    print(f"Saved debug outputs to {save_dir}")

    # Also render posed VAREN mesh with Pyrender and save overlays
    varen_data['global_orient'] = varen_data['global_orient'][0]
    varen_data['pose'] = varen_data['pose'][0]
    varen_data['trans'] = varen_data['trans'][0]
    overlay = renderer.render_mesh_with_pyrender(
        varen_data=varen_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=height,
        width=width,
        base_image=out.get('comp_rgb', None),
    )  # [B, Nv, 3, H, W]

    for b in range(overlay.shape[0]):
        for v in range(overlay.shape[1]):
            img = (overlay[b, v].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"{b}_{v}_pyrender_overlay.jpg"), img)


def test_varen2():
    """
    Render a video using GS3DRenderer from Blender meta.json (outputs/meta.json).
    Assumes all VAREN params (betas, pose, global_orient, trans) are zeros.
    Saves MP4 to ./debug_vis/gs_varen_video/comp_rgb.mp4
    """
    import imageio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load meta
    meta_path = "/data2/lvjin/cvpr26/code/bpy-renderer/examples/object/outputs/meta.json"
    assert os.path.exists(meta_path), f"meta.json not found: {meta_path}"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    width = int(meta.get("width", 512))
    height = int(meta.get("height", 512))
    locations = meta.get("locations", [])
    assert len(locations) > 0, "No locations found in meta.json"

    def compute_fx_fy_from_fov_x(w, h, fov_x_rad):
        fx = (0.5 * float(w)) / math.tan(0.5 * float(fov_x_rad))
        fy = fx
        return float(fx), float(fy)

    # Build minimal renderer config
    cfg = CfgNode(new_allowed=True)
    cfg.RENDERER = CfgNode(new_allowed=True)
    cfg.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
    cfg.RENDERER.SUBDIVIDE_NUM = 1
    cfg.RENDERER.FEAT_DIM = 64
    cfg.RENDERER.QUERY_DIM = 64
    cfg.RENDERER.USE_RGB = True
    cfg.RENDERER.SH_DEGREE = 3
    cfg.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
    cfg.RENDERER.MLP_NETWORK_CONFIG = CfgNode(new_allowed=True)
    cfg.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
    cfg.RENDERER.SHAPE_PARAM_DIM = 39
    cfg.RENDERER.CLIP_SCALING = [100, 1000, 1000, 3000]
    cfg.RENDERER.DECODER_MLP = False
    cfg.RENDERER.SKIP_DECODER = True
    cfg.RENDERER.FIX_OPACITY = False
    cfg.RENDERER.FIX_ROTATION = False
    cfg.RENDERER.DECODE_WITH_EXTRA_INFO = None
    cfg.RENDERER.GRADIENT_CHECKPOINTING = False
    cfg.RENDERER.APPLY_POSE_BLENDSHAPE = False

    renderer: GS3DRenderer = GS3DRenderer(cfg).to(device)

    # Build zero VAREN params
    B = 1
    Nv = len(locations)
    joint_num = renderer.varen_model.varen.joint_num
    shape_dim = renderer.varen_model.varen.layer.SHAPE_SPACE_DIM

    # Load VAREN params
    params_path = os.path.join('data', 'example_params.json')
    assert os.path.exists(params_path), f"Params file not found: {params_path}"
    with open(params_path, 'r') as f:
        params = json.load(f)

    def to_batched_tensor(x, is_betas=False):
        t = torch.tensor(x, dtype=torch.float32, device=device)
        if is_betas:
            if t.dim() == 1:
                t = t.unsqueeze(0)  # [D] -> [1, D]
            return t
        # root/body/trans: expect [1, Nv, ...]
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [3] -> [1, 3]
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [Nv, 3] -> [1, Nv, 3]
        if t.dim() == 3:
            t = t.unsqueeze(0)  # [Nv, J, 3] -> [1, Nv, J, 3]
        return t

    varen_data = {}
    if 'global_orient' in params:
        varen_data['global_orient'] = to_batched_tensor(params['global_orient']).repeat(1, Nv, 1, 1)
    if 'pose' in params:
        varen_data['pose'] = to_batched_tensor(torch.tensor(params['pose']).view(-1, 3)).repeat(1, Nv, 1, 1)
    varen_data['trans'] = to_batched_tensor([0., 0., 0.]).repeat(1, Nv, 1, 1)
    if 'betas' in params:
        varen_data['betas'] = to_batched_tensor(params['betas'], is_betas=True)

    # Query canonical points
    positions, varen_data = renderer.get_query_points(varen_data, device)
    N = positions.shape[1]

    # Zero features
    query_dim = cfg.RENDERER.QUERY_DIM
    gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

    # Per-frame camera matrices and intrinsics
    c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    intrinsic = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)

    fov_x = locations[0].get('camera_angle_x', None)
    if fov_x is not None:
        fx, fy = compute_fx_fy_from_fov_x(width, height, fov_x)
    else:
        fx = fy = 0.75 * float(width)
    intrinsic[:, :, 0, 0] = fx
    intrinsic[:, :, 1, 1] = fy
    intrinsic[:, :, 0, 2] = float(width) * 0.5
    intrinsic[:, :, 1, 2] = float(height) * 0.5

    # If meta contains normalization matrix used on mesh, compensate by applying its inverse to camera
    N_inv = None
    try:
        norm_dict = meta.get('normalization', {})
        N_mat = norm_dict.get('normalization_matrix', None)
        if N_mat is not None:
            N_np = np.array(N_mat, dtype=np.float64)
            N_inv_np = np.linalg.inv(N_np)
            N_inv = torch.from_numpy(N_inv_np).to(device=device, dtype=torch.float32)
    except Exception:
        N_inv = None

    import trimesh
    rot_blender2pyrender = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])).to(device=device, dtype=torch.float32)
    S = torch.tensor([[1., 0., 0., 0.],
                      [0.,-1., 0., 0.],
                      [0., 0.,-1., 0.],
                      [0., 0., 0., 1.]], dtype=torch.float32, device=device)
    for i, loc in enumerate(locations):
        T = loc.get('transform_matrix', loc.get('transform_matrix_original'))
        if T is None:
            T_np = np.eye(4, dtype=np.float32)
        else:
            T_np = np.array(T, dtype=np.float32)
            if T_np.shape != (4, 4):
                T_np = np.eye(4, dtype=np.float32)
        T_torch = torch.from_numpy(T_np).to(device=device, dtype=torch.float32)
        if N_inv is not None:
            T_torch = N_inv @ T_torch
        c2w[0, i] = rot_blender2pyrender @ T_torch @ S

    background_color = torch.zeros((B, Nv, 3), dtype=torch.float32, device=device)

    varen_data['global_orient'] = varen_data['global_orient']
    varen_data['pose'] = varen_data['pose']
    varen_data['trans'] = varen_data['trans']

    out = renderer.forward(
        gs_hidden_features=gs_hidden_features,
        query_points=positions,
        varen_data=varen_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=height,
        width=width,
        additional_features=None,
        background_color=background_color,
        debug=False,
        df_data=None,
    )

    # Write video from comp_rgb
    save_dir = os.path.join('./debug_vis', 'gs_varen_video')
    os.makedirs(save_dir, exist_ok=True)
    out_video = os.path.join(save_dir, 'comp_rgb.mp4')
    fps = int(meta.get('fps', 24))
    writer = imageio.get_writer(out_video, fps=fps)
    comp_rgb = out.get('comp_rgb')  # [B, Nv, 3, H, W]
    comp_rgb_np = (comp_rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(comp_rgb_np.shape[0]):
        writer.append_data(comp_rgb_np[i])
    writer.close()
    print(f"Saved GS VAREN video to {out_video}")

    # Debug for GSRenderer
    # with torch.no_grad():
    #     means = positions[0]  # [N,3]
    #     ones = torch.ones(means.shape[0], 1, device=device)
    #     means4 = torch.cat([means, ones], dim=1)  # [N,4]
    #     w2c = torch.inverse(c2w[0,0])
    #     cam = (means4 @ w2c.T)  # [N,4]
    #     print('z>0 ratio =', (cam[:,2] > 0).float().mean().item())

    # Also render posed VAREN mesh with Pyrender and save overlays
    # rot = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])).to(device=device, dtype=torch.float32)
    # c2w[:, 0] = c2w[:, 0] @ rot
    # varen_data['global_orient'] = varen_data['global_orient'][0][[0]]
    # varen_data['pose'] = varen_data['pose'][0][[0]]
    # varen_data['trans'] = varen_data['trans'][0][[0]]
    # overlay = renderer.render_mesh_with_pyrender(
    #     varen_data=varen_data,
    #     c2w=(rot_blender2pyrender @ c2w[0, 0]).unsqueeze(0).unsqueeze(0),
    #     intrinsic=intrinsic[:, [0]],
    #     height=height,
    #     width=width,
    #     base_image=None,
    # )  # [B, Nv, 3, H, W]

    # import cv2
    # for b in range(overlay.shape[0]):
    #     for v in range(overlay.shape[1]):
    #         img = (overlay[b, v].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    #         cv2.imwrite(os.path.join(save_dir, f"{b}_{v}_pyrender_overlay.jpg"), img)


def test_varen3():
    """
    Render a video using GS3DRenderer from Blender meta.json (outputs/meta.json).
    Assumes all VAREN params (betas, pose, global_orient, trans) are zeros.
    Saves MP4 to ./debug_vis/gs_varen_video/comp_rgb.mp4
    """
    import imageio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Sequence
    seq_dir = "data/20201128_ID_1_0002_hsmal_007685"
    c2w = torch.load(os.path.join(seq_dir, "c2w.pt"), weights_only=True).to(device)
    mvp_matrix = torch.load(os.path.join(seq_dir, "mvp_matrix.pt"), weights_only=True).to(device)
    proj = torch.bmm(mvp_matrix, c2w)

    width, height = 512, 512

    def compute_intrinsics_from_projection(P: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Given an OpenGL-style projection matrix P (with sign-flipped Y as in nvdiffrast):
        P[0,0] = 1/(tan(fovy/2)*aspect), P[1,1] = -1/tan(fovy/2)
        Derive fx, fy for a pinhole camera used by pyrender.
        We assume principal point at (width/2, height/2).
        Returns fx, fy, cx, cy.
        """
        fx = width * 0.5 * abs(P[0, 0])
        fy = height * 0.5 * abs(P[1, 1])
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)
    
    fx, fy, cx, cy = compute_intrinsics_from_projection(proj[0], width, height)
    c2w = c2w.unsqueeze(0)
    B, Nv = c2w.shape[0], c2w.shape[1]
    intrinsic = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    intrinsic[:, :, 0, 0] = fx
    intrinsic[:, :, 1, 1] = fy
    intrinsic[:, :, 0, 2] = cx
    intrinsic[:, :, 1, 2] = cy

    # Build minimal renderer config
    cfg = CfgNode(new_allowed=True)
    cfg.RENDERER = CfgNode(new_allowed=True)
    cfg.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
    cfg.RENDERER.SUBDIVIDE_NUM = 1
    cfg.RENDERER.FEAT_DIM = 64
    cfg.RENDERER.QUERY_DIM = 64
    cfg.RENDERER.USE_RGB = True
    cfg.RENDERER.SH_DEGREE = 3
    cfg.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
    cfg.RENDERER.MLP_NETWORK_CONFIG = CfgNode(new_allowed=True)
    cfg.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
    cfg.RENDERER.SHAPE_PARAM_DIM = 39
    cfg.RENDERER.CLIP_SCALING = [100, 1000, 1000, 3000]
    cfg.RENDERER.DECODER_MLP = False
    cfg.RENDERER.SKIP_DECODER = True
    cfg.RENDERER.FIX_OPACITY = False
    cfg.RENDERER.FIX_ROTATION = False
    cfg.RENDERER.DECODE_WITH_EXTRA_INFO = None
    cfg.RENDERER.GRADIENT_CHECKPOINTING = False
    cfg.RENDERER.APPLY_POSE_BLENDSHAPE = False

    renderer: GS3DRenderer = GS3DRenderer(cfg).to(device)

    # Load VAREN params
    params_path = os.path.join(seq_dir, os.path.basename(seq_dir) + ".npz")
    assert os.path.exists(params_path), f"Params file not found: {params_path}"
    params = np.load(params_path, allow_pickle=True)
    varen_data = {
        'global_orient': params['global_orient'].reshape(1, 1, 3).repeat(Nv, axis=0).tolist(),
        'pose': params['pose'].reshape(1, 37, 3).repeat(Nv, axis=0).tolist(),
        'betas': params['shape'].tolist(),
        'trans': params['transl'].reshape(1, 1, 3).repeat(Nv, axis=0).tolist(),
    }

    def to_batched_tensor(x, is_betas=False):
        t = torch.tensor(x, dtype=torch.float32, device=device)
        if is_betas:
            if t.dim() == 1:
                t = t.unsqueeze(0)  # [D] -> [1, D]
            return t
        # root/body/trans: expect [1, Nv, ...]
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [3] -> [1, 3]
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [Nv, 3] -> [1, Nv, 3]
        if t.dim() == 3:
            t = t.unsqueeze(0)  # [Nv, J, 3] -> [1, Nv, J, 3]
        return t

    varen_data = {k: to_batched_tensor(v) for k, v in varen_data.items() if k != 'betas'}
    varen_data['betas'] = to_batched_tensor(params['shape'], is_betas=True)

    # Query canonical points
    positions, varen_data = renderer.get_query_points(varen_data, device)
    N = positions.shape[1]

    # Zero features
    query_dim = cfg.RENDERER.QUERY_DIM
    gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

    import trimesh
    rot_blender2pyrender = torch.from_numpy(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])).to(device=device, dtype=torch.float32)
    S = torch.tensor([[1., 0., 0., 0.],
                      [0.,-1., 0., 0.],
                      [0., 0.,-1., 0.],
                      [0., 0., 0., 1.]], dtype=torch.float32, device=device)
    for i in range(Nv):
        T_torch = c2w[0, i].to(device=device, dtype=torch.float32)
        c2w[0, i] = T_torch @ S

    background_color = torch.zeros((B, Nv, 3), dtype=torch.float32, device=device)

    out = renderer.forward(
        gs_hidden_features=gs_hidden_features,
        query_points=positions,
        varen_data=varen_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=height,
        width=width,
        additional_features=None,
        background_color=background_color,
        debug=False,
        df_data=None,
    )

    # Write video from comp_rgb
    save_dir = os.path.join('./debug_vis', 'gs_varen_video')
    os.makedirs(save_dir, exist_ok=True)
    out_video = os.path.join(save_dir, 'comp_rgb.mp4')
    fps = 24
    writer = imageio.get_writer(out_video, fps=fps)
    comp_rgb = out.get('comp_rgb')  # [B, Nv, 3, H, W]
    comp_rgb_np = (comp_rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
    for i in range(comp_rgb_np.shape[0]):
        writer.append_data(comp_rgb_np[i])
    writer.close()
    print(f"Saved GS VAREN video to {out_video}")


def test_varen4():
    """
    Render using GS3DRenderer from animer_outputs (e.g., outputs/horse2/refined_results.pt).
    Expects keys: refined_betas, refined_pose (6D), refined_global_orient (6D), refined_cam_t, refined_tail_scale.
    Saves results to ./debug_vis/gs_varen_animer
    """
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build renderer config (minimal, similar to test_varen/test_varen2/test_varen3)
    cfg = CfgNode(new_allowed=True)
    cfg.RENDERER = CfgNode(new_allowed=True)
    cfg.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
    cfg.RENDERER.SUBDIVIDE_NUM = 1
    cfg.RENDERER.FEAT_DIM = 64
    cfg.RENDERER.QUERY_DIM = 64
    cfg.RENDERER.USE_RGB = True
    cfg.RENDERER.SH_DEGREE = 3
    cfg.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
    cfg.RENDERER.MLP_NETWORK_CONFIG = CfgNode(new_allowed=True)
    cfg.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
    cfg.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
    cfg.RENDERER.SHAPE_PARAM_DIM = 39
    cfg.RENDERER.CLIP_SCALING = [100, 1000, 1000, 3000]
    cfg.RENDERER.DECODER_MLP = False
    cfg.RENDERER.SKIP_DECODER = True
    cfg.RENDERER.FIX_OPACITY = False
    cfg.RENDERER.FIX_ROTATION = False
    cfg.RENDERER.DECODE_WITH_EXTRA_INFO = None
    cfg.RENDERER.GRADIENT_CHECKPOINTING = False
    cfg.RENDERER.APPLY_POSE_BLENDSHAPE = False

    renderer: GS3DRenderer = GS3DRenderer(cfg).to(device)

    # Load animer outputs
    animer_path = os.path.join('outputs', 'horse2', 'refined_results.pt')
    assert os.path.exists(animer_path), f"File not found: {animer_path}"
    animer_outputs = torch.load(animer_path, weights_only=True)

    # Extract first frame
    betas = animer_outputs['refined_betas'][[0]].to(device=device, dtype=torch.float32)
    pose6d = animer_outputs['refined_pose'][[0]]  # [1, J*6]
    global6d = animer_outputs['refined_global_orient'][[0]]  # [1, 6]
    tail_scale = animer_outputs.get('refined_tail_scale', None)
    cam_t = animer_outputs.get('refined_cam_t', None)

    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
    # Convert rotations to axis-angle
    pose_mat = rotation_6d_to_matrix(pose6d.view(1, -1, 6).to(device=device, dtype=torch.float32))
    pose_aa = matrix_to_axis_angle(pose_mat)  # [1, J, 3]
    glob_mat = rotation_6d_to_matrix(global6d.view(1, -1, 6).to(device=device, dtype=torch.float32))
    glob_aa = matrix_to_axis_angle(glob_mat).view(1, 1, 3)  # [1, Nv, 3] with Nv=1

    # Build varen_data for a single view
    B, Nv = 1, 1
    varen_data = {
        'betas': betas,  # [1, D]
        'global_orient': glob_aa.unsqueeze(1),  # [1, 1, 1, 3]
        'pose': pose_aa.unsqueeze(1),  # [1, 1, J, 3]
        'trans': torch.zeros((B, Nv, 3), dtype=torch.float32, device=device),
    }
    if cam_t is not None:
        t = cam_t[[0]].to(device=device, dtype=torch.float32)
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, 1, 3] or [1, 3]
        if t.shape[-1] == 3:
            varen_data['trans'] = t.view(B, Nv, 3)
    if tail_scale is not None:
        varen_data['tail_scale'] = tail_scale[[0]].to(device=device, dtype=torch.float32)

    # Query canonical points
    positions, varen_data = renderer.get_query_points(varen_data, device)
    N = positions.shape[1]

    # Zero features
    query_dim = cfg.RENDERER.QUERY_DIM
    gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

    # Camera and intrinsics (identity, pinhole)
    c2w = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    intrinsic = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(B, Nv, 1, 1)
    width = height = 256
    intrinsic[:, :, 0, 0] = 5000.0
    intrinsic[:, :, 1, 1] = 5000.0
    intrinsic[:, :, 0, 2] = float(width) * 0.5
    intrinsic[:, :, 1, 2] = float(height) * 0.5
    background_color = torch.zeros((B, Nv, 3), dtype=torch.float32, device=device)

    out = renderer.forward(
        gs_hidden_features=gs_hidden_features,
        query_points=positions,
        varen_data=varen_data,
        c2w=c2w,
        intrinsic=intrinsic,
        height=height,
        width=width,
        additional_features=None,
        background_color=background_color,
        debug=False,
        df_data=None,
    )

    # Save outputs
    save_dir = os.path.join('./debug_vis', 'gs_varen_animer')
    os.makedirs(save_dir, exist_ok=True)
    comp_rgb = out.get('comp_rgb')  # [B, Nv, 3, H, W]
    if comp_rgb is not None:
        for b in range(comp_rgb.shape[0]):
            for v in range(comp_rgb.shape[1]):
                img = (comp_rgb[b, v].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{b}_{v}_comp_rgb.jpg"), img)
    if '3dgs' in out:
        gs_list = out['3dgs']
        for b_idx in range(len(gs_list)):
            for v_idx in range(len(gs_list[b_idx])):
                out['3dgs'][b_idx][v_idx].save_ply(os.path.join(save_dir, f"{b_idx}_{v_idx}.ply"))

    print(f"Saved outputs to {save_dir}")

if __name__ == "__main__":
    # test_varen()
    # test_varen2()
    test_varen3()
    # test_varen4()
