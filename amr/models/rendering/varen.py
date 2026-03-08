import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import copy
import math
import os.path as osp
import pdb
import pickle

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import trimesh
from pytorch3d.io import load_ply, save_ply
from pytorch3d.ops import SubdivideMeshes, knn_points
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from amr.models.varen.lbs import batch_rigid_transform as varen_batch_rigid_transform
from torch.nn import functional as F

from amr.models.rendering.mesh_utils import Mesh
from amr.models.rendering.smplx import smplx
from amr.models.rendering.smplx.smplx.lbs import blend_shapes
from amr.models.rendering.smplx.vis_utils import render_mesh
from amr.models.varen import VAREN


def avaliable_device():

    import torch

    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


class VAREN_Mesh(object):
    def __init__(
        self,
        varen_model_path,
        shape_param_dim=39,
        subdivide_num=2,
    ):
        """VAREN using dense sampling"""
        super().__init__()
        self.varen_model_path = varen_model_path
        self.shape_param_dim = shape_param_dim
        self.subdivide_num = subdivide_num
        self.layer = VAREN(model_path=self.varen_model_path, use_muscle_deformations=False)

        self.vertex_num = self.layer.v_template.shape[0]
        self.face = self.layer.faces_tensor
        self.joint_num = self.layer.NUM_JOINTS + 1  # Note that the number of joints include the root joint
        self.root_joint_idx = 0

        self.subdivider_list = self.get_subdivider(self.subdivide_num)

    def set_id_info(self, shape_param, joint_offset, locator_offset):
        self.shape_param = shape_param
        self.joint_offset = joint_offset
        self.locator_offset = locator_offset

    def get_joint_offset(self, joint_offset):
        device = joint_offset.device
        batch_size = joint_offset.shape[0]
        weight = torch.ones((batch_size, self.joint_num, 1)).float().to(device)
        weight[:, self.root_joint_idx, :] = 0
        joint_offset = joint_offset * weight
        return joint_offset

    def get_subdivider(self, subdivide_num):
        vert = self.layer.v_template.float()
        face = torch.LongTensor(self.face)
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num - 1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def subdivide_mesh(self, vert, feat_list=None):
        face = self.face
        mesh = Meshes(vert[None, :, :], face[None, :, :])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list,1)
                for subdivider in self.subdivider_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                return vert
            else:
                return vert, *feat_list


class VARENSubdividedMeshModel(nn.Module):
    def __init__(
        self,
        varen_model_path: str,
        subdivide_num: int=2,
        shape_param_dim: int=39,
        apply_pose_blendshape: bool=False,
    ) -> None:
        super().__init__()

        self.varen = VAREN_Mesh(
            varen_model_path=varen_model_path,
            shape_param_dim=shape_param_dim,
            subdivide_num=subdivide_num,
        )
        self.layer = self.varen.layer
        self.subdivide_num = subdivide_num

        # register
        self.apply_pose_blendshape = apply_pose_blendshape
        self.varen_init()

    def varen_init(self):
        """
        Initialize the sub-devided varen model by registering buffers for various attributes
        This method performs the following steps:
        1. Upsamples the mesh and other assets.
        2. Computes skinning weights, pose directions, expression directions, and various flags for different body parts.
        3. Reshapes and permutes the pose and expression directions.
        4. Converts the flags to boolean values.
        5. Registers buffers for the computed attributes.
        Args:
            self: The object instance.
        Returns:
            None
        """
        face_upsampled = self.varen.subdivider_list[-1]._subdivided_faces.cpu() if self.subdivide_num > 0 else self.varen.face
        self.register_buffer("face_upsampled", face_upsampled)

        num_vertices = self.varen.layer.v_template.shape[0]
        attributes = [self.varen.layer.lbs_weights, 
                      self.varen.layer.posedirs.transpose(0, 1).view(num_vertices, -1), 
                      self.varen.layer.shapedirs.view(num_vertices, -1), 
                      self.varen.layer.J_regressor.transpose(0, 1)]
        ret = self.varen.subdivide_mesh(self.varen.layer.v_template.float(), attributes)
        v_template_upsampled, lbs_weights, posedirs, shapedirs, J_regressor = ret

        num_vertices_upsampled = v_template_upsampled.shape[0]
        self.num_vertices_upsampled = num_vertices_upsampled
        posedirs = posedirs.reshape(num_vertices_upsampled * 3, -1).permute(1, 0)
        shapedirs = shapedirs.view(num_vertices_upsampled, 3 , self.varen.layer.SHAPE_SPACE_DIM)
        J_regressor = J_regressor.permute(1, 0)

        self.register_buffer("J_regressor_up", J_regressor.contiguous())
        self.register_buffer("v_template_up", v_template_upsampled.contiguous())
        self.register_buffer("lbs_weights_up", lbs_weights.contiguous())
        self.register_buffer("shapedirs_up", shapedirs.contiguous())
        self.register_buffer("posedirs_up", posedirs.contiguous())

    def get_zero_pose(self, shape_param, device):
        varen = self.varen
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_body_pose = torch.zeros((batch_size, (self.varen.joint_num - 1) * 3)).float().to(device)
        output = self.layer(
            global_orient=zero_pose,
            pose=zero_body_pose,
            betas=shape_param,
        )
        joint_zero_pose = output.joints[:, :self.varen.joint_num, :]  # T-pose joints
        return joint_zero_pose

    def get_transform_mat_joint(
        self, transform_mat_neutral_pose, joint_template, varen_param
    ):
        """_summary_
        Args:
            transform_mat_neutral_pose (_type_): [B, 38, 4, 4]
            joint_template (_type_): [B, 38, 3]
            varen_param (_type_): dict
        Returns:
            _type_: _description_
        """
        # 1. T-Pose to Template
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. Template to Image pose
        root_pose = varen_param["global_orient"]
        body_pose = varen_param["pose"]
        # trans = smplx_param['trans']

        # forward kinematics
        pose = torch.cat(
            (
                root_pose,
                body_pose,
            ),
            dim=1,
        )  # [B, 38, 3]
        pose = axis_angle_to_matrix(pose)  # [B, 38, 3, 3]
        # Optional tail scaling
        tail_scale = varen_param.get("tail_scale", None)
        posed_joints, transform_mat_joint_2 = varen_batch_rigid_transform(
            pose[:, :, :, :], joint_template[:, :, :], self.layer.parents, tail_scale=tail_scale
        )
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 38, 4, 4]

        # combine 1. T-Pose to Template and 2. Template to Image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint):
        batch_size = transform_mat_joint.shape[0]
        skinning_weight = self.lbs_weights_up.unsqueeze(0).repeat(batch_size, 1, 1)
        transform_mat_vertex = torch.matmul(
            skinning_weight,
            transform_mat_joint.view(batch_size, self.varen.joint_num, 16),
        ).view(batch_size, self.num_vertices_upsampled, 4, 4)
        return transform_mat_vertex

    def get_posed_blendshape(self, varen_param: dict):
        # TODO: Why only apply on hand and face?
        # posed_blendshape is only applied on hand and face, which parts are closed to smplx model
        root_pose = varen_param["global_orient"]
        pose = varen_param["pose"]  # [B, 37, 3]
        batch_size = root_pose.shape[0]
 
        # smplx pose-dependent vertex offset
        pose = (
            axis_angle_to_matrix(pose) - torch.eye(3)[None, None, :, :].float().cuda()
        ).view(batch_size, (self.varen.joint_num - 1) * 9)
        # (B, 37 * 9) x (37*9, V)

        pose_offset = torch.matmul(pose.detach(), self.posedirs_up).view(
            batch_size, self.num_vertices_upsampled, 3
        )
        return pose_offset

    def lbs(self, xyz, transform_mat_vertex, trans):
        batch_size = xyz.shape[0]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # T-pose. xyz: [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, self.num_vertices_upsampled, 4
        )[:, :, :3]  # [B, N, 3]
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1) if trans.dim() == 2 else xyz + trans
        return xyz

    def get_template_pose(self, shape_param, device):
        v_template = self.varen.layer.v_template.unsqueeze(0).repeat(shape_param.shape[0], 1, 1).float().to(device)
        v_shaped = v_template + blend_shapes(shape_param, self.layer.shapedirs)
        joints_template = torch.einsum('bik,ji->bjk', [v_shaped, self.varen.layer.J_regressor])
        return joints_template

    def transform_to_posed_verts_from_neutral_pose(
        self, mean_3d, varen_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose (neutral pose is T-Pose).  
            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            varen_data (dict): VAREN data containing body_pose with shape [B*Nv, 37, 3] and betas with shape [B, 39].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.
        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """
        batch_size = mean_3d.shape[0]
        shape_param = varen_data["betas"]

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )

        # compute vertices-LBS function
        transform_mat_null_vertex = self.get_transform_mat_vertex(transform_mat_neutral_pose)  # T-Pose to Template

        null_mean_3d = self.lbs(mean_3d, transform_mat_null_vertex, torch.zeros_like(varen_data["trans"]))

        # blend_shape offset
        blend_shape_offset = blend_shapes(shape_param, self.shapedirs_up)
        null_mean3d_blendshape = null_mean_3d + blend_shape_offset

        # get transformation matrix of the nearest vertex and perform lbs
        joint_null_pose = self.get_template_pose(shape_param, device)  # Must be the joints of Template

        transform_mat_joint, j3d = self.get_transform_mat_joint(
            None, joint_null_pose, varen_data
        )  # Template to Image Pose

        # compute vertices-LBS function
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint
        )  # Template to Image Pose

        posed_mean_3d = self.lbs(
            null_mean3d_blendshape, transform_mat_vertex, varen_data["trans"]
        )  # posed with varen_param

        # as we do not use transform port [...,:,3],so we simply compute chain matrix
        neutral_to_posed_vertex = torch.matmul(
            transform_mat_vertex, transform_mat_null_vertex
        )  # [B, N, 4, 4]

        return posed_mean_3d, neutral_to_posed_vertex

    def get_query_points(self, varen_data, device):
        """transform_mat_neutral_pose is function to warp T-Pose to Template"""
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
            self.get_neutral_pose(
                use_id_info=False,  # we blendshape at zero-pose
                shape_param=varen_data["betas"],
                device=device,
            )
        )

        return (
            mesh_neutral_pose,
            mesh_neutral_pose_wo_upsample,
            transform_mat_neutral_pose,  # T-Pose to Template
        )

    def transform_to_posed_verts(self, varen_data, device):
        """_summary_
        Args:
            varen_data (_type_): e.g., body_pose:[B*Nv, 37, 3], betas:[B*Nv, 39]
        """

        # T-Pose verts
        mesh_neutral_pose, _, transform_mat_neutral_pose = self.get_query_points(
            varen_data, device
        )

        mean_3d, transform_matrix = self.transform_to_posed_verts_from_neutral_pose(
            mesh_neutral_pose,
            varen_data,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
            device,
        )

        return mean_3d, transform_matrix

    def upsample_mesh_batch(
        self,
        varen,
        shape_param,
        neutral_body_pose,
        betas,
        device=None,
    ):
        """using blendshape to offset pts"""

        device = device if device is not None else avaliable_device()

        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size, 1, 3)).float().to(device)

        dense_pts = self.v_template_up.to(device)
        dense_pts = dense_pts.unsqueeze(0).repeat(batch_size, 1, 1)

        blend_shape_offset = blend_shapes(betas, self.shapedirs_up)

        dense_pts = dense_pts + blend_shape_offset

        joint_template = self.get_template_pose(shape_param, device)  # The joints of Template

        neutral_pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose,
            ),
            dim=1,
        )  # [B, 38, 3]

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, self.varen.joint_num, 3)
        )  # [B, 38, 3, 3]
        posed_joints, transform_mat_joint = varen_batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_template[:, :, :], self.layer.parents
        )

        skinning_weight = self.lbs_weights_up.unsqueeze(0).repeat(batch_size, 1, 1)

        # B 38 4,4, B N 38 -> B N 4 4
        transform_mat_vertex = torch.einsum(
            "blij,bnl->bnij", transform_mat_joint, skinning_weight
        )
        mesh_neutral_pose_upsampled = self.lbs(dense_pts, transform_mat_vertex, None)

        return mesh_neutral_pose_upsampled

    def transform_to_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           torch.Tensor: Posed vertices with shape [B*Nv, N, 3] + offset.
        """

        batch_size = mean_3d.shape[0]
        shape_param = smplx_data["betas"]
        face_offset = smplx_data.get("face_offset", None)
        joint_offset = smplx_data.get("joint_offset", None)
        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # smplx facial expression offset
        smplx_expr_offset = (
            smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        mean_3d = mean_3d + smplx_expr_offset  # 大 pose

    def get_neutral_pose(self, shape_param, device, use_id_info=False):
        """Get the T-Pose for VAREN"""
        varen = self.varen
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 1,3)).float().to(device)
        # T-Pose
        neutral_body_pose = torch.zeros((batch_size, self.varen.joint_num - 1, 3)).float().to(device)

        if use_id_info:
            shape_param = shape_param
        else:
            shape_param = (
                torch.zeros((batch_size, self.varen.layer.SHAPE_SPACE_DIM)).float().to(device)
            )

        output = self.varen.layer(
            global_orient=zero_pose,
            pose=neutral_body_pose,
            betas=shape_param,
        )  # Output of T-Pose (neutral pose)

        # using dense sample strategy, and warp to neutral pose
        mesh_neutral_pose_upsampled = self.upsample_mesh_batch(
            self.varen,
            shape_param=shape_param,
            neutral_body_pose=neutral_body_pose,
            betas=shape_param,
            device=device,
        )

        mesh_neutral_pose = output.vertices
        joint_neutral_pose = output.joints[:, : self.varen.joint_num, :]  # T-Pose  [B, 38, 3]

        # compute transformation matrix for making T-Pose to Template
        neutral_body_pose = neutral_body_pose.view(batch_size, self.varen.joint_num - 1, 3)

        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose))
        )

        pose = torch.cat((zero_pose, neutral_body_pose_inv), dim=1)

        pose = axis_angle_to_matrix(pose)  # [B, 38, 3, 3]

        # T-Pose to Template
        _, transform_mat_neutral_pose = varen_batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.layer.parents
        )  # [B, 38, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            transform_mat_neutral_pose,
        )


def generate_varen_mesh():
    """Debug helper for VARENSubdividedMeshModel.

    - Instantiates a VARENSubdividedMeshModel
    - Generates neutral (T-pose) meshes (upsampled and original)
    - Exports OBJ files and saves NPZ/NPY for quick inspection
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model and output paths
    varen_model_path = "data/varen/"
    subdivide_num = 1
    out_dir = "debug/varen_points"
    os.makedirs(out_dir, exist_ok=True)

    # Build model
    model = VARENSubdividedMeshModel(
        varen_model_path=varen_model_path,
        subdivide_num=subdivide_num,
        shape_param_dim=39,
        apply_pose_blendshape=False,
    ).to(device)

    # Prepare minimal inputs (neutral shape)
    betas = torch.zeros((1, model.varen.layer.SHAPE_SPACE_DIM), device=device)
    varen_data = {"betas": betas}

    # Query neutral meshes
    mesh_up, mesh_orig, _ = model.get_query_points(varen_data=varen_data, device=device)

    # Faces
    faces_up = model.face_upsampled.detach().cpu().numpy()
    faces_orig = model.varen.face.detach().cpu().numpy()

    # Export neutral (upsampled)
    verts_up = mesh_up[0].detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts_up, faces=faces_up, process=False).export(
        osp.join(out_dir, f"varen_subdivide{subdivide_num}.obj")
    )
    np.save(osp.join(out_dir, f"varen_subdivide{subdivide_num}.npy"), verts_up)

    # Export neutral (original resolution)
    verts_orig = mesh_orig[0].detach().cpu().numpy()
    trimesh.Trimesh(vertices=verts_orig, faces=faces_orig, process=False).export(
        osp.join(out_dir, "varen.obj")
    )
    np.save(osp.join(out_dir, "varen.npy"), verts_orig)

    # Optionally: produce a trivial posed mesh (all-zero pose/orient/trans)
    global_orient = torch.zeros((1, 1, 3), device=device)
    pose = torch.zeros((1, model.varen.joint_num - 1, 3), device=device)
    trans = torch.zeros((1, 3), device=device)
    varen_pose = {"betas": betas, "global_orient": global_orient, "pose": pose, "trans": trans}
    posed_up, _ = model.transform_to_posed_verts(varen_pose, device=device)
    posed_up_np = posed_up[0].detach().cpu().numpy()
    trimesh.Trimesh(vertices=posed_up_np, faces=faces_up, process=False).export(
        osp.join(out_dir, f"varen_subdivide{subdivide_num}_posed.obj")
    )
    np.save(osp.join(out_dir, f"varen_subdivide{subdivide_num}_posed.npy"), posed_up_np)


def debug_varen_transform(
    varen_param_json: str = "data/example_params.json",
    varen_model_path: str = "data/varen/",
    subdivide_num: int = 1,
    output_dir: str = "debug/varen_transform",
    sample_vertices: int = None,
    tail_scale: torch.Tensor = None,
):
    """Load VAREN params from JSON, run transforms, and save intermediates.

    Saves npz/obj files for: neutral mesh, posed mesh, per-vertex transform chains,
    and reports numerical consistency checks for transform composition.
    """
    import json

    device = avaliable_device()
    os.makedirs(output_dir, exist_ok=True)

    # Build model
    model: VARENSubdividedMeshModel = VARENSubdividedMeshModel(
        varen_model_path=varen_model_path,
        subdivide_num=subdivide_num,
        shape_param_dim=39,
        apply_pose_blendshape=False,
    ).to(device)

    # Load JSON params
    with open(varen_param_json, "r") as f:
        params = json.load(f)

    # Parse params -> tensors
    betas_list = params.get("betas", [])
    pose_list = params.get("pose", [])
    global_orient_list = params.get("global_orient", [0.0, 0.0, 0.0])
    trans_list = params.get("trans", [0.0, 0.0, 0.0])
    tail_scale_list = params.get("tail_scale", None)

    betas = torch.tensor(betas_list, dtype=torch.float32, device=device).view(1, -1)
    # Pad/truncate to model space
    shape_dim = model.varen.layer.SHAPE_SPACE_DIM
    if betas.shape[1] < shape_dim:
        pad = torch.zeros((1, shape_dim - betas.shape[1]), dtype=betas.dtype, device=device)
        betas = torch.cat([betas, pad], dim=1)
    elif betas.shape[1] > shape_dim:
        betas = betas[:, :shape_dim]

    pose = torch.tensor(pose_list, dtype=torch.float32, device=device)
    assert pose.numel() % 3 == 0, "Pose length must be divisible by 3"
    num_pose_rows = pose.numel() // 3
    pose = pose.view(1, num_pose_rows, 3)

    # VAREN expects body pose of size (joint_num-1, 3) and separate global_orient (1, 1, 3)
    joint_num = model.varen.joint_num  # includes root
    if num_pose_rows == joint_num:
        body_pose = pose[:, 1:, :]
    elif num_pose_rows == (joint_num - 1):
        body_pose = pose
    else:
        raise ValueError(f"Unexpected pose rows: {num_pose_rows}, expected {joint_num} or {joint_num-1}")

    global_orient = torch.tensor(global_orient_list, dtype=torch.float32, device=device).view(1, 1, 3)
    trans = torch.tensor(trans_list, dtype=torch.float32, device=device).view(1, 3)
    # Tail scale: prefer explicit arg, fallback to JSON if provided
    if tail_scale is not None:
        ts = tail_scale.to(device)
        if ts.dim() == 1:
            ts = ts.view(1, -1)
        assert ts.shape[-1] == 2, "tail_scale must have 2 values: [lengthening, fatness]"
        tail_scale_tensor = ts
    elif tail_scale_list is not None:
        tail_scale_tensor = torch.tensor(tail_scale_list, dtype=torch.float32, device=device).view(1, -1)
        assert tail_scale_tensor.shape[-1] == 2, "tail_scale must have 2 values: [lengthening, fatness]"
    else:
        tail_scale_tensor = None

    varen_data = {
        "betas": betas,
        "global_orient": global_orient,
        "pose": body_pose,
        "trans": trans,
    }
    if tail_scale_tensor is not None:
        varen_data["tail_scale"] = tail_scale_tensor

    # Get neutral query points and base transform transform_mat_neutral_pose: T-Pose to Template
    mesh_neutral_pose, mesh_neutral_pose_orig, transform_mat_neutral_pose = model.get_query_points(varen_data, device)

    # Full pipeline to get posed vertices and chain transform
    posed_mean_3d, neutral_to_posed_vertex = model.transform_to_posed_verts_from_neutral_pose(
        mesh_neutral_pose, varen_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    )

    # Recompute internal intermediates for checks
    with torch.no_grad():
        zeros = torch.zeros_like(trans)
        transform_mat_null_vertex = model.get_transform_mat_vertex(transform_mat_neutral_pose)   # T-Pose to Template
        null_mean_3d = model.lbs(mesh_neutral_pose, transform_mat_null_vertex, zeros)  

        # Shape offset and joint template
        blend_shape_offset = blend_shapes(betas, model.shapedirs_up)
        null_mean3d_blend = null_mean_3d + blend_shape_offset
        joint_template = model.get_template_pose(betas, device)

        # Pose transform and per-vertex skin transformation
        transform_mat_joint, _ = model.get_transform_mat_joint(None, joint_template, varen_data)  # Template to Posed
        transform_mat_vertex = model.get_transform_mat_vertex(transform_mat_joint)

        # Check pipeline matches posed_mean_3d
        posed_check = model.lbs(null_mean3d_blend, transform_mat_vertex, trans)

        # Optional vertex subsampling for lighter artifacts
        if sample_vertices is not None and sample_vertices > 0:
            idx = torch.randperm(mesh_neutral_pose.shape[1], device=device)[:sample_vertices]

            def sel(x):
                return x[:, idx] if x.dim() >= 3 and x.shape[1] == mesh_neutral_pose.shape[1] else x

            mesh_neutral_pose_s = sel(mesh_neutral_pose)
            posed_mean_3d_s = sel(posed_mean_3d)
            posed_check_s = sel(posed_check)
            null_mean_3d_s = sel(null_mean_3d)
            neutral_to_posed_vertex_s = neutral_to_posed_vertex[:, idx]
            transform_mat_null_vertex_s = transform_mat_null_vertex[:, idx]
            transform_mat_vertex_s = transform_mat_vertex[:, idx]
        else:
            mesh_neutral_pose_s = mesh_neutral_pose
            posed_mean_3d_s = posed_mean_3d
            posed_check_s = posed_check
            null_mean_3d_s = null_mean_3d
            neutral_to_posed_vertex_s = neutral_to_posed_vertex
            transform_mat_null_vertex_s = transform_mat_null_vertex
            transform_mat_vertex_s = transform_mat_vertex

        # Numeric diffs
        diff_pose = (posed_mean_3d_s - posed_check_s).norm(dim=-1)
        diff_pose_mean = float(diff_pose.mean().detach().cpu())
        diff_pose_max = float(diff_pose.max().detach().cpu())

        # Verify chain property with zero betas
        betas_zero = torch.zeros_like(betas)
        joint_template_zero = model.get_template_pose(betas_zero, device)
        transform_mat_joint_zero, _ = model.get_transform_mat_joint(None, joint_template_zero, varen_data)
        transform_mat_vertex_zero = model.get_transform_mat_vertex(transform_mat_joint_zero)
        M = torch.matmul(transform_mat_vertex_zero, transform_mat_null_vertex)
        y_chain = model.lbs(mesh_neutral_pose, M, None)
        y_true_wo_blend = model.lbs(null_mean_3d, transform_mat_vertex_zero, trans)
        diff_chain = (y_true_wo_blend - (y_chain + trans.unsqueeze(1))).norm(dim=-1)
        diff_chain_mean = float(diff_chain.mean().detach().cpu())
        diff_chain_max = float(diff_chain.max().detach().cpu())

    # Save numpy artifacts
    faces_up = model.face_upsampled.detach().cpu().numpy()
    trimesh.Trimesh(
        vertices=mesh_neutral_pose[0].detach().cpu().numpy(), faces=faces_up, process=False
    ).export(osp.join(output_dir, f"neutral_subdivide{subdivide_num}.obj"))
    trimesh.Trimesh(
        vertices=posed_mean_3d[0].detach().cpu().numpy(), faces=faces_up, process=False
    ).export(osp.join(output_dir, f"posed_subdivide{subdivide_num}.obj"))

    np.savez(
        osp.join(output_dir, "varen_transform_debug.npz"),
        mesh_neutral_pose=mesh_neutral_pose[0].detach().cpu().numpy(),
        transform_mat_neutral_pose=transform_mat_neutral_pose[0].detach().cpu().numpy(),
        transform_mat_null_vertex=transform_mat_null_vertex_s[0].detach().cpu().numpy(),
        joint_template=joint_template[0].detach().cpu().numpy(),
        transform_mat_joint=transform_mat_joint[0].detach().cpu().numpy(),
        transform_mat_vertex=transform_mat_vertex_s[0].detach().cpu().numpy(),
        neutral_to_posed_vertex=neutral_to_posed_vertex_s[0].detach().cpu().numpy(),
        posed_mean_3d=posed_mean_3d_s[0].detach().cpu().numpy(),
        posed_check=posed_check_s[0].detach().cpu().numpy(),
        diff_pose_mean=np.array(diff_pose_mean),
        diff_pose_max=np.array(diff_pose_max),
        diff_chain_mean=np.array(diff_chain_mean),
        diff_chain_max=np.array(diff_chain_max),
    )

    # Also write a small text report
    with open(osp.join(output_dir, "report.txt"), "w") as f:
        f.write(f"diff_pose_mean: {diff_pose_mean}\n")
        f.write(f"diff_pose_max: {diff_pose_max}\n")
        f.write(f"diff_chain_mean (betas=0): {diff_chain_mean}\n")
        f.write(f"diff_chain_max (betas=0): {diff_chain_max}\n")


if __name__ == "__main__":
    # generate_varen_mesh()
    debug_varen_transform(
        varen_param_json="data/example_params.json",
        varen_model_path="data/varen/",
        subdivide_num=1,
        output_dir="debug/varen_transform",
        sample_vertices=None
    )
