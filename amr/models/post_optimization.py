import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
import cv2
import pickle
import imageio.v2 as imageio
from tqdm import tqdm
from PIL import Image
from amr.models.predictor import VideoPredictor
from amr.models.varen import VAREN
from amr.configs import get_config
from amr.utils.mesh_renderer import SilhouetteRenderer
from amr.utils.geometry import perspective_projection
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
from amr.utils.renderer import Renderer
from amr.datasets.utils import trans_point2d
from amr.models.losses import (leg_sideway_error, leg_torsion_error, tail_sideway_error, 
                                tail_torsion_error, spine_sideway_error, spine_torsion_error)


class PostProcessRefiner(nn.Module):
	"""Refiner that initializes optimizable parameters from AniMerVAREN outputs
	and provides utilities to derive vertices, 2D keypoints, and silhouettes.

	Optimizable parameters (per-frame):
	- global_orient (axis-angle)
	- pose (axis-angle)
	- betas (shared init as mean over frames, then per-frame copy)
	- cam (weak-perspective params: s, tx, ty)
	"""
	def __init__(
		self,
		cfg,
		animer_outputs: Dict[str, torch.Tensor],
		device: Optional[str] = None,
	):
		super().__init__()
		self.varen_keypoint_idx = [2, 3, 4, 5, 6, 13, 11, 9, 14, 12, 10, 19, 17, 15, 20, 18, 16]
		self.vitpose_keypoint_idx = list(range(0, 17))
		keypoint_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, -1, 1)
		# self.varen_keypoint_idx = [4, 2, 3, 5, 13, 14, 11, 12, 9, 10, 19, 20, 17, 18, 15, 16]
		# self.vitpose_keypoint_idx = list(range(0, 3)) + list(range(4, 17))
		self.register_buffer("keypoint_weight", keypoint_weight)
		self.cfg = cfg
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.varen = VAREN(cfg.SMAL.MODEL_PATH, use_muscle_deformations=False).to(self.device)
		self.to(self.device)
		# Keep original AniMer outputs for optional on-the-fly rendering
		self.animer_outputs = animer_outputs

		with torch.no_grad():
			pred_global_orient = animer_outputs['pred_global_orient']  # rotation matrices
			pred_pose = animer_outputs['pred_pose']  # rotation matrices
			pred_betas = animer_outputs['pred_betas']  # [N, B]
			pred_cam = animer_outputs['pred_cam']  # [N, 3]

		N = pred_global_orient.shape[0]
		# Convert rotmats to axis-angle for optimization
		global_orient_6d = matrix_to_rotation_6d(pred_global_orient.reshape(N, -1, 3, 3)).view(N, -1)
		pose_6d = matrix_to_rotation_6d(pred_pose.reshape(N, -1, 3, 3)).view(N, -1)
		# Mean betas across time, then per-frame copy
		betas_mean = pred_betas.mean(dim=0, keepdim=True).repeat(N, 1)

		# Register as parameters to be optimizable
		self.param_global_orient = nn.Parameter(global_orient_6d.detach().clone().to(self.device))  # [N, 6]
		self.param_pose = nn.Parameter(pose_6d.detach().clone().to(self.device))  # [N, 6*37]
		self.param_betas = nn.Parameter(betas_mean.detach().clone().to(self.device))  # [N, 39]
		self.param_cam = nn.Parameter(pred_cam.detach().clone().to(self.device))  # [N, 3]
		# Trainable tail scale [1, 2] -> expand to [N, 2] when used in forward
		self.param_tail_scale = nn.Parameter(torch.zeros(N, 2, device=self.device))

		# Initial parameters
		self.init_global_orient = global_orient_6d.detach().clone().to(self.device).requires_grad_(False)
		self.init_pose = pose_6d.detach().clone().to(self.device).requires_grad_(False)
		self.init_betas = betas_mean.detach().clone().to(self.device).requires_grad_(False)
		self.init_cam = pred_cam.detach().clone().to(self.device).requires_grad_(False)

		# Cache faces tensor for rendering
		faces_np = self.varen.faces
		faces = faces_np if isinstance(faces_np, torch.Tensor) else torch.from_numpy(faces_np)
		self.register_buffer("faces", faces.to(self.device, dtype=torch.int64))

	def get_optimizable_parameters(self) -> List[nn.Parameter]:
		params: List[nn.Parameter] = []
		params.extend([self.param_pose, self.param_betas, self.param_cam, self.param_tail_scale, self.param_global_orient])
		return params

	def _derive_vertices_keypoints_silhouette(
		self,
		image_size: int = 256,
		chunk_size: int = 32,
		silhouette_renderer: Optional[SilhouetteRenderer] = None,
	):
		"""Shared forward that returns vertices, projected 2D keypoints, and silhouettes.

		Keeps gradients. For no-grad evaluation, call within a no-grad context or
		use `compute_vertices_keypoints_silhouette`.
		"""
		N = self.param_global_orient.shape[0]
		# SMAL forward
		smal_output = self.varen(
			global_orient=rotation_6d_to_matrix(self.param_global_orient.view(N, -1, 6)),
			pose=rotation_6d_to_matrix(self.param_pose.view(N, -1, 6)),
			betas=self.param_betas,
			tail_scale=self.param_tail_scale,
			pose2rot=False,
		)
		vertices = smal_output.vertices  # [N, V, 3]
		keypoints_3d = smal_output.surface_keypoints  # [N, K, 3]

		# Camera translation from weak-perspective cam params
		focal_default = float(getattr(self.cfg.EXTRA, 'FOCAL_LENGTH', 5000))
		focal_px = torch.tensor([focal_default, focal_default], device=self.device, dtype=vertices.dtype).unsqueeze(0).repeat(N, 1)
		pred_cam_t = torch.stack([
			self.param_cam[:, 1],
			self.param_cam[:, 2],
			2 * focal_px[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * self.param_cam[:, 0] + 1e-9),
		], dim=-1)

		# 2D projection (normalized focal) -> coords in [-0.5, 0.5]
		focal_norm = focal_px / float(self.cfg.MODEL.IMAGE_SIZE)
		keypoints_2d = perspective_projection(keypoints_3d, translation=pred_cam_t, focal_length=focal_norm)

		# Silhouette rendering in chunks
		if silhouette_renderer is None:
			silhouette_renderer = SilhouetteRenderer(size=image_size, focal=focal_default, device=self.device)
		vertices_cam = vertices + pred_cam_t.unsqueeze(1)
		silhouettes_chunks: List[torch.Tensor] = []
		for start in range(0, N, chunk_size):
			end = min(N, start + chunk_size)
			verts_chunk = vertices_cam[start:end]
			faces_chunk = self.faces.unsqueeze(0).repeat(verts_chunk.shape[0], 1, 1)
			sil_chunk = silhouette_renderer(verts_chunk, faces_chunk)
			silhouettes_chunks.append(sil_chunk)
		silhouettes = torch.cat(silhouettes_chunks, dim=0)
		# Ensure silhouettes are in [0, 1]
		silhouettes = torch.clamp(silhouettes, 0.0, 1.0)

		return {
			'vertices': vertices,
			'keypoints_2d': keypoints_2d,
			'silhouettes': silhouettes,
		}

	@torch.no_grad()
	def compute_vertices_keypoints_silhouette(
		self,
		image_size: int = 256,
		chunk_size: int = 32,
	):
		"""Compute vertices, 2D keypoints, and silhouettes with current parameters.

		Returns dict with keys:
		- 'vertices': [N, V, 3]
		- 'keypoints_2d': [N, K, 2] (normalized)
		- 'silhouettes': [N, image_size, image_size]
		"""
		return self._derive_vertices_keypoints_silhouette(image_size=image_size, chunk_size=chunk_size)

	def optimize_parameters(
		self,
		keypoints_results: Dict[int, Dict[str, np.ndarray]],
		mask_list: np.ndarray,
		valid_idx: np.ndarray,
		image_size: int = 256,
		num_iters: int = 200,
		lr: float = 5e-3,
		w_silhouette: float = 10.0,
		w_keypoints: float = 1.0,
		w_temporal: float = 5.0,
		w_reg: float = 0.1,
		w_pose_prior: float = 0.0,
		chunk_size: int = 32,
		render_every: Optional[int] = None,
		output_dir: Optional[str] = None,
		params_per_stage: Optional[List[List[str]]] = None,
		use_adaptive_lr: bool = True,
		lr_scheduler_patience: int = 20,
		lr_scheduler_factor: float = 0.5,
	) -> Dict[str, torch.Tensor]:
		"""Optimize SMAL, shape, and camera parameters using silhouette and 2D keypoint losses.

		Args:
			keypoints_results: Dict mapping frame_idx -> {"keypoints": (M, K, 3) array [x, y, score]}
			mask_list: Array/List of binary masks per frame. Will be resized to (image_size, image_size) and normalized to [0, 1].
			valid_idx: Indices of frames that correspond to the temporal model outputs (length N).
			image_size: Silhouette render size.
			num_iters: Optimization iterations (can be list for multi-stage).
			lr: Learning rate for Adam (can be list for multi-stage).
			w_silhouette: Weight for silhouette loss (can be list for multi-stage).
			w_keypoints: Weight for 2D keypoint loss (can be list for multi-stage).
			w_temporal: Weight for temporal smoothness loss (can be list for multi-stage).
			w_reg: Weight for regularization loss (can be list for multi-stage).
			w_pose_prior: Weight for pose prior losses (can be list for multi-stage).
			chunk_size: Chunk size for silhouette rendering.
			render_every: If set to an integer N, render video every N steps; if None, disable.
			output_dir: Directory to save rendered videos when render_every is set.
			params_per_stage: Optional list of parameter groups to optimize per stage.
			use_adaptive_lr: Whether to use adaptive learning rate scheduling.
			lr_scheduler_patience: Number of iterations with no improvement before reducing LR.
			lr_scheduler_factor: Factor by which to reduce learning rate.

		Returns:
			Dict containing final derived outputs after optimization: 'vertices', 'keypoints_2d', 'silhouettes'.
		"""
		device = self.device
		N = self.param_global_orient.shape[0]

		# Prepare GT silhouettes [N, H, W] in [0, 1]
		gt_silhouettes_np: List[np.ndarray] = []
		for i in range(N):
			frame_id = int(valid_idx[i])
			mask_item = mask_list[frame_id]
			if mask_item is None:
				mask = np.zeros((image_size, image_size), dtype=np.float32)
			else:
				if isinstance(mask_item, torch.Tensor):
					mask = mask_item.detach().cpu().numpy()
				elif isinstance(mask_item, np.ndarray):
					mask = mask_item
				else:
					mask = np.array(mask_item)
			if mask.ndim == 3:
				mask = mask[..., 0]
			if mask.dtype != np.float32:
				mask = mask.astype(np.float32)
			# Normalize to [0, 1]
			max_val = float(mask.max()) if mask.size > 0 else 0.0
			if max_val > 1.0:
				mask = mask / 255.0
			mask = np.clip(mask, 0.0, 1.0)
			gt_silhouettes_np.append(mask)
		gt_silhouettes = torch.from_numpy(np.stack(gt_silhouettes_np, axis=0)).to(device=device, dtype=torch.float32)

		# Prepare GT keypoints [N, K, 3] with coords in [-0.5, 0.5]
		# Determine K from first available detection
		K = 0
		for i in range(N):
			frame_id = int(valid_idx[i])
			if frame_id in keypoints_results:
				k = keypoints_results[frame_id]
				if isinstance(k, np.ndarray) and k.size > 0:
					# shape: (M, K, 3)
					K = int(k.shape[1])
					break
		if K == 0:
			w_keypoints = 0.0
			gt_keypoints = torch.zeros((N, 1, 3), device=device, dtype=torch.float32)  # dummy
		else:
			gt_kpts_list: List[np.ndarray] = []
			model_img_size = float(self.cfg.MODEL.IMAGE_SIZE)
			for i in range(N):
				frame_id = int(valid_idx[i])
				k_select = None
				if frame_id in keypoints_results:
					kdata = keypoints_results[frame_id]
					if isinstance(kdata, np.ndarray) and kdata.ndim == 3 and kdata.shape[2] >= 3 and kdata.shape[1] == K and kdata.shape[0] > 0:
						# Choose instance with highest mean score
						scores = kdata[..., 2]
						best_idx = int(np.argmax(scores.mean(axis=1)))
						k_select = kdata[best_idx]
				if k_select is None:
					k_select = np.zeros((K, 3), dtype=np.float32)
				k_xy = k_select[:, :2].astype(np.float32)
				k_conf = k_select[:, 2:3].astype(np.float32)
				# Normalize to [-0.5, 0.5] using model image size
				k_xy_norm = (k_xy / model_img_size) - 0.5
				k_all = np.concatenate([k_xy_norm, k_conf], axis=-1)
				gt_kpts_list.append(k_all)
			gt_keypoints = torch.from_numpy(np.stack(gt_kpts_list, axis=0)).to(device=device, dtype=torch.float32)

		gt_keypoints = gt_keypoints[:, self.vitpose_keypoint_idx, :]

		# Pre-create renderer (outside loop) for efficiency
		silhouette_renderer = SilhouetteRenderer(size=image_size, focal=float(getattr(self.cfg.EXTRA, 'FOCAL_LENGTH', 5000)), device=device)
		global_step: int = 0

		# Multi-stage support: allow lists for iterations, loss weights, and parameter groups
		if isinstance(num_iters, (list, tuple, np.ndarray)):
			stage_iters = [int(x) for x in num_iters]
		else:
			stage_iters = [int(num_iters)]
		def _as_list(x):
			return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]
		w_s_list = [float(x) for x in _as_list(w_silhouette)]
		w_k_list = [float(x) for x in _as_list(w_keypoints)]
		w_t_list = [float(x) for x in _as_list(w_temporal)]
		w_r_list = [float(x) for x in _as_list(w_reg)]
		w_p_list = [float(x) for x in _as_list(w_pose_prior)]
		S = len(stage_iters)
		def _broadcast(lst):
			if len(lst) == 1:
				return lst * S
			if len(lst) != S:
				raise ValueError(f"Inconsistent lengths: num_iters has {S} stages but a weight list has {len(lst)}")
			return lst
		w_s_list = _broadcast(w_s_list)
		w_k_list = _broadcast(w_k_list)
		w_t_list = _broadcast(w_t_list)
		w_r_list = _broadcast(w_r_list)
		w_p_list = _broadcast(w_p_list)
		lr_list = _broadcast(lr)
		# Broadcast parameter groups per stage if provided
		param_name_to_param = {
			"pose": self.param_pose,
			"betas": self.param_betas,
			"global_orient": self.param_global_orient,
			"cam": self.param_cam,
			"tail_scale": self.param_tail_scale,
		}
		if params_per_stage is None:
			params_spec_list: Optional[List[List[str]]]= None
		else:
			if not isinstance(params_per_stage, (list, tuple)) or any(not isinstance(x, (list, tuple)) for x in params_per_stage):
				raise ValueError("params_per_stage must be a list of lists of parameter names")
			params_spec_list = [list(map(str, spec)) for spec in params_per_stage]
			if len(params_spec_list) == 1:
				params_spec_list = params_spec_list * S
			elif len(params_spec_list) != S:
				raise ValueError(f"Inconsistent lengths: num_iters has {S} stages but params_per_stage has {len(params_spec_list)}")

		loss_history: List[float] = []
		loss_components: Dict[str, List[float]] = {}

		for stage_idx, (it_count, ws, wk, wt, wr, wp, stage_lr) in enumerate(zip(stage_iters, w_s_list, w_k_list, w_t_list, w_r_list, w_p_list, lr_list)):
			# Build optimizer over only selected parameters for this stage
			if params_per_stage is None:
				stage_params = self.get_optimizable_parameters()
			else:
				spec = params_spec_list[stage_idx]
				stage_params_raw: List[nn.Parameter] = []
				for name in spec:
					if name not in param_name_to_param:
						raise ValueError(f"Unknown parameter name in params_per_stage: {name}")
					stage_params_raw.append(param_name_to_param[name])
				# Deduplicate while preserving order
				seen: set = set()
				stage_params: List[nn.Parameter] = []
				for p in stage_params_raw:
					pid = id(p)
					if pid in seen:
						continue
					seen.add(pid)
					stage_params.append(p)
			optimizer = torch.optim.Adam(stage_params, lr=stage_lr, fused=True)
			
			# Add learning rate scheduler for adaptive LR
			scheduler = None
			if use_adaptive_lr:
				scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
					optimizer, 
					mode='min', 
					factor=lr_scheduler_factor, 
					patience=lr_scheduler_patience,
					verbose=False
				)
			
			pbar = tqdm(range(int(it_count)), total=int(it_count), desc=f"Stage {stage_idx+1}/{S}", dynamic_ncols=True)
			for it in pbar:
				optimizer.zero_grad(set_to_none=True)

				# Use shared forward for current parameters
				derived_iter = self._derive_vertices_keypoints_silhouette(
					image_size=image_size,
					chunk_size=chunk_size,
					silhouette_renderer=silhouette_renderer,
				)
				keypoints_2d = derived_iter['keypoints_2d'][:, self.varen_keypoint_idx, :]
				silhouettes_pred = derived_iter['silhouettes']

				# Losses
				loss_sil = F.l1_loss(silhouettes_pred, gt_silhouettes) if ws > 0.0 else torch.tensor(0.0, device=device)
				if wk > 0.0:
					# Pred already normalized in [-0.5, 0.5]; use confidence weighting in the loss
					loss_kpt = F.mse_loss(keypoints_2d, gt_keypoints[:, :, :-1], reduction='none')
					loss_kpt = (loss_kpt * gt_keypoints[:, :, -1:] * self.keypoint_weight).mean()
				else:
					loss_kpt = torch.tensor(0.0, device=device)

				# pred_xy = (keypoints_2d[0] + 0.5) * 255
				# gt_xy = (gt_keypoints[0, :, :2] + 0.5) * 255
				# pred_mask = (silhouettes_pred[0].detach().cpu().numpy() * 255).astype(np.uint8)[:, :, None].repeat(3, axis=-1)
				# gt_mask = (gt_silhouettes[0].detach().cpu().numpy() * 255).astype(np.uint8)[:, :, None].repeat(3, axis=-1)
				# loss_kpt = 0.0
				# for i, (pred_xy_, gt_xy_) in enumerate(zip(pred_xy, gt_xy)):
				# 	cv2.circle(pred_mask, (int(pred_xy_[0]), int(pred_xy_[1])), 5, (255, 0, 0), -1)
				# 	cv2.circle(gt_mask, (int(gt_xy_[0]), int(gt_xy_[1])), 5, (0, 0, 255), -1)
				# 	cv2.putText(pred_mask, str(i), (int(pred_xy_[0]), int(pred_xy_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
				# 	cv2.putText(gt_mask, str(i), (int(gt_xy_[0]), int(gt_xy_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

				# cv2.imwrite('pred_mask.png', pred_mask)
				# cv2.imwrite('gt_mask.png', gt_mask)
				# exit()

				# Optional rendering every N global steps
				if render_every is not None:
					try:
						re_int = int(render_every)
						if re_int > 0 and (global_step % re_int == 0) and global_step > 0:
							out_dir = output_dir or (self.cfg.OUTPUT_DIR if hasattr(self.cfg, 'OUTPUT_DIR') else '.')
							out_path = os.path.join(out_dir, f'refined_mesh_s{stage_idx+1}_step{global_step:06d}.mp4')
							self.render_refined_mesh(
								animer_outputs=self.animer_outputs,
								output_path=out_path,
								fps=30,
								image_size=image_size,
							)
					except Exception:
						pass
				
				v_global_orient = self.param_global_orient[:-1] - self.param_global_orient[1:]
				v_pose = self.param_pose[:-1] - self.param_pose[1:]
				loss_temporal = (F.mse_loss(self.param_global_orient[:-1], self.param_global_orient[1:]) +
								 F.mse_loss(self.param_pose[:-1], self.param_pose[1:]) +
								 F.mse_loss(v_global_orient[:-1], v_global_orient[1:]) +
								 F.mse_loss(v_pose[:-1], v_pose[1:])
								 )
				loss_reg = (F.mse_loss(self.param_global_orient, self.init_global_orient) +
							F.mse_loss(self.param_pose, self.init_pose) +
							F.mse_loss(self.param_betas, self.init_betas) +
							F.mse_loss(self.param_cam, self.init_cam)
							)

				# Pose prior losses (leg, tail, spine sideway and torsion)
				if wp > 0.0:
					pose_with_glob = torch.cat([
						rotation_6d_to_matrix(self.param_global_orient.view(N, 1, 6)),
						rotation_6d_to_matrix(self.param_pose.view(N, -1, 6))
					], dim=1)  # [N, 38, 3, 3]
					loss_pose_prior = (leg_sideway_error(pose_with_glob) + leg_torsion_error(pose_with_glob) +
									   tail_sideway_error(pose_with_glob) + tail_torsion_error(pose_with_glob) +
									   spine_sideway_error(pose_with_glob) + spine_torsion_error(pose_with_glob))
				else:
					loss_pose_prior = torch.tensor(0.0, device=device)

				loss_total = (ws * loss_sil + 
							  wk * loss_kpt + 
							  wt * loss_temporal + 
							  wr * loss_reg + 
							  wp * loss_pose_prior)
				loss_total.backward()
				optimizer.step()
				
				# Update learning rate scheduler if enabled
				if scheduler is not None:
					scheduler.step(loss_total.detach().item())

				# Update progress bar with current loss
				try:
					postfix_dict = {
						"loss_total": float(loss_total.detach().item()),
						"loss_sil": float(loss_sil.detach().item()),
						"loss_kpt": float(loss_kpt.detach().item()),
						"loss_temporal": float(loss_temporal.detach().item()),
						"loss_reg": float(loss_reg.detach().item()),
						"loss_prior": float(loss_pose_prior.detach().item()),
					}
					if use_adaptive_lr and scheduler is not None:
						postfix_dict["lr"] = optimizer.param_groups[0]['lr']
					pbar.set_postfix(postfix_dict, refresh=False)
					
					loss_history.append(float(loss_total.detach().item()))
					loss_components["silhouette"] = loss_components.get("silhouette", []) + [float(loss_sil.detach().item())]
					loss_components["keypoints"] = loss_components.get("keypoints", []) + [float(loss_kpt.detach().item())]
					loss_components["temporal"] = loss_components.get("temporal", []) + [float(loss_temporal.detach().item())]
					loss_components["regularization"] = loss_components.get("regularization", []) + [float(loss_reg.detach().item())]
					loss_components["pose_prior"] = loss_components.get("pose_prior", []) + [float(loss_pose_prior.detach().item())]
				except Exception:
					pass

				global_step += 1

		# Compute final derived outputs with current parameters
		with torch.no_grad():
			derived = self.compute_vertices_keypoints_silhouette(image_size=image_size, chunk_size=chunk_size)
		
		# Add loss history and components to the returned dictionary
		derived['loss_history'] = loss_history
		derived['loss_components'] = loss_components
		
		return derived

	@torch.no_grad()
	def save_refined_results(
		self,
		output_dir: str,
		derived: Optional[Dict[str, torch.Tensor]] = None,
		image_size: int = 256,
		chunk_size: int = 32,
	) -> str:
		"""Save refined parameters and optional derived outputs to disk.

		Args:
			output_dir: Directory to save to.
			derived: Optional precomputed outputs from `compute_vertices_keypoints_silhouette`.
			image_size: If `derived` is None, render size used to compute derived outputs.
			chunk_size: Chunk size for silhouette rendering when deriving outputs.

		Returns:
			Path to the saved .pt file containing refined params and derived outputs.
		"""
		os.makedirs(output_dir, exist_ok=True)
		if derived is None:
			derived = self.compute_vertices_keypoints_silhouette(image_size=image_size, chunk_size=chunk_size)

		# Camera translation computed from refined weak-perspective params
		N = self.param_cam.shape[0]
		focal_default = float(getattr(self.cfg.EXTRA, 'FOCAL_LENGTH', 5000))
		focal_px = torch.tensor([focal_default, focal_default], device=self.device, dtype=self.param_cam.dtype).unsqueeze(0).repeat(N, 1)
		refined_cam_t = torch.stack([
			self.param_cam[:, 1],
			self.param_cam[:, 2],
			2 * focal_px[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * self.param_cam[:, 0] + 1e-9),
		], dim=-1)

		to_save: Dict[str, torch.Tensor] = {
			'refined_global_orient': self.param_global_orient.detach().cpu(),
			'refined_pose': self.param_pose.detach().cpu(),
			'refined_betas': self.param_betas.detach().cpu(),
			'refined_cam': self.param_cam.detach().cpu(),
			'refined_cam_t': refined_cam_t.detach().cpu(),
			'refined_tail_scale': self.param_tail_scale.detach().cpu(),
		}
		# Attach derived outputs (already detached/no-grad)
		for k in ['vertices', 'keypoints_2d', 'silhouettes']:
			if k in derived and isinstance(derived[k], torch.Tensor):
				to_save[k] = derived[k].detach().cpu()

		out_path = os.path.join(output_dir, 'refined_results.pt')
		torch.save(to_save, out_path)
		return out_path

	def plot_total_loss_curve(self, loss_history: List[float], output_path: str, title: str = "Total Loss Curve"):
		"""Plot and save the total loss curve.
		
		Args:
			loss_history: List of total loss values per iteration.
			output_path: Path to save the loss curve image.
			title: Title for the plot.
		"""
		import matplotlib.pyplot as plt
		
		plt.figure(figsize=(10, 6))
		plt.plot(loss_history, 'b-', linewidth=2, alpha=0.8)
		plt.xlabel('Iteration')
		plt.ylabel('Total Loss')
		plt.title(title)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		
		# Ensure output directory exists
		os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Saved total loss curve to: {output_path}")

	def plot_loss_components_curves(self, loss_components: Dict[str, List[float]], output_dir: str):
		"""Plot and save individual loss component curves.
		
		Args:
			loss_components: Dictionary mapping loss component names to lists of values per iteration.
			output_dir: Directory to save the loss component curve images.
		"""
		import matplotlib.pyplot as plt
		
		os.makedirs(output_dir, exist_ok=True)
		
		# Plot each loss component separately
		for component_name, loss_values in loss_components.items():
			if not loss_values:  # Skip empty lists
				continue
				
			plt.figure(figsize=(10, 6))
			plt.plot(loss_values, 'r-', linewidth=2, alpha=0.8)
			plt.xlabel('Iteration')
			plt.ylabel(f'{component_name} Loss')
			plt.title(f'{component_name} Loss Curve')
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			
			output_path = os.path.join(output_dir, f'{component_name.lower().replace(" ", "_")}_loss_curve.png')
			plt.savefig(output_path, dpi=300, bbox_inches='tight')
			plt.close()
			print(f"Saved {component_name} loss curve to: {output_path}")
		
		# Plot all components together for comparison
		plt.figure(figsize=(12, 8))
		for component_name, loss_values in loss_components.items():
			if loss_values:  # Only plot non-empty lists
				plt.plot(loss_values, label=component_name, linewidth=2, alpha=0.8)
		
		plt.xlabel('Iteration')
		plt.ylabel('Loss Value')
		plt.title('All Loss Components Comparison')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		
		comparison_path = os.path.join(output_dir, 'all_loss_components_comparison.png')
		plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Saved loss components comparison to: {comparison_path}")

	@torch.no_grad()
	def render_refined_mesh(
		self,
		animer_outputs: Dict[str, torch.Tensor],
		output_path: str,
		fps: int = 30,
		image_size: Optional[int] = None,
	) -> str:
		"""Render refined mesh over the model's cropped input images and save to a video.

		Args:
			animer_outputs: Dict returned by `VideoPredictor.inference()` containing 'img'.
			output_path: Path to output video (.mp4).
			fps: Frames per second for the output video.
			image_size: Render size; defaults to cfg.MODEL.IMAGE_SIZE.

		Returns:
			Path to the saved video.
		"""
		# Derive vertices and camera translation from refined parameters
		N = self.param_global_orient.shape[0]
		image_size = int(image_size or self.cfg.MODEL.IMAGE_SIZE)
		focal_default = float(getattr(self.cfg.EXTRA, 'FOCAL_LENGTH', 5000))
		focal_px = torch.tensor([focal_default, focal_default], device=self.device, dtype=self.param_cam.dtype).unsqueeze(0).repeat(N, 1)

		smal_output = self.varen(
			global_orient=rotation_6d_to_matrix(self.param_global_orient.view(N, -1, 6)),
			pose=rotation_6d_to_matrix(self.param_pose.view(N, -1, 6)),
			betas=self.param_betas.mean(dim=0, keepdim=True).expand(N, -1),
			tail_scale=self.param_tail_scale.mean(dim=0, keepdim=True).expand(N, -1),
			pose2rot=False,
		)
		vertices = smal_output.vertices  # [N, V, 3]
		pred_cam_t = torch.stack([
			self.param_cam[:, 1],
			self.param_cam[:, 2],
			2 * focal_px[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * self.param_cam[:, 0] + 1e-9),
		], dim=-1)

		# Setup renderer
		renderer = Renderer(self.cfg, faces=self.faces)

		# Prepare writer
		os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

		img_seq = animer_outputs.get('img', None)
		if isinstance(img_seq, torch.Tensor):
			limit = min(N, int(img_seq.shape[0]))
		elif isinstance(img_seq, (list, tuple)):
			limit = min(N, len(img_seq))
		else:
			limit = N

		save_frames = []
		for i in tqdm(range(limit), desc="Rendering refined mesh", dynamic_ncols=True):
			img_tensor: torch.Tensor = animer_outputs['img'][i]
			overlay_rgb = renderer(
				vertices[i].detach().cpu().numpy(),
				pred_cam_t[i].detach().cpu().numpy(),
				img_tensor,
				mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
				scene_bg_color=(1, 1, 1),
			)
			side_view = renderer(
				vertices[i].detach().cpu().numpy(),
				pred_cam_t[i].detach().cpu().numpy(),
				img_tensor,
				side_view=True,
				mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
				scene_bg_color=(1, 1, 1),
			)
			# os.makedirs(os.path.join(os.path.dirname(output_path), 'rgba_meshes'), exist_ok=True)
			# overlay_rgb = renderer.render_rgba(
			# 	vertices[i].detach().cpu().numpy(),
			# 	pred_cam_t[i].detach().cpu().numpy(),
			# 	mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
			# 	scene_bg_color=(1, 1, 1),
			# )
			# Image.fromarray((overlay_rgb * 255).astype(np.uint8)).save(os.path.join(os.path.dirname(output_path), 'rgba_meshes', f'{i:06d}.png'))
			save_frames.append(np.concatenate([overlay_rgb * 255, side_view * 255], axis=1).astype(np.uint8))
		imageio.mimsave(output_path, save_frames, fps=fps)
		return output_path

	@torch.no_grad()
	def save_vitpose_skeleton_rgba(
		self,
		keypoints_results: Dict[int, np.ndarray],
		output_dir: str,
		valid_idx: Optional[np.ndarray] = None,
		image_size: int = 256,
		confidence_threshold: float = 0.5,
	) -> List[str]:
		"""Save VitPose++ keypoints as skeleton RGBA PNGs with transparent background.
		
		Args:
			keypoints_results: Dict mapping frame_idx -> detections. Each value can be
				- np.ndarray of shape (M, K, 3) with [x, y, score] per instance, or
				- dict containing 'keypoints' with the same ndarray.
			output_dir: Directory to save images in.
			valid_idx: Optional array mapping model output index to original frame index. If None, uses 0..N-1.
			image_size: Output image size (assumed coordinate frame of keypoints).
			confidence_threshold: Minimum confidence to draw a point/segment.
		
		Returns:
			List of file paths for saved RGBA images.
		"""
		N = self.param_global_orient.shape[0]
		save_dir = os.path.join(output_dir, "rgba_skeleton")
		os.makedirs(save_dir, exist_ok=True)
		
		# Skeleton pairs as requested
		skeleton_pairs = [
			[0, 1], [0, 2], [0, 4], [3, 4], [4, 5], [4, 6], [5, 7], [7, 9],
			[6, 8], [8, 10], [3, 11], [3, 12], [11, 13], [13, 15], [12, 14], [14, 16],
		]
		
		# Color palettes similar to visualization.py
		import matplotlib.pyplot as plt
		try:
			skel_colors = np.round(np.array(plt.get_cmap('Set2').colors) * 255).astype(np.uint8)[:, ::-1].tolist()
		except AttributeError:
			skel_colors = np.round(np.array(plt.get_cmap('Set2')(np.linspace(0, 1, 8))) * 255).astype(np.uint8)[:, -2::-1].tolist()
		try:
			pt_colors = np.round(np.array(plt.get_cmap('tab20').colors) * 255).astype(np.uint8)[:, ::-1].tolist()
		except AttributeError:
			pt_colors = np.round(np.array(plt.get_cmap('tab20')(np.linspace(0, 1, 16))) * 255).astype(np.uint8)[:, -2::-1].tolist()
		
		out_paths: List[str] = []
		for i in tqdm(range(N), desc="Saving RGBA skeleton", dynamic_ncols=True):
			frame_id = int(valid_idx[i]) if valid_idx is not None else i
			kdata = keypoints_results.get(frame_id, None)
			
			k_select = None
			if isinstance(kdata, np.ndarray):
				if kdata.ndim == 3 and kdata.shape[2] >= 3 and kdata.shape[0] > 0:
					scores = kdata[..., 2]
					best_idx = int(np.argmax(scores.mean(axis=1)))
					k_select = kdata[best_idx]
			elif isinstance(kdata, dict):
				arr = kdata.get('keypoints', None)
				if isinstance(arr, list):
					arr = np.array(arr)
				if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] >= 3 and arr.shape[0] > 0:
					scores = arr[..., 2]
					best_idx = int(np.argmax(scores.mean(axis=1)))
					k_select = arr[best_idx]
			
			if k_select is None:
				K = 17
				k_select = np.zeros((K, 3), dtype=np.float32)
			
			K = k_select.shape[0]
			valid_pairs = [pair for pair in skeleton_pairs if pair[0] < K and pair[1] < K]
			
			# points as (y, x, conf)
			points = np.stack([k_select[:, 1], k_select[:, 0], k_select[:, 2]], axis=-1).astype(np.float32)
			
			# Draw on black background (BGR)
			canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
			
			# Draw skeleton: single color like visualization.py (per person)
			person_color = tuple(skel_colors[0])
			for joint in valid_pairs:
				pt1 = points[joint[0]]
				pt2 = points[joint[1]]
				if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
					cv2.line(
						canvas,
						(int(pt1[1]), int(pt1[0])),
						(int(pt2[1]), int(pt2[0])),
						person_color,
						2,
					)
			
			# Draw points
			circle_size = max(1, min(canvas.shape[:2]) // 150)
			for j, pt in enumerate(points):
				if pt[2] > confidence_threshold:
					cv2.circle(canvas, (int(pt[1]), int(pt[0])), circle_size, tuple(pt_colors[j % len(pt_colors)]), -1)
			
			# Convert to BGRA with transparent background outside drawings
			alpha = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
			alpha = (alpha > 0).astype(np.uint8) * 255
			bgra = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
			bgra[..., 3] = alpha
			
			out_path = os.path.join(save_dir, f"kpts_{i:06d}.png")
			cv2.imwrite(out_path, bgra)
			out_paths.append(out_path)
		
		return out_paths


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cfg", type=str, help="Path to AniMerVAREN Config",
						default="logs/train/runs/varen_temporal_head_new_dataset/.hydra/config.yaml")
	parser.add_argument("--animer_outputs", type=str, help="Path to AniMerVAREN outputs",
						default="demo_out/horse2/animer_outputs.pt")
	parser.add_argument("--vitpose_results", type=str, help="Path to VitPose++ results",
						default="demo_out/horse2/vitpose_results.pkl")
	parser.add_argument("--output_dir", type=str, default="outputs/horse2/", help="Directory to save intermediate and final results")
	parser.add_argument("--optimize_length", type=int, default=30, help="Number of frames to optimize")
	args = parser.parse_args()

	animer_outputs = torch.load(args.animer_outputs, weights_only=True)
	animer_outputs = {k: v[range(args.optimize_length)] for k, v in animer_outputs.items()}
	vitpose_results = pickle.load(open(args.vitpose_results, "rb"))
	vitpose_results = {k: v for k, v in vitpose_results.items() if k in range(args.optimize_length)}
	vitpose_results = {k: trans_point2d(v['keypoints'][0], animer_outputs['trans'][k])[None] for k, v in vitpose_results.items()}
	mask_list = animer_outputs['mask'][:args.optimize_length]
	valid_idx = np.arange(args.optimize_length)
	cfg = get_config(args.cfg)

	# 4) Derive vertices, 2D keypoints and silhouettes (256x256) with refiner
	refiner: PostProcessRefiner = PostProcessRefiner(cfg, animer_outputs)
	# derived = refiner.compute_vertices_keypoints_silhouette(image_size=256, chunk_size=32)
	derived_opt = refiner.optimize_parameters(
		keypoints_results=vitpose_results,  # dict: frame_idx -> {"keypoints": (M, K, 3)}
		mask_list=mask_list,                # list/array of per-frame binary masks
		valid_idx=valid_idx,                # indices aligned with model outputs
		image_size=256,                     # silhouette render size
		num_iters=[100, 100],          # optimization iterations
		lr=[5e-3, 5e-3],              # learning rate
		chunk_size=30,                      # batch rendering chunk size
		w_silhouette=[100.0, 10000.0],
		w_keypoints=[10000., 100.0],
		w_temporal=[100.0, 100.0],
		w_reg=[100.0, 300.0],
		w_pose_prior=[1000.0, 100.0],    # per-joint pose limits (head: 36°, legs: variable)
		params_per_stage=[
			["pose", "betas", "cam", "tail_scale", "global_orient"],
			["tail_scale", "betas"],
		],
		output_dir=args.output_dir,
		render_every=100,
		use_adaptive_lr=True,              # enable adaptive learning rate
		lr_scheduler_patience=20,           # reduce LR after 20 iters with no improvement
		lr_scheduler_factor=0.5,            # reduce LR by factor of 0.5
	)
	# 5) Save refined parameters and derived outputs
	refine_save_path = refiner.save_refined_results(
		output_dir=args.output_dir,
		derived=derived_opt,
		image_size=256,
		chunk_size=30,
	)
	print(f"Saved refined results to: {refine_save_path}")

	# 5.5) Plot and save loss curves
	if 'loss_history' in derived_opt and 'loss_components' in derived_opt:
		# Plot total loss curve
		total_loss_path = os.path.join(args.output_dir, 'total_loss_curve.png')
		refiner.plot_total_loss_curve(
			loss_history=derived_opt['loss_history'],
			output_path=total_loss_path,
			title="Total Loss Curve During Optimization"
		)
		
		# Plot individual loss component curves
		loss_curves_dir = os.path.join(args.output_dir, 'loss_curves')
		refiner.plot_loss_components_curves(
			loss_components=derived_opt['loss_components'],
			output_dir=loss_curves_dir
		)
		print(f"Saved loss curves to: {args.output_dir}")
	else:
		print("Warning: Loss history not found in optimization results")

	# 6) Render refined mesh video on cropped inputs
	refined_video_path = os.path.join(args.output_dir, 'refined_mesh.mp4')
	refined_video_path = refiner.render_refined_mesh(
		animer_outputs=animer_outputs,
		output_path=refined_video_path,
		fps=30,
		image_size=256,
	)
	print(f"Saved refined mesh video to: {refined_video_path}")
	# refiner.save_vitpose_skeleton_rgba(
	# 	keypoints_results=vitpose_results,
	# 	output_dir=args.output_dir,
	# 	valid_idx=valid_idx,
	# 	image_size=256,
	# 	confidence_threshold=0.5,
	# )
