import os
import glob
import argparse
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import pickle
from tqdm import tqdm
import imageio.v2 as imageio

from amr.utils.track_bboxes import (
	track_bboxes,
	detect_bboxes_using_sam3,
	track_bboxes_using_sam3,
	load_data,
)
from amr.models import load_amr
from amr.models.predictor import VideoPredictor
from amr.utils.geometry import perspective_projection
from amr.utils.renderer import Renderer, cam_crop_to_full
from pytorch3d.transforms import matrix_to_axis_angle
from amr.models.pose_models import ViTPose

import warnings
warnings.filterwarnings("ignore")


class PostProcessPipeline:
	"""
	Unified pipeline for horse bbox tracking, VitPose++ keypoints, and AniMerVAREN outputs and Optimization.
	"""
	def __init__(
		self,
		device: Optional[str] = None,
		prompt: Union[str, List[str]] = "horse",
		tracker_backend: str = "sam3",
		detection_score_thresh: float = 0.1,
		checkpoint_path: Optional[str] = None,
	):
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.prompt = prompt
		self.tracker_backend = tracker_backend.lower()
		if self.tracker_backend not in {"samurai", "sam3"}:
			raise ValueError("tracker_backend must be one of ['samurai', 'sam3'].")
		self.detection_score_thresh = detection_score_thresh

		self._vitpose_model = None
		self._amr_model = None
		self._amr_cfg = None

		# Initialize AniMerVAREN model if checkpoint provided
		if checkpoint_path is not None:
			model, model_cfg = load_amr(checkpoint_path, "AniMerVAREN")
			model = model.to(self.device)
			model.eval()
			self._amr_model = model
			self._amr_cfg = model_cfg

	def _load_frames(self, video_path: str) -> List[np.ndarray]:
		if os.path.isdir(video_path):
			frames_path = (
				glob.glob(os.path.join(video_path, "*.jpg"))
				+ glob.glob(os.path.join(video_path, "*.jpeg"))
				+ glob.glob(os.path.join(video_path, "*.JPG"))
				+ glob.glob(os.path.join(video_path, "*.JPEG"))
				+ glob.glob(os.path.join(video_path, "*.png"))
				+ glob.glob(os.path.join(video_path, "*.PNG"))
			)
			frames_path.sort()
			frames = [cv2.imread(frame_path) for frame_path in frames_path]
		else:
			cap = cv2.VideoCapture(video_path)
			frames = []
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				frames.append(frame)
			cap.release()
		return frames

	def _detect_init_bbox(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Run SAM3 detection on the first frame of the provided video path.

		Returns:
			best_bbox: np.ndarray of shape (4,) in [x1, y1, x2, y2]
			video_frames: np.ndarray of stacked frames used for detection/tracking
		"""
		boxes, scores, frames = detect_bboxes_using_sam3(
			video_path, prompt=self.prompt, device=self.device
		)
		if boxes is None or len(boxes) == 0:
			raise ValueError(
				"No detections found by SAM3 on the first frame for prompt: "
				f"{self.prompt}"
			)

		scores = np.asarray(scores).reshape(-1)
		boxes = np.asarray(boxes)
		if boxes.ndim == 1:
			boxes = boxes.reshape(1, -1)

		valid_indices = np.where(scores >= self.detection_score_thresh)[0]
		if valid_indices.size == 0:
			valid_indices = np.arange(scores.shape[0])

		best_idx = int(valid_indices[np.argmax(scores[valid_indices])])
		best_bbox = boxes[best_idx].astype(np.float32)
		return best_bbox, frames

	def track_bounding_boxes(
		self,
		video_path: str,
		init_bbox: Optional[np.ndarray] = None,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Track the horse bounding box across the video.

		Returns:
			valid_idx: np.ndarray of frame indices considered valid by the tracker
			detected_bboxes: np.ndarray of shape (num_frames, 4) in [x1, y1, x2, y2]
		"""
		assert os.path.isdir(video_path) or video_path.endswith(".mp4")
		if self.tracker_backend == "sam3":
			return self._track_with_sam3(video_path, init_bbox)
		return self._track_with_samurai(video_path, init_bbox)

	def _track_with_sam3(
		self,
		video_path: str,
		init_bbox: Optional[np.ndarray] = None,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		assert os.path.isdir(video_path) or video_path.endswith(".mp4")
		if init_bbox is None:
			init_bbox, frames_np = self._detect_init_bbox(video_path)
		else:
			frames_np = load_data(video_path)

		if frames_np is None or len(frames_np) == 0:
			raise ValueError("No frames loaded from the given video path.")

		init_bbox = np.asarray(init_bbox, dtype=np.float32).reshape(1, 4)

		raw_valid_idx, raw_bboxes, raw_masks = track_bboxes_using_sam3(
			init_bbox,
			frames_np,
			device=self.device,
		)

		target_id = 0
		valid_idx_list = raw_valid_idx.get(target_id, [])
		bbox_list = raw_bboxes.get(target_id, [])
		mask_entries = raw_masks.get(target_id, [])

		num_frames = frames_np.shape[0]
		all_bboxes = np.zeros((num_frames, 4), dtype=np.float32)
		height, width = frames_np.shape[1:3]
		empty_mask = np.zeros((height, width), dtype=np.uint8)
		full_masks: List[np.ndarray] = [empty_mask.copy() for _ in range(num_frames)]

		for fid, bbox in zip(valid_idx_list, bbox_list):
			all_bboxes[int(fid)] = np.asarray(bbox, dtype=np.float32)
		for fid, mask in zip(valid_idx_list, mask_entries):
			full_masks[int(fid)] = np.asarray(mask, dtype=np.uint8)

		valid_idx_array = np.array(valid_idx_list, dtype=np.int32)
		mask_array = np.stack(full_masks, axis=0) if len(full_masks) > 0 else np.empty((0, height, width), dtype=np.uint8)

		return valid_idx_array, all_bboxes, mask_array

	def _track_with_samurai(
		self,
		video_path: str,
		init_bbox: Optional[np.ndarray] = None,
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		frames = self._load_frames(video_path)
		if len(frames) == 0:
			raise ValueError("No frames loaded from the given video path.")

		if init_bbox is None:
			init_bbox, _ = self._detect_init_bbox(video_path)

		init_bbox = np.asarray(init_bbox, dtype=np.float32)

		try:
			valid_idx, detected_bboxes, mask_list = track_bboxes(
				video_path, init_bbox, device=self.device
			)
		except Exception:
			height, width = frames[0].shape[:2]
			valid_idx, detected_bboxes, mask_list = track_bboxes(
				video_path,
				np.array([0, 0, width, height], dtype=np.float32),
				device=self.device,
			)

		mask_array = (
			np.stack(mask_list, axis=0).astype(np.uint8)
			if len(mask_list) > 0
			else np.empty((0,), dtype=np.uint8)
		)
		return np.array(valid_idx), np.array(detected_bboxes), mask_array

	def _ensure_vitpose(self) -> ViTPose:
		if self._vitpose_model is not None:
			return self._vitpose_model
		model = ViTPose(
			cfg_path="third-party/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py",
			device=self.device,
			return_pose_image=True
		)
		return model

	@staticmethod
	def _xyxy_to_coco(xyxy: np.ndarray) -> np.ndarray:
		bbox_coco = xyxy.astype(np.float32).copy()
		bbox_coco[2] = bbox_coco[2] - bbox_coco[0]
		bbox_coco[3] = bbox_coco[3] - bbox_coco[1]
		return bbox_coco

	def infer_vitposepp(
		self,
		video_path: str,
		valid_idx: np.ndarray,
		detected_bboxes: np.ndarray,
		return_pose_image: bool = False,
	):
		self._vitpose_model = self._ensure_vitpose()
		frames = self._load_frames(video_path)
		results: Dict[int, Dict[str, np.ndarray]] = {}
		pose_images = []
		with torch.no_grad():
			for frame_index in tqdm(valid_idx.tolist(), total=len(valid_idx), desc="Inference VitPose++", dynamic_ncols=True):
				bbox_xyxy = detected_bboxes[frame_index]
				if bbox_xyxy[2] <= 0 or bbox_xyxy[3] <= 0:
					continue
				bbox_coco = self._xyxy_to_coco(bbox_xyxy)

				frame_bgr = frames[frame_index]
				frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
				pose_results = self._vitpose_model(frame_rgb, bbox_coco)

				image_pose_result = pose_results[0]
				if len(image_pose_result) == 0:
					results[frame_index] = {
						"keypoints": np.zeros((0, 0, 2), dtype=np.float32),
						"scores": np.zeros((0, 0), dtype=np.float32),
					}
					continue

				if return_pose_image:
					pose_images.append(pose_results[1])
				# Convert to numpy with consistent shapes
				keypoints_list: List[np.ndarray] = []
				scores_list: List[np.ndarray] = []
				for pose_result in image_pose_result:
					kpts = pose_result["keypoints"][:, :2]
					scrs = pose_result["keypoints"][:, 2]
					if isinstance(kpts, torch.Tensor):
						kpts = kpts.detach().cpu().numpy()
					if isinstance(scrs, torch.Tensor):
						scrs = scrs.detach().cpu().numpy()
					keypoints_list.append(kpts.astype(np.float32))
					scores_list.append(scrs.astype(np.float32))

				keypoints_arr = (
					np.stack(keypoints_list, axis=0) if len(keypoints_list) > 0 else np.zeros((0, 0, 2), dtype=np.float32)
				)
				scores_arr = (
					np.stack(scores_list, axis=0) if len(scores_list) > 0 else np.zeros((0, 0), dtype=np.float32)
				)
				keypoints_with_score = np.concatenate([keypoints_arr, scores_arr[:, :, np.newaxis]], axis=-1)
				results[frame_index] = {"keypoints": keypoints_with_score}

		return results if not return_pose_image else (results, pose_images)

	def infer_animer_varen(
		self,
		video_path: str,
		valid_idx: np.ndarray,
		detected_bboxes: np.ndarray,
		mask_list: Optional[List[np.ndarray]] = None,
	) -> Dict[str, torch.Tensor]:
		"""Run AniMerVAREN temporal inference and return raw outputs from the predictor.

		Common keys include: 'pred_global_orient', 'pred_pose', 'pred_betas', 'pred_cam',
		and others as defined by the model's predictor.
		"""
		frames_bgr = self._load_frames(video_path)

		if self._amr_model is None or self._amr_cfg is None:
			raise ValueError("AniMerVAREN model is not initialized. Provide checkpoint_path in PostProcessPipeline constructor.")

		predictor = VideoPredictor(
			self._amr_model,
			self._amr_cfg,
			frames_bgr,
			valid_idx,
			detected_bboxes,
			device=self.device,
			mask_list=mask_list,
		)
		results = predictor.inference()
		return results

	def save_vitpose_results(self, vitpose_results: Dict[int, Dict[str, np.ndarray]], output_dir: str) -> str:
		"""Save VitPose++ results as a pickle file.

		Args:
			vitpose_results: Dict mapping frame_idx -> {"keypoints": (M, K, 3) np.ndarray}
			output_dir: Directory to save to.

		Returns:
			Path to the saved file.
		"""
		os.makedirs(output_dir, exist_ok=True)
		out_path = os.path.join(output_dir, "vitpose_results.pkl")
		with open(out_path, "wb") as f:
			pickle.dump(vitpose_results, f)
		return out_path

	def save_animer_outputs(self, animer_outputs: Dict[str, torch.Tensor], output_dir: str) -> str:
		"""Save AniMerVAREN outputs (moved to CPU) as a torch .pt file.

		Args:
			animer_outputs: Dict of torch tensors.
			output_dir: Directory to save to.

		Returns:
			Path to the saved file.
		"""
		os.makedirs(output_dir, exist_ok=True)
		cpu_outputs: Dict[str, torch.Tensor] = {}
		for k, v in animer_outputs.items():
			if torch.is_tensor(v):
				cpu_outputs[k] = v.detach().cpu()
			else:
				cpu_outputs[k] = v
		out_path = os.path.join(output_dir, "animer_outputs.pt")
		torch.save(cpu_outputs, out_path)
		return out_path

	def save_mask_list(self, mask_list: np.ndarray, output_dir: str) -> str:
		"""Save all masks into a single non-image file.

		Saves as a pickle containing a list of numpy arrays.
		"""
		os.makedirs(output_dir, exist_ok=True)
		# Convert to a Python list to avoid potential object dtype pitfalls
		masks_pylist: List[np.ndarray] = []
		for m in mask_list:
			masks_pylist.append(m)
		out_path = os.path.join(output_dir, "mask_list.pkl")
		with open(out_path, "wb") as f:
			pickle.dump(masks_pylist, f)
		return out_path

	def render_keypoints_video(
		self,
		video_path: str,
		vitpose_results: Dict[int, Dict[str, np.ndarray]],
		output_path: str,
		fps: int = 30,
		score_threshold: float = 0.3,
	) -> str:
		"""Draw VitPose keypoints on frames and save to a video.

		Args:
			video_path: Path to the video or directory of frames.
			vitpose_results: Mapping frame_idx -> {"keypoints": (M, K, 3) array [x, y, score]}.
			output_path: Path to the output .mp4 video.
			fps: Frames per second of the output video.
			score_threshold: Minimum keypoint score to draw.

		Returns:
			Path to the saved video.
		"""
		frames = self._load_frames(video_path)
		if len(frames) == 0:
			raise ValueError("No frames loaded from the given video path.")

		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
		height, width = frames[0].shape[:2]

		save_frames = []
		for frame_index in tqdm(range(len(frames)), desc="Rendering keypoints", dynamic_ncols=True):
			frame = frames[frame_index].copy()
			res = vitpose_results.get(frame_index)
			if res is not None and "keypoints" in res:
				kps = res["keypoints"]  # (M, K, 3)
				if isinstance(kps, torch.Tensor):
					kps = kps.detach().cpu().numpy()
				if kps.ndim == 3:
					for person_idx in range(kps.shape[0]):
						for joint_idx in range(kps.shape[1]):
							x, y, s = kps[person_idx, joint_idx]
							if float(s) < score_threshold:
								continue
							x_i, y_i = int(round(float(x))), int(round(float(y)))
							if 0 <= x_i < width and 0 <= y_i < height:
								cv2.circle(frame, (x_i, y_i), 5, (0, 0, 255), -1)

			# Convert BGR to RGB for imageio
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			save_frames.append(frame_rgb)
		imageio.mimsave(output_path, save_frames, fps=fps)
		return output_path

	def render_mesh_video(
		self,
		video_path: str,
		animer_outputs: Dict[str, torch.Tensor],
		valid_idx: np.ndarray,
		output_path: str,
		fps: int = 30,
		full_frame: bool = True,
	) -> str:
		"""Render AniMerVAREN mesh on frames and save to a video.

		Args:
			video_path: Path to the video or directory of frames.
			animer_outputs: Raw outputs from VideoPredictor.inference().
			valid_idx: Indices of frames corresponding to the predictions timeline.
			output_path: Path to the output .mp4 video.
			fps: Frames per second of the output video.
			full_frame: If True, overlay on original frames; otherwise, render on cropped model inputs.

		Returns:
			Path to the saved video.
		"""
		if self._amr_model is None or self._amr_cfg is None:
			raise ValueError("AniMerVAREN model is not initialized. Provide checkpoint_path in PostProcessPipeline constructor.")

		frames_bgr = self._load_frames(video_path)
		if len(frames_bgr) == 0:
			raise ValueError("No frames loaded from the given video path.")
		frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]

		device = self.device
		faces = torch.from_numpy(self._amr_model.smal.faces).to(device) if isinstance(self._amr_model.smal.faces, np.ndarray) else self._amr_model.smal.faces.to(device)
		renderer = Renderer(self._amr_cfg, faces=faces)

		pred_global_orient: torch.Tensor = animer_outputs["pred_global_orient"]
		pred_pose: torch.Tensor = animer_outputs["pred_pose"]
		pred_betas: torch.Tensor = animer_outputs["pred_betas"]
		pred_cam: torch.Tensor = animer_outputs["pred_cam"]

		sequence_length = pred_global_orient.shape[0]
		pred_betas_filled = pred_betas.mean(dim=0, keepdim=True).repeat(sequence_length, 1)

		pred_smal_params = dict(
			global_orient=matrix_to_axis_angle(pred_global_orient).view(sequence_length, -1),
			pose=matrix_to_axis_angle(pred_pose).view(sequence_length, -1),
			betas=pred_betas_filled,
		)
		smal_output = self._amr_model.smal(**pred_smal_params)
		vertices = smal_output.vertices.detach().cpu().numpy()

		LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

		os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

		save_frames = []
		if full_frame:
			height, width = frames_rgb[0].shape[:2]
			img_size = torch.tensor([[frames_rgb[0].shape[1], frames_rgb[0].shape[0]]], device=device)
			scaled_focal_length = self._amr_cfg.EXTRA.FOCAL_LENGTH / self._amr_cfg.MODEL.IMAGE_SIZE * img_size.max()
			box_center: torch.Tensor = animer_outputs["box_center"]
			bbox_size: torch.Tensor = animer_outputs["bbox_size"]
			pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, bbox_size, img_size, scaled_focal_length)
			pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

			# Ensure alignment between predictions and valid indices
			limit = min(len(vertices), len(valid_idx))
			for i in tqdm(range(limit), desc="Rendering mesh", dynamic_ncols=True):
				fid = int(valid_idx[i])
				cam_view = renderer.render_rgba_multiple(
					[vertices[i]],
					cam_t=[pred_cam_t_full[i]],
					render_res=img_size[0].detach().cpu().numpy(),
					mesh_base_color=LIGHT_BLUE,
					scene_bg_color=(0, 0, 0),
					focal_length=scaled_focal_length,
				)
				input_img = frames_rgb[fid].astype(np.float32) / 255.0
				alpha = cam_view[:, :, 3:]  
				overlay_rgb = input_img[:, :, :3] * (1 - alpha) + cam_view[:, :, :3] * alpha
				save_frames.append((overlay_rgb * 255).astype(np.uint8))
		else:
			# Render on model's cropped input images
			render_size = self._amr_cfg.MODEL.IMAGE_SIZE

			limit = min(len(vertices), animer_outputs.get("pred_cam_t", pred_cam).shape[0])
			for i in tqdm(range(limit), desc="Rendering mesh", dynamic_ncols=True):
				input_img = animer_outputs["img"][i]
				cam_t = animer_outputs.get("pred_cam_t", pred_cam)[i].detach().cpu().numpy()
				img_overlay = renderer(
					vertices[i],
					cam_t,
					input_img,
					mesh_base_color=LIGHT_BLUE,
					scene_bg_color=(0, 0, 0),
				)
				save_frames.append(img_overlay)
		imageio.mimsave(output_path, save_frames, fps=fps)
		return output_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--video_path", type=str, required=True, help="Path to the video or directory of frames")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to AniMerVAREN checkpoint")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save intermediate and final results")
	parser.add_argument("--prompt", type=str, default="horse", help="Text prompt for SAM3 detection")
	parser.add_argument(
		"--tracker_backend",
		type=str,
		default="sam3",
		choices=["samurai", "sam3"],
		help="Select tracker backend for bbox/mask propagation",
	)
	args = parser.parse_args()

	pipeline = PostProcessPipeline(
		device="cuda" if torch.cuda.is_available() else "cpu",
		checkpoint_path=args.checkpoint,
		prompt=args.prompt,
		tracker_backend=args.tracker_backend,
	)

	# 1) Track bounding boxes
	valid_idx, detected_bboxes, mask_list = pipeline.track_bounding_boxes(args.video_path)
	mask_path = pipeline.save_mask_list(mask_list, args.output_dir)
	imageio.mimsave(os.path.join(args.output_dir, "mask.mp4"), mask_list, fps=30)

	# 2) Inference VitPose++ (keypoints only, no drawing) and save
	vitpose_results, pose_images = pipeline.infer_vitposepp(args.video_path, valid_idx, detected_bboxes, return_pose_image=True)
	vitpose_path = pipeline.save_vitpose_results(vitpose_results, args.output_dir)
	imageio.mimsave(os.path.join(args.output_dir, "pose_images.mp4"), pose_images, fps=30)

	# 3) Inference AniMerVAREN (temporal model) and save
	animer_outputs = pipeline.infer_animer_varen(args.video_path, valid_idx, detected_bboxes, mask_list)
	animer_path = pipeline.save_animer_outputs(animer_outputs, args.output_dir)

	# pipeline.render_keypoints_video(args.video_path, vitpose_results, os.path.join(args.output_dir, "keypoints.mp4"))
	pipeline.render_mesh_video(args.video_path, animer_outputs, valid_idx, os.path.join(args.output_dir, "mesh.mp4"))

	print(f"Tracked frames: {len(valid_idx)}")
	print(f"VitPose frames returned: {len(vitpose_results)}")
	print(f"Saved VitPose to: {vitpose_path}")
	print(f"AniMerVAREN outputs: {list(animer_outputs.keys())}")
	print(f"Saved AniMerVAREN outputs to: {animer_path}")
	print(f"Saved mask list to: {mask_path}")
