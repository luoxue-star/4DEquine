import os
import numpy as np
import torch
import gc
from typing import List, Union
from transformers import Sam3Processor, Sam3Model, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from PIL import Image
import imageio.v2 as imageio
import sys
from pathlib import Path


def _build_sam2_video_predictor(*args, **kwargs):
    samurai_root = Path(__file__).resolve().parents[2] / "third-party" / "samurai"
    samurai_root_str = str(samurai_root)
    if samurai_root_str not in sys.path:
        sys.path.append(samurai_root_str)
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(*args, **kwargs)

color = [(255, 0, 0)]


def track_bboxes(video_path: str, init_bbox: np.ndarray, device: str = "cuda"):
    """
    Args:
        video_path: str, path to the video or directory of frames
        init_bbox: np.ndarray, initial bounding box
    Returns:
        List[int], list of valid bounding boxes
        List[np.ndarray], list of detected bounding boxes
		List[np.ndarray], list of detected masks
    """
    model_cfg = "configs/samurai/sam2.1_hiera_l.yaml"
    model_path = "third-party/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"
    predictor = _build_sam2_video_predictor(model_cfg, model_path, device=device)
    prompts = (init_bbox, 0)

    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16):
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        bbox, track_label = prompts
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        detected_bboxes = []
        valid_bboxes = []
        mask_list = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                mask_list.append(mask.astype(np.uint8) * 255)
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max, y_max]
                detected_bboxes.append(bbox)
                if bbox[2] > 0 and bbox[3] > 0:
                    valid_bboxes.append(frame_idx)

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    return valid_bboxes, detected_bboxes, mask_list


def load_data(path: str):
	if path.endswith(".mp4"):
		video = imageio.get_reader(path)
		frames = []
		for frame in video:
			frames.append(frame)
		video.close()
	elif path.endswith(".jpg") or path.endswith(".png"):
		frame = imageio.imread(path)
		frames = [frame]
	else:
		if os.path.isdir(path):
			frames = [imageio.imread(os.path.join(path, f)) for f in sorted(os.listdir(path))]
		else:
			raise ValueError(f"Unsupported file type: {path}")
	return np.stack(frames, axis=0)


def detect_bboxes_using_sam3(
	path: str, 
	prompt: Union[str, List[str]] = "horse", 
	device: str = "cuda",
	model: Sam3Model = None,
	processor: Sam3Processor = None,
):
	"""
	Detect the bboxes of first frame using SAM3
	Args:
		path: str, path to the video or image
		prompt: str or list of str, prompt to detect the bboxes
		device: str, device to use
	Returns:
		bbox: np.ndarray, detected bbox
		frames: np.ndarray, All frames of the video/image/directory
	"""
	if model is None:
		model = Sam3Model.from_pretrained("facebook/sam3").to(device)
	if processor is None:
		processor = Sam3Processor.from_pretrained("facebook/sam3")

	if isinstance(prompt, str):
		text_prompts = [prompt]
	else:
		text_prompts = list(prompt)

	frames = load_data(path)
	inputs = processor(images=[Image.fromarray(frames[0])], text=text_prompts, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	results = processor.post_process_instance_segmentation(
			outputs,
			threshold=0.5,
			mask_threshold=0.5,
			target_sizes=inputs.get("original_sizes").tolist()
	)
	scores = results[0]['scores'].cpu().numpy()
	boxes = results[0]['boxes'].cpu().numpy()
	return boxes, scores, frames


def track_bboxes_using_sam3(boxes: np.ndarray,
							video_frames: np.ndarray,
							device: str = "cuda",
							tracker: Sam3TrackerVideoModel = None,
							processor: Sam3TrackerVideoProcessor = None,
							):
	"""
	Track the bboxes of the video using SAM3
	Args:
		boxes: np.ndarray, detected bboxes
		video_frames: np.ndarray, video frames
		device: str, device to use
	Returns:
		valid_idx: list of int, indices of valid frames
		valid_bboxes: list of np.ndarray, detected bboxes
		mask_list: list of np.ndarray, detected masks
	"""
	# 1. Initialize the tracker
	if tracker is None:
		tracker = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
	if processor is None:
		processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

	num_boxes = len(boxes)
	if num_boxes == 0:
		raise ValueError("No bounding boxes detected on the first frame for the given prompt.")
	obj_ids = list(range(num_boxes))

	# 2. Initialize the tracker and track the bboxes of the video
	inference_session = processor.init_video_session(
						video=video_frames,
						inference_device=device,
						dtype=torch.bfloat16,
					)
	processor.add_inputs_to_inference_session(
		inference_session=inference_session,
		frame_idx=0,
		obj_ids=obj_ids,
		# SAM3 expects boxes in shape [image, boxes, coords]; keep single image batch.
		input_boxes=[boxes.tolist()],
		# input_labels=[1 for _ in range(len(boxes))],
	)

	valid_idx, valid_bboxes, mask_list = {i: [] for i in range(len(boxes))}, {i: [] for i in range(len(boxes))}, {i: [] for i in range(len(boxes))}
	for sam3_tracker_video_output in tracker.propagate_in_video_iterator(
		inference_session,
		start_frame_idx=0,  # first conditioning frame already populated via add_inputs_to_inference_session
	):
		video_res_masks = processor.post_process_masks(
			[sam3_tracker_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
		)[0]

		video_res_masks = (video_res_masks.float().cpu().numpy() > 0.0).astype(np.uint8) * 255
		for i, mask in enumerate(video_res_masks):
			y, x = np.where(mask[0] > 0)
			if len(y) > 0 and len(x) > 0:
				boxes = np.array([x.min(), y.min(), x.max(), y.max()])
				valid_idx[i].append(sam3_tracker_video_output.frame_idx)
				valid_bboxes[i].append(boxes)
				mask_list[i].append(mask[0])
			else:
				continue

	return valid_idx, valid_bboxes, mask_list


if __name__ == "__main__":
	valid_idx, valid_bboxes, mask_list = track_bboxes_using_sam3("data/bedroom.mp4", device="cuda", prompt="kids")
	for i in range(len(valid_idx)):
		masks = mask_list[i]
		writer = imageio.get_writer(f"mask_{i:02d}.mp4", fps=30)
		for mask in masks:
			writer.append_data(mask)
		writer.close()
		