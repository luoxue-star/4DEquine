#!/usr/bin/env python3
import argparse
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import torch

from amr.configs import get_config
from amr.datasets.utils import trans_point2d
from amr.models.pre_post_optimization import PostProcessPipeline
from amr.models.post_optimization import PostProcessRefiner

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _is_image_dir(path):
    """Return True when `path` is a directory containing image frames.

    A valid image directory is treated as one video sequence represented by
    multiple image files. This check is intentionally lightweight: it only
    verifies that at least one supported image extension exists in the folder.
    """
    return os.path.isdir(path) and any(name.lower().endswith(IMAGE_EXTS) for name in os.listdir(path))


def _collect_inputs(video_path):
    """Expand the user input into one or more processable items.

    Supported cases:
    1. `video_path` is a single `.mp4` file.
    2. `video_path` is a single directory of images.
    3. `video_path` is a parent directory containing multiple `.mp4` files
       and/or multiple image directories.

    The returned list always contains the leaf inputs that should be processed
    individually by the pipeline.
    """
    if os.path.isfile(video_path) or _is_image_dir(video_path):
        return [video_path]
    if not os.path.isdir(video_path):
        raise ValueError(f"Invalid video_path: {video_path}")
    # Treat each child mp4 or image directory as one independent sequence.
    inputs = [
        os.path.join(video_path, name)
        for name in sorted(os.listdir(video_path))
        if os.path.isfile(os.path.join(video_path, name)) and name.lower().endswith(".mp4")
        or _is_image_dir(os.path.join(video_path, name))
    ]
    if not inputs:
        raise ValueError(f"No valid videos or image folders found in: {video_path}")
    return inputs


def _item_output_dir(output_dir, input_path, multiple):
    """Choose the output directory for one input sequence.

    For a single input, the user-provided output directory is reused directly.
    For batched inputs, each item gets its own subdirectory named after the
    file or folder stem so results do not overwrite each other.
    """
    if not multiple:
        return output_dir
    name = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]
    return os.path.join(output_dir, name)


def parse_args():
    """Parse command-line arguments for post-optimization.

    The script can run stage 1, stage 2, or both. `video_path` may point to a
    single video, a single image folder, or a directory that contains multiple
    processable inputs.
    """
    parser = argparse.ArgumentParser(
        description="Process one video for post-optimization, then optimize it."
    )
    parser.add_argument("--video_path", type=str, default=None, help="Input video path")
    parser.add_argument("--checkpoint", type=str, default=None, help="AniMerVAREN checkpoint")
    parser.add_argument("--cfg", type=str, default=None, help="AniMerVAREN config yaml")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--stage1", action="store_true", help="Whether to run stage 1")
    parser.add_argument("--stage2", action="store_true", help="Whether to run stage 2. Note: must be run the first stage before running the second stage")
    return parser.parse_args()


def run_one(args: argparse.Namespace, video_path: str, output_dir: str, pipeline: PostProcessPipeline | None):
    """Run the configured stages for one resolved input sequence.

    Args:
        args: Parsed command-line arguments shared by the whole script.
        video_path: One concrete input item, either an `.mp4` path or an image
            directory path.
        output_dir: Directory where intermediate and final outputs for this
            item should be written.
        pipeline: PostProcessPipeline instance

    Stage 1 performs tracking, keypoint inference, and AniMerVAREN inference,
    then saves the intermediate results to disk. Stage 2 reloads those saved
    results and runs the optimization/refinement stage.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing: {video_path}")
    print("[Note]: If you encounter OOM error, please run first stage and second stage separately.")
    if args.stage1:
        print("-------------------------------- Running Stage 1 --------------------------------")
        # Prepare all per-frame observations needed by the optimizer.
        valid_idx, detected_bboxes, mask_list = pipeline.track_bounding_boxes(video_path)
        vitpose_results = pipeline.infer_vitposepp(
            video_path, valid_idx, detected_bboxes, return_pose_image=False
        )
        animer_outputs = pipeline.infer_animer_varen(
            video_path, valid_idx, detected_bboxes, mask_list
        )

        pipeline.save_mask_list(mask_list, output_dir)
        pipeline.save_vitpose_results(vitpose_results, output_dir)
        pipeline.save_animer_outputs(animer_outputs, output_dir)
        print("-------------------------------- Stage 1 Completed --------------------------------")

    if args.stage2:
        print("-------------------------------- Running Stage 2 --------------------------------")
        # Optimize model parameters from the saved stage-1 predictions.
        # Reload stage-1 outputs from disk so this stage can run independently
        # and starts from CPU memory instead of reusing live GPU tensors.
        animer_outputs = torch.load(os.path.join(output_dir, "animer_outputs.pt"), map_location="cpu", weights_only=True)
        mask_list = animer_outputs["mask"]
        valid_idx = np.arange(animer_outputs["pred_cam"].shape[0], dtype=np.int32)
        vitpose_results = pickle.load(open(os.path.join(output_dir, "vitpose_results.pkl"), "rb"))
        vitpose_results = {k: v for k, v in vitpose_results.items()}
        # Convert keypoints back into the coordinate system expected by the refiner.
        vitpose_results = {k: trans_point2d(v['keypoints'][0], animer_outputs['trans'][k])[None] for k, v in vitpose_results.items()}
        refiner = PostProcessRefiner(get_config(args.cfg), animer_outputs, device=args.device)
        derived_opt = refiner.optimize_parameters(
            keypoints_results=vitpose_results,
            mask_list=mask_list,
            valid_idx=valid_idx,
            image_size=args.image_size,
            num_iters=[75, 100],
            lr=[5e-3, 5e-3],
            chunk_size=args.chunk_size,
            w_silhouette=[100.0, 10000.0],
            w_keypoints=[10000.0, 100.0],
            w_temporal=[100.0, 100.0],
            w_reg=[1000.0, 300.0],
            w_pose_prior=[1000.0, 300.0],
            params_per_stage=[
                ["pose", "betas", "cam", "tail_scale", "global_orient"],
                ["tail_scale", "betas"],
            ],
            output_dir=output_dir,
            render_every=None,
            use_adaptive_lr=True,
            lr_scheduler_patience=20,
            lr_scheduler_factor=0.5,
        )

        postrefine_dir = os.path.join(output_dir, "postrefine")
        refiner.save_refined_results(
            output_dir=postrefine_dir,
            derived=derived_opt,
            image_size=args.image_size,
            chunk_size=args.chunk_size,
        )
        refiner.render_refined_mesh(
            animer_outputs=animer_outputs,
            output_path=os.path.join(postrefine_dir, "refined_mesh.mp4"),
            fps=args.fps,
            image_size=args.image_size,
        )
        print("-------------------------------- Stage 2 Completed --------------------------------")


def main(args: argparse.Namespace):
    """Resolve user input and process all discovered sequences.

    This is the top-level dispatcher for the script. It expands `video_path`
    into one or more concrete inputs, decides whether per-item output folders
    are needed, and then runs each item sequentially.
    """
    inputs = _collect_inputs(args.video_path)
    multiple = len(inputs) > 1
    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = PostProcessPipeline(device=args.device, checkpoint_path=args.checkpoint) if args.stage1 else None
    for video_path in inputs:
        run_one(args, video_path, _item_output_dir(args.output_dir, video_path, multiple), pipeline)


if __name__ == "__main__":
    args = parse_args()
    main(args)

