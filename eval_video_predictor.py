import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from amr.configs import get_config
from amr.models.varen_amr import AniMerVAREN
from amr.models.varen import VAREN
from amr.models.predictor import VideoPredictor
from amr.utils.evaluate_metric import Evaluator
from amr.utils.geometry import perspective_projection
from amr.datasets.utils import trans_point2d
from pytorch3d.transforms import matrix_to_axis_angle


APT36K_JSON_DIR = Path("data/apt36k_eval")
APT36K_ROOT = Path("/data2/lvjin/cvpr26/data/AP-36k-patr1")
VARENPOSER_TO_APT36K = [2, 3, 4, 5, 6, 13, 11, 9, 14, 12, 10, 19, 17, 15, 20, 18, 16]

ANIMAL4D_JSON_DIR = Path("data/animal4d_eval")
ANIMAL4D_ROOT = Path("/data2/lvjin/cvpr26/data/Animal4D/horse")
VARENPOSER_TO_ANIMAL4D = [2, 3, 4, 5, 6, 13, 11, 9, 14, 12, 10, 19, 17, 15, 20, 18, 16]

VARENPOSER_JSON_DIR = Path("data/varenposer_eval")
VARENPOSER_ROOT = Path("/data2/lvjin/cvpr26/data/VarenPoser/testseq")
VARENPOSER_TO_VARENPOSER = list(range(21))


def _compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor):
    """
    Batched Procrustes: returns S1 transformed to S2, along with R, scale, t.
    Shapes: S1, S2: (B, N, 3)
    """
    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)

    # Handle inputs that include additional channels (e.g. confidence scores)
    if S1.shape[-1] > 3:
        S1 = S1[..., :3]
    if S2.shape[-1] > 3:
        S2 = S2[..., :3]
    batch_size = S1.shape[0]
    S1_t = S1.permute(0, 2, 1)
    S2_t = S2.permute(0, 2, 1)
    mu1 = S1_t.mean(dim=2, keepdim=True)
    mu2 = S2_t.mean(dim=2, keepdim=True)
    X1 = S1_t - mu1
    X2 = S2_t - mu2
    var1 = (X1**2).sum(dim=(1,2))
    K = torch.matmul(X1, X2.permute(0, 2, 1))
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    t = mu2 - scale*torch.matmul(R, mu1)
    S1_hat = scale*torch.matmul(R, S1_t) + t
    return S1_hat.permute(0, 2, 1), R, scale, t


def _procrustes_align(source_points: torch.Tensor, selected_source_points: torch.Tensor,
                      selected_target_points: torch.Tensor):
    """
    Aligns source_points to target using selected correspondences.
    source_points: (B, Ns, 3)
    selected_source_points: (B, K, 3)
    selected_target_points: (B, K, 3)
    Returns: transformed_source_points (B, Ns, 3)
    """
    selected_transformed_source, R, scale, t = _compute_similarity_transform(selected_source_points, selected_target_points)
    # apply to full source points
    source_points_t = source_points.permute(0, 2, 1)  # (B, 3, Ns)
    rotated = torch.matmul(R, source_points_t)  # (B, 3, Ns)
    source_transformed = scale * rotated + t
    source_transformed = source_transformed.permute(0, 2, 1)
    return source_transformed


def _compute_chamfer(pred_vertices_aligned: torch.Tensor, gt_vertices: torch.Tensor):
    from pytorch3d.loss import chamfer_distance
    # Ensure shapes (B, N, 3)
    if pred_vertices_aligned.dim() == 2:
        pred_vertices_aligned = pred_vertices_aligned.unsqueeze(0)
    if gt_vertices.dim() == 2:
        gt_vertices = gt_vertices.unsqueeze(0)
    cd, _ = chamfer_distance(pred_vertices_aligned, gt_vertices)
    return cd


def load_model(checkpoint: str, cfg_path: str, device: str):
    cfg = get_config(cfg_path)
    model = AniMerVAREN.load_from_checkpoint(checkpoint_path=checkpoint, cfg=cfg, strict=False)
    model.to(device).eval()
    return model, cfg


def collect_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files = sorted([path for path in directory.glob("*.json") if path.is_file()])
    return files


def _xywh_to_xyxy(b: List[float]) -> List[float]:
    # JSON stores [x, y, w, h]
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _load_sequence(
    json_path: Path,
    root_image: Path,
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    with open(json_path, "r") as f:
        data = json.load(f)  # type: ignore

    items = data["data"]
    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    bboxes_xyxy: List[List[float]] = []
    gt_keypoints_list: List[np.ndarray] = []
    vis_list: List[np.ndarray] = []

    for it in items:
        img_path = root_image / it["img_path"]
        mask_path = root_image / it["mask_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        frames.append(img)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        masks.append(mask)

        bboxes_xyxy.append(_xywh_to_xyxy(it["bbox"]))

        kps = np.array(it["keypoint_2d"], dtype=np.float32)  # (K,3): x,y,conf
        gt_keypoints_list.append(kps[:, :2])
        vis_list.append((kps[:, 2] > 0.5).astype(np.float32))

    bboxes_xyxy_np = np.array(bboxes_xyxy, dtype=np.float32)
    gt_kp_np = np.stack(gt_keypoints_list, axis=0)  # (T,K,2)
    vis_np = np.stack(vis_list, axis=0)  # (T,K)
    return frames, bboxes_xyxy_np, masks, gt_kp_np, vis_np


def _to_axis_angle(global_orient_mt: torch.Tensor, pose_mt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    go_aa = matrix_to_axis_angle(global_orient_mt.reshape(global_orient_mt.shape[0], -1, 3, 3)).view(global_orient_mt.shape[0], -1)
    pose_aa = matrix_to_axis_angle(pose_mt.reshape(pose_mt.shape[0], -1, 3, 3)).view(pose_mt.shape[0], -1)
    return go_aa, pose_aa


def evaluate_sequence(
    json_path: Path,
    root_image: Path,
    model: AniMerVAREN,
    cfg,
    evaluator: Evaluator,
    device: str,
    compute_3d: bool = False,
    varen_model: Optional[VAREN] = None,
    keypoint_indices: Optional[Sequence[int]] = None,
    save_renders: bool = False,
    save_input: bool = False,
    render_dir: Optional[Path] = None,
    sequence_name: str = "",
) -> Dict[str, List[float]]:
    frames, bboxes_xyxy, mask_list, gt_kp_px, vis = _load_sequence(json_path, root_image)
    valid_idx = np.arange(len(frames), dtype=np.int32)

    predictor = VideoPredictor(
        model=model,
        model_cfg=cfg,
        frames=frames,  # BGR frames
        valid_idx=valid_idx,
        bboxes=bboxes_xyxy,
        mask_list=mask_list,
        device=device,
    )
    with torch.no_grad():
        results = predictor.inference()

    # Prepare SMAL inputs (axis-angle) and compute keypoints/verts
    pred_global_orient_mt: torch.Tensor = results["pred_global_orient"]
    pred_pose_mt: torch.Tensor = results["pred_pose"]
    pred_betas: torch.Tensor = results["pred_betas"]
    pred_cam: torch.Tensor = results["pred_cam"]
    pred_cam_t: torch.Tensor = results["pred_cam_t"]

    # Stabilize betas across time
    seq_len = pred_global_orient_mt.shape[0]
    pred_betas = pred_betas.mean(dim=0, keepdim=True).repeat(seq_len, 1)

    go_aa, pose_aa = _to_axis_angle(pred_global_orient_mt, pred_pose_mt)
    smal_out = model.smal(global_orient=go_aa, pose=pose_aa, betas=pred_betas)
    pred_kp3d: torch.Tensor = smal_out.surface_keypoints  # (T,K,3)
    if keypoint_indices is not None:
        kp_idx = torch.tensor(keypoint_indices, device=pred_kp3d.device, dtype=torch.long)
        pred_kp3d = pred_kp3d.index_select(dim=1, index=kp_idx)

    # Reconstruct pred_keypoints_2d in model's normalized crop coordinates
    focal = cfg.SMAL.get("FOCAL_LENGTH", 5000)
    focal_norm = torch.tensor([focal, focal], device=device, dtype=pred_kp3d.dtype).unsqueeze(0).repeat(seq_len, 1) / cfg.MODEL.IMAGE_SIZE
    pred_kp2d_norm: torch.Tensor = perspective_projection(pred_kp3d, translation=pred_cam_t, focal_length=focal_norm)

    # Build GT keypoints in normalized crop coordinates using saved trans
    trans: torch.Tensor = results["trans"].detach().cpu()  # (T,2,3)
    gt_kp_norm_list: List[np.ndarray] = []
    for t in range(seq_len):
        kpxy = gt_kp_px[t]
        conf = vis[t][:, None]  # (K,1)
        kp = np.concatenate([kpxy, conf], axis=1)
        kp_crop = trans_point2d(kp, trans[t].numpy())  # (K,3)
        # Normalize to [-0.5, 0.5] over model crop size
        kp_xy = kp_crop[:, :2] / float(cfg.MODEL.IMAGE_SIZE) - 0.5
        kp_norm = np.concatenate([kp_xy, kp_crop[:, 2:]], axis=1)
        gt_kp_norm_list.append(kp_norm.astype(np.float32))
    gt_kp_norm = torch.from_numpy(np.stack(gt_kp_norm_list, axis=0)).to(device)  # (T,K,3)

    # Prepare batch/output dicts expected by Evaluator
    output = {"pred_keypoints_2d": pred_kp2d_norm}
    batch = {"keypoints_2d": gt_kp_norm, "mask": results.get("mask", torch.zeros(seq_len, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.IMAGE_SIZE, device=device))}

    # Render and save images if requested
    if save_renders or save_input:
        if render_dir is not None:
            # Setup renderer
            faces_np = model.smal.faces
            mesh_renderer = model.mesh_renderer

            # Setup output directory
            seq_render_dir = render_dir / sequence_name
            seq_render_dir.mkdir(parents=True, exist_ok=True)

            # Get vertices and camera translation for all frames
            pred_vertices = smal_out.vertices.detach().cpu().numpy()  # (T, V, 3)
            pred_cam_t_np = pred_cam_t.detach().cpu().numpy()  # (T, 3)

            # Denormalize frames for rendering
            mean = torch.tensor(cfg.MODEL.IMAGE_MEAN).reshape(3, 1, 1).to(device)
            std = torch.tensor(cfg.MODEL.IMAGE_STD).reshape(3, 1, 1).to(device)

            for t in range(seq_len):
                frame_tensor = results['img'][t]
                frame_np = (frame_tensor * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()

                # Render overlay
                over_np = mesh_renderer(
                    pred_vertices[t],
                    pred_cam_t_np[t],
                    frame_np,
                    focal_length=focal,
                    side_view=False,
                    baseColorFactor=(0.65098039, 0.74117647, 0.85882353, 1.0),
                )

                # Save images
                frame_idx = f"{t:06d}"
                if save_input:
                    cv2.imwrite(str(seq_render_dir / f"{frame_idx}_input.jpg"), (frame_np[:, :, ::-1] * 255.0).astype(np.uint8))
                if save_renders:
                    over_bgr = (over_np[:, :, ::-1] * 255.0).astype(np.uint8)
                    cv2.imwrite(str(seq_render_dir / f"{frame_idx}_overlay.jpg"), over_bgr)

    # 2D metrics per batch of the whole sequence
    pck, auc, accel = evaluator.eval_2d(output, batch, pck_threshold=[0.05, 0.10])

    # Sequence acceleration in normalized crop space
    pred_np = pred_kp2d_norm.detach().cpu().numpy()  # (T,K,2)
    gt_np = gt_kp_norm[:, :, :2].detach().cpu().numpy()
    vis_np = gt_kp_norm[:, :, 2].detach().cpu().numpy() > 0.5

    metrics = {
        "PCK@0.05": [float(pck[1])] * seq_len,
        "PCK@0.10": [float(pck[2])] * seq_len,
        "AUC": [float(auc)] * seq_len,
        "ACCEL": [float(accel) * (30 ** 2)] * (seq_len - 2),
    }

    # Optional 3D metric: PA-Chamfer (when GT SMAL params available)
    if compute_3d:
        # Load GT SMAL params and 3D keypoints from JSON
        with open(json_path, "r") as f:
            j = json.load(f)  # type: ignore
        items = j["data"]
        pose = torch.tensor([it["pose"] for it in items], dtype=torch.float32, device=device)
        betas = torch.tensor([it["shape"] + it.get("shape_extra", []) for it in items], dtype=torch.float32, device=device)
        transl = torch.tensor([it["trans"] for it in items], dtype=torch.float32, device=device)

        # Extract GT 3D keypoints (T, K, 3)
        gt_keypoints_3d = torch.tensor([it["keypoint_3d"] for it in items], dtype=torch.float32, device=device)

        go = pose[:, :3]
        ps = pose[:, 3:]
        with torch.no_grad():
            gt_vertices = varen_model(global_orient=go, pose=ps, betas=betas, transl=transl).vertices  # (T,V,3)

        # Align pred_vertices to GT using 3D keypoints via procrustes
        pred_vertices = smal_out.vertices.detach() + pred_cam_t.unsqueeze(1)
        pred_vertices_aligned = _procrustes_align(pred_vertices, pred_kp3d + pred_cam_t.unsqueeze(1), gt_keypoints_3d)

        # Compute chamfer distance on aligned vertices (per-sequence metric, replicate for all images)
        pa_chamfer_distance = _compute_chamfer(pred_vertices_aligned, gt_vertices).mean().cpu().numpy() * 1000
        # Replicate the sequence-level metric for each image
        metrics["PA-Chamfer"] = [float(pa_chamfer_distance)] * seq_len

    return metrics


def summarize(dataset_name: str, results: Dict[str, Dict[str, List[float]]], default_keys: Iterable[str]) -> None:
    if not results:
        print(f"Dataset: {dataset_name} (0 sequences)")
        print("  No results")
        return

    keys = list(default_keys)
    # Count total images across all sequences (use PCK as reference)
    total_images = 0
    for metrics in results.values():
        for k in metrics:
            if "PCK" in k or "AUC" in k:
                total_images += len(metrics[k])
                break
    print(f"Dataset: {dataset_name} ({len(results)} sequences, {total_images} images)")
    for k in keys:
        # Collect all per-image values across all sequences
        all_vals = []
        for metrics in results.values():
            if k in metrics:
                all_vals.extend(metrics[k])
        if not all_vals:
            continue
        print(f"  {k}: {np.mean(all_vals):.4f} (n={len(all_vals)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="logs/train/runs/mocapv2/.hydra/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="logs/train/runs/mocapv2/checkpoints/last.ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render_dir", type=str, default="outputs/mocap", help="Directory to save rendered images")
    parser.add_argument("--save_renders", type=bool, default=True, help="Save rendered overlay images")
    parser.add_argument("--save_input", type=bool, default=False, help="Save input images along with renders")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=["apt36k", "animal4d", "varenposer"],
        help="Datasets to evaluate (subset of apt36k, animal4d, varenposer)",
    )
    args = parser.parse_args()

    device = args.device
    model, cfg = load_model(args.checkpoint, args.config, device)
    evaluator = Evaluator(smal_model=model.smal, image_size=cfg.MODEL.IMAGE_SIZE)

    dataset_definitions = {
        "apt36k": {
            "json_dir": APT36K_JSON_DIR,
            "root": APT36K_ROOT,
            "compute_3d": False,
            "keypoint_indices": VARENPOSER_TO_APT36K,
        },
        "animal4d": {
            "json_dir": ANIMAL4D_JSON_DIR,
            "root": ANIMAL4D_ROOT,
            "compute_3d": False,
            "keypoint_indices": VARENPOSER_TO_ANIMAL4D,
        },
        "varenposer": {
            "json_dir": VARENPOSER_JSON_DIR,
            "root": VARENPOSER_ROOT,
            "compute_3d": True,
            "keypoint_indices": VARENPOSER_TO_VARENPOSER,
        },
    }

    # Load VAREN only if any dataset needs 3D metrics
    varen_model = model.smal.eval()

    metric_keys = ["PCK@0.05", "PCK@0.10", "AUC", "ACCEL", "PA-Chamfer"]

    # Setup render directory if saving is enabled
    render_dir = None
    if args.save_renders or args.save_input:
        render_dir = Path(args.render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        if name not in dataset_definitions:
            print(f"[WARN] Unknown dataset '{name}', skipping")
            continue

        definition = dataset_definitions[name]
        json_dir = definition["json_dir"]
        root_image = definition["root"]
        seq_files = collect_json_files(json_dir)

        dataset_results: Dict[str, Dict[str, List[float]]] = {}
        for jp in tqdm(seq_files, desc=f"{name}", dynamic_ncols=True):
            metrics = evaluate_sequence(
                jp,
                root_image,
                model,
                cfg,
                evaluator,
                device,
                compute_3d=definition["compute_3d"],
                varen_model=varen_model,
                keypoint_indices=definition["keypoint_indices"],
                save_renders=args.save_renders,
                save_input=args.save_input,
                render_dir=render_dir,
                sequence_name=jp.stem,
            )
            dataset_results[jp.stem] = metrics

        summarize(name, dataset_results, metric_keys)


if __name__ == "__main__":
    main()


