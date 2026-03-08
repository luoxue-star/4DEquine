import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
from tqdm import tqdm

from amr.configs import get_config
from amr.utils.evaluate_metric import Evaluator
from amr.models.varen import VAREN
from amr.datasets.utils import trans_point2d

VARENPOSERTOAPT36K = [2, 3, 4, 5, 6, 13, 11, 9, 14, 12, 10, 19, 17, 15, 20, 18, 16]
VARENPOSERTOANIMAL4D = [2, 3, 4, 5, 6, 13, 11, 9, 14, 12, 10, 19, 17, 15, 20, 18, 16]
VARENPOSERTOVARENPOSER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def compute_error_accel(joints_gt, joints_pred, vis=None, return_per_frame=False, fps: float = 1.0):
    """
    Computes 2D acceleration error:
        1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (NxKx2 or 3): Ground truth 2D or 3D joints. N frames, K joints, (x,y) or (x,y,z) coords.
        joints_pred (NxKx2 or 3): Predicted 2D or 3D joints. N frames, K joints, (x,y) or (x,y,z) coords.
        vis (N, K): Visibility mask for each frame.
    Returns:
        If return_per_frame is True, returns a tensor of per-frame acceleration
        errors with shape (N-2,). Otherwise returns the mean acceleration error as
        a scalar float.
    """
    # (N-2)xKx2 or 3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    if isinstance(accel_gt, np.ndarray):
        accel_gt = torch.from_numpy(accel_gt)
    if isinstance(accel_pred, np.ndarray):
        accel_pred = torch.from_numpy(accel_pred)
    if isinstance(vis, np.ndarray):
        vis = torch.from_numpy(vis)

    normed = torch.linalg.norm(accel_pred - accel_gt, dim=2)
    
    if vis is None:
        new_vis = torch.ones((len(normed), normed.shape[1]), dtype=torch.bool, 
                            device=normed.device)  # (N-2)x14
    else:
        if vis.dim() == 1:
            invis = torch.logical_not(vis)
            invis1 = torch.roll(invis, shifts=-1, dims=0)
            invis2 = torch.roll(invis, shifts=-2, dims=0)
            new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]
            new_vis = torch.logical_not(new_invis)
            new_vis = new_vis.unsqueeze(1).expand(-1, normed.shape[1])  # (N-2)x14
        else:
            invis = torch.logical_not(vis)  # (N,K)
            invis1 = torch.roll(invis, shifts=-1, dims=0)  # (N,K)
            invis2 = torch.roll(invis, shifts=-2, dims=0)  # (N,K)
            new_invis = torch.logical_or(invis, torch.logical_or(invis1, invis2))[:-2]  # (N-2,K)
            new_vis = torch.logical_not(new_invis)  # (N-2,K)
    
    masked_normed = normed * new_vis.float()
    
    valid_count = torch.sum(new_vis, dim=1).float()
    valid_count = torch.clamp(valid_count, min=1.0)
    
    frame_errors = torch.sum(masked_normed, dim=1) / valid_count
    frame_errors = frame_errors * (fps ** 2)
    
    if return_per_frame:
        return frame_errors
    return frame_errors.mean().item()


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


class MetricStore:
    def __init__(self, include_3d: bool):
        self.include_3d = include_3d
        self.two_d = {
            "sum_pck": None,
            "sum_accel": 0.0,
            "count": 0,
        }
        self.three_d = None
        if include_3d:
            self.three_d = {
                "sum_pa_chamfer": 0.0,
                "count": 0,
            }

    def update_2d(self, pck: List[float], accel: float, num_samples: int) -> None:
        if num_samples <= 0:
            return
        pck_array = np.array(pck, dtype=np.float32)
        if self.two_d["sum_pck"] is None:
            self.two_d["sum_pck"] = np.zeros_like(pck_array, dtype=np.float64)
        self.two_d["sum_pck"] += pck_array * num_samples
        self.two_d["sum_accel"] += float(accel) * num_samples
        self.two_d["count"] += num_samples

    def update_3d(
        self,
        pa_chamfer_distance: float,
        num_samples: int,
    ) -> None:
        if not self.include_3d or self.three_d is None or num_samples <= 0:
            return

        self.three_d["sum_pa_chamfer"] += float(pa_chamfer_distance) * num_samples
        self.three_d["count"] += num_samples

    def summarize(self, default_cfg) -> Dict[str, Optional[Dict[str, float]]]:
        summary: Dict[str, Optional[Dict[str, float]]] = {"2d": None, "3d": None}

        if self.two_d["count"] > 0 and self.two_d["sum_pck"] is not None:
            count = self.two_d["count"]
            mean_pck = self.two_d["sum_pck"] / count
            accel_mean = self.two_d["sum_accel"] / count

            pck_summary: Dict[str, float] = {}
            for idx, threshold in enumerate(default_cfg.METRIC.PCK_THRESHOLD):
                key = f"PCK@{threshold}"
                pck_summary[key] = float(mean_pck[idx])
            summary["2d"] = {
                **pck_summary,
                "ACCEL": float(accel_mean),
            }

        if self.include_3d and self.three_d and self.three_d["count"] > 0:
            count = self.three_d["count"]
            summary["3d"] = {
                "PA-Chamfer": float(self.three_d["sum_pa_chamfer"] / count),
            }

        return summary


def load_json_ground_truth(json_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[Dict]]:
    """Load ground truth data from JSON file.
    
    Returns:
        gt_keypoints_2d: List of keypoints (K, 2)
        vis: List of visibility masks (K,)
        smal_params: Dict with pose, betas, transl, global_orient (for 3D eval) or None
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_keypoints_2d = []
    gt_keypoints_3d = []
    vis = []
    smal_params = None
    has_3d = False
    
    items = data['data']
    for item in items:
        # Load keypoints
        kps = np.array(item['keypoint_2d'], dtype=np.float32)  # (K, 3): x, y, conf
        gt_keypoints_2d.append(kps[:, :2])
        vis.append((kps[:, 2] > 0.5).astype(np.float32))

        keypoints_3d = np.array(item['keypoint_3d'], dtype=np.float32) if 'keypoint_3d' in item else np.zeros((len(kps), 3), dtype=np.float32)
        gt_keypoints_3d.append(keypoints_3d)
        
        # Check if 3D data is available
        if 'pose' in item and not has_3d:
            has_3d = True
            smal_params = {
                'pose': [],
                'betas': [],
                'transl': [],
                'global_orient': []
            }
        
        if has_3d:
            pose = np.array(item['pose'], dtype=np.float32)
            try:
                betas = np.array(item['shape'] + item['shape_extra'], dtype=np.float32)
            except:
                betas = np.array(item['shape'], dtype=np.float32)
            transl = np.array(item['trans'], dtype=np.float32)
            global_orient = pose[:3]
            pose = pose[3:]
            
            smal_params['pose'].append(pose)
            smal_params['betas'].append(betas)
            smal_params['transl'].append(transl)
            smal_params['global_orient'].append(global_orient)
    
    if has_3d:
        for key in smal_params:
            smal_params[key] = np.stack(smal_params[key], axis=0)
    
    return gt_keypoints_2d, gt_keypoints_3d, vis, smal_params


def load_results(results_path: Path) -> Dict[str, torch.Tensor]:
    """Load refined results from postrefine optimization."""
    if not results_path.exists():
        raise FileNotFoundError(f"Refined results not found: {results_path}")
    
    refined_data = torch.load(results_path, map_location='cpu', weights_only=True)
    return refined_data


def evaluate_sequence(
    json_path: Path,
    results_path: Path,
    evaluator: Evaluator,
    varen: VAREN,
    default_cfg,
    device: str,
    metric_store: MetricStore,
    compute_3d: bool,
    animal3d_mapping: List[int],
    image_size: int = 256,
) -> None:
    """Evaluate a single sequence."""
    
    # Load ground truth
    gt_keypoints_2d, gt_keypoints_3d, vis, smal_params = load_json_ground_truth(json_path)
    
    # Load refined results
    refined_data = load_results(results_path)
    # Load w/o refined results
    wo_refine_data = load_results(Path(results_path.parent.parent / "animer_outputs.pt"))
    trans_matrix = wo_refine_data['trans'].numpy()
    
    # Extract predictions
    pred_keypoints_2d = refined_data['keypoints_2d']  # (N, 21, 2)
    pred_vertices = refined_data.get('vertices', None)
    
    # Map keypoints to target dataset format
    pred_keypoints_2d_mapped = pred_keypoints_2d[:, animal3d_mapping, :]
    
    # Prepare batch data for evaluation
    num_frames = len(gt_keypoints_2d)
    # Transform GT keypoints to crop space with visibility
    gt_kp_crop = [trans_point2d(np.concatenate((gt_keypoints_2d[i], vis[i][:, None]), axis=1), trans_matrix[i]) 
                  for i in range(num_frames)]
    gt_kp_crop_array = np.stack(gt_kp_crop, axis=0)  # (N, K, 3)
    
    # Normalize GT keypoints: extract x,y, normalize, then concatenate back with visibility
    gt_kp_tensor = torch.from_numpy(gt_kp_crop_array.copy())
    gt_kp_xy_normalized = gt_kp_tensor[:, :, :2] / image_size - 0.5  # (N, K, 2)
    gt_kp_vis = gt_kp_tensor[:, :, 2:3]  # (N, K, 1)
    gt_kp_tensor = torch.cat([gt_kp_xy_normalized, gt_kp_vis], dim=-1)  # (N, K, 3)
    
    gt_kp_3d_tensor = torch.stack([torch.from_numpy(kp) for kp in gt_keypoints_3d], dim=0).to(device)  # (N, K, 3)
    
    # Prepare batch dict for evaluator
    batch = {
        'keypoints_2d': gt_kp_tensor.to(device),
        'has_keypoints_2d': torch.ones(num_frames, device=device),
    }
    
    # Prepare output dict
    output = {
        'pred_keypoints_2d': pred_keypoints_2d_mapped.to(device),
    }
    
    # Compute 3D metrics if available
    if compute_3d and pred_vertices is not None and smal_params is not None:
        # Get predicted 3D keypoints from refined parameters
        from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
        
        refined_pose_6d = refined_data['refined_pose']
        refined_global_orient_6d = refined_data['refined_global_orient']
        refined_betas = refined_data['refined_betas']
        refined_cam_t = refined_data['refined_cam_t']
        
        # Convert 6D to rotation matrix
        refined_pose_rotmat = matrix_to_axis_angle(rotation_6d_to_matrix(refined_pose_6d.reshape(-1, 6)).reshape(num_frames, -1, 3, 3))
        refined_global_orient_rotmat = matrix_to_axis_angle(rotation_6d_to_matrix(refined_global_orient_6d.reshape(-1, 6)).reshape(num_frames, -1, 3, 3))
        
        # Get 3D keypoints from VAREN
        with torch.no_grad():
            varen_output = varen(
                global_orient=refined_global_orient_rotmat.to(device),
                pose=refined_pose_rotmat.to(device),
                betas=refined_betas.to(device),
                transl=torch.zeros_like(refined_cam_t).to(device)
            )
            pred_keypoints_3d = varen_output.surface_keypoints[:, animal3d_mapping, :]
        
        # Get GT 3D keypoints
        gt_pose = torch.from_numpy(smal_params['pose']).to(device)
        gt_betas = torch.from_numpy(smal_params['betas']).to(device)
        gt_transl = torch.from_numpy(smal_params['transl']).to(device)
        gt_global_orient = torch.from_numpy(smal_params['global_orient']).to(device)
        
        with torch.no_grad():
            gt_varen_output = varen(
                global_orient=gt_global_orient.reshape(num_frames, 1, 3),
                pose=gt_pose.reshape(num_frames, -1, 3),
                betas=gt_betas,
                transl=gt_transl
            )
            gt_vertices = gt_varen_output.vertices
        
        # Procrustes alignment
        pred_vertices_aligned = _procrustes_align(
            pred_vertices.to(device).float(),
            pred_keypoints_3d,
            gt_kp_3d_tensor
        )
        
        # Compute Chamfer distance
        pa_chamfer_distance = _compute_chamfer(pred_vertices_aligned, gt_vertices).mean().cpu().numpy() * 1000
        
        metric_store.update_3d(
            pa_chamfer_distance,
            num_samples=num_frames,
        )
    
    # Compute 2D metrics
    pck = evaluator.compute_pck_bbox(output, batch, pck_threshold=default_cfg.METRIC.PCK_THRESHOLD)
    
    # Compute acceleration error
    pred_seq_2d = pred_keypoints_2d_mapped.numpy()
    gt_seq_2d = gt_kp_xy_normalized.numpy()
    vis_seq = np.array(vis)
    
    if pred_seq_2d.shape[0] >= 3:
        accel_err = compute_error_accel(gt_seq_2d, pred_seq_2d, vis=vis_seq, fps=30.0)
    else:
        accel_err = 0.0
    
    metric_store.update_2d(pck, accel_err, num_samples=num_frames)


def collect_json_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    files = sorted([path for path in directory.glob("*.json") if path.is_file()])
    return files


def summarize_and_print(dataset_name: str, store: MetricStore, default_cfg, num_sequences: int) -> None:
    """Print summary of metrics."""
    summary = store.summarize(default_cfg)
    print(f"Dataset: {dataset_name} ({num_sequences} sequences)")

    if summary["3d"]:
        print("  3D metrics:")
        for key, value in summary["3d"].items():
            print(f"    {key}: {value:.4f}")
    else:
        print("  3D metrics: N/A")

    if summary["2d"]:
        print("  2D metrics:")
        for key, value in summary["2d"].items():
            print(f"    {key}: {value:.4f}")
    else:
        print("  2D metrics: N/A")

    print("")


def main():
    parser = argparse.ArgumentParser(description="Evaluate postrefine results on multiple datasets")
    parser.add_argument("--config", type=str, 
                        default="logs/train/runs/varen/.hydra/config.yaml",
                        help="Path to config file")
    parser.add_argument("--default_eval_config", type=str,
                        default="amr/configs_hydra/experiment/default_val.yaml",
                        help="Path to default evaluation config")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use for computation")
    
    args = parser.parse_args()
    
    cfg = get_config(args.config)
    default_cfg = get_config(args.default_eval_config)
    
    device = args.device
    
    # Load VAREN model for evaluation
    varen = VAREN(model_path=cfg.SMAL.MODEL_PATH, use_muscle_deformations=False).to(device).eval()
    evaluator = Evaluator(smal_model=varen, image_size=cfg.MODEL.IMAGE_SIZE)
    
    # Dataset definitions
    dataset_definitions = [
        {
            "name": "apt36k",
            "json_dir": Path(default_cfg.DATASETS.APT36K.JSON_DIR),
            "compute_3d": False,
            "animal3d_mapping": VARENPOSERTOAPT36K,
            "refined_root": Path(default_cfg.DATASETS.APT36K.REFINED_ROOT),
        },
        {
            "name": "animal4d",
            "json_dir": Path(default_cfg.DATASETS.ANIMAL4D.JSON_DIR),
            "compute_3d": False,
            "animal3d_mapping": VARENPOSERTOANIMAL4D,
            "refined_root": Path(default_cfg.DATASETS.ANIMAL4D.REFINED_ROOT),
        },
        {
            "name": "varenposer",
            "json_dir": Path(default_cfg.DATASETS.VARENPOSER.JSON_DIR),
            "compute_3d": True,
            "animal3d_mapping": VARENPOSERTOVARENPOSER,
            "refined_root": Path(default_cfg.DATASETS.VARENPOSER.REFINED_ROOT),
        },
    ]
    
    for definition in dataset_definitions:
        json_files = collect_json_files(definition["json_dir"])
        metric_store = MetricStore(include_3d=definition["compute_3d"])
        
        evaluated_count = 0
        skipped_count = 0
        
        for json_path in tqdm(json_files, desc=f"{definition['name']}", total=len(json_files), leave=False):
            refined_path = definition["refined_root"] / json_path.stem / "postrefine" / "refined_results.pt"
            if not refined_path.exists():
                print(f"[SKIP] {json_path.stem} - no refined results found")
                skipped_count += 1
                continue

            try:
                evaluate_sequence(
                    json_path=json_path,
                    results_path=refined_path,
                    evaluator=evaluator,
                    varen=varen,
                    default_cfg=default_cfg,
                    device=device,
                    metric_store=metric_store,
                    compute_3d=definition["compute_3d"],
                    animal3d_mapping=definition["animal3d_mapping"],
                    image_size=cfg.MODEL.IMAGE_SIZE,
                )
                evaluated_count += 1
            except Exception as e:
                print(f"[ERROR] {json_path.stem} - {str(e)}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
        
        print(f"\n{definition['name']}: Evaluated {evaluated_count}/{len(json_files)} sequences (skipped {skipped_count})")
        summarize_and_print(
            dataset_name=definition["name"],
            store=metric_store,
            default_cfg=default_cfg,
            num_sequences=evaluated_count,
        )


if __name__ == "__main__":
    main()

