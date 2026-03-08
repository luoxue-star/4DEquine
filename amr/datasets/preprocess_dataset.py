import os
import json
import re
import cv2
import numpy as np
from tqdm import tqdm


def read_json(json_path: str):
    with open(json_path, 'r') as f:
        return json.load(f)

def write_json(json_path: str, data: object):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def process_apt36k(json_path: str):
    data = read_json(json_path)

    images = data.get('images', [])
    annotations = data.get('annotations', [])

    # Map image_id -> {image_path, video_id}
    image_id_to_info = {}
    for image in images:
        image_id = image.get('id')
        if image_id is None:
            continue
        image_id_to_info[image_id] = {
            'image_path': '/'.join(image.get('file_name').split("\\")[-3:]),
            'video_id': image.get('video_id'),
        }

    # Group horse annotations by (video_id, track_id) and collect their frames
    grouped_by_track = {}
    for ann in annotations:
        # Keep only horse annotations (category_id == 4)
        if ann.get('category_id') != 4:
            continue
        img_id = ann.get('image_id')
        info = image_id_to_info.get(img_id)
        if info is None:
            continue
        video_id = ann.get('video_id')
        track_id = ann.get('track_id')
        if video_id is None or track_id is None:
            continue
        key = (video_id, track_id)
        entry = {
            'image_path': info['image_path'],
            'image_id': img_id,
            'annotation': ann,
        }
        if key not in grouped_by_track:
            grouped_by_track[key] = []
        grouped_by_track[key].append(entry)

    # Convert to a list and sort frames within each track by image_id
    grouped_tracks = []
    for (video_id, track_id), items in grouped_by_track.items():
        items.sort(key=lambda x: x.get('image_id', 0))
        grouped_tracks.append({
            'video_id': video_id,
            'track_id': track_id,
            'frames': items,
        })

    return grouped_tracks


def draw_bbox_and_keypoints(image, annotation):
    bbox = annotation.get('bbox')
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    keypoints = annotation.get('keypoints', [])
    for i in range(0, len(keypoints), 3):
        if i + 2 >= len(keypoints):
            break
        kx, ky, kv = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if kv is None or kv <= 0:
            continue
        if kx is None or ky is None or kx <= 0 or ky <= 0:
            continue
        cx = int(round(kx))
        cy = int(round(ky))
        cv2.circle(image, (cx, cy), 3, (255, 0, 0), 3)
        cv2.putText(image, str(i // 3), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def render_and_save_tracks(grouped_tracks, dataset_root: str, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    for track in tqdm(grouped_tracks, desc="Rendering and saving tracks", total=len(grouped_tracks), dynamic_ncols=True):
        video_id = track.get('video_id')
        track_id = track.get('track_id')
        track_dir = os.path.join(output_root, f"videoid{video_id}_trackid{track_id}")
        os.makedirs(track_dir, exist_ok=True)

        for frame in track.get('frames', []):
            rel_path = frame.get('image_path')
            if not rel_path:
                continue
            rel_path = rel_path.replace('\\', '/')
            abs_path = os.path.join(dataset_root, rel_path)
            if not os.path.exists(abs_path):
                if os.path.isabs(rel_path) and os.path.exists(rel_path):
                    abs_path = rel_path
                else:
                    continue

            image = cv2.imread(abs_path)
            if image is None:
                continue

            vis_image = image.copy()
            draw_bbox_and_keypoints(vis_image, frame.get('annotation', {}))

            out_name = f"image_{frame.get('image_id')}.jpg"
            out_path = os.path.join(track_dir, out_name)
            cv2.imwrite(out_path, vis_image)


def process_animal4d(sequence_root: str):
    sequences = [d for d in os.listdir(sequence_root) if os.path.isdir(os.path.join(sequence_root, d))]
    sequences.sort()

    grouped_tracks = []

    for video_idx, seq_name in enumerate(tqdm(sequences, desc="Parsing Animal4D sequences", dynamic_ncols=True)):
        seq_dir = os.path.join(sequence_root, seq_name)
        # Collect rgb frames
        rgb_files = [f for f in os.listdir(seq_dir) if f.endswith('_rgb.png')]
        # Sort by numeric prefix
        def frame_index(name: str):
            stem = os.path.splitext(name)[0]  # 00000001_rgb
            fid = stem.split('_rgb')[0]
            try:
                return int(re.sub(r'[^0-9]', '', fid))
            except Exception:
                return 0
        rgb_files.sort(key=frame_index)

        frames = []
        for f_idx, rgb_name in enumerate(rgb_files):
            stem = os.path.splitext(rgb_name)[0]
            frame_id_str = stem.split('_rgb')[0]
            keypoint_txt = os.path.join(seq_dir, f"{frame_id_str}_keypoint.txt")
            mask_path = os.path.join(seq_dir, f"{frame_id_str}_mask.png")
            rgb_path = os.path.join(seq_dir, rgb_name)

            if not os.path.exists(rgb_path) or not os.path.exists(mask_path) or not os.path.exists(keypoint_txt):
                continue

            # Compute bbox from mask
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            if mask_img.ndim == 3 and mask_img.shape[2] == 4:
                gray = mask_img[:, :, 3]
            elif mask_img.ndim == 3:
                gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = mask_img
            ys, xs = np.where(gray > 0)
            if ys.size == 0 or xs.size == 0:
                continue
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            bbox_x = float(x_min)
            bbox_y = float(y_min)
            bbox_w = float(x_max - x_min + 1)
            bbox_h = float(y_max - y_min + 1)

            # Read keypoints from txt (each row: x, y, conf)
            keypoints_flat = []
            try:
                with open(keypoint_txt, 'r') as kf:
                    for line in kf:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p for p in re.split(r'[\s,]+', line) if p]
                        if len(parts) < 3:
                            continue
                        try:
                            kx = float(parts[0])
                            ky = float(parts[1])
                            kv = float(parts[2])
                        except Exception:
                            continue
                        keypoints_flat.extend([kx, ky, kv])
            except Exception:
                continue
            if len(keypoints_flat) == 0:
                continue

            # Relative image path to sequence_root
            rel_image_path = '/'.join([seq_name, rgb_name])
            rel_mask_path = '/'.join([seq_name, mask_path.split("/")[-1]])
            try:
                image_id = int(re.sub(r'[^0-9]', '', frame_id_str))
            except Exception:
                image_id = f_idx

            annotation = {
                'bbox': [bbox_x, bbox_y, bbox_w, bbox_h],
                'keypoints': keypoints_flat,
                'track_id': 0,
                'category_id': 4,
                'image_id': image_id,
                'video_id': video_idx,
                'num_keypoints': int(sum(1 for i in range(2, len(keypoints_flat), 3) if keypoints_flat[i] > 0)),
                'is_crowd': 0,
                'area': float(bbox_w * bbox_h),
            }

            frames.append({
                'image_path': rel_image_path,
                'mask_path': rel_mask_path,
                'image_id': image_id,
                'annotation': annotation,
            })

        # Sort frames by image_id
        frames.sort(key=lambda x: x.get('image_id', 0))
        if len(frames) == 0:
            continue
        grouped_tracks.append({
            'video_id': video_idx,
            'track_id': 0,
            'frames': frames,
        })

    return grouped_tracks


def process_varenposer(sequence_root: str):
    sequences = [d for d in os.listdir(sequence_root) if os.path.isdir(os.path.join(sequence_root, d))]
    sequences.sort()

    grouped_tracks = []

    for video_idx, seq_name in enumerate(tqdm(sequences, desc="Parsing VarenPoser sequences", dynamic_ncols=True)):
        seq_dir = os.path.join(sequence_root, seq_name)
        # Find metadata json file in sequence directory
        json_files = [f for f in os.listdir(seq_dir) if f.lower().endswith('.json')]
        if not json_files:
            continue
        meta_path = os.path.join(seq_dir, json_files[0])
        meta = read_json(meta_path)

        img_paths = meta.get('img_path', [])
        mask_paths = meta.get('mask_path', [])
        bboxes = meta.get('bbox', [])
        keypoints_2d_list = meta.get('keypoint_2d', [])
        keypoints_3d_list = meta.get('keypoint_3d', [])
        pose_list = meta.get('pose', [])
        betas_list = meta.get('shape', [])
        transl_list = meta.get('trans', [])

        num_frames = min(len(img_paths), len(keypoints_2d_list))
        frames = []

        def to_rel_under_root(path_str: str):
            if path_str is None:
                return None
            p = str(path_str).replace('\\', '/').lstrip('./')
            if os.path.isabs(p):
                try:
                    abs_p = os.path.abspath(p)
                    root_abs = os.path.abspath(sequence_root)
                    seq_abs = os.path.abspath(seq_dir)
                    common = os.path.commonpath([abs_p, root_abs])
                    if common == root_abs:
                        return os.path.relpath(abs_p, start=sequence_root).replace('\\', '/')
                    common2 = os.path.commonpath([abs_p, seq_abs])
                    if common2 == seq_abs:
                        return os.path.relpath(abs_p, start=sequence_root).replace('\\', '/')
                except Exception:
                    pass
                return '/'.join([seq_name, os.path.basename(p)])
            if p.startswith(seq_name + '/'):
                return p
            return '/'.join([seq_name, p])

        for i in range(num_frames):
            img_rel = to_rel_under_root(img_paths[i])
            mask_rel = to_rel_under_root(mask_paths[i]) if i < len(mask_paths) else None
            if img_rel is None:
                continue

            # Determine bbox: use provided if valid, else compute from mask, else from keypoints
            bbox = bboxes[i] if i < len(bboxes) else None
            if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                # Try from mask
                bbox = None
                if mask_rel is not None:
                    abs_mask = os.path.join(sequence_root, mask_rel)
                    if os.path.exists(abs_mask):
                        m = cv2.imread(abs_mask, cv2.IMREAD_UNCHANGED)
                        if m is not None:
                            if m.ndim == 3 and m.shape[2] == 4:
                                gray = m[:, :, 3]
                            elif m.ndim == 3:
                                gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                            else:
                                gray = m
                            ys, xs = np.where(gray > 0)
                            if ys.size > 0 and xs.size > 0:
                                y_min, y_max = int(ys.min()), int(ys.max())
                                x_min, x_max = int(xs.min()), int(xs.max())
                                bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
                # Fallback from keypoints
                if bbox is None:
                    kps = keypoints_2d_list[i]
                    try:
                        kps_arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
                        valid = kps_arr[:, 2] > 0
                        if valid.any():
                            xs = kps_arr[valid, 0]
                            ys = kps_arr[valid, 1]
                            x_min, x_max = float(xs.min()), float(xs.max())
                            y_min, y_max = float(ys.min()), float(ys.max())
                            bbox = [x_min, y_min, x_max - x_min + 1.0, y_max - y_min + 1.0]
                    except Exception:
                        pass
            if bbox is None:
                continue

            # Keypoints: ensure flat [x,y,v,...]
            kps = keypoints_2d_list[i]
            if isinstance(kps, list) and len(kps) > 0 and not isinstance(kps[0], (int, float)):
                flat_kps = []
                for pt in kps:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                        flat_kps.extend([float(pt[0]), float(pt[1]), float(pt[2])])
                keypoints_flat = flat_kps
            else:
                # Already flat
                keypoints_flat = [float(v) for v in kps]
            if len(keypoints_flat) == 0:
                continue

            # image_id from filename digits; fallback to index
            base_name = os.path.basename(str(img_paths[i]))
            digits = re.findall(r'\d+', base_name)
            if digits:
                try:
                    image_id = int(digits[-1])
                except Exception:
                    image_id = i
            else:
                image_id = i

            # Build annotation
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            annotation = {
                'bbox': [x, y, w, h],
                'keypoints': keypoints_flat,
                'track_id': 0,
                'category_id': 4,
                'image_id': image_id,
                'video_id': video_idx,
                'num_keypoints': int(sum(1 for j in range(2, len(keypoints_flat), 3) if keypoints_flat[j] > 0)),
                'is_crowd': 0,
                'area': float(max(0.0, w) * max(0.0, h)),
                'keypoints_3d': keypoints_3d_list[i],
                'global_orient': pose_list[i][:3],
                'pose': pose_list[i][3:],
                'betas': betas_list[i],
                'transl': transl_list[i],
            }

            frames.append({
                'image_path': img_rel,
                'mask_path': mask_rel,
                'image_id': image_id,
                'annotation': annotation,
            })

        frames.sort(key=lambda x: x.get('image_id', 0))
        if len(frames) == 0:
            continue
        grouped_tracks.append({
            'video_id': video_idx,
            'track_id': 0,
            'frames': frames,
        })

    return grouped_tracks 


if __name__ == "__main__":
    # input_json = "/data2/lvjin/cvpr26/data/AP-36k-patr1/apt36k_annotations.json"
    # grouped_tracks = process_apt36k(input_json)
    # output_json = os.path.join(os.path.dirname(input_json), "apt36k_horse_tracks.json")
    # write_json(output_json, grouped_tracks)
    # output_dir = os.path.join("outputs/", "apt36k_horse_tracks")
    # render_and_save_tracks(grouped_tracks, os.path.dirname(input_json), output_dir)

    # sequence_root = "/data2/lvjin/cvpr26/data/Animal4D/horse"
    # animal4d_tracks = process_animal4d(sequence_root)
    # os.makedirs("outputs/", exist_ok=True)
    # out_json = os.path.join(sequence_root, "animal4d_horse_tracks.json")
    # write_json(out_json, animal4d_tracks)
    # render_and_save_tracks(animal4d_tracks, sequence_root, os.path.join("outputs/", "animal4d_horse_tracks"))

    sequence_root = "/data2/lvjin/cvpr26/data/VarenPoser/testseq"
    varen_tracks = process_varenposer(sequence_root)
    out_json = os.path.join(sequence_root, "varenposer_horse_tracks.json")
    write_json(out_json, varen_tracks)
    render_and_save_tracks(varen_tracks, sequence_root, os.path.join("outputs/", "varenposer_horse_tracks"))