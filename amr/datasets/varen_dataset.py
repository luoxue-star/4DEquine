import copy
import os
import numpy as np
from yacs.config import CfgNode
import cv2
import re
import math
import torch
import logging
import torchvision.transforms as transforms
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import glob
import io
import webdataset as wds
import tarfile
import json
import random
from tqdm import tqdm
from PIL import Image
from typing import Iterable, List, Dict
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from amr.datasets.utils import (get_example, 
                                expand_to_aspect_ratio, 
                                do_augmentation, 
                                get_example_video,
                                gen_trans_from_patch_cv,
                                convert_cvimg_to_tensor)



class VarenPoserWebDataset:
    def __init__(self, cfg, dataset_name, is_train: bool):
        self.dataset_name = dataset_name
        self.focal_length = cfg.SMAL.get("FOCAL_LENGTH", 5000)
        
        # Get tar file pattern (support both single file and sharded patterns)
        if is_train:
            self.tar_pattern = cfg.DATASETS[dataset_name].TAR_PATTERN.TRAIN
        else:
            self.tar_pattern = cfg.DATASETS[dataset_name].TAR_PATTERN.TEST
        
        self.mask_iterator = None
        self.mask_webdataset = None
        self.mask_prob = cfg.DATASETS[dataset_name].get('MASK_PROB', 0.0)
        
        self.is_train = is_train
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

        self.estimated_size = self.estimate_dataset_size()
        print(f"Estimated dataset size: {self.estimated_size}")

    def estimate_dataset_size(self):
        """Estimate the total number of samples in the dataset"""
        try:
            # Try to load dataset info if available
            if "{" in self.tar_pattern:
                # Extract directory from pattern
                pattern_dir = os.path.dirname(self.tar_pattern)
                info_file = os.path.join(pattern_dir, "dataset_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    return info['total_samples']
            
            # Fallback: count samples in first shard/file
            import glob
            tar_files = glob.glob(self.tar_pattern.replace("{", "").replace("}", "").replace("..", "*"))
            if tar_files:
                with tarfile.open(tar_files[0], 'r') as tar:
                    json_files = [name for name in tar.getnames() if name.endswith('.json')]
                    samples_per_shard = len(json_files)
                    return samples_per_shard * len(tar_files)
            
            return 10000  # Default fallback
        except:
            return 10000  # Default fallback

    def get_random_mask(self):
        """Get a random mask image, with fallback to WebDataset if needed"""
        # If no preloaded masks, try to get from WebDataset iterator
        if self.mask_iterator:
            try:
                return next(self.mask_iterator)
            except StopIteration:
                # Reset iterator if exhausted
                self.mask_iterator = iter(self.mask_webdataset)
                try:
                    return next(self.mask_iterator)
                except StopIteration:
                    return None
        
        return None
    
    def get_object_bbox(self, mask_img):
        """Extract bounding box of the object from mask image"""
        if mask_img.shape[2] == 4:  # RGBA
            alpha_mask = mask_img[:, :, 3]
        else:  # RGB - use brightness
            alpha_mask = np.mean(mask_img, axis=2)
        
        coords = np.where(alpha_mask > 0.1 * 255)
        
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    def apply_random_mask(self, img_rgba, mask_patch, keypoints_2d, horse_bbox):
        """Apply random masking using one of the loaded mask images within horse bbox"""
        if random.random() > self.mask_prob:
            return img_rgba, mask_patch, keypoints_2d
        
        # Get a random mask image
        mask_img = self.get_random_mask()
        if mask_img is None:
            return img_rgba, mask_patch, keypoints_2d
        
        mask_img = mask_img.copy()
        
        # Get object bbox from mask image
        obj_bbox = self.get_object_bbox(mask_img)
        if obj_bbox is None:
            return img_rgba, mask_patch, keypoints_2d
        
        # Extract object region from mask image
        obj_x, obj_y, obj_w, obj_h = obj_bbox
        obj_region = mask_img[obj_y:obj_y+obj_h, obj_x:obj_x+obj_w]
        
        # Horse bbox dimensions
        horse_x, horse_y, horse_w, horse_h = horse_bbox
        
        # Calculate scaling factor to make object cover only part of horse (20% to 60% of horse size)
        coverage_factor = random.uniform(0.4, 1)
        
        # Calculate target size for object to cover specified portion of horse
        target_w = int(horse_w * coverage_factor)
        target_h = int(horse_h * coverage_factor)
        
        # Maintain aspect ratio of object while fitting within target size
        obj_aspect = obj_w / obj_h
        target_aspect = target_w / target_h
        
        if obj_aspect > target_aspect:
            # Object is wider, scale by width
            final_w = target_w
            final_h = int(target_w / obj_aspect)
        else:
            # Object is taller, scale by height
            final_h = target_h
            final_w = int(target_h * obj_aspect)
        
        # Ensure final size is valid
        if final_w <= 0 or final_h <= 0:
            return img_rgba, mask_patch, keypoints_2d
        
        # Resize object to final size
        obj_resized = cv2.resize(obj_region, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
        
        # Random position within horse bbox
        max_x = max(0, horse_w - final_w)
        max_y = max(0, horse_h - final_h)
        
        if max_x <= 0 or max_y <= 0:
            return img_rgba, mask_patch, keypoints_2d
        
        rel_x = random.randint(0, max_x)
        rel_y = random.randint(0, max_y)
        
        # Absolute position in full image
        abs_x = horse_x + rel_x
        abs_y = horse_y + rel_y
        
        # Get image dimensions
        img_h, img_w = img_rgba.shape[:2]
        
        # Ensure position is within image bounds
        abs_x = max(0, min(abs_x, img_w - final_w))
        abs_y = max(0, min(abs_y, img_h - final_h))
        
        # Extract alpha channel as mask
        if obj_resized.shape[2] == 4:  # RGBA
            alpha_mask = obj_resized[:, :, 3] / 255.0
            rgb_mask = obj_resized[:, :, :3]
        else:  # RGB - use brightness as mask
            alpha_mask = np.mean(obj_resized, axis=2) / 255.0
            rgb_mask = obj_resized
        
        # Create binary mask (threshold to make it more distinct)
        binary_mask = (alpha_mask > 0.5).astype(np.float32)
        
        # Calculate actual region that will be modified
        end_y = min(abs_y + final_h, img_h)
        end_x = min(abs_x + final_w, img_w)
        actual_h = end_y - abs_y
        actual_w = end_x - abs_x
        
        # Create occlusion mask for keypoint visibility
        occlusion_mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        if actual_h > 0 and actual_w > 0:
            # Crop the mask to fit the actual region
            region_mask = binary_mask[:actual_h, :actual_w]
            region_rgb = rgb_mask[:actual_h, :actual_w]
            
            # Create mask for blending
            mask_3d = np.stack([region_mask] * 3, axis=2)
            
            # Apply the mask to RGB channels
            img_rgba[abs_y:end_y, abs_x:end_x, :3] = (
                img_rgba[abs_y:end_y, abs_x:end_x, :3] * (1 - mask_3d) + 
                region_rgb * mask_3d
            )
            
            # Update the alpha channel (horse mask) - occluded areas become invisible
            mask_patch[abs_y:end_y, abs_x:end_x] = (
                mask_patch[abs_y:end_y, abs_x:end_x] * (1 - region_mask)
            )
            
            # Update occlusion mask
            occlusion_mask[abs_y:end_y, abs_x:end_x] = region_mask
        
        # Update keypoint visibility based on occlusion
        # keypoints_2d_updated = keypoints_2d.copy()
        # for i, keypoint in enumerate(keypoints_2d):
        #     if len(keypoint) >= 3:  # Has visibility information
        #         x, y = int(keypoint[0]), int(keypoint[1])
        #         # Check if keypoint is within image bounds
        #         if 0 <= x < img_w and 0 <= y < img_h:
        #             # Check if keypoint is occluded (with small radius for robustness)
        #             radius = 3  # Check 3x3 area around keypoint
        #             y_min, y_max = max(0, y - radius), min(img_h, y + radius + 1)
        #             x_min, x_max = max(0, x - radius), min(img_w, x + radius + 1)
                    
        #             # If any part of the keypoint area is occluded, mark as invisible
        #             if np.any(occlusion_mask[y_min:y_max, x_min:x_max] > 0.5):
        #                 keypoints_2d_updated[i, 2] = 0.0  # Set visibility to 0
        
        return img_rgba, mask_patch, keypoints_2d

    def decode_sample(self, sample):
        """Decode a sample from WebDataset format"""
        try:
            # Handle both .jpg and .png extensions
            if 'jpg' in sample:
                image_data = sample['jpg']
            elif 'png' in sample:
                image_data = sample['png']
            else:
                raise ValueError("No image data found in sample")
                
            # Load image
            image = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))
            
            # Load mask
            mask = np.array(Image.open(io.BytesIO(sample['mask.png'])).convert('L'))
            
            # Load metadata
            metadata = json.loads(sample['json'].decode('utf-8'))
            
            # Extract data from metadata
            keypoint_2d = np.array(metadata['keypoint_2d'], dtype=np.float32)
            keypoint_3d = np.array(metadata['keypoint_3d'], dtype=np.float32)
            apply_mask = False
            if keypoint_3d.shape[1] == 3:
                keypoint_3d = np.concatenate((keypoint_3d, np.ones((len(keypoint_3d), 1))), axis=-1).astype(np.float32)
                apply_mask = True  # apply mask if dataset is varenposer

            bbox = metadata['bbox']  # [x, y, w, h]
            center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
            pose = np.array(metadata['pose'], dtype=np.float32) if metadata['pose'] else np.zeros(114, dtype=np.float32)
            betas = np.array(metadata['shape'], dtype=np.float32) if metadata['shape'] else np.zeros(39, dtype=np.float32)
            translation = np.array(metadata['trans'], dtype=np.float32) if metadata['trans'] else np.zeros(3, dtype=np.float32)
            
            has_pose = np.array(1., dtype=np.float32) if not (pose == 0).all() else np.array(0., dtype=np.float32)
            has_betas = np.array(1., dtype=np.float32) if not (betas == 0).all() else np.array(0., dtype=np.float32)
            has_translation = np.array(1., dtype=np.float32) if not (translation == 0).all() else np.array(0., dtype=np.float32)
            
            ori_keypoint_2d = keypoint_2d.copy()
            center_x, center_y = center[0], center[1]
            bbox_size = max([bbox[2], bbox[3]])

            smal_params = {'global_orient': pose[:3],
                           'pose': pose[3:],
                           'betas': betas,
                           'transl': translation,
                           }
            has_smal_params = {'global_orient': has_pose,
                               'pose': has_pose,
                               'betas': has_betas,
                               'transl': has_translation,
                               }

            augm_config = copy.deepcopy(self.augm_config)
            img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
            
            # Apply random masking before cropping and augmentation
            if self.is_train and apply_mask:
                img_rgba, mask_modified, keypoint_2d = self.apply_random_mask(
                    img_rgba, mask.astype(np.float32) / 255.0, keypoint_2d, bbox
                )
                # Update the alpha channel with the modified mask
                img_rgba[:, :, 3] = (mask_modified * 255).astype(np.uint8)
            
            # Call your existing get_example function
            img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans, img_border_mask = get_example(
                img_rgba,
                center_x, center_y,
                bbox_size, bbox_size,
                keypoint_2d, keypoint_3d,
                smal_params, has_smal_params,
                self.IMG_SIZE, self.IMG_SIZE,
                self.MEAN, self.STD, self.is_train, augm_config,
                is_bgr=False, return_trans=True,
                use_skimage_antialias=self.use_skimage_antialias,
                border_mode=self.border_mode
            )
            
            img_patch = (img_patch_rgba[:3, :, :])
            mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
            if (mask_patch < 0.5).all():
                mask_patch = np.ones_like(mask_patch)

            return {'img': img_patch,
                    'mask': mask_patch,
                    'keypoints_2d': keypoints_2d,
                    'keypoints_3d': keypoints_3d,
                    'orig_keypoints_2d': ori_keypoint_2d,
                    'box_center': np.array(center.copy(), dtype=np.float32),
                    'box_size': float(bbox_size),
                    'img_size': np.array(1.0 * img_size[::-1].copy(), dtype=np.float32),
                    'global_orient': smal_params['global_orient'],
                    'pose': smal_params['pose'],
                    'betas': smal_params['betas'],
                    'has_global_orient': has_smal_params['global_orient'],
                    'has_pose': has_smal_params['pose'],
                    'has_betas': has_smal_params['betas'],
                    'focal_length': np.array([self.focal_length, self.focal_length], dtype=np.float32),}
        
        except Exception as e:
            print(f"Error decoding sample: {e}")
            # Return a dummy sample or re-raise the exception
            raise e

    def create_webdataset(self):
        """Create WebDataset from tar files with proper length handling"""
        # Create the base dataset
        dataset = wds.WebDataset(self.tar_pattern, shardshuffle=True,)  # Shuffle for training
        
        # Add decoding and processing
        dataset = dataset.map(self.decode_sample)  # Apply our decoding function

        # Add shuffling for training
        if self.is_train:
            dataset = dataset.shuffle(100)
        return dataset


class VarenPoserVideoWebDataset(VarenPoserWebDataset):
    def __init__(self, cfg, dataset_name, is_train: bool):
        assert is_train, "VarenPoserVideoWebDataset is only support for training"
        super().__init__(cfg, dataset_name, is_train)
        self.cfg = cfg
        self.img_key = [f"{i:04d}.jpg" for i in range(256)] if "V2" not in dataset_name else [f"{i:04d}.jpg" for i in range(600)]
        self.mask_key = [f"mask.{i:04d}.png" for i in range(256)] if "V2" not in dataset_name else [f"mask.{i:04d}.png" for i in range(600)]
        self.fps_choices = [4, 5, 6, 10, 15, 20, 30, 60] if "V2" not in dataset_name else [2, 3, 4, 5, 6, 10, 15, 20, 30, 60]

    #     self.webdataset = wds.WebDataset(self.tar_pattern, shardshuffle=False).map(self.cache_decode)
    #     self.metadata = dict()
    #     print("Cache decoding...")
    #     for key, sample in tqdm(self.webdataset, total=self.estimate_dataset_size()):
    #         self.metadata[key] = sample
    #     print("Cache decoding done")

    # def cache_decode(self, sample):
    #     meta = json.loads(sample['meta.json'].decode('utf-8'))
    #     return sample['__key__'], meta

    def estimate_dataset_size(self):
        return len(os.listdir(os.path.dirname(self.tar_pattern)))

    def sample_video_frames(self,
                            video_len: int = 256,
                            num_frames: int = 16,
                            orig_fps: float = 60.0,
                            target_fps: float = None,
    ) -> np.ndarray:
        """
        Simulate different frame rates by uniform sampling along the time axis.
        Args:
            video_len (int): Length of the video in frames (default: 256).
            num_frames (int): Number of frames to sample (default: 16).
            orig_fps (float): Original frame rate of the video (default: 60.0).
            target_fps (float, optional): Desired frame rate to simulate.
                If None, randomly chosen from fps_choices.
        Returns:
            np.ndarray: Sampled video indices of shape (num_frames,).
        """
        # 1. Randomly choose target fps if not provided
        if target_fps is None:
            target_fps = float(np.random.choice(self.fps_choices))

        # 2. Compute sampling step in frame units
        step = orig_fps / target_fps  # e.g., 60/15 = 4 → pick every 4th frame on average

        # 3. Compute maximum valid start so that start + step*(num_frames-1) < T
        max_start = video_len - step * (num_frames - 1)
        if max_start <= 0:
            start = 0.0
        else:
            start = float(np.random.uniform(0, max_start))

        # 4. Generate indices and clamp to [0, T-1]
        indices = [int(start + step * i) for i in range(num_frames)]
        indices = np.clip(indices, 0, video_len-1)

        return indices

    # def decode_sample(self, sample):
    #     # Acquire a video sequence with annotations
    #     try:
    #         meta = json.loads(sample['meta.json'].decode('utf-8'))
    #         # meta = self.metadata[sample['__key__']]
    #         sample_indices = self.sample_video_frames(video_len=len(self.img_key), num_frames=self.cfg.TRAIN.SEQ_LEN)
    #         frames = [sample[self.img_key[i]] for i in sample_indices]
    #         masks = [sample[self.mask_key[i]] for i in sample_indices]

    #         items = []
    #         augm_config = do_augmentation(self.augm_config)
    #         trans = self.get_trans(augm_config)
    #         for i in range(self.cfg.TRAIN.SEQ_LEN):
    #             item = dict()
    #             # Decode the video sequence
    #             video_frame = np.array(Image.open(io.BytesIO(frames[i])).convert('RGB'))
    #             video_mask = np.array(Image.open(io.BytesIO(masks[i])).convert('L'))
    #             keypoint_2d = np.array(meta['keypoint_2d'][sample_indices[i]], dtype=np.float32)
    #             keypoint_3d = np.array(meta['keypoint_3d'][sample_indices[i]], dtype=np.float32)
    #             apply_mask = False
    #             if keypoint_3d.shape[1] == 3:
    #                 keypoint_3d = np.concatenate((keypoint_3d, np.ones((len(keypoint_3d), 1))), axis=-1).astype(np.float32)
    #                 apply_mask = True  # apply mask if dataset is varenposer
    #             bbox = meta['bbox'][sample_indices[i]]
    #             center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
    #             bbox_size = max([bbox[2], bbox[3]])
    #             pose = np.array(meta['pose'][sample_indices[i]], dtype=np.float32) if meta['pose'][sample_indices[i]] else np.zeros(114, dtype=np.float32)
    #             betas = np.array(meta['shape'][sample_indices[i]], dtype=np.float32) if meta['shape'][sample_indices[i]] else np.zeros(39, dtype=np.float32)
    #             transl = np.array(meta['trans'][sample_indices[i]], dtype=np.float32) if meta['trans'][sample_indices[i]] else np.zeros(3, dtype=np.float32)
    #             has_pose = np.array(1., dtype=np.float32) if not (pose == 0).all() else np.array(0., dtype=np.float32)
    #             has_betas = np.array(1., dtype=np.float32) if not (betas == 0).all() else np.array(0., dtype=np.float32)
    #             has_translation = np.array(1., dtype=np.float32) if not (transl == 0).all() else np.array(0., dtype=np.float32)

    #             ori_keypoint_2d = keypoint_2d.copy()
    #             center_x, center_y = center[0], center[1]
    #             bbox_size = max([bbox[2], bbox[3]])

    #             smal_params = {'global_orient': pose[:3],
    #                         'pose': pose[3:],
    #                         'betas': betas,
    #                         'transl': transl,
    #                         }
    #             has_smal_params = {'global_orient': has_pose,
    #                             'pose': has_pose,
    #                             'betas': has_betas,
    #                             'transl': has_translation,
    #                             }
    #             img_rgba = np.concatenate([video_frame, video_mask[:, :, None]], axis=2)

    #             # Apply random masking before cropping and augmentation
    #             if self.is_train and apply_mask:
    #                 img_rgba, mask_modified, keypoint_2d = self.apply_random_mask(
    #                     img_rgba, video_mask.astype(np.float32) / 255.0, keypoint_2d, bbox
    #                 )
    #                 # Update the alpha channel with the modified mask
    #                 img_rgba[:, :, 3] = (mask_modified * 255).astype(np.uint8)

    #             img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans, img_border_mask = get_example_video(
    #                 img_rgba,
    #                 center_x, center_y,
    #                 bbox_size, bbox_size,
    #                 keypoint_2d, keypoint_3d,
    #                 smal_params, has_smal_params,
    #                 self.IMG_SIZE, self.IMG_SIZE,
    #                 self.MEAN, self.STD, self.is_train, augm_config,
    #                 is_bgr=False, return_trans=True,
    #                 use_skimage_antialias=self.use_skimage_antialias,
    #                 border_mode=self.border_mode
    #             )
                
    #             img_patch = (img_patch_rgba[:3, :, :])
    #             mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
    #             if (mask_patch < 0.5).all():
    #                 mask_patch = np.ones_like(mask_patch)

    #             item['img'] = img_patch  # (T, C, H, W)
    #             item['mask'] = mask_patch  # (T, H, W)
    #             item['keypoints_2d'] = keypoints_2d  # (T, 21, 3)
    #             item['keypoints_3d'] = keypoints_3d  # (T, 21, 4)
    #             item['orig_keypoints_2d'] = ori_keypoint_2d  # (T, 21, 3)
    #             item['box_center'] = np.array(center.copy(), dtype=np.float32)  # (T, 2)
    #             item['box_size'] = float(bbox_size)  # (T,)
    #             item['focal_length'] = np.array([self.focal_length, self.focal_length], dtype=np.float32)  # (T, 2)
    #             item['img_size'] = np.array(1.0 * img_size[::-1].copy(), dtype=np.float32)  # (T, 2)
    #             item['global_orient'] = smal_params['global_orient']  # (T, 3)
    #             item['pose'] = smal_params['pose']  # (T, 111)
    #             item['betas'] = smal_params['betas']  # (T, 39)
    #             item['has_global_orient'] = has_smal_params['global_orient']  # (T,)
    #             item['has_pose'] = has_smal_params['pose']  # (T,)
    #             item['has_betas'] = has_smal_params['betas']  # (T,)

    #             items.append(item)

    #         keys = [k for k in item]
    #         sequence = {k: np.stack([item[k] for item in items]) for k in keys}
    #         return sequence
    #     except Exception as e:
    #         print(f"Error decoding sample {sample['__url__']}: {e}")

    def decode_sample(self, sample):
        try:
            meta = json.loads(sample['meta.json'].decode('utf-8'))
            # meta = self.metadata[sample['__key__']]
            # Augmentation is all the same for all frames
            augm_config = do_augmentation(self.augm_config)
            trans = self.get_trans(augm_config)

            items = dict(img=[], 
                        mask=[], 
                        keypoints_2d=[], 
                        keypoints_3d=[], 
                        orig_keypoints_2d=[], 
                        box_center=np.ones(shape=(self.cfg.TRAIN.SEQ_LEN, 2), dtype=np.float32) * self.IMG_SIZE // 2, 
                        box_size=np.ones(shape=(self.cfg.TRAIN.SEQ_LEN,), dtype=np.float32) * self.IMG_SIZE, 
                        focal_length=np.ones(shape=(self.cfg.TRAIN.SEQ_LEN, 2), dtype=np.float32) * self.focal_length, 
                        img_size=np.ones(shape=(self.cfg.TRAIN.SEQ_LEN, 2), dtype=np.float32) * self.IMG_SIZE, 
                        global_orient=[], 
                        pose=[], 
                        betas=[], 
                        has_global_orient=np.ones(self.cfg.TRAIN.SEQ_LEN, dtype=np.float32), 
                        has_pose=np.ones(self.cfg.TRAIN.SEQ_LEN, dtype=np.float32), 
                        has_betas=np.ones(self.cfg.TRAIN.SEQ_LEN, dtype=np.float32))
            sample_indices = self.sample_video_frames(video_len=len(self.img_key), num_frames=self.cfg.TRAIN.SEQ_LEN)
            for i in range(self.cfg.TRAIN.SEQ_LEN):
                frame = np.array(Image.open(io.BytesIO(sample[self.img_key[sample_indices[i]]])).convert('RGB'))
                mask = np.array(Image.open(io.BytesIO(sample[self.mask_key[sample_indices[i]]])).convert('L'))
                img_rgba = np.concatenate([frame, mask[:, :, None]], axis=2)

                # Apply random mask
                img_rgba, mask_modified, keypoint_2d = self.apply_random_mask(
                    img_rgba, mask.astype(np.float32) / 255.0, meta['keypoint_2d'][sample_indices[i]], meta['bbox'][sample_indices[i]]
                )
                # Update the alpha channel with the modified mask
                img_rgba[:, :, 3] = (mask_modified * 255).astype(np.uint8)

                # Transform image
                img_patch_rgba = self.transform_image(img_rgba, trans)
                img_patch_rgba = convert_cvimg_to_tensor(img_patch_rgba)
                for n_c in range(min(img_patch_rgba.shape[0], 3)):
                    img_patch_rgba[n_c, :, :] = np.clip(img_patch_rgba[n_c, :, :] * augm_config[-3][n_c], 0, 255)
                    if self.MEAN is not None and self.STD is not None:
                        img_patch_rgba[n_c, :, :] = (img_patch_rgba[n_c, :, :] - self.MEAN[n_c]) / self.STD[n_c]
                img_patch = (img_patch_rgba[:3, :, :])
                mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
                if (mask_patch < 0.5).all():
                    mask_patch = np.ones_like(mask_patch)

                items['img'].append(img_patch)
                items['mask'].append(mask_patch)
                items['keypoints_2d'].append(meta['keypoint_2d'][sample_indices[i]])
                items['keypoints_3d'].append(meta['keypoint_3d'][sample_indices[i]])
                items['orig_keypoints_2d'].append(meta['keypoint_2d'][sample_indices[i]])
                items['global_orient'].append(meta['pose'][sample_indices[i]][:3])
                items['pose'].append(meta['pose'][sample_indices[i]][3:])
                items['betas'].append(meta['shape'][sample_indices[i]])

            items['img'] = np.stack(items['img'])
            items['mask'] = np.stack(items['mask'])
            items['keypoints_2d'] = np.stack(items['keypoints_2d']).astype(np.float32)
            items['keypoints_3d'] = np.stack(items['keypoints_3d']).astype(np.float32)
            items['orig_keypoints_2d'] = np.stack(items['orig_keypoints_2d']).astype(np.float32)
            items['global_orient'] = np.stack(items['global_orient']).astype(np.float32)
            items['pose'] = np.stack(items['pose']).astype(np.float32)
            items['betas'] = np.stack(items['betas']).astype(np.float32)

            # Transform 2d keypoints
            items['keypoints_2d'] = self.transform_keypoints_2d(items['keypoints_2d'], trans).astype(np.float32)
            # Transform 3d keypoints
            items['keypoints_3d'] = self.transform_keypoints_3d(items['keypoints_3d'], augm_config[1]).astype(np.float32)
            # Transform smal params
            items['global_orient'] = self.transform_smal_params(items['global_orient'], augm_config[1]).astype(np.float32)
            
            return items

        except Exception as e:
            print(f"Error decoding sample: {e}")
            return None

    def transform_keypoints_2d(self, keypoints_2d: np.ndarray, trans: np.ndarray):
        """
        Args:
            keypoints_2d: (N, 21, 3)
            trans: (2, 3)
        Returns:
            keypoints_2d: (N, 21, 3)
        """
        coords = keypoints_2d[:, :, :2]
        vis = keypoints_2d[:, :, [2]]
        coords_homo = np.concatenate((coords, np.ones((coords.shape[0], coords.shape[1], 1))), axis=-1)
        coords_homo = np.einsum("ij, kpj->kpi", trans, coords_homo)  # ij:[2, 3] kpj:[N, 21, 3] -> kpi:[N, 21, 2]
        coords = coords_homo / self.IMG_SIZE - 0.5
        return np.concatenate((coords, vis), axis=-1)

    def transform_keypoints_3d(self, keypoints_3d: np.ndarray, rot: float):
        """
        Args:
            keypoints_3d: (N, 21, 3)
            rot: float
        Returns:
            keypoints_3d: (N, 21, 4)
        """
        ones = np.ones((keypoints_3d.shape[0], keypoints_3d.shape[1], 1), dtype=keypoints_3d.dtype)
        if rot == 0:
            return np.concatenate((keypoints_3d, ones), axis=-1)

        rot_mat = np.eye(3, dtype=np.float32)
        rot_rad = -rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        keypoints_3d = np.einsum('ij,kpj->kpi', rot_mat, keypoints_3d)  # ij:[3, 3], kpj:[N, 21, 3] -> kpi:[N, 21, 3]
        return np.concatenate((keypoints_3d, ones), axis=-1)

    def transform_smal_params(self, global_orient: np.ndarray, rot: float):
        """
        Args:
            global_orient: (N, 3)
            rot: float
        Returns:
            global_orient: (N, 3)
        """
        if rot == 0:
            return global_orient

        R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                    [0, 0, 1]], dtype=np.float32)
        global_mat = Rotation.from_rotvec(global_orient).as_matrix()  # (N, 3, 3)
        global_mat = np.einsum("ij, kjp->kip", R, global_mat)  # ij:[3, 3], kjp:[N, 3, 3] -> kip: [N, 3, 3]
        global_mat = Rotation.from_matrix(global_mat)
        return global_mat.as_rotvec()

    def get_trans(self, augm_config):
        center_x, center_y, width, height, patch_width, patch_height = self.IMG_SIZE // 2, self.IMG_SIZE // 2, self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.IMG_SIZE
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = augm_config
        # scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = 1.0, 0, False, False, 0, [1.0,
        #                                                                                                         1.0,
        #                                                                                                         1.0], 0., 0.
        center_x += width * tx
        center_y += height * ty
        trans = gen_trans_from_patch_cv(center_x, center_y, width, height, patch_width, patch_height, scale, rot)
        return trans

    def transform_image(self, image: np.ndarray, trans: np.ndarray):
        """
        Args:
            image: (H, W, 3)
            trans: (2, 3)
        Returns:
            img_patch_cv: (H, W, 3)
        """
        img_patch_cv = cv2.warpAffine(  image, trans, (self.IMG_SIZE, self.IMG_SIZE),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=self.border_mode,
                                        borderValue=0,
                                        )
        # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
        if (image.shape[2] == 4) and (self.border_mode != cv2.BORDER_CONSTANT):
            img_patch_cv[:, :, 3] = cv2.warpAffine( image[:, :, 3], trans, (self.IMG_SIZE, self.IMG_SIZE),
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT)
        return img_patch_cv

    def create_webdataset(self):
        """Create WebDataset from tar files with proper length handling"""
        # Create the base dataset
        dataset = wds.WebDataset(self.tar_pattern, shardshuffle=100)  # Shuffle for training
        dataset = dataset.map(self.decode_sample)
        return dataset


class VarenEvalDataset(Dataset):
    def __init__(self, cfg: CfgNode, dataset_name: str="ANIMAL3D"):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_image = cfg.DATASETS[dataset_name].ROOT_IMAGE
        self.focal_length = cfg.SMAL.get("FOCAL_LENGTH", 5000)

        json_file = cfg.DATASETS[dataset_name].JSON_FILE.TEST
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = False
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L'))
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32) if 'keypoint_2d' in data else np.zeros((21, 3), dtype=np.float32)
        keypoint_3d = np.concatenate(
            (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32) if 'keypoint_3d' in data else np.zeros((21, 4), dtype=np.float32)

        bbox = data['bbox']  # [x, y, w, h]
        center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
        pose = np.zeros(114, dtype=np.float32)
        betas = np.zeros(39, dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
        has_pose = np.array(1., dtype=np.float32) if not (pose == 0).all() else np.array(0., dtype=np.float32)
        has_betas = np.array(1., dtype=np.float32) if not (betas == 0).all()  else np.array(0., dtype=np.float32)
        has_translation = np.array(1., dtype=np.float32) if not (translation == 0).all() else np.array(0., dtype=np.float32)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max([bbox[2], bbox[3]])

        smal_params = {'global_orient': pose[:3],
                       'pose': pose[3:],
                       'betas': betas,
                       'transl': translation,
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           }
        smal_params_is_axis_angle = {'global_orient': True,
                                     'pose': True,
                                     'betas': False,
                                     'transl': False,
                                     }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans, img_border_mask = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {'img': img_patch,
                'mask': mask_patch,
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'box_center': np.array(center.copy(), dtype=np.float32),
                'box_size': float(bbox_size),
                'img_size': np.array(1.0 * img_size[::-1].copy(), dtype=np.float32),
                'global_orient': smal_params['global_orient'],
                'pose': smal_params['pose'],
                'betas': smal_params['betas'],
                'has_global_orient': has_smal_params['global_orient'],
                'has_pose': has_smal_params['pose'],
                'has_betas': has_smal_params['betas'],
                'focal_length': np.array([self.focal_length, self.focal_length], dtype=np.float32),
                "img_border_mask": img_border_mask.astype(np.float32),
                "has_mask": np.array(0, dtype=np.float32)}
        return item


def create_dataloader(dataset, batch_size=16, num_workers=1, prefetch_factor=2, pin_memory=True, is_train=True):
    """Create DataLoader for the WebDataset"""    
    return wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=is_train,
        persistent_workers=True,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


class MaskWebDataset:
    def __init__(self, cfg: CfgNode):
        self.mask_pattern = cfg.MASK_PATTERN
        self.mask_webdataset = None
        self.mask_iterator = None
        self.load_mask_images()

    def load_mask_images(self):
        """Load mask images from tar file or sharded pattern"""
        if self.mask_pattern:
            # Handle sharded mask pattern
            print(f"Loading masks from sharded pattern: {self.mask_pattern}")
            try:
                max_preload = 100  # Number of masks to preload into memory 
                # Create WebDataset for mask loading
                self.mask_webdataset = (
                    wds.WebDataset(self.mask_pattern, shardshuffle=10).map(self._decode_mask_sample).shuffle(max_preload)
                )
                
                # Keep the WebDataset for additional mask loading during training
                self.mask_iterator = iter(self.mask_webdataset)
                
            except Exception as e:
                print(f"Error loading sharded masks: {e}")
        else:
            print("No mask pattern or tar file specified")

    def _decode_mask_sample(self, sample):
        """Decode a mask sample from WebDataset format"""
        try:
            # Load mask image
            mask_data = sample['png']
            mask_img = np.array(Image.open(io.BytesIO(mask_data)).convert('RGBA'))
            return mask_img
        except Exception as e:
            print(f"Error decoding mask sample: {e}")
            # Return a dummy mask or skip
            return np.zeros((256, 256, 4), dtype=np.uint8)


class MixedDataLoader:
    def __init__(self, dataloaders: List[wds.WebLoader], mixing_prob: torch.FloatTensor, total_len: int):
        """
        Args:
            dataloaders (List[wds.WebLoader]): List of WebLoaders to be mixed.
            mixing_prob (torch.FloatTensor): Probability of each dataloader to be sampled from
            total_len (int): Total length of the dataset
        """
        assert len(dataloaders) == mixing_prob.shape[0]
        self.dataloaders = dataloaders
        self.mixing_prob = mixing_prob
        # Iterator state
        self._iter_dls = None
        self._iter_mixing_prob = None
        self.random_generator = torch.Generator()
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __iter__(self):
        # Synchronize dataloader seeds
        self.random_generator.manual_seed(42)
        self._iter_dls = [iter(loader) for loader in self.dataloaders]
        self._iter_mixing_prob = self.mixing_prob.clone()
        return self

    def __next__(self):
        """
        Sample a dataloader to sample from based on mixing probabilities. 
        If one of the dataloaders is exhausted, we continue sampling from the other loaders until all are exhausted.
        """
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")

        while self._iter_mixing_prob.any():  # at least one D-Loader with non-zero prob.
            dataset_idx = self._iter_mixing_prob.multinomial(
                1, generator=self.random_generator
            ).item()
            try:
                item = next(self._iter_dls[dataset_idx])
                return item
            except StopIteration:
                # No more iterations for this dataset, set it's mixing probability to zero and try again.
                self._iter_mixing_prob[dataset_idx] = 0
            except Exception as e:
                # log and raise any other unexpected error.
                logging.error(e)
                raise e

        # Exhausted all iterators
        raise StopIteration


class VARENTrainMixedDataset:
    def __init__(self, cfg: CfgNode, is_train: bool=True) -> None:
        dataset_list = cfg.DATASETS
        self.mask_webdataset = MaskWebDataset(cfg)

        self.datasets = []
        self.batch_size = []
        dataset_size = []
        for dataset, v in dataset_list.items():
            if dataset != "CONFIG" and "TAR_PATTERN" in v:
                if "VIDEO" in dataset:
                    dataset = VarenPoserVideoWebDataset(cfg, dataset, is_train=is_train)
                    self.batch_size.append(cfg.TRAIN.BATCH_SIZE.get("VIDEO", 1))
                    dataset_size.append(dataset.estimate_dataset_size())
                else:
                    dataset = VarenPoserWebDataset(cfg, dataset, is_train=is_train)
                    self.batch_size.append(cfg.TRAIN.BATCH_SIZE.get("IMAGE", 16))
                dataset_size.append(dataset.estimate_dataset_size())
                dataset.mask_webdataset = self.mask_webdataset.mask_webdataset
                dataset.mask_iterator = self.mask_webdataset.mask_iterator
                self.datasets.append(dataset.create_webdataset())  
        self.is_train = is_train
        self.num_workers = cfg.GENERAL.NUM_WORKERS
        self.prefetch_factor = cfg.GENERAL.PREFETCH_FACTOR
        self.pin_memory = True
        self.drop_last = True
        assert len(self.datasets) > 0

        # Assign each dataset a probability proportional to its length.
        dataset_lens = [
            (math.floor(d / bs) if self.drop_last else math.ceil(d / bs))
            for d, bs in zip(dataset_size, self.batch_size)
        ]
        total_len = sum(dataset_lens)
        dataset_prob = torch.tensor([d_len / total_len for d_len in dataset_lens])
        logging.info(f"Dataset mixing probabilities: {dataset_prob.tolist()}")
        self.dataset_prob = dataset_prob
        self.total_len = total_len

    def get_loader(self) -> Iterable:
        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(
            zip(self.datasets, self.batch_size)
        ):
            dataloaders.append(create_dataloader(dataset, 
                                                 batch_size=batch_size, 
                                                 num_workers=self.num_workers, 
                                                 prefetch_factor=self.prefetch_factor, 
                                                 pin_memory=self.pin_memory, 
                                                 is_train=self.is_train))
        return MixedDataLoader(dataloaders, self.dataset_prob, self.total_len)


class VarenEvalTemporalDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cfg: CfgNode,
                 dataset_name: str = "ANIMAL3D",
                 train: bool = False,
                 **kwargs):
        super().__init__()
        assert train is False, "VarenEvalTemporalDataset is only for inference"

        keypoints_map = {"APT36K": np.array([-1, -1, 0, 1, 2, 3, 4, -1, -1, 7, 10, 6, 9, 5, 8, 13, 16, 12, 15, 11, 14]),
                         "ANIMAL4D": np.array([-1, -1, 0, 1, 2, 3, 4, -1, -1, 7, 10, 6, 9, 5, 8, 13, 16, 12, 15, 11, 14]),
                         'VARENPOSER': np.array(range(21))}
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.keypoints_map = keypoints_map[dataset_name]
        self.train = train

        # Read image root and json from cfg
        self.root_image = cfg.DATASETS[dataset_name].ROOT_IMAGE
        self.json_file = cfg.DATASETS[dataset_name].get('JSON_FILE').get('TEST', None)
        assert self.json_file is not None and os.path.exists(self.json_file), f"Invalid json file: {self.json_file}"

        with open(self.json_file, 'r') as f:
            self.tracks: List[Dict] = json.load(f)

        # Model/image processing params
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]
        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        track = self.tracks[idx]
        video_id = track.get('video_id')
        track_id = track.get('track_id')
        frames = track.get('frames', [])

        imgs: List[np.ndarray] = []
        keypoints_seq: List[np.ndarray] = []
        keypoints_3d_seq: List[np.ndarray] = []
        smal_params_seq: List[Dict] = []
        has_smal_params_seq: List[Dict] = []

        for fr in frames:
            rel_path = fr.get('image_path', '')
            rel_path = rel_path.replace('\\', '/')
            img_path = os.path.join(self.root_image, rel_path)
            ann = fr.get('annotation', {})

            # COCO-style bbox [x, y, w, h]
            bbox = ann.get('bbox', [0, 0, 1, 1])
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            center_x, center_y = x + w * 0.5, y + h * 0.5
            bbox_size = max(w, h)

            # Keypoints: flat list [x1, y1, v1, x2, y2, v2, ...]
            kpts_flat = ann.get('keypoints', [])
            if len(kpts_flat) == 0:
                continue
            if len(kpts_flat) % 3 != 0:
                kpts_flat = kpts_flat[: len(kpts_flat) // 3 * 3]
            keypoints_2d = np.array(kpts_flat, dtype=np.float32).reshape(-1, 3)

            # Dummy placeholders for unused fields required by get_example
            keypoints_3d = np.zeros((keypoints_2d.shape[0], 4), dtype=np.float32) if 'keypoints_3d' not in ann else np.array(ann['keypoints_3d'], dtype=np.float32)
            smal_params = {
                'global_orient': np.zeros(3, dtype=np.float32) if 'global_orient' not in ann else np.array(ann['global_orient'], dtype=np.float32),
                'pose': np.zeros(111, dtype=np.float32) if 'pose' not in ann else np.array(ann['pose'], dtype=np.float32),
                'betas': np.zeros(39, dtype=np.float32) if 'betas' not in ann else np.array(ann['betas'], dtype=np.float32),
                'transl': np.zeros(3, dtype=np.float32) if 'trans' not in ann else np.array(ann['trans'], dtype=np.float32),
            }
            has_smal_params = {
                'global_orient': np.array(0., dtype=np.float32) if 'global_orient' not in ann else np.array(1., dtype=np.float32),
                'pose': np.array(0., dtype=np.float32) if 'pose' not in ann else np.array(1., dtype=np.float32),
                'betas': np.array(0., dtype=np.float32) if 'betas' not in ann else np.array(1., dtype=np.float32),
                'transl': np.array(0., dtype=np.float32) if 'trans' not in ann else np.array(1., dtype=np.float32),
            }

            img_patch, kp2d, kp3d, smal_params, has_smal_params, _, _ = get_example(
                img_path,
                center_x, center_y,
                bbox_size, bbox_size,
                keypoints_2d, keypoints_3d,
                smal_params, has_smal_params,
                self.IMG_SIZE, self.IMG_SIZE,
                self.MEAN, self.STD,
                False, self.augm_config,
                is_bgr=True,
                use_skimage_antialias=self.use_skimage_antialias,
                border_mode=self.border_mode,
                return_trans=False,
            )

            imgs.append(img_patch)
            kp2d = kp2d[self.keypoints_map, :]
            kp2d[np.where(self.keypoints_map == -1)] = np.array([0, 0, 0])
            keypoints_seq.append(kp2d.astype(np.float32))
            kp3d = keypoints_3d[self.keypoints_map, :]
            kp3d[np.where(self.keypoints_map == -1)] = np.array([0, 0, 0, 0])
            keypoints_3d_seq.append(kp3d.astype(np.float32))
            smal_params_seq.append(smal_params)
            has_smal_params_seq.append(has_smal_params)

        # Stack along time dimension: (T, C, H, W) and (T, K, 3)
        imgs = np.stack(imgs, axis=0) if len(imgs) > 0 else np.zeros((0, 3, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32)
        keypoints_seq = np.stack(keypoints_seq, axis=0) if len(keypoints_seq) > 0 else np.zeros((0, 0, 3), dtype=np.float32)
        keypoints_3d_seq = np.stack(keypoints_3d_seq, axis=0) if len(keypoints_3d_seq) > 0 else np.zeros((0, 0, 4), dtype=np.float32)
        smal_params_seq_ = dict()
        for key in smal_params_seq[0]:
            smal_params_seq_[key] = np.stack([smal_params[key] for smal_params in smal_params_seq], axis=0)
        has_smal_params_seq_ = dict()
        for key in has_smal_params_seq[0]:
            has_smal_params_seq_[key] = np.stack([has_smal_params[key] for has_smal_params in has_smal_params_seq], axis=0)

        return {
            'img': imgs,
            'keypoints_2d': keypoints_seq,
            'keypoints_3d': keypoints_3d_seq,
            'focal_length': np.ones((imgs.shape[0], 2), dtype=np.float32) * self.cfg.EXTRA.FOCAL_LENGTH,
            'mask': np.ones((imgs.shape[0], self.IMG_SIZE, self.IMG_SIZE), dtype=np.float32),
            'smal_params': smal_params_seq_,
            'has_smal_params': has_smal_params_seq_,
        }


if __name__ == "__main__":
    import hydra, torch
    @hydra.main(version_base="1.2", config_path=str(root / "amr/configs_hydra"), config_name="train.yaml")
    def main(cfg):
        dataset = VARENTrainMixedDataset(cfg, is_train=True)
        dataloader = dataset.get_loader()
        for item in tqdm(dataloader, total=len(dataloader)):
            continue
    main()