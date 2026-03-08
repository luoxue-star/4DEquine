from typing import Dict, List, Optional
import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])


class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 rescale_factor=1,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.boxes = boxes

        assert train is False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        flip = False

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size * 1.0) / patch_width)
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans, _ = generate_image_patch_cv2(cvimg,
                                                        center_x, center_y,
                                                        bbox_size, bbox_size,
                                                        patch_width, patch_height,
                                                        flip, 1.0, 0.0,
                                                        border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
            'box_center': self.center[idx].copy(),
            'box_size': bbox_size,
            'img_size': 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]]),
            'focal_length': np.array([self.cfg.EXTRA.FOCAL_LENGTH, self.cfg.EXTRA.FOCAL_LENGTH]),
        }
        return item


class ViTDetDatasetTemporal(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 frames: np.array,
                 bboxes: np.array,
                 mask_list: Optional[List[np.ndarray]] = None,
                 rescale_factor=1,
                 train: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.frames = frames
        self.bboxes = bboxes
        self.mask_list = mask_list
        assert train is False, "ViTDetDatasetTemporal is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        bboxes = bboxes.astype(np.float32)
        self.center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2]) / 200.0

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center_x = self.center[idx][0]
        center_y = self.center[idx][1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        flip = False

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.frames[idx].copy()
        if self.mask_list is not None:
            mask = self.mask_list[idx]
            cvimg = np.concatenate((cvimg, mask[:, :, np.newaxis]), axis=2)
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size * 1.0) / patch_width)

            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans, _ = generate_image_patch_cv2(cvimg,
                                                       center_x, center_y,
                                                       bbox_size, bbox_size,
                                                       patch_width, patch_height,
                                                       flip, 1.0, 0.0,
                                                       border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)
        mask = img_patch[0] if self.mask_list is not None else None
        img_patch = img_patch[1:] if self.mask_list is not None else img_patch

        # apply normalization
        for n_c in range(min(self.frames[idx].shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'mask': mask,
            'box_center': self.center[idx].copy(),
            'box_size': bbox_size,
            'img_size': 1.0 * np.array([self.frames[idx].shape[1], self.frames[idx].shape[0]]),
            'focal_length': np.array([self.cfg.EXTRA.FOCAL_LENGTH, self.cfg.EXTRA.FOCAL_LENGTH]),
            'trans': trans,
        }
        return item