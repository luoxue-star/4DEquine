import os
import random
import numpy as np
import torch
import json
import cv2
import traceback
import glob

from PIL import Image
from collections import defaultdict
from typing import Union
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
from megfile import smart_open, smart_path_join, smart_exists
from torchvision import transforms

from yacs.config import CfgNode
from omegaconf import DictConfig


class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dirs: str, meta_path: Optional[Union[list, str]]):
        super().__init__()
        self.root_dirs = root_dirs
        self.uids = self._load_uids(meta_path)

    def __len__(self):
        return len(self.uids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            traceback.print_exc()
            print(f"[DEBUG-DATASET] Error when loading {self.uids[idx]}")
            # raise e
            return self.__getitem__((idx + 1) % self.__len__())

    @staticmethod
    def _load_uids(meta_path: Optional[Union[list, str]]):
        # meta_path is a json file
        if isinstance(meta_path, str):
            with open(meta_path, 'r') as f:
                uids = json.load(f)
        else:
            uids_lst = []
            max_total = 0
            for pth, weight in meta_path:
                with open(pth, 'r') as f:
                    uids = json.load(f)
                    max_total = max(len(uids) / weight, max_total)
                uids_lst.append([uids, weight, pth])
            merged_uids = []
            for uids, weight, pth in uids_lst:
                repeat = 1
                if len(uids) < int(weight * max_total):
                    repeat = int(weight * max_total) // len(uids)
                cur_uids = uids * repeat
                merged_uids += cur_uids
                print("Data Path:", pth, "Repeat:", repeat, "Final Length:", len(cur_uids))
            uids = merged_uids
            print("Total UIDs:", len(uids))
        return uids

    @staticmethod
    def _load_rgba_image(file_path, bg_color: float = 1.0):
        ''' Load and blend RGBA image to RGB with certain background, 0-1 scaled '''
        rgba = np.array(Image.open(smart_open(file_path, 'rb')))
        rgba = torch.from_numpy(rgba).float() / 255.0
        rgba = rgba.permute(2, 0, 1).unsqueeze(0)
        rgb = rgba[:, :3, :, :] * rgba[:, 3:4, :, :] + bg_color * (1 - rgba[:, 3:, :, :])
        rgba[:, :3, ...] * rgba[:, 3:, ...] + (1 - rgba[:, 3:, ...])
        return rgb

    @staticmethod
    def _locate_datadir(root_dirs, uid, locator: str):
        for root_dir in root_dirs:
            datadir = smart_path_join(root_dir, uid, locator)
            if smart_exists(datadir):
                return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for uid {uid}")


class AvatarDataset(BaseDataset):
    def __init__(self, 
                 root_dirs: str, 
                 meta_path: Optional[Union[str, list]],
                 sample_side_views: int,
                 render_image_res_low: int, 
                 render_image_res_high: int, 
                 render_region_size: int,
                 source_image_res: int,
                 repeat_num: int = 1,
                 crop_range_ratio_hw: list = [1.0, 1.0],
                 aspect_standard: float = 1.0,  # h/w
                 enlarge_ratio: list = [0.8, 1.2],
                 debug: bool = False,
                 is_val: bool = False,
                 **kwargs):
        """
        Args:
            root_dirs: str, directory path of data  
            meta_path: Optional[Union[str, list]], path to meta data
            sample_side_views: int, number of side views
            render_image_res_low: int, resolution of rendered image
            render_image_res_high: int, resolution of rendered image
            render_region_size: int, size of rendered region
            source_image_res: int, resolution of source image
            repeat_num: int = 1, number of times to repeat the data
            crop_range_ratio_hw: list = [1.0, 1.0], ratio of crop range
            aspect_standard: float = 1.0,  # h/w
            enlarge_ratio: list = [0.8, 1.2], ratio of enlarge range
            debug: bool = False, whether to debug
            is_val: bool = False, whether to validate
            kwargs: dict, additional arguments
        """
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        if not (isinstance(render_region_size, list) or isinstance(render_region_size, tuple)): 
            render_region_size = render_region_size, render_region_size  # [H, W]
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        
        self.uids = self.uids * repeat_num
        self.crop_range_ratio_hw = crop_range_ratio_hw
        self.debug = debug
        self.aspect_standard = aspect_standard
        
        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = enlarge_ratio
        print(f"AvatarDataset, data_len:{len(self.uids)}, repeat_num:{repeat_num}, debug:{debug}, is_val:{is_val}")
        self.multiply = kwargs.get("multiply", 16)
        # set data deterministic
        self.is_val = is_val

    @staticmethod
    def _load_pose(frame_info, transpose_R=False):
        """
        TODO: Implement this function
        """
        c2w = torch.eye(4)
        c2w = np.array(frame_info["transform_matrix"])
        c2w[:3, 1:3] *= -1
        c2w = torch.FloatTensor(c2w)
        """
        if transpose_R:
            w2c = torch.inverse(c2w)
            w2c[:3, :3] = w2c[:3, :3].transpose(1, 0).contiguous()
            c2w = torch.inverse(w2c)
        """
        
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = frame_info["fl_x"]
        intrinsic[1, 1] = frame_info["fl_y"]
        intrinsic[0, 2] = frame_info["cx"]
        intrinsic[1, 2] = frame_info["cy"]
        intrinsic = intrinsic.float()
        
        return c2w, intrinsic

    def img_center_padding(self, img_np, pad_ratio):
        
        ori_w, ori_h = img_np.shape[:2]
        
        w = round((1 + pad_ratio) * ori_w)
        h = round((1 + pad_ratio) * ori_h)
        
        if len(img_np.shape) > 2:
            img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
        else:
            img_pad_np = np.zeros((w, h), dtype=np.uint8)
        offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
        img_pad_np[offset_h: offset_h + img_np.shape[0]:, offset_w: offset_w + img_np.shape[1]] = img_np
        
        return img_pad_np
    
    def resize_image_keepaspect_np(self, img, max_tgt_size):
        """
        similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio  
        """
        h, w = img.shape[:2]
        ratio = max_tgt_size / max(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """ 
            img: [H, W, 3]
            mask: [H, W]
        """ 
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)
        
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        
        half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
        half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
        aspect = half_h / half_w

        if aspect >= aspect_standard:                
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)
            
        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03
        
        offset_x = center_x - half_w
        offset_y = center_y - half_h
        
        new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        
        return  new_img, new_mask, offset_x, offset_y        
        
    def load_rgb_image_with_aug_bg(self, rgb_path, mask_path, bg_color, pad_ratio, max_tgt_size, aspect_standard, enlarge_ratio,
                                   render_tgt_size, multiply, intr):
        rgb = np.array(Image.open(rgb_path))
        interpolation = cv2.INTER_AREA
        if rgb.shape[0] != 1024 and rgb.shape[0] == rgb.shape[1]:
            rgb = cv2.resize(rgb, (1024, 1024), interpolation=interpolation)
        if pad_ratio > 0:
            rgb = self.img_center_padding(rgb, pad_ratio)
        
        rgb = rgb / 255.0
        if mask_path is not None:
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path)) > 180
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                assert pad_ratio == 0
                # if pad_ratio > 0:
                #     mask = self.img_center_padding(mask, pad_ratio)
                # mask = mask / 255.0
            else:
                # print("no mask file")
                mask = (rgb >= 0.99).sum(axis=2) == 3
                mask = np.logical_not(mask)
                # erode
                mask = (mask * 255).astype(np.uint8)
                kernel_size, iterations = 3, 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=iterations) / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
    
        # crop image to enlarge face area.
        try:
            rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)
        except Exception as ex:
            print(rgb_path, mask_path, ex)

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                                         aspect_standard=aspect_standard,
                                                                         tgt_size=render_tgt_size, multiply=multiply)
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
        
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)

        return rgb, mask, intr
            
    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs
    
    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list
    
    def calc_new_tgt_size(self, cur_hw, tgt_size, multiply):
        ratio = tgt_size / min(cur_hw)
        tgt_size = int(ratio * cur_hw[0]), int(ratio * cur_hw[1])
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x
    
    def load_flame_params(self, flame_file_path, teeth_bs=None):
        
        flame_param = dict(np.load(flame_file_path), allow_pickle=True)

        flame_param_tensor = {}
        flame_param_tensor['expr'] = torch.FloatTensor(flame_param['expr'])[0]
        flame_param_tensor['rotation'] = torch.FloatTensor(flame_param['rotation'])[0]
        flame_param_tensor['neck_pose'] = torch.FloatTensor(flame_param['neck_pose'])[0]
        flame_param_tensor['jaw_pose'] = torch.FloatTensor(flame_param['jaw_pose'])[0]
        flame_param_tensor['eyes_pose'] = torch.FloatTensor(flame_param['eyes_pose'])[0]
        flame_param_tensor['translation'] = torch.FloatTensor(flame_param['translation'])[0]
        if teeth_bs is not None:
            flame_param_tensor['teeth_bs'] = torch.FloatTensor(teeth_bs)
            # flame_param_tensor['expr'] = torch.cat([flame_param_tensor['expr'], flame_param_tensor['teeth_bs']], dim=0)
    
        return flame_param_tensor
    
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        crop_ratio_h, crop_ratio_w = self.crop_range_ratio_hw
        
        uid = self.uids[idx]
        if len(uid.split('/')) == 1:
            uid = os.path.join(self.root_dirs, uid)
        mode_str = "train" if not self.is_val else "test"
        transforms_json = os.path.join(uid, f"transforms_{mode_str}.json")
        
        with open(transforms_json) as fp:
            data = json.load(fp)    
        cor_flame_path = transforms_json.replace('transforms_{}.json'.format(mode_str),'canonical_flame_param.npz')
        flame_param = np.load(cor_flame_path)
        shape_param = torch.FloatTensor(flame_param['shape'])
        # data['static_offset'] = flame_param['static_offset']
                        
        all_frames = data["frames"]

        sample_total_views = self.sample_side_views + 1
        if len(all_frames) >= self.sample_side_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(len(all_frames)))
                else:
                    replace = len(all_frames) < sample_total_views
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(len(all_frames)))
                else:
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = len(all_frames) < sample_total_views
                frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > 1:
                    frame_id_list = np.linspace(0, len(all_frames) - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for i in range(sample_total_views)]
        
        cam_id_list = frame_id_list
        
        assert self.sample_side_views + 1 == len(frame_id_list)

        # source images
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        flame_params = []
        teeth_bs_pth = os.path.join(uid, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None
        for cam_id, frame_id in zip(cam_id_list, frame_id_list):
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path or "tiktok_v34" in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])

            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_path = os.path.join(uid, frame_info["flame_param_path"])
            flame_param = self.load_flame_params(flame_path, teeth_bs)

            # if cam_id == 0:
            #     shape_param = flame_param["betas"]

            c2w, ori_intrinsic = self._load_pose(frame_info, transpose_R="nersemble" in frame_path)

            bg_color = random.choice([0.0, 0.5, 1.0])  # 1.0
            # if self.is_val:
            #     bg_color = 1.0       
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(frame_path, mask_path=mask_path,
                                                                    bg_color=bg_color, 
                                                                    pad_ratio=0,
                                                                    max_tgt_size=None,
                                                                    aspect_standard=self.aspect_standard,
                                                                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                                                                    render_tgt_size=self.render_image_res,
                                                                    multiply=16,
                                                                    intr=ori_intrinsic.clone())
            c2ws.append(c2w)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            intrs.append(intrinsic)
            flame_params.append(flame_param)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]

        flame_params_tmp = defaultdict(list)
        for flame in flame_params:
            for k, v in flame.items():
                flame_params_tmp[k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        flame_params = flame_params_tmp
        # TODO check different betas for same person
        flame_params["betas"] = shape_param
        
        # reference images
        prob_refidx = np.ones(self.sample_side_views + 1)
        if not self.is_val:
            prob_refidx[0] = 0.5  # front_prob
        else:
            prob_refidx[0] = 1.0
        # print(frame_id_list, kinect_color_list, prob_refidx[0])
        prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        ref_idx = np.random.choice(self.sample_side_views + 1, p=prob_refidx)
        cam_id_source_list = cam_id_list[ref_idx: ref_idx + 1]
        frame_id_source_list = frame_id_list[ref_idx: ref_idx + 1]

        source_c2ws, source_intrs, source_rgbs, source_flame_params = [], [], [], []
        for cam_id, frame_id in zip(cam_id_source_list, frame_id_source_list):
            frame_info = all_frames[frame_id]
            frame_path = os.path.join(uid, frame_info["file_path"])
            if 'nersemble' in frame_path:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])
            else:
                mask_path = os.path.join(uid, frame_info["fg_mask_path"]).replace("/export/", "/mask/").replace("/fg_masks/", "/mask/").replace(".png", ".jpg")
            flame_path = os.path.join(uid, frame_info["flame_param_path"])
            
            teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
            flame_param = self.load_flame_params(flame_path, teeth_bs)

            c2w, ori_intrinsic = self._load_pose(frame_info)
            
            # bg_color = 1.0
            # bg_color = 0.0
            bg_color = random.choice([0.0, 0.5, 1.0])   # 1. 
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(frame_path, mask_path=mask_path, 
                                                                    bg_color=bg_color,
                                                                    pad_ratio=0,
                                                                    max_tgt_size=None, 
                                                                    aspect_standard=self.aspect_standard,
                                                                    enlarge_ratio=self.enlarge_ratio if (not self.is_val) or ("nersemble" in frame_path) else [1.0, 1.0],
                                                                    render_tgt_size=self.source_image_res,
                                                                    multiply=self.multiply,
                                                                    intr=ori_intrinsic.clone())

            source_c2ws.append(c2w)
            source_intrs.append(intrinsic)
            source_rgbs.append(rgb)
            source_flame_params.append(flame_param)

        source_c2ws = torch.stack(source_c2ws, dim=0)
        source_intrs = torch.stack(source_intrs, dim=0)
        source_rgbs = torch.cat(source_rgbs, dim=0)

        flame_params_tmp = defaultdict(list)
        for flame in source_flame_params:
            for k, v in flame.items():
                flame_params_tmp['source_'+k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        source_flame_params = flame_params_tmp
        # TODO check different betas for same person
        source_flame_params["source_betas"] = shape_param
    
        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]   # [H, W]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,  # [N1, 4, 4]
            'source_intrs': source_intrs,  # [N1, 4, 4]
            'source_rgbs': source_rgbs.clamp(0, 1),   # [N1, 3, H, W]
            'render_image': render_image.clamp(0, 1), # [N, 3, H, W]
            'render_mask': render_mask.clamp(0, 1), #[ N, 1, H, W]
            'c2ws': c2ws,  # [N, 4, 4]
            'intrs': intrs,  # [N, 4, 4]
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),  # [N, 2]
            'render_bg_colors': bg_colors, # [N, 3]
            'pytorch3d_transpose_R': torch.Tensor(["nersemble" in frame_path]), # [1]
        }
        
        #['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas']
        # 'flame_params': flame_params, # dict: body_pose:[N, 21, 3], 
        ret.update(flame_params)
        ret.update(source_flame_params)
            
        return ret


class VarenAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode):
        """
        Dataset for VAREN-based avatar data. A directory represents one sequence and must contain:
            - c2w.pt: [Nv, 4, 4] camera-to-world matrices
            - mvp.pt: [Nv, 4, 4] intrinsic matrices
            - params.npz: VAREN parameters including keys: 'shape', 'global_orient', 'pose', 'transl'
            - images and optional masks (either co-located or in a 'masks' subfolder)
        """
        super().__init__()
        self.sample_side_views = cfg.DATASETS.VarenAvatar.get("SAMPLE_SIDE_VIEWS", 4)
        self.render_image_res_low = cfg.DATASETS.VarenAvatar.get("RENDER_IMAGE_RES_LOW", 512)
        self.render_image_res_high = cfg.DATASETS.VarenAvatar.get("RENDER_IMAGE_RES_HIGH", 512)
        self.render_region_size = cfg.DATASETS.VarenAvatar.get("RENDER_REGION_SIZE", 512)
        self.source_image_res = cfg.DATASETS.VarenAvatar.get("SOURCE_IMAGE_RES", 512)

        self.crop_range_ratio_hw = cfg.DATASETS.VarenAvatar.get("CROP_RANGE_RATIO_HW", [1.0, 1.0])
        self.debug = cfg.DATASETS.VarenAvatar.get("DEBUG", False)
        self.aspect_standard = cfg.DATASETS.VarenAvatar.get("ASPECT_STANDARD", 1.0)

        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = cfg.DATASETS.VarenAvatar.get("ENLARGE_RATIO", [0.8, 1.2])
        self.multiply = cfg.DATASETS.VarenAvatar.get("MULTIPLY", 16)
        # set data deterministic
        self.is_val = cfg.DATASETS.VarenAvatar.get("IS_VAL", False)

        self.root_dirs = cfg.DATASETS.VarenAvatar.ROOT
        self.sequence = os.listdir(self.root_dirs)

    def _scan_images_and_masks(self, uid_dir):
        image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        masks = []

        def is_image(p):
            ext = os.path.splitext(p)[1].lower()
            return ext in image_exts

        # Prefer structured subfolders if present
        images_dir = os.path.join(uid_dir, 'images')
        masks_dir = os.path.join(uid_dir, 'masks')
        if os.path.isdir(images_dir):
            for f in sorted(os.listdir(images_dir)):
                p = os.path.join(images_dir, f)
                if os.path.isfile(p) and is_image(p) and ('mask' not in f.lower()):
                    images.append(p)
        if os.path.isdir(masks_dir):
            for f in sorted(os.listdir(masks_dir)):
                p = os.path.join(masks_dir, f)
                if os.path.isfile(p) and is_image(p):
                    masks.append(p)

        # Fallback: scan root directory for images and masks
        if len(images) == 0:
            for f in sorted(os.listdir(uid_dir)):
                p = os.path.join(uid_dir, f)
                if os.path.isfile(p) and is_image(p):
                    if ('mask' in f.lower()) or ('fg' in f.lower() and 'mask' in f.lower()):
                        masks.append(p)
                    else:
                        images.append(p)

        return images, masks

    def _find_mask_for_image(self, img_path, masks_list):
        if len(masks_list) == 0:
            return None
        img_name = os.path.basename(img_path)
        img_stem, img_ext = os.path.splitext(img_name)
        # exact name match in masks folder
        for m in masks_list:
            m_name = os.path.basename(m)
            if os.path.splitext(m_name)[0] == img_stem:
                return m
        # suffix variants
        candidates = [
            img_stem + '_mask' + img_ext,
            img_stem + '-mask' + img_ext,
            img_stem + '_fg' + img_ext,
            img_stem + '_fg_mask' + img_ext,
        ]
        for cand in candidates:
            for m in masks_list:
                if os.path.basename(m) == cand:
                    return m
        return None

    def _load_varen_params(self, params_path, num_views):
        params_npz = np.load(params_path, allow_pickle=True)
        varen = {}
        # betas
        if 'betas' in params_npz:
            betas = np.array(params_npz['betas'])
            betas = betas.reshape(-1)
            varen['betas'] = torch.from_numpy(betas).float()
        # global_orient
        if 'global_orient' in params_npz:
            go = np.array(params_npz['global_orient'])
            if go.ndim == 1:
                go = go.reshape(1, 3)
            if go.shape[0] == 1 and num_views > 1:
                go = np.repeat(go[None, ...], num_views, axis=0)  # [Nv, 1, 3] or [Nv, 3]
            if go.ndim == 2:
                go = go[:, None, :]  # [Nv, 1, 3]
            varen['global_orient'] = torch.from_numpy(go).float()
        # pose
        if 'pose' in params_npz:
            pose = np.array(params_npz['pose'])
            if pose.ndim == 2 and pose.shape[-1] == 3:  # [J, 3]
                pose = np.repeat(pose[None, ...], num_views, axis=0)  # [Nv, J, 3]
            varen['pose'] = torch.from_numpy(pose).float()
        # trans
        if 'trans' in params_npz:
            trans = np.array(params_npz['trans'])
            if trans.ndim == 1:
                trans = np.repeat(trans[None, ...], num_views, axis=0)  # [Nv, 3]
            varen['trans'] = torch.from_numpy(trans).float()
        return varen

    def compute_intrinsics_from_projection(self, P: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Given an OpenGL-style projection matrix P (with sign-flipped Y as in nvdiffrast):
        P[0,0] = 1/(tan(fovy/2)*aspect), P[1,1] = -1/tan(fovy/2)
        Derive fx, fy for a pinhole camera used by pyrender.
        We assume principal point at (width/2, height/2).
        Returns fx, fy, cx, cy.
        """
        fx = width * 0.5 * abs(P[0, 0])
        fy = height * 0.5 * abs(P[1, 1])
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)
    
    def inner_get_item(self, idx):
        sample_total_views = self.sample_side_views + 1

        uid = self.sequence[idx]
        uid = os.path.join(self.root_dirs, uid)

        # Load camera extrinsics and intrinsics
        c2w_path = os.path.join(uid, 'c2w.pt')
        mvp_matrix_path = os.path.join(uid, 'mvp_matrix.pt')
        assert os.path.exists(c2w_path), f"Missing c2w.pt in {uid}"
        assert os.path.exists(mvp_matrix_path), f"Missing mvp_matrix.pt in {uid}"

        c2ws_all = torch.load(c2w_path)
        mvp_matrix_all = torch.load(mvp_matrix_path)
        proj_all = torch.bmm(mvp_matrix_all, c2ws_all)
        c2ws_all = c2ws_all.float()
        proj_all = proj_all.float()
        assert c2ws_all.shape[0] == proj_all.shape[0], "c2w and mvp must have same Nv"

        Nv = c2ws_all.shape[0]
        intrinsic = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(Nv, 1, 1)
        for i, proj in enumerate(proj_all):
            fx, fy, cx, cy = self.compute_intrinsics_from_projection(proj[0], self.render_image_res, self.render_image_res)
            intrinsic[i, 0, 0] = fx
            intrinsic[i, 1, 1] = fy
            intrinsic[i, 0, 2] = cx
            intrinsic[i, 1, 2] = cy

        # Scan images and masks
        image_list, mask_list = self._scan_images_and_masks(uid)
        assert len(image_list) > 0, f"No images found in {uid}"

        # Align counts
        Nv = min(len(image_list), c2ws_all.shape[0])
        image_list = image_list[:Nv]
        c2ws_all = c2ws_all[:Nv]
        intrs_all = intrinsic[:Nv]

        # Sample view indices
        if Nv >= sample_total_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(Nv))
                else:
                    replace = Nv < sample_total_views
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(Nv))
                else:
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = Nv < sample_total_views
                frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > 1:
                    frame_id_list = np.linspace(0, Nv - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for _ in range(sample_total_views)]

        frame_id_list = list(map(int, frame_id_list))
        cam_id_list = frame_id_list
        assert self.sample_side_views + 1 == len(frame_id_list)

        # Load VAREN parameters
        params_path = os.path.join(uid, 'params.npz')
        assert os.path.exists(params_path), f"Missing params.npz in {uid}"
        varen_params_all = self._load_varen_params(params_path, num_views=Nv)

        # Build source/render batches
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        for frame_id in frame_id_list:
            img_path = image_list[frame_id]
            mask_path = self._find_mask_for_image(img_path, mask_list)
            c2w = c2ws_all[frame_id]
            ori_intrinsic = intrinsic[frame_id]

            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                img_path,
                mask_path=mask_path,
                bg_color=bg_color,
                pad_ratio=0,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio,
                render_tgt_size=self.render_image_res,
                multiply=16,
                intr=ori_intrinsic.clone(),
            )

            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)
        intrs = torch.stack(intrs, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        masks = torch.cat(masks, dim=0)
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)

        # Subset varen params according to selected views
        varen_params = {}
        if 'betas' in varen_params_all:
            varen_params['betas'] = varen_params_all['betas']
        if 'global_orient' in varen_params_all:
            varen_params['global_orient'] = varen_params_all['global_orient'][frame_id_list]
        if 'pose' in varen_params_all:
            varen_params['pose'] = varen_params_all['pose'][frame_id_list]
        if 'trans' in varen_params_all:
            varen_params['trans'] = varen_params_all['trans'][frame_id_list]

        # Reference/source images (pick one view)
        prob_refidx = np.ones(self.sample_side_views + 1)
        if not self.is_val:
            prob_refidx[0] = 0.5
        else:
            prob_refidx[0] = 1.0
        prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        ref_idx = int(np.random.choice(self.sample_side_views + 1, p=prob_refidx))

        source_c2ws = c2ws[ref_idx: ref_idx + 1]
        source_intrs = intrs[ref_idx: ref_idx + 1]
        source_rgbs = rgbs[ref_idx: ref_idx + 1]

        source_varen_params = {}
        if 'betas' in varen_params:
            source_varen_params['source_betas'] = varen_params['betas']
        if 'global_orient' in varen_params:
            source_varen_params['source_global_orient'] = varen_params['global_orient'][ref_idx: ref_idx + 1]
        if 'pose' in varen_params:
            source_varen_params['source_pose'] = varen_params['pose'][ref_idx: ref_idx + 1]
        if 'trans' in varen_params:
            source_varen_params['source_trans'] = varen_params['trans'][ref_idx: ref_idx + 1]

        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 2.5, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 2.5, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,
            'source_intrs': source_intrs,
            'source_rgbs': source_rgbs.clamp(0, 1),
            'render_image': render_image.clamp(0, 1),
            'render_mask': render_mask.clamp(0, 1),
            'c2ws': c2ws,
            'intrs': intrs,
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': bg_colors,
            'pytorch3d_transpose_R': torch.tensor([0.0]),
        }

        ret.update(varen_params)
        ret.update(source_varen_params)

        return ret


class MiniVarenAvatarDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode, root_dir: str, is_train: bool = True):
        """
        Dataset for VAREN-based avatar data. A directory represents one sequence and must contain:
            - c2w.pt: [Nv, 4, 4] camera-to-world matrices
            - mvp_matrix.pt: [Nv, 4, 4] intrinsic matrices
            - params.npz: VAREN parameters including keys: 'shape', 'global_orient', 'pose', 'transl'
            - images and optional masks (either co-located or in a 'masks' subfolder)
        """
        super().__init__()
        self.sample_side_views = cfg.DATASETS.get("SAMPLE_SIDE_VIEWS", 4)
        self.render_image_res_low = cfg.DATASETS.get("RENDER_IMAGE_RES_LOW", 512)
        self.render_image_res_high = cfg.DATASETS.get("RENDER_IMAGE_RES_HIGH", 512)
        self.render_region_size = cfg.DATASETS.get("RENDER_REGION_SIZE", 512)
        self.source_image_res = cfg.DATASETS.get("SOURCE_IMAGE_RES", 512)

        self.crop_range_ratio_hw = cfg.DATASETS.get("CROP_RANGE_RATIO_HW", [1.0, 1.0])
        self.aspect_standard = cfg.DATASETS.get("ASPECT_STANDARD", 1.0)

        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = cfg.DATASETS.get("ENLARGE_RATIO", [0.8, 1.2])
        self.multiply = cfg.DATASETS.get("MULTIPLY", 16)
        # set data deterministic
        self.is_val = not is_train

        self.root_dirs = root_dir
        self.sequence = os.listdir(self.root_dirs)

    def _load_varen_params(self, params_path, num_views):
        params_npz = np.load(params_path, allow_pickle=True)
        varen = {}
        # betas
        if 'shape' in params_npz:
            betas = np.array(params_npz['shape'])
            betas = betas.reshape(-1)
            varen['betas'] = torch.from_numpy(betas).float()
        # global_orient
        if 'global_orient' in params_npz:
            go = np.array(params_npz['global_orient'])
            if go.ndim == 1:
                go = go.reshape(1, 3)
            if go.shape[0] == 1 and num_views > 1:
                go = np.repeat(go[None, ...], num_views, axis=0)  # [Nv, 1, 3] or [Nv, 3]
            if go.ndim == 2:
                go = go[:, None, :]  # [Nv, 1, 3]
            varen['global_orient'] = torch.from_numpy(go).float()
        # pose
        if 'pose' in params_npz:
            pose = np.array(params_npz['pose'])
            if pose.ndim == 1:
                pose = pose.reshape(-1, 3)
            if pose.ndim == 2 and pose.shape[-1] == 3:  # [J, 3]
                pose = np.repeat(pose[None, ...], num_views, axis=0)  # [Nv, J, 3]
            varen['pose'] = torch.from_numpy(pose).float()
        # trans
        if 'transl' in params_npz:
            trans = np.array(params_npz['transl'])
            if trans.ndim == 1:
                trans = np.repeat(trans[None, ...], num_views, axis=0)  # [Nv, 3]
            varen['trans'] = torch.from_numpy(trans).float()
        if 'tail_scale' in params_npz:
            tail_scale = np.array(params_npz['tail_scale'])
            tail_scale = tail_scale.reshape(-1)
            varen['tail_scale'] = torch.from_numpy(tail_scale).float()
        return varen

    @staticmethod
    def compute_intrinsics_from_projection(P, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Given an OpenGL-style projection matrix P (with sign-flipped Y as in nvdiffrast):
        P[0,0] = 1/(tan(fovy/2)*aspect), P[1,1] = -1/tan(fovy/2)
        Derive fx, fy for a pinhole camera used by pyrender.
        We assume principal point at (width/2, height/2).
        Returns fx, fy, cx, cy.
        """
        if isinstance(P, torch.Tensor):
            P = P.detach().cpu().numpy()
        fx = width * 0.5 * abs(P[0, 0])
        fy = height * 0.5 * abs(P[1, 1])
        cx = width * 0.5
        cy = height * 0.5
        return float(fx), float(fy), float(cx), float(cy)

    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """ 
            img: [H, W, 3]
            mask: [H, W]
        """ 
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)
        
        center_x, center_y = img.shape[1]//2, img.shape[0]//2
        
        half_w = max(abs(center_x - x_min), abs(center_x -  x_max))
        half_h = max(abs(center_y - y_min), abs(center_y -  y_max))
        aspect = half_h / half_w

        if aspect >= aspect_standard:                
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) >  0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(enlarge_ratio_cur * half_w)
            
        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03
        
        offset_x = center_x - half_w
        offset_y = center_y - half_h
        
        new_img = img[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        new_mask = mask[offset_y: offset_y + 2*half_h, offset_x: offset_x + 2*half_w]
        
        return  new_img, new_mask, offset_x, offset_y      

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = int(tgt_size[0] / multiply) * multiply, int(tgt_size[1] / multiply) * multiply
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def load_rgb_image_with_aug_bg(self, rgb, mask, bg_color, max_tgt_size, aspect_standard, enlarge_ratio,
                                   render_tgt_size, multiply, intr):
        interpolation = cv2.INTER_AREA
        assert rgb.shape[0] == mask.shape[0] and rgb.shape[1] == mask.shape[1], f"RGB and mask shape mismatch: {rgb.shape}, {mask.shape}"
        # if rgb.shape[0] != self.render_image_res and rgb.shape[0] == rgb.shape[1]:
        #     rgb = cv2.resize(rgb, (self.render_image_res, self.render_image_res), interpolation=interpolation)
        #     mask = cv2.resize(mask, (self.render_image_res, self.render_image_res), interpolation=interpolation)
        
        rgb = rgb / 255.0
        mask = (mask > 127).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
    
        try:
            rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)
        except Exception as ex:
            print(ex)

        try:
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y
        except Exception as ex:
            print(ex)
            print(intr.shape)

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                                         aspect_standard=aspect_standard,
                                                                         tgt_size=render_tgt_size, multiply=multiply)
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
        
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)

        return rgb, mask, intr

    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list
    
    def inner_get_item(self, idx):
        sample_total_views = self.sample_side_views + 1

        uid = self.sequence[idx]
        uid = os.path.join(self.root_dirs, uid)

        try:
            # Scan images and masks
            image_file = os.path.join(uid, "image.jpg") if os.path.exists(os.path.join(uid, "image.jpg")) else os.path.join(uid, "image.png")
            image = np.array(Image.open(image_file).convert('RGB'))
            images = [image[:, i*image.shape[0]:(i+1)*image.shape[0]] for i in range(image.shape[1] // image.shape[0])]
            masks_file = os.path.join(uid, 'mask.png')
            mask = np.array(Image.open(masks_file).convert('L'))
            mask_images = [mask[:, i*image.shape[0]:(i+1)*image.shape[0]] for i in range(mask.shape[1] // image.shape[0])]
            assert len(images) == len(mask_images), f"Number of images and masks mismatch: {len(images)}, {len(mask_images)}"
            Nv = len(images)
        except:
            image_file = sorted(glob.glob(os.path.join(uid, "images", "*.jpg")))
            mask_file = sorted(glob.glob(os.path.join(uid, "masks", "*.png")))
            assert len(image_file) == len(mask_file), f"Number of images and masks mismatch: {len(image_file)}, {len(mask_file)}"
            Nv = len(image_file)

        # Sample view indices
        if Nv >= sample_total_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(Nv))
                else:
                    replace = Nv < sample_total_views
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(Nv))
                else:
                    frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = Nv < sample_total_views
                frame_id_list = np.random.choice(Nv, size=sample_total_views, replace=replace)
            else:
                if Nv > 1:
                    frame_id_list = np.linspace(0, Nv - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for _ in range(sample_total_views)]
        frame_id_list = list(map(int, frame_id_list))
        assert self.sample_side_views + 1 == len(frame_id_list)

        # Load camera extrinsics and intrinsics
        if os.path.exists(os.path.join(uid, 'c2w.pt')) and os.path.exists(os.path.join(uid, 'mvp_matrix.pt')):
            c2w_path = os.path.join(uid, 'c2w.pt')
            mvp_matrix_path = os.path.join(uid, 'mvp_matrix.pt')

            c2ws_all = torch.load(c2w_path, weights_only=True, map_location="cpu")[frame_id_list]
            mvp_matrix_all = torch.load(mvp_matrix_path, weights_only=True, map_location="cpu")[frame_id_list]
            proj_all = torch.bmm(mvp_matrix_all, c2ws_all)
            c2ws_all = c2ws_all.float()
            proj_all = proj_all.float()
            assert c2ws_all.shape[0] == proj_all.shape[0], "c2w and mvp must have same Nv"

            images_ = [images[i] for i in frame_id_list]
            images = images_
            mask_images_ = [mask_images[i] for i in frame_id_list]
            mask_images = mask_images_

            intrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(sample_total_views, 1, 1)
            for i, proj in enumerate(proj_all):
                fx, fy, cx, cy = self.compute_intrinsics_from_projection(proj, images[i].shape[1], images[i].shape[0])
                intrinsics[i, 0, 0] = fx
                intrinsics[i, 1, 1] = fy
                intrinsics[i, 0, 2] = cx
                intrinsics[i, 1, 2] = cy
        else:
            c2ws_all = torch.eye(4, dtype=torch.float32)
            c2ws_all[1, 1] = c2ws_all[2, 2] = -1.0
            c2ws_all = c2ws_all.unsqueeze(0).repeat(sample_total_views, 1, 1)
            images = [np.array(Image.open(image_file[i]).convert('RGB')) for i in frame_id_list]
            mask_images = [np.array(Image.open(mask_file[i]).convert('L')) for i in frame_id_list]
            intrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics[0, 2] = images[0].shape[1] // 2
            intrinsics[1, 2] = images[0].shape[0] // 2
            # TODO: hardcode for now, need to change to dynamic
            intrinsics[0, 0] = intrinsics[1, 1] = 5000. / 256. * max(images[0].shape[0], images[0].shape[1])
            intrinsics = intrinsics.unsqueeze(0).repeat(sample_total_views, 1, 1)

        # Load VAREN parameters
        params_path = os.path.join(uid, f'{self.sequence[idx]}.npz')
        assert os.path.exists(params_path), f"Missing {self.sequence[idx]}.npz in {uid}"
        varen_params_all = self._load_varen_params(params_path, num_views=Nv)

        # Build source/render batches
        c2ws, intrs, rgbs, bg_colors, masks = [], [], [], [], []
        S = torch.tensor([[1., 0., 0., 0.],
                         [0.,-1., 0., 0.],
                         [0., 0.,-1., 0.],
                         [0., 0., 0., 1.]], dtype=torch.float32)
        for frame_id in range(len(frame_id_list)):
            c2w = c2ws_all[frame_id] @ S  # Convert OpenGL to Gaussian
            ori_intrinsic = intrinsics[frame_id]

            bg_color = random.choice([0.0, 0.5, 1.0])
            rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                images[frame_id],
                mask_images[frame_id],
                bg_color=bg_color,
                max_tgt_size=None,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio,
                render_tgt_size=self.render_image_res,
                multiply=self.multiply,
                intr=ori_intrinsic.clone(),
            )

            c2ws.append(c2w)
            intrs.append(intrinsic)
            rgbs.append(rgb)
            bg_colors.append(bg_color)
            masks.append(mask)

        c2ws = torch.stack(c2ws, dim=0)
        intrs = torch.stack(intrs, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        masks = torch.cat(masks, dim=0)
        bg_colors = torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)

        # Subset varen params according to selected views
        varen_params = {}
        if 'betas' in varen_params_all:
            varen_params['betas'] = varen_params_all['betas']
        if 'global_orient' in varen_params_all:
            varen_params['global_orient'] = varen_params_all['global_orient'][frame_id_list]
        if 'pose' in varen_params_all:
            varen_params['pose'] = varen_params_all['pose'][frame_id_list]
        if 'trans' in varen_params_all:
            varen_params['trans'] = varen_params_all['trans'][frame_id_list]
        if 'tail_scale' in varen_params_all:
            varen_params['tail_scale'] = varen_params_all['tail_scale']

        # Reference/source images (pick one view)
        prob_refidx = np.ones(self.sample_side_views + 1)
        if not self.is_val:
            prob_refidx[0] = 0.5
        else:
            prob_refidx[0] = 1.0
        prob_refidx[1:] = (1 - prob_refidx[0]) / len(prob_refidx[1:])
        ref_idx = int(np.random.choice(self.sample_side_views + 1, p=prob_refidx))

        source_c2ws = c2ws[ref_idx: ref_idx + 1]
        source_intrs = intrs[ref_idx: ref_idx + 1]
        source_rgbs = rgbs[ref_idx: ref_idx + 1]

        source_varen_params = {}
        if 'betas' in varen_params:
            source_varen_params['betas'] = varen_params['betas']
        if 'global_orient' in varen_params:
            source_varen_params['global_orient'] = varen_params['global_orient'][ref_idx: ref_idx + 1]
        if 'pose' in varen_params:
            source_varen_params['pose'] = varen_params['pose'][ref_idx: ref_idx + 1]
        if 'trans' in varen_params:
            source_varen_params['trans'] = varen_params['trans'][ref_idx: ref_idx + 1]
        if 'tail_scale' in varen_params:
            source_varen_params['tail_scale'] = varen_params['tail_scale']

        render_image = rgbs
        render_mask = masks
        tgt_size = render_image.shape[2:4]
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 2.5, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 2.5, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        ret = {
            'uid': uid,
            'source_c2ws': source_c2ws,
            'source_intrs': source_intrs,
            'source_rgbs': source_rgbs.clamp(0, 1),
            'render_image': render_image.clamp(0, 1),
            'render_mask': render_mask.clamp(0, 1),
            'c2ws': c2ws,
            'intrs': intrs,
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': bg_colors,
            'pytorch3d_transpose_R': torch.tensor([0.0]),
        }

        ret["smal_params"] = varen_params
        ret["source_smal_params"] = source_varen_params

        return ret
    
    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        if len(self.sequence) == 0:
            raise IndexError("MiniVarenAvatarDataset has no samples to load.")

        # Bounded retry to avoid unbounded recursion when samples are bad.
        max_trials = len(self.sequence)
        cur_idx = idx % len(self.sequence)
        for _ in range(max_trials):
            try:
                return self.inner_get_item(cur_idx)
            except Exception:
                traceback.print_exc()
                print(f"[DEBUG-DATASET] Error when loading {self.sequence[cur_idx]}")
                cur_idx = (cur_idx + 1) % len(self.sequence)

        # If all attempts fail, surface a clear error.
        raise RuntimeError("MiniVarenAvatarDataset failed to load any sample; all trials errored.")


class MergedMiniVarenAvatarDataset(torch.utils.data.Dataset):
    """
    Mix multiple MiniVarenAvatarDataset instances using per-dataset sampling weights.
    The weight in cfg.DATASETS.<name>.WEIGHT controls sampling probability during training.
    """

    def __init__(self, cfg: CfgNode, is_train: bool = True):
        super().__init__()
        self.cfg = cfg
        self.is_val = not is_train
        # deterministic base seed so DDP / DistributedSampler can rely on __getitem__(idx)
        self.base_seed = int(cfg.get("SEED", 42)) if hasattr(cfg, "get") else 42
        # track epoch to make sampling differ across epochs while still reproducible
        self.epoch = 0
        self._epoch_perm = None
        self._epoch_perm_version = -1

        dataset_entries = []
        weights_provided = []
        for name, node in cfg.DATASETS.items():
            if isinstance(node, (CfgNode, DictConfig, dict)):
                root = node.get("ROOT", None)
                is_val_entry = bool(node.get("IS_VAL", False))
                weight = node.get("WEIGHT", None)
            elif hasattr(node, "get"):
                try:
                    root = node.get("ROOT", None)
                    is_val_entry = bool(node.get("IS_VAL", False))
                    weight = node.get("WEIGHT", None)
                except Exception:
                    continue
            else:
                continue

            if root is None or is_val_entry != self.is_val:
                continue

            weights_provided.append(weight is not None)
            dataset_entries.append((name, root, weight))

        if len(dataset_entries) == 0:
            stage = "val" if self.is_val else "train"
            raise ValueError(
                f"MergedMiniVarenAvatarDataset requires at least one dataset with ROOT and IS_VAL={self.is_val} in cfg.DATASETS for {stage} stage."
            )

        self.datasets: List[MiniVarenAvatarDataset] = []
        self.dataset_names: List[str] = []
        probs = []
        lengths = []
        for name, root_dir, weight in dataset_entries:
            ds = MiniVarenAvatarDataset(cfg, root_dir=root_dir, is_train=is_train)
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.dataset_names.append(name)
            # If weight is missing, fall back to dataset length to preserve coverage.
            if weight is None:
                probs.append(float(len(ds)))
            else:
                probs.append(max(float(weight), 0.0))
            lengths.append(len(ds))

        if len(self.datasets) == 0:
            raise ValueError("MergedMiniVarenAvatarDataset found only empty datasets.")

        prob_sum = float(sum(probs))
        if prob_sum <= 0:
            self.sample_probs = np.ones(len(probs), dtype=np.float32) / len(probs)
        else:
            self.sample_probs = np.array(probs, dtype=np.float32) / prob_sum

        self.lengths = lengths
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_len = int(sum(self.lengths))
        # If all weights were missing, prefer length-based deterministic sampling to ensure full coverage each epoch.
        self.length_based_sampling = len(weights_provided) > 0 and all(not w for w in weights_provided)

    def set_epoch(self, epoch: int):
        """Allow external samplers (e.g., DistributedSampler) to inject epoch for per-epoch shuffling."""
        self.epoch = int(epoch)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if len(self.datasets) == 0:
            raise IndexError("MergedMiniVarenAvatarDataset has no datasets to sample from.")

        # Make sampling deterministic w.r.t. idx for DDP compatibility
        if self.length_based_sampling and not self.is_val:
            # Shuffle the flattened index space once per epoch so ordering changes every epoch
            if self._epoch_perm_version != self.epoch or self._epoch_perm is None:
                rng = np.random.default_rng(self.base_seed + self.epoch)
                self._epoch_perm = rng.permutation(self.total_len)
                self._epoch_perm_version = self.epoch
            wrapped_idx = idx % self.total_len
            mapped_idx = int(self._epoch_perm[wrapped_idx])
            ds_idx = int(np.searchsorted(self.cumulative_lengths, mapped_idx, side="right"))
            prev_end = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
            sample_idx = int((mapped_idx - prev_end) % self.lengths[ds_idx])
        elif self.is_val:
            wrapped_idx = idx % self.total_len
            ds_idx = int(np.searchsorted(self.cumulative_lengths, wrapped_idx, side="right"))
            prev_end = 0 if ds_idx == 0 else self.cumulative_lengths[ds_idx - 1]
            sample_idx = int((wrapped_idx - prev_end) % self.lengths[ds_idx])
        else:
            # Probability sampling: change randomness each epoch so sequence differs across epochs
            rng = np.random.default_rng(self.base_seed + self.epoch * 1000003 + idx)
            ds_idx = int(rng.choice(len(self.datasets), p=self.sample_probs))
            sample_idx = int(rng.integers(0, self.lengths[ds_idx]))

        sample = self.datasets[ds_idx][sample_idx]
        sample["dataset_name"] = self.dataset_names[ds_idx]
        sample["dataset_idx"] = ds_idx
        return sample


class Bbox:
    def __init__(self, box, mode="whwh"):

        assert len(box) == 4
        assert mode in ["whwh", "xywh"]
        self.box = box
        self.mode = mode

    def to_xywh(self):

        if self.mode == "whwh":

            l, t, r, b = self.box

            center_x = (l + r) / 2
            center_y = (t + b) / 2
            width = r - l
            height = b - t
            return Bbox([center_x, center_y, width, height], mode="xywh")
        else:
            return self

    def to_whwh(self):

        if self.mode == "whwh":
            return self
        else:

            cx, cy, w, h = self.box
            l = cx - w // 2
            t = cy - h // 2
            r = cx + w - (w // 2)
            b = cy + h - (h // 2)

            return Bbox([l, t, r, b], mode="whwh")

    def area(self):

        box = self.to_xywh()
        _, __, w, h = box.box

        return w * h

    def get_box(self):
        return list(map(int, self.box))

    def scale(self, scale, width, height):
        new_box = self.to_xywh()
        cx, cy, w, h = new_box.get_box()
        w = w * scale
        h = h * scale

        l = cx - w // 2
        t = cy - h // 2
        r = cx + w - (w // 2)
        b = cy + h - (h // 2)

        l = int(max(l, 0))
        t = int(max(t, 0))
        r = int(min(r, width))
        b = int(min(b, height))

        return Bbox([l, t, r, b], mode="whwh")

    def __repr__(self):
        box = self.to_whwh()
        l, t, r, b = box.box

        return f"BBox(left={l}, top={t}, right={r}, bottom={b})"


class AvatarInferDataset(torch.utils.data.Dataset):
    def __init__(self, image: List[np.ndarray], mask: List[np.ndarray], intrs: Optional[np.ndarray] = None, render_image_res: int = 512):
        super().__init__()
        self.image = image
        self.mask = mask
        assert len(self.image) == len(self.mask), "Image and mask must have the same length"
        self.intrs = intrs
        self.render_image_res = render_image_res

    def __getitem__(self, idx):
        image = self.image[idx]
        image_height, image_width = image.shape[:2]
        mask = self.mask[idx]
        intr = self.intrs[idx] if self.intrs is not None else None
        image, mask, intr = self.infer_preprocess_image(image, mask, intr, render_tgt_size=self.render_image_res, max_tgt_size=self.render_image_res)
        if intr is None:
            item = {
                "source_rgbs": image,
                "source_masks": mask,
                "image_ori_height": image_height,
                "image_ori_width": image_width,
            }
        else:
            item = {
                "source_rgbs": image,
                "source_masks": mask,
                "source_intrs": intr,
                "image_ori_height": image_height,
                "image_ori_width": image_width,
            }
        return item

    def __len__(self):
        return len(self.image)

    def get_bbox(self, mask):
        height, width = mask.shape
        # Normalize mask to [0, 1], robust to boolean or 0-255 inputs
        pha = mask.astype(np.float32)
        if pha.max() > 1.0:
            pha = pha / 255.0
        pha[pha < 0.5] = 0.0
        pha[pha >= 0.5] = 1.0

        # obtain bbox
        _h, _w = np.where(pha == 1)

        whwh = [
            _w.min().item(),
            _h.min().item(),
            _w.max().item(),
            _h.max().item(),
        ]

        box = Bbox(whwh)
        # scale box to 1.05
        scale_box = box.scale(1.1, width=width, height=height)
        return scale_box

    def resize_image_keepaspect_np(self, img, max_tgt_size):
        """
        similar to ImageOps.contain(img_pil, (img_size, img_size)) # keep the same aspect ratio
        """
        h, w = img.shape[:2]
        ratio = max_tgt_size / max(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)
        return cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)

    def center_crop_according_to_mask(self, img, mask, aspect_standard, enlarge_ratio):
        """
        img: [H, W, 3]
        mask: [H, W]
        """
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("empty mask")

        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)

        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        half_w = max(abs(center_x - x_min), abs(center_x - x_max))
        half_h = max(abs(center_y - y_min), abs(center_y - y_max))
        half_w_raw = half_w
        half_h_raw = half_h
        aspect = half_h / half_w

        if aspect >= aspect_standard:
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        # not exceed original image
        if half_h > center_y:
            half_w = round(half_h_raw / aspect_standard)
            half_h = half_h_raw
        if half_w > center_x:
            half_h = round(half_w_raw * aspect_standard)
            half_w = half_w_raw

        if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
            enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
            enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
            enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
            enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
            enlarge_ratio_cur = (
                np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
                + enlarge_ratio_min
            )
            half_h, half_w = round(enlarge_ratio_cur * half_h), round(
                enlarge_ratio_cur * half_w
            )

        assert half_h <= center_y
        assert half_w <= center_x
        assert abs(half_h / half_w - aspect_standard) < 0.03

        offset_x = center_x - half_w
        offset_y = center_y - half_h

        new_img = img[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]
        new_mask = mask[offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w]

        return new_img, new_mask, offset_x, offset_y

    def calc_new_tgt_size_by_aspect(self, cur_hw, aspect_standard, tgt_size, multiply):
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
        tgt_size = tgt_size * aspect_standard, tgt_size
        tgt_size = (
            int(tgt_size[0] / multiply) * multiply,
            int(tgt_size[1] / multiply) * multiply,
        )
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def scale_intrs(self, intrs, ratio_x, ratio_y):
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y  
        return intrs

    def infer_preprocess_image(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        intr: Optional[np.ndarray] = None,
        bg_color: float = 1.0,
        max_tgt_size: int = 512,
        aspect_standard: float = 1.0,
        enlarge_ratio: List[float] = [1.0, 1.0],
        render_tgt_size: int = 512,
        multiply: int = 16,
    ):
        rgb_raw = rgb.copy()

        bbox = self.get_bbox(mask)
        bbox_list = bbox.get_box()

        rgb = rgb[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]
        mask = mask[bbox_list[1] : bbox_list[3], bbox_list[0] : bbox_list[2]]

        h, w, _ = rgb.shape
        cur_ratio = h / w
        scale_ratio = cur_ratio / aspect_standard


        target_w = int(min(w * scale_ratio, h))
        if target_w - w >0:
            offset_w = (target_w - w) // 2

            rgb = np.pad(
                rgb,
                ((0, 0), (offset_w, offset_w), (0, 0)),
                mode="constant",
                constant_values=255,
            )

            mask = np.pad(
                mask,
                ((0, 0), (offset_w, offset_w)),
                mode="constant",
                constant_values=0,
            )
        else:
            target_h = w * aspect_standard
            offset_h = int(target_h - h)

            rgb = np.pad(
                rgb,
                ((offset_h, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=255,
            )

            mask = np.pad(
                mask,
                ((offset_h, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        rgb = rgb / 255.0  # normalize to [0, 1]
        mask = mask / 255.0

        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # resize to specific size require by preprocessor of smplx-estimator.
        rgb = self.resize_image_keepaspect_np(rgb, max_tgt_size)
        mask = self.resize_image_keepaspect_np(mask, max_tgt_size)

        # crop image to enlarge human area.
        rgb, mask, offset_x, offset_y = self.center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio
        )
        if intr is not None:
            intr[0, 2] -= offset_x
            intr[1, 2] -= offset_y

        # resize to render_tgt_size for training

        tgt_hw_size, ratio_y, ratio_x = self.calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=render_tgt_size,
            multiply=multiply,
        )

        rgb = cv2.resize(
            rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )

        if intr is not None:

            # ******************** Merge *********************** #
            intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
            assert (
                abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5
            ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
            assert (
                abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5
            ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"

            # ******************** Merge *********************** #
            intr[0, 2] = rgb.shape[1] // 2
            intr[1, 2] = rgb.shape[0] // 2

        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # [3, H, W]
        mask = (
            torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1)
        )  # [1, H, W]
        return rgb, mask, intr


def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith("."):
            valid_id_list.append(file)
    print(len(valid_id_list), valid_id_list[:2])
    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)


def gen_valid_id_json():
    root_dir = "./train_data/vfhq_vhap/export"
    mask_root_dir = "./train_data/vfhq_vhap/mask"
    save_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    valid_id_list = []
    for file in os.listdir(root_dir):
        if not file.startswith(".") and ".txt" not in file:
            valid_id_list.append(file)
    print("raw:", len(valid_id_list), valid_id_list[:2])

    mask_valid_id_list = []
    for file in os.listdir(mask_root_dir):
        if not file.startswith(".") and ".txt" not in file:
            mask_valid_id_list.append(file)
    print("mask:", len(mask_valid_id_list), mask_valid_id_list[:2])

    valid_id_list = list(set(valid_id_list).intersection(set(mask_valid_id_list)))
    print("intesection:", len(mask_valid_id_list), mask_valid_id_list[:2])

    with open(save_path, "w") as fp:
        json.dump(valid_id_list, fp)
        
    save_train_path = "./train_data/vfhq_vhap/label/valid_id_train_list.json"
    save_val_path = "./train_data/vfhq_vhap/label/valid_id_val_list.json"
    valid_id_list = sorted(valid_id_list)
    idxs = np.linspace(0, len(valid_id_list)-1, num=20, endpoint=True).astype(np.int64)
    valid_id_train_list = []
    valid_id_val_list = []
    for i in range(len(valid_id_list)):
        if i in idxs:
            valid_id_val_list.append(valid_id_list[i])
        else:
            valid_id_train_list.append(valid_id_list[i])

    print(len(valid_id_train_list), len(valid_id_val_list), valid_id_val_list)
    with open(save_train_path, "w") as fp:
        json.dump(valid_id_train_list, fp)
        
    with open(save_val_path, "w") as fp:
        json.dump(valid_id_val_list, fp)


def debug_minivaren_avatar_dataset():
    """
    Debug MiniVarenAvatarDataset by loading one sample and saving images/masks.

    - ROOT: /data2/lvjin/cvpr26/data/VarenAvatar/
    - SAMPLE_SIDE_VIEWS: 4
    - Other arguments use dataset defaults
    """
    import os
    import traceback

    # Build minimal config
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar.ROOT = "/data2/lvjin/4DEquine/Animal4DEval/"
    # MiniVarenAvatarDataset reads shared settings from cfg.DATASETS directly.
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4
    cfg.DATASETS.RENDER_IMAGE_RES_LOW = 448
    cfg.DATASETS.RENDER_IMAGE_RES_HIGH = 448
    cfg.DATASETS.RENDER_REGION_SIZE = 448
    cfg.DATASETS.SOURCE_IMAGE_RES = 448

    dataset = MiniVarenAvatarDataset(cfg, root_dir=cfg.DATASETS.VarenAvatar.ROOT, is_train=False)

    print(f"[MiniVarenAvatarDataset] length: {len(dataset)}")
    save_root = "./debug_vis/mini_varen_avatar_dataset"
    os.makedirs(save_root, exist_ok=True)

    try:
        data = dataset[0]
    except Exception as e:
        traceback.print_exc()
        print(f"[DEBUG] Failed to load first sample: {e}")
        try:
            seq0 = dataset.sequence[0] if hasattr(dataset, "sequence") and len(dataset.sequence) > 0 else None
            print(f"First sequence: {seq0}")
        except Exception:
            pass
        return

    # Print keys and shapes
    print("[MiniVarenAvatarDataset] Sample 0 keys and shapes:")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            try:
                print(f"  - {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
            except Exception:
                print(f"  - {k}: tensor")
        else:
            print(f"  - {k}: type={type(v)}")

    def save_tensor_as_image(t: torch.Tensor, path: str):
        img = t.detach().cpu().clamp(0, 1)
        if img.dim() == 3 and img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        img = (img * 255.0).astype(np.uint8)
        # OpenCV expects BGR
        img = img[..., ::-1]
        cv2.imwrite(path, img)

    # Save source images
    if isinstance(data.get("source_rgbs"), torch.Tensor):
        src = data["source_rgbs"]
        for i in range(min(4, src.shape[0])):
            save_tensor_as_image(src[i], os.path.join(save_root, f"source_{i}.jpg"))

    # Save render images and masks
    if isinstance(data.get("render_image"), torch.Tensor):
        rgbs = data["render_image"]
        for i in range(min(4, rgbs.shape[0])):
            save_tensor_as_image(rgbs[i], os.path.join(save_root, f"render_{i}.jpg"))

    if isinstance(data.get("render_mask"), torch.Tensor):
        mks = data["render_mask"]
        for i in range(min(4, mks.shape[0])):
            m = mks[i]
            if m.shape[0] == 1:
                m = m.repeat(3, 1, 1)
            save_tensor_as_image(m, os.path.join(save_root, f"mask_{i}.jpg"))

    try:
        import sys
        ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from amr.models.rendering.gs_varen_renderer import GS3DRenderer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Minimal renderer config
        cfg_render = CfgNode(new_allowed=True)
        cfg_render.RENDERER = CfgNode(new_allowed=True)
        cfg_render.RENDERER.VAREN_MODEL_PATH = 'data/varen/'
        cfg_render.RENDERER.SUBDIVIDE_NUM = 1
        cfg_render.RENDERER.FEAT_DIM = 64
        cfg_render.RENDERER.QUERY_DIM = 64
        cfg_render.RENDERER.USE_RGB = True
        cfg_render.RENDERER.SH_DEGREE = 3
        cfg_render.RENDERER.XYZ_OFFSET_MAX_STEP = 1.0
        cfg_render.RENDERER.MLP_NETWORK_CONFIG = CfgNode(new_allowed=True)
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.activation = 'silu'
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.n_hidden_layers = 2
        cfg_render.RENDERER.MLP_NETWORK_CONFIG.n_neurons = 512
        cfg_render.RENDERER.SHAPE_PARAM_DIM = 39
        cfg_render.RENDERER.CLIP_SCALING = [100, 1000, 1000, 3000]
        cfg_render.RENDERER.DECODER_MLP = False
        cfg_render.RENDERER.SKIP_DECODER = True
        cfg_render.RENDERER.FIX_OPACITY = False
        cfg_render.RENDERER.FIX_ROTATION = False
        cfg_render.RENDERER.DECODE_WITH_EXTRA_INFO = None
        cfg_render.RENDERER.GRADIENT_CHECKPOINTING = False
        cfg_render.RENDERER.APPLY_POSE_BLENDSHAPE = False

        renderer = GS3DRenderer(cfg_render).to(device)

        # Build varen_data from dataset output
        def to_batched_tensor(t, is_betas=False):
            t = t.to(dtype=torch.float32, device=device)
            if is_betas:
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # [D] -> [1, D]
                return t
            if t.dim() == 2 and t.shape[-1] == 3:
                # [Nv, 3] -> [1, Nv, 1, 3]
                t = t.unsqueeze(0).unsqueeze(2)
            elif t.dim() == 3 and t.shape[-1] == 3:
                # [Nv, J, 3] or [Nv, 1, 3] -> [1, Nv, J/1, 3]
                t = t.unsqueeze(0)
            return t

        varen_data = {}
        params = data['smal_params']
        if 'betas' in params:
            varen_data['betas'] = to_batched_tensor(params['betas'], is_betas=True)
        if 'global_orient' in params:
            varen_data['global_orient'] = to_batched_tensor(params['global_orient'])
        if 'pose' in params:
            varen_data['pose'] = to_batched_tensor(params['pose'])
        if 'trans' in params:
            varen_data['trans'] = to_batched_tensor(params['trans'])
        if 'tail_scale' in params:
            varen_data['tail_scale'] = to_batched_tensor(params['tail_scale'], is_betas=True)

        # Query canonical points
        positions, varen_data = renderer.get_query_points(varen_data, device)

        B = positions.shape[0]
        N = positions.shape[1]
        query_dim = cfg_render.RENDERER.QUERY_DIM
        gs_hidden_features = torch.zeros((B, N, query_dim), dtype=torch.float32, device=device)

        # Prepare cameras and intrinsics
        c2w = data['c2ws'].to(device=device, dtype=torch.float32).unsqueeze(0)
        intr = data['intrs'].to(device=device, dtype=torch.float32).unsqueeze(0)
        H = int(data['render_image'].shape[-2])
        W = int(data['render_image'].shape[-1])
        bg = data.get('render_bg_colors', torch.ones(c2w.shape[1], 3, dtype=torch.float32)).to(device=device)
        bg = bg.unsqueeze(0)

        out = renderer.forward(
            gs_hidden_features=gs_hidden_features,
            query_points=positions,
            varen_data=varen_data,
            c2w=c2w,
            intrinsic=intr,
            height=H,
            width=W,
            additional_features=None,
            background_color=bg,
            debug=False,
            df_data=None,
        )

        # Save GS render outputs
        gs_save = os.path.join(save_root, "gs_render")
        os.makedirs(gs_save, exist_ok=True)
        comp_rgb = out.get('comp_rgb')  # [B, Nv, 3, H, W]
        if isinstance(comp_rgb, torch.Tensor):
            comp_rgb_np = (comp_rgb[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(min(4, comp_rgb_np.shape[0])):
                # OpenCV expects BGR
                cv2.imwrite(os.path.join(gs_save, f"gs_render_{i}.jpg"), comp_rgb_np[i][..., ::-1])
        comp_mask = out.get('comp_mask')  # [B, Nv, 1, H, W]
        if isinstance(comp_mask, torch.Tensor):
            mask_np = (comp_mask[0, :, 0].detach().cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(min(4, mask_np.shape[0])):
                cv2.imwrite(os.path.join(gs_save, f"gs_mask_{i}.jpg"), mask_np[i])
        print(f"[MiniVarenAvatarDataset] GS render saved to {gs_save}")
    except Exception as e:
        traceback.print_exc()
        print(f"[DEBUG] GS render failed: {e}")

    print(f"[MiniVarenAvatarDataset] Saved debug outputs to {save_root}")


def debug_minivaren_avatar_dataset_2():
    """
    For loop all samples in the dataset
    - ROOT: /data/lvjin/cvpr26/data/VarenTex/
    - SAMPLE_SIDE_VIEWS: 4
    - Other arguments use dataset defaults
    """
    # Build minimal config
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar = CfgNode(new_allowed=True)
    cfg.DATASETS.VarenAvatar.ROOT = "/data/lvjin/cvpr26/data/VarenTex/"
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4

    dataset = MiniVarenAvatarDataset(cfg, root_dir=cfg.DATASETS.VarenAvatar.ROOT, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, 
                                            num_workers=8, pin_memory=True, prefetch_factor=8)

    from tqdm import tqdm
    for item in tqdm(dataloader, total=len(dataloader)):
        continue


def debug_merged_minivaren_avatar_dataset():
    """
    Iterate over a merged multi-root MiniVarenAvatarDataset for quick sanity checks.
    Uses the hrm24g config-style fields: three roots with equal weights.
    """
    cfg = CfgNode(new_allowed=True)
    cfg.DATASETS = CfgNode(new_allowed=True)
    # Multiple datasets with weights to test merged sampling
    for name, root in [
        ("VarenAvatar", "/data2/lvjin/cvpr26/data/VarenTex"),
        ("VarenWan", "/data2/lvjin/4DEquine/VarenWan"),
        ("Animal4D", "/data2/lvjin/4DEquine/Animal4D"),
    ]:
        cfg.DATASETS[name] = CfgNode(new_allowed=True)
        cfg.DATASETS[name].ROOT = root
        cfg.DATASETS[name].WEIGHT = 1.0
    cfg.DATASETS.SAMPLE_SIDE_VIEWS = 4
    cfg.DATASETS.RENDER_IMAGE_RES_LOW = 448
    cfg.DATASETS.RENDER_IMAGE_RES_HIGH = 448
    cfg.DATASETS.RENDER_REGION_SIZE = 448
    cfg.DATASETS.SOURCE_IMAGE_RES = 448
    cfg.SEED = 42

    dataset = MergedMiniVarenAvatarDataset(cfg, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                             num_workers=8, pin_memory=True, prefetch_factor=2,
                                             drop_last=True)

    from tqdm import tqdm
    for item in tqdm(dataloader, total=len(dataset) // 8, dynamic_ncols=True):
        # Just iterate a few batches to verify shape/key consistency
        assert "dataset_name" in item, "Merged dataset should include dataset_name key"
        assert "source_rgbs" in item, "Expected source_rgbs in sample"


if __name__ == "__main__":
    # debug_minivaren_avatar_dataset()
    # debug_minivaren_avatar_dataset_2()
    debug_merged_minivaren_avatar_dataset()