import numpy as np
import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import default_collate
from amr.utils import recursive_to
from amr.datasets.vitdet_dataset import ViTDetDatasetTemporal
from typing import Optional


class VideoPredictor:
    def __init__(self, 
                model, 
                model_cfg,
                frames: List[np.ndarray], 
                valid_idx: np.ndarray, 
                bboxes: np.ndarray,
                mask_list: Optional[List[np.ndarray]] = None,
                device: str = "cuda"):
        self.model = model
        self.model_cfg = model_cfg
        self.frames = np.stack(frames, axis=0)
        self.valid_idx = valid_idx
        self.bboxes = bboxes
        self.device = device
        self.mask_list = mask_list

    def parse_chunks(self, min_len=16):
        """ 
        If a track disappear in the middle, 
        we separate it to different segments to estimate the HPS independently. 
        If a segment is less than 16 frames, we get rid of it for now. 
        """
        frame_chunks = []
        boxes_chunks = []
        step = self.valid_idx[1:] - self.valid_idx[:-1]
        step = np.concatenate([[0], step])
        breaks = np.where(step != 1)[0]

        start = 0
        for bk in breaks:
            f_chunk = self.valid_idx[start:bk]
            b_chunk = self.bboxes[start:bk]
            start = bk
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

            if bk == breaks[-1]:  # last chunk
                f_chunk = self.valid_idx[bk:]
                b_chunk = self.bboxes[bk:]
                if len(f_chunk) >= min_len:
                    frame_chunks.append(f_chunk)
                    boxes_chunks.append(b_chunk)

        return frame_chunks, boxes_chunks

    def inference(self):
        frame_chunks, boxes_chunks = self.parse_chunks(min_len=min(len(self.frames), len(self.valid_idx), len(self.bboxes)))

        if len(frame_chunks) == 0:
            return

        img = []
        mask = []
        pred_cam = []
        pred_cam_t = []
        pred_global_orient = []
        pred_pose = []
        pred_betas = []
        bbox_size = []
        box_center = []
        trans = []

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            img_ck = self.frames[frame_ck]
            results = self.inference_chunk(img_ck, boxes_ck)

            img.append(results['img'])
            mask.append(results['mask'])
            pred_cam.append(results['pred_cam'])
            pred_cam_t.append(results['pred_cam_t'])
            pred_global_orient.append(results['pred_global_orient'])
            pred_pose.append(results['pred_pose'])
            pred_betas.append(results['pred_betas'])
            bbox_size.append(results['bbox_size'])
            box_center.append(results['box_center'])
            trans.append(results['trans'])

        results = {'img': torch.cat(img),
                   'mask': torch.cat(mask),
                   'pred_cam': torch.cat(pred_cam),
                   'pred_cam_t': torch.cat(pred_cam_t),
                   'pred_global_orient': torch.cat(pred_global_orient),
                   'pred_pose': torch.cat(pred_pose),
                   'pred_betas': torch.cat(pred_betas),
                   'bbox_size': torch.cat(bbox_size),
                   'box_center': torch.cat(box_center),
                   'trans': torch.cat(trans)}
        return results

    def inference_chunk(self, image: List[np.ndarray], boxes: List[np.ndarray]):
        dataset = ViTDetDatasetTemporal(self.model_cfg, image, boxes, mask_list=self.mask_list)

        # Results
        img = []
        mask = []
        pred_cam = []
        pred_cam_t = []
        pred_global_orient = []
        pred_pose = []
        pred_betas = []
        bbox_size = []
        box_center = []
        trans = []

        items = []
        if len(dataset) < 16:
            # Process all items at once when dataset is smaller than batch size
            items = [dataset[i] for i in range(len(dataset))]
            batch = default_collate(items)

            with torch.no_grad():
                batch = recursive_to(batch, self.device)
                batch['img'] = batch['img'].unsqueeze(0)
                out = self.model(batch)
                expected_out = dict(pred_cam=out['pred_cam'],
                                    mask=batch['mask'],
                                    pred_cam_t=out['pred_cam_t'],
                                    pred_global_orient=out['pred_smal_params']['global_orient'],
                                    pred_pose=out['pred_smal_params']['pose'],
                                    pred_betas=out['pred_smal_params']['betas'],
                                    img=batch['img'].squeeze(0),
                                    box_size=batch['box_size'],
                                    box_center=batch['box_center'],
                                    trans=batch['trans'])

            results = {'mask': expected_out['mask'],
                       'pred_cam': expected_out['pred_cam'],
                       'pred_cam_t': expected_out['pred_cam_t'],
                       'pred_global_orient': expected_out['pred_global_orient'],
                       'pred_pose': expected_out['pred_pose'],
                       'pred_betas': expected_out['pred_betas'],
                       'img': expected_out['img'],
                       'bbox_size': expected_out['box_size'],
                       'box_center': expected_out['box_center'],
                       'trans': expected_out['trans']}
            return results

        for i in tqdm(range(len(dataset)), desc="Inference"):
            item = dataset[i]
            items.append(item)

            if len(items) < 16:
                continue
            elif len(items) == 16:
                batch = default_collate(items)
            else:
                items.pop(0)
                batch = default_collate(items)

            with torch.no_grad():
                batch = recursive_to(batch, self.device)
                batch['img'] = batch['img'].unsqueeze(0)
                out = self.model(batch)
                expected_out = dict(pred_cam=out['pred_cam'],
                                    mask=batch['mask'],
                                    pred_cam_t=out['pred_cam_t'],
                                    pred_global_orient=out['pred_smal_params']['global_orient'], 
                                    pred_pose=out['pred_smal_params']['pose'], 
                                    pred_betas=out['pred_smal_params']['betas'],
                                    img=batch['img'].squeeze(0),
                                    box_size=batch['box_size'],
                                    box_center=batch['box_center'],
                                    trans=batch['trans'])

            if len(dataset) == 16:  # The video only has 16 frames
                out = expected_out
            elif i == 15:  # First 16 frames
                out = {k: v[:9] for k,v in expected_out.items()}
            elif i == len(dataset) - 1:  # Last 16 frames
                out = {k: v[8:] for k,v in expected_out.items()}
            else:  # Middle 16 frames
                out = {k: v[[8]] for k,v in expected_out.items()}
                
            mask.append(out['mask'])
            pred_cam.append(out['pred_cam'])
            pred_cam_t.append(out['pred_cam_t'])
            pred_global_orient.append(out['pred_global_orient'])
            pred_pose.append(out['pred_pose'])
            pred_betas.append(out['pred_betas'])
            img.append(out['img'])
            bbox_size.append(out['box_size'])
            box_center.append(out['box_center'])
            trans.append(out['trans'])

        results = {'mask': torch.cat(mask),
                   'pred_cam': torch.cat(pred_cam),
                   'pred_cam_t': torch.cat(pred_cam_t),
                   'pred_global_orient': torch.cat(pred_global_orient),
                   'pred_pose': torch.cat(pred_pose),
                   'pred_betas': torch.cat(pred_betas),
                   'img': torch.cat(img),
                   'bbox_size': torch.cat(bbox_size),
                   'box_center': torch.cat(box_center),
                   'trans': torch.cat(trans)}
        return results