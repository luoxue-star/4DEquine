# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video
[**Arxiv**]() | [**Project Page**](https://luoxue-star.github.io/4DEquine_Project_Page/)

## Environment Setup
```bash
git clone --recursive https://github.com/luoxue-star/4DEquine.git
cd 4DEquine
conda create -n 4DEquine python=3.10
conda activate 4DEquine
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .[all]
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Inference
Download the checkpoints from [here](https://drive.google.com/drive/folders/1YDtNaKmueDoyP-NgySrp-BtJR0Pv9ftk?usp=sharing).

First, run AniMoFormer:
```bash
python post_optimization_from_video.py --video_path /path/to/video --checkpoint /path/to/checkpoint --cfg /path/to/config --output_dir /path/to/output_folder --stage1 --stage2
```

After AniMoFormer finishes, run EquineGS:
```bash
python demo_avatar.py --animation_params_path /path/to/refined_results.pt --track_mask_file /path/to/mask_list.pkl --img_path /path/to/image --checkpoint /path/to/checkpoint --out_folder /path/to/output_folder
```

## Evaluation
### AniMoFormer
Download the dataset JSON files from [here](https://drive.google.com/drive/folders/1YDtNaKmueDoyP-NgySrp-BtJR0Pv9ftk?usp=sharing).

Generate AniMoFormer predictions:
```bash
python post_optimization_from_video.py --video_path /path/to/sequences_folder --checkpoint /path/to/checkpoint --cfg /path/to/config --output_dir /path/to/output_folder --stage1 --stage2
```

Then set the dataset paths in `amr/configs_hydra/experiment/default_val.yaml` and run:
```bash
python eval_pose.py --config /path/to/config
```

### EquineGS
```bash
python eval_avatar.py --checkpoint /path/to/checkpoint --out_folder /path/to/output_folder --image_dir /path/to/image_sequences_dir --postrefine_dir /path/to/post_optimization_output_dir
```

## Training
Download the training data from [here](https://drive.google.com/drive/folders/1YDtNaKmueDoyP-NgySrp-BtJR0Pv9ftk?usp=sharing), then update the dataset paths in `amr/configs_hydra/experiment/pose.yaml` and `amr/configs_hydra/experiment/hrm.yaml`.

### Train AniMoFormer
```bash
bash training_scripts/animoformer.sh
```

### Train EquineGS
```bash
bash training_scripts/equinegs.sh
```

## Citation
If you find this code useful for your research, please consider citing the following paper:
```bash

```

## Contact
For questions about this implementation, please contact [Jin Lyu](mailto:lvjin1766@gmail.com).
