python main_avatar.py exp_name=test experiment=hrm trainer=gpu_avatar launcher=local \
       trainer=ddp trainer.devices=4 \
       GENERAL.VAL_STEPS=10000 \
       TRAIN.BATCH_SIZE=2 \
       WANDB.MODE=offline
