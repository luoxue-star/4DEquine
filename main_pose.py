from typing import Optional
from tqdm import tqdm
import hydra
import pyrootutils
from amr.models.varen_amr import AniMerVAREN
from amr.datasets import VARENDataModule


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import TQDMProgressBar
from amr.utils.pylogger import get_pylogger
from amr.utils.misc import task_wrapper, log_hyperparameters
import signal
import wandb

os.environ['WANDB_BASE_URL'] = "https://api.bandw.top"
# Set wandb to offline mode for local logging only
os.environ['WANDB_MODE'] = 'offline'
signal.signal(signal.SIGUSR1, signal.SIG_DFL)


class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()
        
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ncols = 150
        bar.dynamic_ncols=False
        return bar

    def init_validation_tqdm(self):
        bar = tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            dynamic_ncols= False,
            ncols = 150,
        )
        return bar

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        if 'v_num' in items:
            items.pop('v_num')
        return items


@hydra.main(version_base="1.2", config_path=str(root / "amr/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule = VARENDataModule(cfg)
    model = AniMerVAREN(cfg)

    # Setup wandb logger
    wandb_logger_kwargs = dict(
        name=cfg.WANDB.get("NAME", cfg.exp_name),
        save_dir=cfg.paths.output_dir,
        project=cfg.WANDB.get("PROJECT_NAME", "AniMer_VAREN"),
        entity=cfg.get("ENTITY"),
        log_model=False,
        job_type="train",
        # Convert the OmegaConf object to a primitive dict to avoid serialization errors
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    resume_id = cfg.WANDB.get("ID") or cfg.WANDB.get("RUN_ID")
    resume_mode = cfg.WANDB.get("RESUME", "allow")
    if resume_id:
        wandb_logger_kwargs["id"] = resume_id
        wandb_logger_kwargs["resume"] = resume_mode
        os.environ.setdefault("WANDB_RUN_ID", resume_id)
        os.environ.setdefault("WANDB_RESUME", resume_mode)

    logger = WandbLogger(**wandb_logger_kwargs)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'),
        every_n_epochs=cfg.GENERAL.CHECKPOINT_EPOCHS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor="val/loss",
        mode="min"
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        MyTQDMProgressBar()
    ]

    log = get_pylogger(__name__)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher', None) is not None) else None),
        sync_batchnorm=True,
        use_distributed_sampler=False,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        # This function will still work, but the primary config is already logged by WandbLogger
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
    log.info("Fitting done")

    # Finalize wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
