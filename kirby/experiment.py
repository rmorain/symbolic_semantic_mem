__all__ = ["Experiment"]

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
import wandb


class Experiment:
    def __init__(self, run_params, model):
        self.run_params = run_params
        self.model = model
        pl.utilities.seed.seed_everything(seed=self.run_params.random_seed, workers=True)

        # Log config
        wandb.init(config={
            "batch_size" : self.run_params.batch_size,
            "lr" : self.run_params.lr,
            "patience" : self.run_params.patience,
            "run_name" : self.run_params.run_name,
            "seq_length" : self.run_params.seq_length,
            "knowledge_buffer" : self.run_params.knowledge_buffer,
            "momentum" : self.run_params.momentum,
            "model" : self.run_params.model,
            "pretrained" : self.run_params.pretrained,
            "num_workers" : self.run_params.num_workers,
            "num_gpus" : self.run_params.num_gpus,
            "random_seed" : self.run_params.random_seed,
            })


    def run(self):
        trainer = pl.Trainer(
            gpus=self.run_params.num_gpus,
            plugins=DDPPlugin(find_unused_parameters=False),
            max_epochs=self.run_params.max_epochs,
            fast_dev_run=self.run_params.debug,
            accelerator="ddp",
            logger=WandbLogger(
                name=self.run_params.run_name, project=self.run_params.project_name
            ),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=self.run_params.patience),
                ModelCheckpoint(
                    dirpath=os.getcwd() + "/checkpoints",
                    filename="{epoch}-{val_loss:.2f}-" + self.run_params.run_name,
                    monitor="val_loss",
                    save_top_k=1,
                ),
            ],
            default_root_dir=os.getcwd() + "/checkpoints",
        )

        trainer.fit(self.model)
