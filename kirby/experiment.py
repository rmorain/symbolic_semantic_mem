__all__ = ["Experiment"]

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


class Experiment:
    def __init__(self, run_params, model):
        self.run_params = run_params
        self.model = model

    def run(self):
        trainer = pl.Trainer(
            gpus=self.run_params.num_gpus,
            plugins=DDPPlugin(find_unused_parameters=False),
            max_epochs=self.run_params.max_epochs,
            fast_dev_run=self.run_params.debug,
            logger=WandbLogger(
                name=self.run_params.run_name, project=self.run_params.project_name
            ),
            callbacks=[
                EarlyStopping(monitor="val_loss"),
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
