__all__ = ["Experiment"]

import os
import pytorch_lightning as pl
from .basic_model import BasicModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Experiment:
    def __init__(self, run_params):
        self.run_params = run_params
        self.model = BasicModel(run_params)

    def run(self):
        trainer = pl.Trainer(
            gpus=self.run_params.num_gpus,
            max_epochs=self.run_params.max_epochs,
            fast_dev_run=self.run_params.debug,
            logger=WandbLogger(
                name=self.run_params.run_name, project=self.run_params.project_name
            ),
            callbacks=[EarlyStopping(monitor="val_loss")],
            default_root_dir=os.getcwd() + "/../checkpoints",
        )

        trainer.fit(self.model)
        trainer.save_checkpoint(self.run_params.run_name())
