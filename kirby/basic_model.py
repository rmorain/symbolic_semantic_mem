# AUTOGENERATED! DO NOT EDIT! File to edit: 06_basic_model.ipynb (unless otherwise specified).

__all__ = ['BasicModel']

# Cell
import pytorch_lightning as pl
from .run_params import RunParams

# Cell
class BasicModel(pl.LightningModule):
    def __init__(self, run_params):
        super().__init__()
        pass

    def prepare_data(self):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def configure_optimizer(self):
        pass