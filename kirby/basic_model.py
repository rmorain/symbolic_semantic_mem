__all__ = ["BasicModel"]

import pytorch_lightning as pl
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .data_manager import DataManager


class BasicModel(pl.LightningModule):
    def __init__(self, run_params):
        super().__init__()
        self.run_params = run_params
        if self.run_params.pretrained:
            self.model = GPT2LMHeadModel.from_pretrained(self.run_params.model)
        else:
            config = GPT2Config()
            self.model = GPT2LMHeadModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def prepare_data(self):
        data_manager = DataManager(self.run_params)
        self.train_ds, self.val_ds = data_manager.prepare_data()

    def forward(self, x):
        loss = self.model(
            x["input_ids"], attention_mask=x["attention_mask"], labels=x["input_ids"]
        )[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, losses):
        loss = torch.cat([loss.unsqueeze(0) for loss in losses], 0).mean()
        self.log("val_loss", loss)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.run_params.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.run_params.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.run_params.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.run_params.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
