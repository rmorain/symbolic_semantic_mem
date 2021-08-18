import torch
from pytorch_lightning import LightningModule
from transformers import GPT2Config

from .basic_model import BasicModel
from .knowledge_gpt2_model import KnowledgeGPT2LMHeadModel


class KnowledgeModel(BasicModel, LightningModule):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params):
        super().__init__(run_params)
        self.run_params = run_params
        config = GPT2Config()
        config.knowledge_buffer = self.run_params.knowledge_buffer
        self.model = KnowledgeGPT2LMHeadModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, x):
        input_ids = torch.tensor(x["input_ids"][0]).to(self.model.device).long()
        attention_mask = (
            torch.tensor(x["attention_mask"][0]).to(self.model.device).long()
        )
        labels = torch.tensor(x["labels"][0]).to(self.model.device).long()
        loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )[0]
        return loss
