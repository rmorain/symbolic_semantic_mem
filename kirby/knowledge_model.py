import torch
from transformers import GPT2Config

from .basic_model import BasicModel
from .knowledge_gpt2_model import KnowledgeGPT2LMHeadModel


class KnowledgeModel(BasicModel):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params):
        super().__init__(run_params)
        self.run_params = run_params
        config = GPT2Config()
        config.knowledge_buffer = self.run_params.knowledge_buffer
        self.model = KnowledgeGPT2LMHeadModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, x):
        loss = self.model(
            x["input_ids"][0],
            attention_mask=x["attention_mask"][0],
            labels=x["labels"][0],
        )[0]
        return loss
