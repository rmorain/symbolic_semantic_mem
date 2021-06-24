import copy

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
        combined_input = torch.cat((x["input_ids"], x["knowledge"]), 1)
        labels = copy.deepcopy(combined_input)
        # Mask knowledge tokens so they don't contribute to the loss
        labels[:, -self.run_params.knowledge_buffer :] = -100
        combined_attention = torch.cat((x["attention_mask"], x["knowledge_mask"]), 1)
        loss = self.model(
            combined_input, attention_mask=combined_attention, labels=labels
        )[0]
        return loss
