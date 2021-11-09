import torch
from pytorch_lightning import LightningModule
from transformers import GPT2Config, GPT2Tokenizer

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
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.run_params.model)

    def forward(self, x):
        input_ids = x["input_ids"][0]
        attention_mask = x["attention_mask"][0]
        labels = x["labels"][0]
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=self.run_params.output_attentions,
        )
        loss = outputs[0]
        if self.run_params.output_attentions:
            return outputs.attentions
        return loss
