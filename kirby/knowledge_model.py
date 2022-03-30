import torch
from pytorch_lightning import LightningModule
from transformers import GPT2Config, GPT2Tokenizer

from .basic_model import BasicModel
from .knowledge_gpt2_model import KnowledgeGPT2LMHeadModel
from .run_params import RunParams


class KnowledgeModel(BasicModel, LightningModule):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params=RunParams()):
        super().__init__(run_params)
        self.run_params = run_params
        if self.run_params.pretrained:
            self.model = KnowledgeGPT2LMHeadModel.from_pretrained(
                "gpt2", run_params=self.run_params
            )
        else:
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

    def generate(self, prompt, max_new_tokens):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
