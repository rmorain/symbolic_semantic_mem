import pandas as pd
import torch
from datasets import Dataset
from pytorch_lightning import LightningModule
from transformers import GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer

from .basic_model import BasicModel


class QAModel(BasicModel, LightningModule):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params):
        super().__init__(run_params)
        self.run_params = run_params
        config = GPT2Config()
        self.model = GPT2DoubleHeadsModel(config)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Add [CLS] to the vocabulary
        self.tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        # Update the model embeddings with the new vocabulary size
        self.model.resize_token_embeddings(len(self.tokenizer))

    def setup(self, stage):
        df = pd.read_pickle(self.run_params.data_files["train"])
        if self.run_params.debug:
            df = df.iloc[: self.run_params.batch_size * 3 * torch.cuda.device_count()]
        self.train_ds = Dataset.from_pandas(df)
        self.train_ds = self.train_ds.map(
            self.tokenize,
            batched=False,
            # num_proc=4,
            remove_columns=["question", "correct", "distractors"],
        )

    def tokenize(self, x):
        question = x["question"]
        answer = x["correct"] + " [CLS]"
        distractors = [d + " [CLS]" for d in x["distractors"]]
        question_tokens = self.tokenizer(
            question,
            truncation=True,
            max_length=self.run_params.seq_length,
        )
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.run_params.seq_length,
        )
        distractor_tokens = [
            self.tokenizer(
                d,
                truncation=True,
                max_length=self.run_params.seq_length,
            )
            for d in distractors
        ]
        result = self.get_result(question_tokens, answer_tokens, distractor_tokens)

        return result

    def get_result(self, question_tokens, answer_tokens, distractor_tokens):
        input_ids = question_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = (
            question_tokens["attention_mask"] + answer_tokens["attention_mask"]
        )
        for d in distractor_tokens:
            input_ids = input_ids + d["input_ids"]
            attention_mask = attention_mask + d["attention_mask"]
        cls_token_location = [
            i if (token == self.tokenizer.cls_token_id) else None
            for i, token in enumerate(input_ids)
        ]
        cls_token_location = list(filter(None, cls_token_location))
        mc_token_ids = torch.tensor([cls_token_location])
        return {
            "input_ids": torch.tensor(input_ids).unsqueeze(0),
            "mc_token_ids": mc_token_ids,
            "attention_mask": torch.tensor([attention_mask]).unsqueeze(0),
        }

    def forward(self, x):
        input_ids = torch.tensor(x["input_ids"][0]).to(self.model.device).long()
        attention_mask = (
            torch.tensor(x["attention_mask"][0]).to(self.model.device).long()
        )
        mc_token_ids = torch.tensor(x["mc_token_ids"][0]).to(self.model.device).long()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            mc_token_ids=mc_token_ids,
        )
        return outputs
