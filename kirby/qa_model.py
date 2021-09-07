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
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        )
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=self.run_params.seq_length,
        )
        distractor_tokens = [
            self.tokenizer(
                d,
                truncation=True,
                padding="max_length",
                max_length=self.run_params.seq_length,
            )
            for d in distractors
        ]

        result = self.get_result(question_tokens, answer_tokens, distractor_tokens)
        return result

    def get_result(self, question_tokens, answer_tokens, distractor_tokens):
        result = {
            "question_tokens_ids": question_tokens["input_ids"],
            "question_tokens_mask": question_tokens["attention_mask"],
            "answer_tokens_ids": answer_tokens["input_ids"],
            "answer_tokens_mask": answer_tokens["attention_mask"],
        }

        result["distractor_token_ids"] = [d["input_ids"] for d in distractor_tokens]
        result["distractor_token_mask"] = [
            d["attention_mask"] for d in distractor_tokens
        ]

        return result

    def forward(self, x):
        input_ids = [x["question_tokens_ids"] + d for d in x["distractor_token_ids"]]
        attention_mask = [
            x["question_tokens_mask"] + d for d in x["distractor_token_mask"]
        ]
        input_ids.append(x["question_tokens_ids"] + x["answer_tokens_ids"])
        attention_mask.append(x["question_tokens_mask"] + x["answer_tokens_mask"])
        input_ids = torch.tensor(input_ids).to(self.model.device).long()
        attention_mask = torch.tensor(attention_mask).to(self.model.device).long()
        attention_mask = torch.reshape(attention_mask, input_ids.shape)
        mc_token_ids = (input_ids == 50257).nonzero(as_tuple=True)[1]
        labels = torch.tensor([0, 0, 0, 1]).to(self.model.device).long()
        # Shuffle everything
        random_indices = torch.randperm(input_ids.shape[0])
        input_ids = input_ids[random_indices]
        attention_mask = attention_mask[random_indices]
        mc_token_ids = mc_token_ids[random_indices]
        labels = labels[random_indices]
        labels = torch.argmax(labels)
        # Unsqueeze
        input_ids.unsqueeze_(0)
        attention_mask.unsqueeze_(0)
        mc_token_ids.unsqueeze_(0)
        # Forward
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            mc_token_ids=mc_token_ids,
            mc_labels=labels,
        )
        return outputs
