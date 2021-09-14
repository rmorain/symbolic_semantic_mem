import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from datasets import Dataset
from pytorch_lightning import LightningModule
from transformers import GPT2Config, GPT2DoubleHeadsModel, GPT2Tokenizer

from .basic_model import BasicModel


class QAModel(BasicModel, LightningModule):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params):
        super().__init__(run_params)
        self.run_params = run_params
        self.model = GPT2DoubleHeadsModel.from_pretrained(self.run_params.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add [CLS] to the vocabulary
        self.tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        # Update the model embeddings with the new vocabulary size
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        self.train_ds = self.prepare_ds(self.run_params.data_files["train"])
        self.val_ds = self.prepare_ds(self.run_params.data_files["valid"])

    def prepare_ds(self, path):
        df = pd.read_pickle(path)
        if self.run_params.debug:
            df = df.iloc[: self.run_params.batch_size * 3 * torch.cuda.device_count()]
        ds = Dataset.from_pandas(df)
        ds = ds.map(
            self.tokenize,
            batched=False,
            # num_proc=4,
            remove_columns=["question", "correct", "distractors"],
        )
        return ds

    def tokenize(self, x):
        question = x["question"] + " "
        answer = x["correct"] + " [CLS]"
        distractors = [d + " [CLS]" for d in x["distractors"]]
        question_tokens = self.tokenizer(
            question,
        )
        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=self.run_params.seq_length - len(question_tokens["input_ids"]),
        )
        distractor_tokens = [
            self.tokenizer(
                d,
                truncation=True,
                padding="max_length",
                max_length=self.run_params.seq_length
                - len(question_tokens["input_ids"]),
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
        input_ids = [
            result["question_tokens_ids"] + d for d in result["distractor_token_ids"]
        ]
        attention_mask = [
            result["question_tokens_mask"] + d for d in result["distractor_token_mask"]
        ]
        input_ids.append(result["question_tokens_ids"] + result["answer_tokens_ids"])
        attention_mask.append(
            result["question_tokens_mask"] + result["answer_tokens_mask"]
        )
        input_ids = torch.tensor(input_ids).to(self.model.device).long()
        mc_token_ids = (input_ids == 50257).nonzero(as_tuple=True)[1]
        input_ids = input_ids.tolist()
        mc_token_ids = mc_token_ids.tolist()
        labels = 3
        for i in input_ids:
            assert len(i) == self.run_params.seq_length
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mc_token_ids": mc_token_ids,
            "labels": labels,
        }

        return result

    def forward(self, x):
        input_ids = self.fix_batch(x["input_ids"])
        attention_mask = self.fix_batch(x["attention_mask"])
        mc_token_ids = [
            [sequence.tolist() for sequence in choice] for choice in x["mc_token_ids"]
        ]
        mc_token_ids = torch.tensor(mc_token_ids, device=self.model.device)
        mc_token_ids = mc_token_ids.transpose(0, 1)
        mc_token_ids = mc_token_ids.reshape((self.run_params.batch_size, 4))
        mc_token_ids = mc_token_ids.contiguous()
        labels = x["labels"]
        # Forward
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            mc_token_ids=mc_token_ids,
            mc_labels=labels,
        )
        preds = F.softmax(outputs.mc_logits, dim=1)

        loss = outputs.mc_loss
        targets = labels.clone().detach()
        return loss, preds, targets

    def fix_batch(self, choices):
        result = [
            [[t.tolist() for t in sequence] for sequence in choice]
            for choice in choices
        ]
        result = torch.tensor(result, device=self.model.device)
        result = result.transpose(2, 1)
        result = result.transpose(1, 0)
        result = result.contiguous()
        return result

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.forward(batch)
        self.log("train_loss", loss)
        return loss, preds, targets

    def training_step_end(self, batch):
        loss, preds, targets = batch
        self.train_acc(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.forward(batch)

        self.log("val_loss", loss)
        return loss, preds, targets

    def validation_step_end(self, batch):
        loss, preds, targets = batch
        self.val_acc(preds, targets)
        return loss

    def training_epoch_end(self, losses):
        self.log("train_acc_epoch", self.train_acc.compute())

    def validation_epoch_end(self, losses):
        loss = torch.cat([loss.unsqueeze(0) for loss in losses], 0).mean()
        self.log("val_loss", loss)
        self.log("val_acc_epoch", self.val_acc.compute())
