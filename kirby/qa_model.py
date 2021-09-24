import logging
from logging.handlers import RotatingFileHandler

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

        log_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s"
        )

        logFile = "logs/qa_log"

        my_handler = RotatingFileHandler(
            logFile,
            mode="a",
            maxBytes=5 * 1024 * 1024,
            backupCount=2,
            encoding=None,
            delay=0,
        )
        my_handler.setFormatter(log_formatter)
        my_handler.setLevel(logging.INFO)

        self.app_log = logging.getLogger("root")
        self.app_log.setLevel(logging.INFO)

        self.app_log.addHandler(my_handler)

    def prepare_data(self):
        self.setup()

    def setup(self, stage=None):
        ds = self.prepare_ds(self.run_params.data_files)
        self.train_ds = ds[: int(ds.shape[0] * 0.8)]
        self.val_ds = ds[int(-1 * ds.shape[0] * 0.2) :]
        self.train_ds = Dataset.from_dict(self.train_ds)
        self.val_ds = Dataset.from_dict(self.val_ds)

    def prepare_ds(self, path):
        df = pd.read_pickle(path)
        if self.run_params.debug:
            df = df.iloc[
                : self.run_params.batch_size * 3 * 2 * torch.cuda.device_count()
            ]
        ds = Dataset.from_pandas(df)
        ds = ds.map(
            self.tokenize,
            batched=False,
            # num_proc=4,
            remove_columns=["question", "correct", "distractors", "knowledge"],
        )
        return ds

    def tokenize(self, x):
        question = x["question"]
        knowledge = " " + x["knowledge"] + " "
        answer = x["correct"]
        distractors = x["distractors"]
        question_tokens = self.tokenizer(
            question,
        )
        knowledge_tokens = self.tokenizer(
            knowledge,
        )
        if self.run_params.knowledge_tokenize:
            if (
                len(knowledge_tokens["input_ids"]) + len(question_tokens)
                < self.run_params.seq_length
            ):
                knowledge_length = len(knowledge_tokens["input_ids"])
                # Modify attention mask
                if self.run_params.mask_percent != 0.0:
                    perm = torch.randperm(knowledge_length)
                    mask_indices = perm[
                        : round(self.run_params.mask_percent * knowledge_length)
                    ]
                    for i in mask_indices:
                        knowledge_tokens["attention_mask"][i] = 0

                question_tokens["input_ids"] += knowledge_tokens["input_ids"]
                question_tokens["attention_mask"] += knowledge_tokens["attention_mask"]

        answer_tokens = self.tokenizer(
            answer,
            truncation=True,
            padding="max_length",
            max_length=self.run_params.seq_length
            - len(question_tokens["input_ids"])
            - 1,
        )
        distractor_tokens = [
            self.tokenizer(
                d,
                truncation=True,
                padding="max_length",
                max_length=self.run_params.seq_length
                - len(question_tokens["input_ids"])
                - 1,
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
        cls_token_indices = []
        for seq in input_ids:
            try:
                index = seq.index(50256)
            except ValueError:
                index = len(seq)
            cls_token_indices.append(index)

        for i, ids in enumerate(input_ids):
            ids.insert(cls_token_indices[i], 50257)
            attention_mask[i].insert(cls_token_indices[i], 1)
        input_ids = torch.tensor(input_ids).to(self.model.device).long()
        mc_token_ids = (input_ids == 50257).nonzero(as_tuple=True)[1]
        input_ids = input_ids.tolist()
        mc_token_ids = mc_token_ids.tolist()
        labels = 3
        for i, ids in enumerate(input_ids):
            try:
                assert len(ids) == self.run_params.seq_length
                assert len(attention_mask[i]) == self.run_params.seq_length
                try:
                    index = ids.index(50256) - 1
                    assert mc_token_ids[i] == index
                except ValueError:
                    assert mc_token_ids[i] == self.run_params.seq_length - 1
            except Exception:
                __import__("pudb").set_trace()
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
        self.datalog(input_ids, attention_mask, mc_token_ids, labels)
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

    def datalog(self, input_ids, attention_mask, mc_token_ids, labels):
        for i, batch in enumerate(input_ids):
            for j, choice in enumerate(batch):
                id_str = self.tokenizer.decode(choice[: mc_token_ids[i][j]])
                self.app_log.debug("input_ids: %s", id_str)
                self.app_log.debug("attention_mask: %s", attention_mask[i][j])
                self.app_log.debug("mc_token_ids: %s", mc_token_ids[i][j])
            self.app_log.debug("label: %s", labels[i])

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
