__all__ = ["DataManager"]

import copy
import random

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class DataManager:
    def __init__(self, run_params):
        self.run_params = run_params
        self.block_size = 128

    # Load, Tokenize, and Augment data
    def prepare_data(self):
        train_ds, val_ds = map(self.prepare_ds, ("train", "valid"))
        return train_ds, val_ds

    def prepare_ds(self, split):
        tokenizer = AutoTokenizer.from_pretrained(self.run_params.model)
        if not self.run_params.bert:
            tokenizer.pad_token = tokenizer.eos_token
        df = pd.read_pickle(self.run_params.data_files[split][0])

        if self.run_params.debug:
            # df = df.iloc[: self.run_params.batch_size * 3 * torch.cuda.device_count()]
            df = df.iloc[: self.run_params.batch_size * 3 * 1]

        ds = Dataset.from_pandas(df)
        ds = ds.filter(function=self.criteria)
        tokenize_func = (
            self.tokenize_knowledge
            if self.run_params.knowledge_tokenize
            else self.tokenize
        )

        ds = ds.map(
            tokenize_func,
            batched=False,
            num_proc=self.run_params.num_workers,
            remove_columns=self.get_remove_columns(ds),
            fn_kwargs={"tokenizer": tokenizer},
        )
        ds = ds.filter(function=self.right_length)
        format_settings = {"type": "torch", "format_kwargs": {"dtype": torch.long}}
        ds.set_format(**format_settings)
        return ds

    def get_remove_columns(self, ds):
        keys = [x for x in ds.features.keys()]
        keys.remove("entity_index")
        return keys

    def get_num_rows(self, num_rows):
        if self.run_params.debug:
            return self.run_params.batch_size

    # Tokenize a sequence
    def tokenize(self, x, tokenizer=None):
        text_length = self.run_params.seq_length
        text_tokens = tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=text_length,
            return_tensors="pt",
        )
        return text_tokens

    def tokenize_knowledge(self, x, tokenizer=None):
        text_length = self.run_params.seq_length
        know_length = self.run_params.knowledge_buffer
        total_length = text_length + know_length
        if self.run_params.knowledge_front:
            knowledge_tokens = tokenizer(
                x["knowledge"],
                truncation=True,
                max_length=text_length,
                return_tensors="pt",
            )
            text_tokens = tokenizer(
                x["text"],
                truncation=True,
                padding="max_length",
                max_length=total_length - len(knowledge_tokens["input_ids"][0]),
                return_tensors="pt",
            )
            # Combine knowledge and text tokens
            input_ids = torch.cat(
                (knowledge_tokens["input_ids"], text_tokens["input_ids"]), 1
            )
            attention_mask = torch.cat(
                (knowledge_tokens["attention_mask"], text_tokens["attention_mask"]),
                1,
            )
            labels = copy.deepcopy(input_ids)
            labels[:, : len(knowledge_tokens["input_ids"][0])] = -100
        else:
            input_ids, attention_mask, labels = None, None, None
            keys = [i for i in x.keys()]
            if "entity_index" in keys:
                keys.remove("entity_index")
            text_index = keys.index("text")
            if text_index != 0:
                keys[0], keys[text_index] = keys[text_index], keys[0]
            for key in keys:
                if type(x[key]) != str:
                    continue
                if input_ids is None:
                    tokens = tokenizer(
                        x[key],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.run_params.seq_length,
                        padding="max_length",
                    )
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]
                    # Text is tokenized first
                    labels = copy.deepcopy(tokens["input_ids"])
                    # Use the excess text tokens as knowledge tokens
                    if len(keys) == 1:
                        tokens = tokenizer(
                            x[key],
                            return_tensors="pt",
                        )
                        knowledge_tokens = tokens["input_ids"][
                            :, self.run_params.seq_length : total_length
                        ]
                        knowledge_mask = tokens["attention_mask"][
                            :, self.run_params.seq_length : total_length
                        ]
                        input_ids = torch.cat((input_ids, knowledge_tokens), 1)
                        attention_mask = torch.cat(
                            (
                                attention_mask,
                                knowledge_mask,
                            ),
                            1,
                        )
                else:
                    tokens = tokenizer(
                        x[key],
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.run_params.knowledge_buffer
                        // (len(x.keys()) - 1),
                    )
                    input_ids = torch.cat((input_ids, tokens["input_ids"]), 1)
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            tokens["attention_mask"],
                        ),
                        1,
                    )
            knowledge_length = input_ids.shape[-1] - labels.shape[-1]
            prediction_mask = torch.tensor(
                [-100 for _ in range(knowledge_length)]
            ).unsqueeze(0)
            labels = torch.cat((labels, prediction_mask), 1)
            # Add padding if necessary
            if input_ids.shape[-1] < total_length:
                pad_length = total_length - input_ids.shape[-1]
                padding = torch.tensor(
                    [tokenizer.pad_token_id for _ in range(pad_length)]
                ).unsqueeze(0)
                attention_padding = torch.tensor(
                    [0 for _ in range(pad_length)]
                ).unsqueeze(0)
                label_padding = torch.tensor(
                    [-100 for _ in range(pad_length)]
                ).unsqueeze(0)
                input_ids = torch.cat((input_ids, padding), 1)
                attention_mask = torch.cat((attention_mask, attention_padding), 1)
                labels = torch.cat((labels, label_padding), 1)
            elif input_ids.shape[-1] > total_length:
                input_ids = input_ids[:, :total_length]
                attention_mask = attention_mask[:, :total_length]
                labels = labels[:, :total_length]
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        return result

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported
        # it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        if total_length == 0:
            total_length = self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def load(self, split):
        split = self.get_split(split)
        ds = load_dataset(
            self.run_params.data_file_type,
            data_files=self.run_params.data_files,
            split=split,
        )
        ds = ds.filter(function=self.criteria)
        return ds

    def right_length(self, x):
        if (
            self.run_params.knowledge_tokenize
            and len(x["input_ids"][0])
            != self.run_params.seq_length + self.run_params.knowledge_buffer
        ):
            return False
        elif (
            not self.run_params.knowledge_tokenize
            and len(x["input_ids"][0]) != self.run_params.seq_length
        ):
            return False
        return True

    def criteria(self, x):
        x = x["text"]
        # Remove blanks
        if len(x) == 1:
            return False
        # Remove headings
        if x[0:2] == " =":
            return False
        return True
