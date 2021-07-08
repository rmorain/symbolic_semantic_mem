__all__ = ["DataManager"]

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer


class DataManager:
    def __init__(self, run_params):
        self.run_params = run_params
        self.block_size = 128

    # Load, Tokenize, and Augment data
    def prepare_data(self):
        train_ds, val_ds = map(self.prepare_ds, ("train", "valid"))
        return train_ds, val_ds

    def prepare_ds(self, split):
        tokenizer = GPT2Tokenizer.from_pretrained(self.run_params.model)
        tokenizer.pad_token = tokenizer.eos_token
        split = self.get_split(split)
        ds = load_dataset(
            self.run_params.data_file_type,
            data_files=self.run_params.data_files,
            split=split,
        )
        ds = ds.filter(function=self.criteria)
        ds = ds.map(
            self.tokenize,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
            fn_kwargs={"tokenizer": tokenizer},
        )
        ds = ds.map(
            self.group_texts,
            batched=True,
            #             batch_size=self.block_size,
            #             num_proc=self.run_params.num_workers
        )
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return ds

    def prepare_knowledge_ds(self):
        """Specifically for loading a knowledge dataset"""
        df = pd.read_pickle(self.run_params.data_files)
        ds = Dataset.from_pandas(df)
        tokenizer = GPT2Tokenizer.from_pretrained(self.run_params.model)
        tokenizer.pad_token = tokenizer.eos_token
        ds = ds.map(
            self.tokenize_knowledge,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
            fn_kwargs={"tokenizer": tokenizer},
        )

    def get_split(self, split):
        if self.run_params.debug:
            split += f"[:{self.run_params.batch_size*self.block_size}]"
        else:
            split += f"[:{self.run_params.data_set_percentage}%]"
        return split

    # Tokenize a sequence
    def tokenize(self, x, tokenizer=None):
        tokens = tokenizer(x["text"])
        return tokens

    def tokenize_knowledge(self, x, tokenizer=None):
        __import__("pudb").set_trace()
        tokens = tokenizer(x["knowledge"])
        return tokens

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
        result["labels"] = result["input_ids"].copy()
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

    def criteria(self, x):
        x = x["text"]
        # Remove blanks
        if len(x) == 1:
            return False
        # Remove headings
        if x[0:2] == " =":
            return False
        return True
