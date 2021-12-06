import torch
from pytorch_lightning import LightningModule
from transformers import BatchEncoding, BertForMaskedLM, BertTokenizer

from .basic_model import BasicModel


class KnowledgeBert(BasicModel, LightningModule):
    """GPT2 model that uses knowledge inputs"""

    def __init__(self, run_params):
        super().__init__(run_params)
        self.run_params = run_params
        self.model = BertForMaskedLM.from_pretrained(self.run_params.model)
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = BertTokenizer.from_pretrained(self.run_params.model)
        self.collate_fn = DataCollatorForLanguageModeling(
            self.tokenizer, run_params, mlm_probability=0.15
        )

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"].squeeze(1)
        labels = x["labels"]
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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.run_params.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.run_params.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.run_params.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.run_params.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        pass


class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    def __init__(
        self,
        tokenizer,
        run_params,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.run_params = run_params

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10%
        random, 10% original.
        """
        inputs.squeeze_(1)
        labels = inputs.clone()
        """
        We sample a few tokens in each sequence for MLM training (with probability 
        `self.mlm_probability`)
        """
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        special_tokens_mask[:, -self.run_params.knowledge_buffer :] = True
        probability_matrix = probability_matrix.masked_fill(
            special_tokens_mask, value=0.0
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        # ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens
        # unchanged
        return inputs, labels
