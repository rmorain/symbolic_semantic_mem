import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from torch import nn
from transformers import GPT2Config, GPT2Tokenizer
from transformers.generation_beam_constraints import (Constraint,
                                                      DisjunctiveConstraint,
                                                      PhrasalConstraint)
from transformers.generation_beam_search import (BeamScorer, BeamSearchScorer,
                                                 ConstrainedBeamSearchScorer)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor, ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor, HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor, LogitsProcessorList, MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor, NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor, RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria, MaxTimeCriteria, StoppingCriteria, StoppingCriteriaList,
    validate_stopping_criteria)
from transformers.generation_utils import (BeamSampleOutput, BeamSearchOutput,
                                           GreedySearchOutput, SampleOutput)
from transformers.utils import logging

from kirby.basic_model import BasicModel
from kirby.knowledge_gpt2_model import KnowledgeGPT2LMHeadModel
from kirby.run_params import RunParams

logger = logging.get_logger(__name__)


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
        leti = x["entity_index"]
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=self.run_params.output_attentions,
            head_mask=leti,
        )
        loss = outputs[0]
        if self.run_params.output_attentions:
            return outputs.attentions
        return loss

    # @torch.no_grad()
    # def generate(self, prompt, max_new_tokens):
    # __import__("pudb").set_trace()
    # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
    # outputs = self.model.generate(
    # input_ids=input_ids, max_new_tokens=max_new_tokens
    # )
    # return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
