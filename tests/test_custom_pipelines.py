import unittest

from kirby.data_manager import DataManager
from kirby.knowledge_model import KnowledgeGPT2LMHeadModel, KnowledgeModel
from kirby.run_params import RunParams
from pipelines.custom_pipelines import KnowledgeTextGenerationPipeline
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline,
                          set_seed)


class TestKnowledgeTextGenerationPipeline(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        # Loaded model
        CHECKPOINT = "checkpoints/epoch=24-val_loss=0.91-pretrained_min_attention.ckpt"
        self.run_params = RunParams(pretrained=True)
        self.knowledge_model = KnowledgeModel.load_from_checkpoint(
            CHECKPOINT, run_params=self.run_params
        ).model
        # Baseline pretrained model
        self.pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Pipeline for knowledge model
        self.pipe1 = KnowledgeTextGenerationPipeline(
            self.run_params, self.knowledge_model, tokenizer=self.tokenizer
        )
        # Pipeline for normal text generation
        self.pipe2 = pipeline(
            "text-generation", self.pretrained_model, tokenizer=self.tokenizer
        )
        self.prompt = "Harry Potter raised his wand and said"
        self.knowledge = (
            '{"label" : "Harry Potter", "instance of":"Gryffindor student"}'
        )

        self.x = {"text": self.prompt, "knowledge": self.knowledge}

    def test_generate_text(self):
        generated_text1 = self.pipe1(
            self.x,
            max_length=30,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=False,
            do_sample=True,
        )
        self.assertIsNotNone(generated_text1)
        generated_text2 = self.pipe2(
            self.prompt,
            max_length=30,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=False,
        )
        __import__("pudb").set_trace()
        self.assertIsNotNone(generated_text2)

    @staticmethod
    def print_weights(model):
        for params in model.parameters():
            print(params)
            break


if __name__ == "__main__":
    unittest.main()
