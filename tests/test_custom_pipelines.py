import unittest

from kirby.knowledge_model import KnowledgeGPT2LMHeadModel
from kirby.run_params import RunParams
from pipelines.custom_pipelines import KnowledgeTextGenerationPipeline
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline,
                          set_seed)


class TestKnowledgeTextGenerationPipeline(unittest.TestCase):
    def setUp(self):
        __import__("pudb").set_trace()
        set_seed(42)
        self.run_params = RunParams(pretrained=True)
        config = GPT2Config()
        self.config_model = KnowledgeGPT2LMHeadModel(config, knowledge_buffer_length=0)
        self.print_weights(self.config_model)
        self.knowledge_pretrained_model = KnowledgeGPT2LMHeadModel.from_pretrained(
            "gpt2"
        )
        self.print_weights(self.knowledge_pretrained_model)
        self.pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.print_weights(self.pretrained_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.pipe1 = KnowledgeTextGenerationPipeline(
            self.run_params, self.knowledge_pretrained_model, tokenizer=self.tokenizer
        )
        self.pipe2 = KnowledgeTextGenerationPipeline(
            self.run_params, self.pretrained_model, tokenizer=self.tokenizer
        )
        self.pipe3 = pipeline("text-generation", model="gpt2")
        self.prompt = "Harry Potter raised his wand and said"

    def test_generate_text(self):
        generated_text1 = self.pipe1(
            self.prompt,
            max_length=30,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=False,
        )
        self.assertIsNotNone(generated_text1)
        generated_text2 = self.pipe2(
            self.prompt,
            max_length=30,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=False,
        )
        self.assertIsNotNone(generated_text2)
        generated_text3 = self.pipe3(
            self.prompt,
            max_length=30,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=False,
        )
        self.assertIsNotNone(generated_text3)
        __import__("pudb").set_trace()

    @staticmethod
    def print_weights(model):
        for params in model.parameters():
            print(params)
            break


if __name__ == "__main__":
    unittest.main()
