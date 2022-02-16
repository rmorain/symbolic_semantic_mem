import pprint
import unittest

from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline,
                          set_seed)


class TestPretrained(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
        config_model = GPT2LMHeadModel(GPT2Config())
        self.pretrained_pipe = pipeline(
            task="text-generation", model=pretrained_model, tokenizer=tokenizer
        )
        self.config_pipe = pipeline(
            task="text-generation", model=config_model, tokenizer=tokenizer
        )
        self.prompt = "Harry Potter raised his wand and said "

    def test_compare_generation(self):
        # Pretrained
        pretrained_generated_text = self.pretrained_pipe(self.prompt)
        config_generated_text = self.config_pipe(self.prompt)
        pprint.pprint(pretrained_generated_text)
        pprint.pprint(config_generated_text)


if __name__ == "__main__":
    unittest.main()
