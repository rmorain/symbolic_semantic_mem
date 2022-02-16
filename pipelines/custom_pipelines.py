from transformers import TextGenerationPipeline


class KnowledgeTextGenerationPipeline(TextGenerationPipeline):
    ALLOWED_MODELS = ["KnowledgeGPT2LMHeadModel", "GPT2LMHeadModel"]

    def __init__(self, run_params, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.run_params = run_params
