from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType


class KnowledgeTextGenerationPipeline(TextGenerationPipeline):
    ALLOWED_MODELS = ["KnowledgeGPT2LMHeadModel", "GPT2LMHeadModel"]

    def __init__(self, run_params, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.run_params = run_params

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        knowledge=None,
        handle_long_generation=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if knowledge is not None:
            preprocess_params["knowledge"] = knowledge
        if knowledge:
            knowledge_inputs = self.tokenizer(
                knowledge,
                padding=False,
                add_special_tokens=False,
                return_tensors=self.framework,
            )
            knowledge_length = knowledge_inputs["input_ids"].shape[-1]

            if "max_new_tokens" in generate_kwargs:
                pass
            elif "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += knowledge_length
            else:
                generate_kwargs["max_length"] = (
                    self.model.config.max_length + knowledge_length
                )

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += knowledge_length
        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        preprocess_params.update(generate_kwargs)
        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            return_type = (
                ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            )
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params[
                "clean_up_tokenization_spaces"
            ] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self, prompt_text, knowledge="", handle_long_generation=None, **generate_kwargs
    ):
        __import__("pudb").set_trace()
        inputs = self.tokenizer(
            prompt_text + knowledge,
            padding=False,
            add_special_tokens=False,
            return_tensors=self.framework,
        )
        inputs["prompt_text"] = prompt_text

        if handle_long_generation == "hole":
            cur_len = inputs["input_ids"].shape[-1]
            if "max_new_tokens" in generate_kwargs:
                new_tokens = generate_kwargs["max_new_tokens"]
            else:
                new_tokens = (
                    generate_kwargs.get("max_length", self.model.config.max_length)
                    - cur_len
                )
                if new_tokens < 0:
                    raise ValueError("We cannot infer how many new tokens are expected")
            if cur_len + new_tokens > self.tokenizer.model_max_length:
                keep_length = self.tokenizer.model_max_length - new_tokens
                if keep_length <= 0:
                    raise ValueError(
                        "We cannot use `hole` to handle this generation the number of desired tokens exceeds the models max length"
                    )

                inputs["input_ids"] = inputs["input_ids"][:, -keep_length:]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][
                        :, -keep_length:
                    ]

        return inputs
