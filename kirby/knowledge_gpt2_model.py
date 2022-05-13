import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (GPT2MLP, GPT2Attention,
                                                    GPT2Block, GPT2LMHeadModel,
                                                    GPT2Model)

from kirby.run_params import RunParams


class KnowledgeAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, knowledge_buffer_length=64):
        super().__init__(config, is_cross_attention)
        self.knowledge_buffer = knowledge_buffer_length
        pass

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        leti = head_mask.squeeze().long()

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            batch_size = query.size(0)
            causal_mask = self.bias[
                :,
                :,
                key_length - query_length : key_length,
                :key_length,
            ].bool()
            # Expand causal mask to always look at knowledge buffer
            causal_mask = causal_mask.repeat(batch_size, 1, 1, 1)
            if leti is not None:
                for i in range(batch_size):
                    causal_mask[i, :, leti[i].item() :, -self.knowledge_buffer :] = True
            else:
                causal_mask[i, :, :, -self.knowledge_buffer :] = True

            # Don't attend knowledge to text ids
            causal_mask[:, :, -self.knowledge_buffer :, :] = False
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        # if head_mask is not None:
        # attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class KnowledgeGPT2Block(GPT2Block):
    def __init__(self, config, knowledge_buffer_length=64):
        super().__init__(config)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = KnowledgeAttention(
            config, knowledge_buffer_length=knowledge_buffer_length
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)


class KnowledgeGPT2Model(GPT2Model):
    def __init__(self, config, knowledge_buffer_length=64):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                KnowledgeGPT2Block(
                    config, knowledge_buffer_length=knowledge_buffer_length
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


class KnowledgeGPT2LMHeadModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(self, config, run_params=RunParams()):
        super().__init__(config)
        self.transformer = KnowledgeGPT2Model(
            config, knowledge_buffer_length=run_params.knowledge_buffer
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
