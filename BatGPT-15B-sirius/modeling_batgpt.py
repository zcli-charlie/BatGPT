# This code serves as a port of the models described in BatGPT. 
# It is based on the bloom codebase, which provides the initial framework for our model implementation.
# To understand how to use these models, please refer to the documentation and usage instructions provided in the bloom models repository.
# Additionally, we draw inspiration from the ChatGLM and Baichuan codebase, which includes implementations for prefix encoder, chat, and stream_chat functionalities. These components are utilized in our ported models.
# Feel free to explore the ChatGLM and Baichuan codebase for further insights on how these components can be utilized effectively.

import math
import warnings
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from torch.nn.utils import skip_init

import copy
import re
import sys

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_batgpt import BatGPTConfig

logger = logging.get_logger(__name__)


# flags required to enable jit fusion kernels

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)


# For faster llm model initilization
def module_init(cls, empty_init, *args, **kwargs):
    if empty_init:
        return skip_init(cls, *args, **kwargs)
    else:
        return cls(*args, **kwargs)

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: BatGPTConfig):
        super().__init__()
        self.prefix_proj = config.prefix_proj
        self.head_dim = config.hidden_size // config.n_head
        if self.prefix_proj:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.n_layer * self.head_dim * config.num_heads_per_kv * 2
            self.embedding = torch.nn.Embedding(config.prefix_size, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prefix_size,
                                                config.n_layer * self.head_dim * config.num_heads_per_kv * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_proj:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = torch.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

def _build_position_ids(input_ids, device):
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids

def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    """used in training only"""
    dim = tensor.size(0)
    _future_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1
    )   
    _future_mask = _future_mask.unsqueeze(0) + alibi
    _future_mask = _future_mask.to(tensor)
    return _future_mask[:tensor.shape[1] * attn_heads, :maxpos, :maxpos]

@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)





class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class SelfAttention(torch.nn.Module):
    def __init__(self, config: BatGPTConfig, device=None):
        super(SelfAttention, self).__init__()

        self.num_heads = config.n_head
        self.use_multi_query_attn = config.use_multi_query_attn
        self.num_heads_per_kv = config.num_heads_per_kv
        self.qkv_bias = config.qkv_bias
        self.use_native_attn_impl = config.use_native_attn_impl
        if not self.use_multi_query_attn:
            assert self.num_heads_per_kv == self.num_heads, "num_heads_per_kv must equal to num_heads when not use_multi_query_attn"
        
        self.head_dim = config.hidden_size // config.n_head

        self.query_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=self.qkv_bias, 
            device=device, **_config_to_kwargs(config)
        )

        self.key_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_heads_per_kv, bias=self.qkv_bias,
            device=device, **_config_to_kwargs(config)
        )
        self.value_proj = nn.Linear(
            config.hidden_size, self.head_dim * self.num_heads_per_kv, bias=self.qkv_bias,
            device=device, **_config_to_kwargs(config)
        )

        # Output.
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False,
            device=device, **_config_to_kwargs(config)
        )
    
    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        rotary_pos_emb, 
        kv_cache=None, 
        use_cache=True
    ):
        # 1. query/key/value mapping
        # hidden_states: [seq_len, batch_size, hidden_size]
        seq_len, batch_size, hidden_size = hidden_states.shape
        query_layer = self.query_proj(hidden_states)
        key_layer = self.key_proj(hidden_states)
        value_layer = self.value_proj(hidden_states)

        query_layer = query_layer.view(seq_len, batch_size, self.num_heads, self.head_dim)

        key_layer = key_layer.view(seq_len, batch_size, self.num_heads_per_kv, self.head_dim)

        value_layer = value_layer.view(seq_len, batch_size, self.num_heads_per_kv, self.head_dim)

        # 2. apply the rotary position embedding
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # 3. adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        # 4. repeat the key and value for attention
        if self.num_heads_per_kv != self.num_heads:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_heads // self.num_heads_per_kv, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_heads, self.head_dim)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_heads // self.num_heads_per_kv, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_heads, self.head_dim)
            )

        # 5. attention [seq_len, batch_size, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

        pytorch_version = int(torch.__version__.split('.')[0])
        if self.use_native_attn_impl and pytorch_version >= 2:
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                attention_mask)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                if seq_len == 1: # inference with cache
                    if len(attention_mask.size()) == 4:
                        attention_mask = attention_mask[:, :, -1:, :]   
                    else:
                        attention_mask = attention_mask[:, -1:, :]
                attention_scores = attention_scores + attention_mask
                attention_scores = torch.max(attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min))

            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

            context_layer = torch.matmul(attention_probs, value_layer)

        # [batch_size, num_heads, seq_len, head_dim] -> [seq_len, batch_size, num_heads, head_dim]
        context_layer = context_layer.permute(2, 0, 1, 3)

        # [seq_len, batch_size, hidden_size]
        context_layer = context_layer.reshape(seq_len, batch_size, hidden_size)

        # 
        output = self.dense(context_layer)

        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    def __init__(self, config: BatGPTConfig, device=None):
        super(MLP, self).__init__()
        self.mlp_activation = config.mlp_activation

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        
        def silu(x):
            return F.silu(x)

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.mlp_activation == "swiglu":
            self.activation_func = swiglu

            self.gate_proj = None

            self.dense_h_to_4h = nn.Linear(
                config.hidden_size,
                config.ffn_hidden_size * 2,
                bias=False,
                device=device,
                **_config_to_kwargs(config)
            )
        elif self.mlp_activation == "silu":
            self.activation_func = silu

            self.gate_proj = nn.Linear(
                config.hidden_size, 
                config.ffn_hidden_size, 
                bias=False,
                device=device,
                **_config_to_kwargs(config)
            )

            self.dense_h_to_4h = nn.Linear(
                config.hidden_size,
                config.ffn_hidden_size,
                bias=False,
                device=device,
                **_config_to_kwargs(config)
            )
        else:
            raise NotImplementedError("mlp_activation {} not supported".format(self.mlp_activation))

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        if self.mlp_activation == "swiglu":
            intermediate_parallel = self.activation_func(intermediate_parallel)
        elif self.mlp_activation == "silu":
            gated_weight = self.activation_func(self.gate_proj(hidden_states))
            intermediate_parallel = gated_weight * intermediate_parallel
        else:
            raise NotImplementedError("mlp_activation {} not supported".format(self.mlp_activation))
        
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class BatGPTLayer(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: BatGPTConfig, device=None):
        super(BatGPTLayer, self).__init__()

        # Layernorm on the input data.
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, device=device)

        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        rotary_pos_emb, 
        kv_cache=None, 
        use_cache=True,
    ):
        # hidden_states: [s, b, h]
        residual = hidden_states

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)

        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)

        output = residual + output

        return output, kv_cache


class BatGPTTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: BatGPTConfig, device=None):
        super(BatGPTTransformer, self).__init__()

        # Number of layers.
        self.num_layers = config.n_layer

        # Transformer layers.
        def build_layer():
            return BatGPTLayer(config, device=device)

        self.layers = torch.nn.ModuleList([build_layer() for i in range(self.num_layers)])

        # final layer norm before output.
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon, device=device,
                                                dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        rotary_pos_emb,
        kv_caches=None,
        use_cache: Optional[bool] = True,
        output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class BatGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = BatGPTConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["BatGPTLayer"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return



    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BatGPTTransformer):
            module.gradient_checkpointing = value



class BatGPTModel(BatGPTPreTrainedModel):
    def __init__(self, config: BatGPTConfig, device=None):
        super().__init__(config)

        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.hidden_size // config.n_head
        self.max_seq_len = config.max_seq_len
        self.pos_emb_impl = config.pos_emb_impl
        self.model_cache_seq_len = 1024

        # word embedding
        self.word_embeddings = module_init(nn.Embedding,
            config.empty_init,
            config.vocab_size,
            config.emb_dim,
            dtype=config.torch_dtype,
            device=device
        )

        self.emb_fact = None
        if config.use_emb_factorization or config.emb_dim != config.hidden_size:
            self.emb_fact = nn.Linear(config.emb_dim, config.hidden_size, bias=False,
                                      dtype=config.torch_dtype, device=device)

        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        
        self.encoder = module_init(BatGPTTransformer, config.empty_init, config, **init_kwargs)

        self.first_run = True
        self.alibi_mask = None

        self.prefix_size = config.prefix_size
        self.prefix_proj = config.prefix_proj
        if self.prefix_size is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.prefix_size).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_size,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def get_rotary_tensor(self, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
    
        n_elem = head_dim // 2
        
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()

        return cache

    def get_causal_mask(self, input_ids, past_key_values, attention_mask=None) -> torch.BoolTensor:

        batch_size, seq_length = input_ids.shape

        # B x L x L
        causal_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        causal_mask.tril_()

        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        
        if past_length:
            causal_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), causal_mask), dim=-1)
        
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask.unsqueeze(1)
        
        if not past_length and attention_mask is not None:
            causal_mask -= attention_mask.unsqueeze(-1) - 1
        
        causal_mask = (causal_mask < 0.5).bool()
        causal_mask.unsqueeze_(1)

        return causal_mask

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.training:
            slopes = torch.Tensor(_get_interleave(self.num_heads))
            alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_length_with_past).unsqueeze(0).unsqueeze(0).expand(
                self.num_heads,
                -1, -1) 
            alibi = alibi.view(self.num_heads, 1, seq_length_with_past)
            mask = _buffered_future_mask(tensor, seq_length_with_past, alibi, self.num_heads)
        else:
            if self.first_run:
                self.first_run = False
                self.register_buffer("future_mask", _gen_alibi_mask(self.num_heads, self.model_cache_seq_len).to(tensor), persistent=False)
            if seq_length_with_past > self.model_cache_seq_len:
                self.model_cache_seq_len = seq_length_with_past
                self.register_buffer("future_mask", _gen_alibi_mask(self.num_heads, self.model_cache_seq_len).to(tensor), persistent=False)
            mask = self.future_mask[:self.num_heads, :seq_length_with_past, :seq_length_with_past] 
        return mask


    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        seq_length_with_past = seq_length

        # -> word embedding
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            # [b s h] --> [s b h].
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()

        if self.prefix_size is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.prefix_size)),
                                            attention_mask], dim=-1)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[0]
            seq_length_with_past = seq_length_with_past + past_key_values_length


        full_attention_mask = None
        rotary_pos_emb=None
        if self.pos_emb_impl == "alibi":
            if self.training:
                if self.alibi_mask is None or self.alibi_mask.shape[-1] != seq_length_with_past:
                    self.alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
                alibi_mask = self.alibi_mask
            else:
                alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

            
            if attention_mask is not None:

                if len(attention_mask.shape) == 2:
                    expanded_mask = attention_mask.to(alibi_mask.dtype)
                    expanded_mask = torch.tril(torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                                    ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
                else:
                    expanded_mask = attention_mask
                src_len, tgt_len = alibi_mask.size()[-2:]
                expanded_mask = expanded_mask.unsqueeze(1).expand(batch_size, 1, src_len, tgt_len).to(alibi_mask.dtype)
                # Target sizes: [1, 1, 41, 41].  Tensor sizes: [1, 1, 8, 8]
                inverted_mask = 1.0 - expanded_mask
                inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min)
                full_attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
            else:
                full_attention_mask = alibi_mask
        elif self.pos_emb_impl == "rope":
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                # B x 1 x L x L
                full_attention_mask = self.get_causal_mask(input_ids, past_key_values, attention_mask)
            
            # Rotary positional embeddings
            rotary_pos_emb = self.get_rotary_tensor(self.max_seq_len, self.head_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            if position_ids is not None:
                rotary_pos_emb = rotary_pos_emb[position_ids]
            else:
                rotary_pos_emb = rotary_pos_emb[None, :seq_length]
            rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError("position embedding type: {} not supported!".format(self.pos_emb_impl))


        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, 
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, 
            use_cache=use_cache, 
            output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BatGPTForCausalLM(BatGPTPreTrainedModel):
    def __init__(self, config: BatGPTConfig, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length

        self.model = BatGPTModel(config, device=device)

        self.lm_head = module_init(nn.Linear, config.empty_init, config.hidden_size, config.vocab_size, bias=False, 
                                        dtype=config.torch_dtype, device=device)

        self.config = config

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:

        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = _build_position_ids(input_ids, device=input_ids.device)
        
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encodings = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encodings[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        
        lm_logits = self.lm_head(hidden_states)
        
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + encodings[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=encodings.past_key_values,
            hidden_states=encodings.hidden_states,
            attentions=encodings.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        return response

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, system_prompt = None):
        inputs = tokenizer.build_inputs(query, history=history, system_prompt=system_prompt)
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, system_prompt = None):
        inputs = tokenizer.build_stream_inputs(query, history=history, system_prompt=system_prompt)
        inputs = inputs.to(self.device)
        return inputs

    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, system_prompt=None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, **kwargs} #, "logits_processor": logits_processor
        inputs = self.build_inputs(tokenizer, query, history=history, system_prompt=system_prompt)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs, skip_special_tokens=True) #
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, system_prompt=None, past_key_values=None,
                    max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                    return_past_key_values=False, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history, system_prompt=system_prompt)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history, system_prompt=system_prompt)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.model.prefix_size is not None:
                past_length -= self.transformer.prefix_size
            inputs.position_ids += past_length
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @torch.no_grad()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

