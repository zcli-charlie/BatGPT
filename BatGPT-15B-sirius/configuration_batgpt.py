from transformers import PretrainedConfig


class BatGPTConfig(PretrainedConfig):
    
    model_type = "batgpt"

    def __init__(
        self,
        vocab_size=65024,
        emb_dim=5632,
        hidden_size=5632,
        n_layer=48,
        n_head=44,
        layer_norm_epsilon=1e-5,
        use_multi_query_attn=True,
        num_heads_per_kv=2,
        qkv_bias=True,
        use_native_attn_impl=True,
        mlp_activation="swiglu",
        hidden_dropout=0.0,
        ffn_hidden_size=13696,
        prefix_size=None,
        prefix_proj=False,
        max_seq_len=32768,
        pos_emb_impl="rope",
        use_emb_factorization=False,
        empty_init=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_multi_query_attn = use_multi_query_attn
        self.num_heads_per_kv = num_heads_per_kv
        self.qkv_bias = qkv_bias
        self.use_native_attn_impl = use_native_attn_impl
        self.mlp_activation = mlp_activation
        self.hidden_dropout = hidden_dropout
        self.ffn_hidden_size = ffn_hidden_size
        self.prefix_size = prefix_size
        self.prefix_proj = prefix_proj
        self.max_seq_len = max_seq_len
        self.pos_emb_impl = pos_emb_impl
        self.use_emb_factorization = use_emb_factorization
        self.empty_init = empty_init
        super().__init__(**kwargs)