import torch

from torch import nn
from transformers import Qwen2Config

from nanonona.layers.rmsnorm import RMSnorm
from nanonona.layers.linear import Linear, MergedLinear, QKVMergedLinear
from nanonona.layers.activation import SwiGLU
from nanonona.layers.rotary_embedding import get_rope
from nanonona.layers.attention import Attention
from nanonona.layers.embed_head import VocabEmbedding, ParallelLMHead

class Grouped_Attention_Block(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        num_kv_heads: int,
        max_position: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVMergedLinear(
            hidden_size, 
            [self.q_size, self.kv_size, self.kv_size], 
            bias=qkv_bias
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        if not self.qkv_bias:
            self.q_norm = RMSnorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSnorm(self.head_dim, eps=rms_norm_eps)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, positions):
        qkv = self.qkv_proj(x) # num_tokens, q+k+v_size
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v) # num_tokens, num_heads*head_dim
        o = self.o_proj(attn_output.flatten(1, -1))
        return o

class Swi_MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_up_proj = MergedLinear(hidden_size, [intermediate_size]*2) # 先gate，再up
        assert hidden_act == "silu", "Only support silu activation for now, other activations can be implemented later if needed"
        self.down_proj = Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.gate_up_proj(x)
        gate, up = x.chunk(2, -1)
        x = SwiGLU(gate, up)
        x = self.down_proj(x)
        return x

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.input_layernorm = RMSnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Grouped_Attention_Block(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.hidden_size // config.num_attention_heads, 
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.post_attention_layernorm = RMSnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Swi_MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        
    def forward(self, x, residual, positions):
        if residual is None:
            x, residual = self.input_layernorm(x), x
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, positions)

        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)

        return x, residual

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.norm = RMSnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            x, residual = layer(x, residual, positions)
        x, _ = self.norm(x, residual)

        return x

class Qwen2ForCasualLM(nn.Module):
    # 需要合并计算的linear层的映射关系，key是需要合并的linear层的名字，在safetensors里会出现的名字，value是一个tuple，第一个元素是实现上线性层实际的名称，第二个元素是该linear层在合并后的linear层中的位置（从0开始）
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), # weight_name中把“q_proj”映射为“qkv_proj”这个变量名
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        last_token = self.model(input_ids, positions)
        # 不用单独取最后一个token了，lm_head会精准切片
        logits = self.lm_head(last_token)

        return logits