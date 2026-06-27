from dataclasses import dataclass
import torch

# @dataclass 更优雅地定义一个"主要用于存储数据"的类，自动生成 __init__、__repr__、__eq__ 等，减少模板代码
# slots=True，不再有 __dict__来存数据，省内存、加速属性访问，但禁止动态添加新属性
@dataclass(slots=True)
class Context:

    # 以下为prefill模式下特有的参数
    is_prefill:     bool = False

    cu_seqlens_q:   torch.Tensor | None = None
    cu_seqlens_k:   torch.Tensor | None = None
    # cumulative sequence lengths, shape (batch_size+1,), 例如 [0, 5, 11] 表示 batch_size=2，第一个序列长度5，第二个序列长度6。同时因为prefix cach机制，cu_seqlens_q只包含新加入的token的长度，而cu_seqlens_k包含了所有token的长度。即假设Prompt是 "A B C D E"，其中 "A B C" 已经存在于 Prefix Cache 中，本次只需要处理新输入 "D E"，那么cu_seqlens_q 就是 [0, 2]，而 cu_seqlens_k 是 [0, 5]。

    max_seqlen_q:   int = 0
    max_seqlen_k:   int = 0
    # batch 内最长的序列长度，用于分配足够的显存来存储新的token的key和value。对于cu_seqlens_q来说，max_seqlen_q 是新输入token的最大长度；对于cu_seqlens_k来说，max_seqlen_k 是包括prefix cache在内的总长度。

    # 以下参数在prefill和decode模式下都可能用到
    slot_mapping:   torch.Tensor | None = None
    # [num_tokens]，精细到 Token 级别的“内存门牌号”，用于寻址写入kv_cache

    block_tables:   torch.Tensor | None = None
    # [batch_size, max_num_blocks_per_seq]，显存管理系统的“页表”，建立逻辑块到物理块的映射关系，用于读kv_cache

    # 以下只在decode模式下用到
    context_lens:   torch.Tensor | None = None
    # [batch_size]，记录 batch 内每个序列当前的实际总长度

# 单例模式
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT # 申明使用全局变量
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables, context_lens) # LEARN: 易错点，context_lens和block_tables的顺序容易写反

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()