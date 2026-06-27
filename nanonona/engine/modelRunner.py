from nanonona.engine.sequence import Sequence
from nanonona.layers.sample import Sampler
from nanonona.models.Qwen2 import Qwen2ForCasualLM
from nanonona.utils.config import Config
from nanonona.utils.loader import load_model
from nanonona.utils.context import reset_context, set_context

import os
import torch
from transformers import Qwen2Config

class ModelRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        assert os.path.abspath(Config.model.path) == os.path.abspath(model_path), "Model path in config does not match the provided model path."
        self.block_size = Config.engine.block_size

        # torch.cude.set_device(rank)
        torch.set_default_device("cuda") # LEARN：通过CUDA_VISIBLE_DEVICES=2 python example.py设置使用哪张物理显卡，代码内与gpu_id解耦；然后这里设置了default_device后，后续实例化的model会默认放在该设备上，无需再调用model.to(device)。这么做主要是因为llm权重很大，如果（多个进程）先全加载到cpu内存上，会撑爆（CPU OOM），所以现在gpu空出需要的显存，再按需写入加载。
        hf_config = Qwen2Config.from_pretrained(model_path)
        self.hf_config = hf_config
        original_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)

        self.model = Qwen2ForCasualLM(hf_config)
        load_model(self.model, model_path) # 会直接加载到GPU上

        self.sampler = Sampler()

        # 先warmup，再分配KV Cache。因为要统计推理时峰值显存，以计算推理的激活值显存占用情况，进而合理分配KV Cache的大小。
        self.warmup_model()
        self.allocate_kv_cache()

        # LEARN：重要！！模型加载完成后，恢复默认设备和数据类型，避免对后续代码造成影响。
        # torch.set_default_device("cuda") 在模型和 KV Cache 这两个显存大头都在 GPU 妥善安置好后，使命就完成了。在大模型推理引擎中，除了核心的模型矩阵乘法（推理）外，还有大量的控制逻辑、数据同步、统计信息、创建使用 锁页内存（Pin Memory）的张量以及多进程通信这些逻辑是在 CPU 上跑的，需要保持常规的设备和精度。必须立刻收回这个全局特权，让整个系统回到常规、安全的 CPU 控制流中。
        torch.set_default_device("cpu")
        torch.set_default_dtype(original_default_dtype)

    def warmup_model(self):
        # =====================================================================
        # 第一阶段：遍历触发 Triton Autotune (不计入 Peak Memory)
        # =====================================================================
        torch.cuda.empty_cache()
        print("开始第一阶段warmup：遍历触发 Triton Autotune (不计入 Peak Memory)...")
        
        # 1. 提取你定义的 M_BUCKET 的所有边界值
        buckets = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        # 2. 结合配置，限制最大 token 数，防止越界
        max_tokens = min(Config.engine.max_num_running_batched_tokens, Config.engine.max_model_len)
        valid_buckets = [b for b in buckets if b <= max_tokens]
        
        # 3. 确保系统允许的极端最大值也被 Tune 过（如果它不在 bucket 里）
        if max_tokens not in valid_buckets:
            valid_buckets.append(max_tokens)
        valid_buckets.reverse()
            
        # 4. 遍历所有 bucket，构造假序列跑推理。
        # 这里 Linear 算子看到拍平的 M 就是 seq.num_scheduled_tokens，
        # 会精准命中并编译对应的 Triton kernel。
        for m in valid_buckets:
            print(f"Warmup with {m} tokens...")
            seq = Sequence([0] * m)
            # seq.num_scheduled_tokens = m
            self.run([seq], is_prefill=True)

        # =====================================================================
        # 过渡阶段：清理战场，重置水位线
        # =====================================================================
        torch.cuda.empty_cache()
        # 【最核心的一行】：把刚才成百上千个 Triton Config 试跑造成的虚高显存清零！
        torch.cuda.reset_peak_memory_stats() 

        print("开始第二阶段warmup：测算真实的极限推理 Peak Memory...")
        # =====================================================================
        # 第二阶段：测算真实的极限推理 Peak Memory
        # =====================================================================
        # 用配置允许的最大并发度构造极端用例

        # LEARN：有 prefix cache 命中和没有命中时，前向传播产生的动态激活值（Peak Activation Memory）大小几乎是完全一样的。 完全不需要在 warmup 阶段为了追求更极中的显存范围而去特意模拟 prefix cache 命中的场景。现有的、不带 cache 命中的最大尺寸的 warmup 已经能够精准抓住整个系统生命周期中最大、最极端的激活值水位线。因为唯一跟 cached_len 相关的计算过程就只有flashAtten，但因为是fused kernel，不会显式的创建激活中间值，所以实际上 cached_len 并不会影响 peak memory 的计算，也就不用在这考虑了。
        # TODO：写完后检查一下我的代码实现是不是真的之产生跟 num_batched_tokens 大小有关的激活值大小，万一那个layer的实现还是牵扯到cached_len大小的激活值了呢？
        max_num_batched_tokens, max_model_len = Config.engine.max_num_running_batched_tokens, Config.engine.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, Config.engine.max_num_running_seqs)
        
        print(f"Warmup with {num_seqs} sequences of {seq_len} tokens each...")
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        # for seq in seqs:
        #     seq.num_scheduled_tokens = seq_len
            
        # 带着刚才已经选好的最优算子，真正跑一次
        # 这次产生的 Peak 就是纯净的、分配 KV Cache 时必须避让的激活显存！
        self.run(seqs, is_prefill=True)
        
        # 测算完毕，释放掉这些临时激活张量，交接给 allocate_kv_cache 去分配
        torch.cuda.empty_cache()

    # LEARN: 分配KV cache的逻辑是跑一次最长序列的warmup，记录这次run的peak memory，其中会包括峰值的激活值内存占用+模型参数+其他，我们就是要求的这个峰值的激活值内存占用。然后根据这个峰值的激活值内存占用，去计算KV cache的大小，保证KV cache的大小+峰值的激活值内存占用+模型参数+其他<总显存*Config.engine.gpu_memory_utilization
    def allocate_kv_cache(self):
        hf_config = self.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        Config.engine.max_num_kvcache_blocks = int(total * Config.engine.gpu_memory_utilization - used - (peak - current)) // block_bytes
        assert Config.engine.max_num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, Config.engine.max_num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 构建（batchs, max_num_block）的页表，同时转到GPU上
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [[block.block_id for block in seq.block_table] + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 根据 seqs 中的信息来准备预填充的 inputs_ids 和 positions，以及预填充的上下文context，都要转到GPU上
        slots = []
        input_ids = []
        positions = []
        
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        block_tables = None

        for seq in seqs:
            start_block = seq.num_cached_blocks # 从0开始计数
            start = start_block * self.block_size
            end = len(seq)
            end_block = (end +  self.block_size - 1) // self.block_size

            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + (end - start))
            cu_seqlens_k.append(cu_seqlens_k[-1] + len(seq))

            max_seqlen_q = max(max_seqlen_q, end - start)
            max_seqlen_k = max(max_seqlen_k, len(seq))

            if len(seq.block_table) == 0: # warmup不用计算slots，因为slots是用来存kv cache的
                continue

            for i in range(start_block, end_block):
                slots_start = seq.block_table[i].block_id * self.block_size

                if i == end_block - 1:
                    slots_end = seq.block_table[i].block_id * self.block_size + end - i * self.block_size
                else:
                    slots_end = seq.block_table[i].block_id * self.block_size + self.block_size

                slots.extend(range(slots_start, slots_end))

            if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # cached，才会通过页表访问历史kv_cache
                block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions


    def prepare_decode(self, seqs: list[Sequence]):
        # 根据 seqs 中的信息来准备推理的 inputs_ids 和 positions，以及推理的上下文context，都要转到GPU上
        slots = []
        context_lens = []
        input_ids = []
        positions = []

        for seq in seqs:
            slots.append(seq.block_table[-1].block_id * self.block_size + (len(seq) - 1) % self.block_size)
            context_lens.append(len(seq))
            input_ids.append(seq.token_ids[-1])
            positions.append(len(seq) - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slots = torch.tensor(slots, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slots, context_lens=context_lens, block_tables=block_tables)

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 根据 seqs 中的信息来准备采样的 temperature，也要转到GPU上，因为采样层也在GPU上跑
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def run(self, seqs: list[Sequence], is_prefill=False):
        # 构建上下文，把必要tensor转到GPU上
        inputs_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # run，返回的是词表中每个token的取值概率
        logits = self.model(input_ids=inputs_ids, positions=positions)

        temperatures = self.prepare_sample(seqs)
        token_ids = self.sampler(logits, temperatures).tolist()

        reset_context()

        return token_ids



