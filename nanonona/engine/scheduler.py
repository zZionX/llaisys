from collections import deque

from nanonona.engine.blockManager import BlockManager
from nanonona.engine.sequence import Sequence, SequenceStatus
from nanonona.utils.config import Config

class Scheduler:
    def __init__(self):
        self.max_num_running_seqs = Config.engine.max_num_running_seqs
        self.max_num_running_batched_tokens = Config.engine.max_num_running_batched_tokens

        self.running = deque()
        self.waiting = deque()
        self.blockManager = BlockManager(Config.engine.max_num_kvcache_blocks, Config.engine.block_size)

        self.eos = Config.model.eos_token_id
    
    def add_sequence(self, sequence):
        self.waiting.append(sequence)
    
    def schedule(self): # -> seqs, is_prefill
        scheduled_seqs = []
        num_scheduled_seqs = 0
        num_batched_tokens = 0

        # 当前显存能容下队首的seq的话就尽可能prefill
        while self.waiting and num_scheduled_seqs < self.max_num_running_seqs:
            seq: Sequence = self.waiting[0]
            # TODO：这里逻辑可以优化，先blockManager算有多少cached，再对比够不够剩余空间
            if num_batched_tokens + len(seq) > self.max_num_running_batched_tokens or not self.blockManager.can_allocate(seq):
                break

            self.blockManager.allocate(seq)
            num_scheduled_seqs += 1
            assert seq.num_cached_blocks * self.blockManager.block_size <= len(seq)
            num_batched_tokens += len(seq) - seq.num_cached_blocks * Config.engine.block_size
            
            self.waiting.popleft()
            scheduled_seqs.append(seq)
            self.running.append(seq)
            seq.status = SequenceStatus.RUNNING
        if num_scheduled_seqs != 0:
            return scheduled_seqs, True
            
        # 否则进行decode
        while self.running and num_scheduled_seqs < self.max_num_running_seqs:
            seq = self.running.popleft()
            while not self.blockManager.can_append(seq):
                if self.running:
                    out = self.running.pop()
                    self.preempt(out)
                else:
                    self.preempt(seq)
                    break
            else:
                num_scheduled_seqs += 1
                self.blockManager.flexibly_append(seq)
                scheduled_seqs.append(seq)
        assert num_scheduled_seqs, "decode阶段至少应该安排一个询问，可能是当前配置的显存池太小了，或者说当前的询问太长了"
        self.running.extend(reversed(scheduled_seqs))
        return scheduled_seqs, False
    
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.blockManager.deallocate(seq)
        self.waiting.appendleft(seq)

    def is_finished(self):
        return len(self.running) == 0 and len(self.waiting) == 0
    
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.COMPLETED
                self.blockManager.deallocate(seq)
                self.running.remove(seq)