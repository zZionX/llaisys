from collections import deque
import xxhash
import numpy as np

from nanonona.engine.sequence import Sequence
from nanonona.engine.block import Block

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.all_blocks: Block = [Block(i) for i in range(num_blocks)]
        self.free_blocks = deque(range(num_blocks))
        self.allocated_blocks = set()
        self.hash_2_block_id = {}

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix = -1):
        hash = xxhash.xxh64()
        if prefix != -1:
            hash.update(prefix.to_bytes(8, 'little'))
        hash.update(np.array(token_ids).tobytes())
        return hash.intdigest()

    def _num_cached_blocks(self, seq: Sequence):
        num_cached_blocks = 0
        h = -1
        seq.block_table.clear()
        for i in range(seq.blocks_num):
            if i == seq.blocks_num - 1 and len(seq.blocked_slice(i)) != self.block_size:
                break
            h = self.compute_hash(seq.blocked_slice(i), prefix = h)
            if h in self.hash_2_block_id and seq.blocked_slice(i) == self.all_blocks[self.hash_2_block_id[h]].token_ids:
                seq.block_table.append(self.all_blocks[self.hash_2_block_id[h]])
                num_cached_blocks += 1
            else:
                break
        seq.num_cached_blocks = num_cached_blocks
        return num_cached_blocks

    def can_allocate(self, seq: Sequence):
        return len(self.free_blocks) >= (seq.blocks_num - self._num_cached_blocks(seq))
    
    def can_append(self, seq: Sequence):
        return len(self.free_blocks) >= (len(seq) % self.block_size == 1)
    
    def allocate(self, seq: Sequence):
        # cache hit
        for i in range(seq.num_cached_blocks):
            cached_block = seq.block_table[i]
            if cached_block.block_id not in self.allocated_blocks:
                # 之前被分配了但现在被释放了的块，但释放的时候只是把ref_count归零了，内容没有变，所以仍然可以直接使用
                self.allocated_blocks.add(cached_block.block_id)
                self.free_blocks.remove(cached_block.block_id)
                
                cached_block.ref_count = 1
            else:
                # 被使用过的，直接增加引用计数
                cached_block.ref_count += 1

        # cache miss
        h = -1
        if seq.num_cached_blocks > 0:
            h = seq.block_table[seq.num_cached_blocks - 1].hash
        for i in range(seq.num_cached_blocks, seq.blocks_num):
            block_id = self.free_blocks.popleft()
            block = self.all_blocks[block_id]
            self.allocated_blocks.add(block_id)
            seq.block_table.append(block)

            block.ref_count = 1
            # token_ids 和 hash 在满block时写入

            block_tokens = seq.blocked_slice(i)
            if i != seq.blocks_num - 1 or len(block_tokens) == self.block_size:
                block.token_ids = block_tokens
                h = self.compute_hash(block.token_ids, prefix = h)
                block.hash = h
                self.hash_2_block_id[h] = block_id

    def deallocate(self, seq: Sequence):
        # 倒序删除是为了尊重 Block 之间的 Hash的树状依赖关系，先删除子节点再删除父节点
        # 在严谨的树状缓存管理器（如 vLLM 底层的 Radix Tree）中这很重要
        for block in reversed(seq.block_table):
            block.ref_count -= 1
            if block.ref_count == 0:
                self.allocated_blocks.remove(block.block_id)
                self.free_blocks.append(block.block_id)
        seq.num_cached_blocks = 0
        seq.block_table.clear()

    def flexibly_append(self, seq: Sequence):
        last_block = seq.block_table[-1]
        if len(seq) % self.block_size == 0: # 上一个块满了，更新上一个块的内容和哈希
            last_block.token_ids = seq.blocked_slice(seq.blocks_num - 1)
            last_block.hash = self.compute_hash(last_block.token_ids, prefix = seq.block_table[-2].hash if seq.blocks_num > 1 else -1)
            self.hash_2_block_id[last_block.hash] = last_block.block_id
            seq.num_cached_blocks += 1 # 这个参数其实没必要，我现在不引入chunked prefill，这个参数目前用不着
        elif len(seq) % self.block_size == 1: # 新建一个块
            allocated_block_id = self.free_blocks.popleft()
            self.allocated_blocks.add(allocated_block_id)

            target_block = self.all_blocks[allocated_block_id]
            target_block.ref_count = 1
            target_block.token_ids = []
            target_block.hash = -1
            seq.block_table.append(target_block)
        else:
            assert last_block.hash == -1
