import unittest
from unittest.mock import patch
from collections import deque
import sys
import os

# ----------------- 环境 Mock 设置 -----------------
# 为了脱离实际项目目录运行，我们构建一个 Mock 的 Config
class MockEngineConfig:
    max_num_running_seqs = 4
    max_num_running_batched_tokens = 256
    max_num_blocks = 10
    block_size = 4

class MockConfig:
    engine = MockEngineConfig()

# 假设你的文件都在 nanonona 包下，这里将其 mock 掉以保证测试独立运行
sys.modules['nanonona.utils.config'] = unittest.mock.MagicMock()
sys.modules['nanonona.utils.config'].Config = MockConfig

# 导入你实现的三个类 (实际运行环境请替换为正确的 import 路径)
from nanonona.engine.sequence import Sequence, SequenceStatus
from nanonona.engine.blockManager import BlockManager
from nanonona.engine.scheduler import Scheduler

# 修复原代码中的一处小拼写错误 (Sequence._id_gen -> Sequence._ID_GEN)
Sequence._id_gen = Sequence._ID_GEN


class TestSequence(unittest.TestCase):
    def test_sequence_basics(self):
        # 假设 block_size = 4
        # tokens: [10, 20, 30, 40] (Block 0) + [50] (Block 1)
        Sequence.BLOCK_SIZE = 4
        seq = Sequence([10, 20, 30, 40, 50], None)
        
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.blocks_num, 2)
        self.assertEqual(seq.blocked_slice(0), [10, 20, 30, 40])
        self.assertEqual(seq.blocked_slice(1), [50])


class TestBlockManager(unittest.TestCase):
    def setUp(self):
        # 初始化 10 个 block，每个大小为 4
        self.manager = BlockManager(num_blocks=10, block_size=4)

    def test_allocate_and_deallocate(self):
        seq = Sequence([1, 2, 3, 4, 5], None) # 需要 2 个 block
        
        self.assertTrue(self.manager.can_allocate(seq))
        self.manager.allocate(seq)
        
        self.assertEqual(len(self.manager.free_blocks), 8)
        self.assertEqual(len(self.manager.allocated_blocks), 2)
        
        self.manager.deallocate(seq)
        self.assertEqual(len(self.manager.free_blocks), 10)
        self.assertEqual(len(self.manager.allocated_blocks), 0)

    def test_prefix_caching(self):
        # Seq1: 前 4 个 token 组成完整的 Block 0
        seq1 = Sequence([10, 20, 30, 40, 50], None)
        self.manager.can_allocate(seq1) # 触发 _num_cached_blocks 计算
        self.manager.allocate(seq1)
        
        # Seq2: 前 4 个 token 与 Seq1 完全一致
        seq2 = Sequence([10, 20, 30, 40, 60], None)
        self.assertTrue(self.manager.can_allocate(seq2)) # 这里会发现 Cache Hit = 1
        
        self.assertEqual(seq2.num_cached_blocks, 1)
        self.manager.allocate(seq2)
        
        # 验证显存共享：Seq1 和 Seq2 总共应该只占用 3 个物理 Block (共享1个，各自独立1个)
        self.assertEqual(len(self.manager.free_blocks), 7)
        # 验证共享 Block 的引用计数为 2
        shared_block = seq2.block_table[0]
        self.assertEqual(shared_block.ref_count, 2)

    def test_flexibly_append_decode_phase(self):
        seq = Sequence([1, 2, 3], None)
        self.manager.can_allocate(seq)
        self.manager.allocate(seq)
        
        # 初始只占用 1 个 block
        self.assertEqual(len(seq.block_table), 1)
        last_block = seq.block_table[-1]
        self.assertEqual(last_block.hash, -1) # 未满，未计算 hash

        # 模拟 Decode 追加第 4 个 Token (刚好填满 Block)
        seq.token_ids.append(4)
        self.manager.flexibly_append(seq)
        self.assertEqual(len(seq.block_table), 1)
        self.assertNotEqual(last_block.hash, -1) # 已满，计算了 hash

        # 模拟 Decode 追加第 5 个 Token (需要申请新 Block)
        seq.token_ids.append(5)
        self.manager.flexibly_append(seq)
        self.assertEqual(len(seq.block_table), 2) # 分配了新 Block
        self.assertEqual(len(self.manager.free_blocks), 8)


class TestScheduler(unittest.TestCase):
    def setUp(self):
        # 为调度器测试配置极小的环境
        MockConfig.engine.max_num_blocks = 3
        MockConfig.engine.block_size = 2
        MockConfig.engine.max_num_running_seqs = 2
        self.scheduler = Scheduler()
        Sequence.BLOCK_SIZE = MockConfig.engine.block_size

    def test_continuous_batching_schedule(self):
        seq1 = Sequence([1, 2], None)
        self.scheduler.add_sequence(seq1)
        
        # 第一次 Schedule：应该执行 Prefill
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduled_seqs), 1)
        self.assertEqual(self.scheduler.running[0], seq1)
        self.assertEqual(seq1.status, SequenceStatus.RUNNING)
        
        # 此时增加一个 Seq2，同时准备让 Seq1 进入 Decode 阶段
        seq2 = Sequence([3, 4], None)
        self.scheduler.add_sequence(seq2)
        
        # 第二次 Schedule：只要显存足够，应该继续优先 Prefill Seq2
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduled_seqs), 1)
        self.assertEqual(scheduled_seqs[0], seq2)
        self.assertEqual(len(self.scheduler.running), 2)

    def test_preemption_due_to_oom(self):
        # 显存总共 3 个 Block
        self.assertEqual(len(self.scheduler.blockManager.free_blocks), 3)
        seq1 = Sequence([1, 2, 3], None) # 需要 2 个 Block
        seq2 = Sequence([4, 5], None)    # 需要 1 个 Block
        self.scheduler.add_sequence(seq1)
        self.scheduler.add_sequence(seq2)
        
        # 第一次 Schedule：把 3 个 Block 全部占满
        seqs, is_prefill = self.scheduler.schedule()
        self.assertEqual(len(self.scheduler.blockManager.free_blocks), 0)
        
        # 模拟 Decode 生成，seq1 和 seq2 各追加一个 token
        seq1.token_ids.append(6) # 需要新 Block (第4个token填入Block 1，无需新块... 等等，3是第2块第1个，4填满第2块)
        seq2.token_ids.append(6) # 当前长度2，追加第3个token需要新 Block (第2块)
        
        # 第二次 Schedule：触发 Decode，此时 seq2 申请显存会失败
        scheduled_seqs, is_prefill = self.scheduler.schedule()
        self.assertFalse(is_prefill)
        
        # 由于显存不足，其中一个必须被 Preempt (挂起)
        self.assertEqual(len(self.scheduler.waiting), 1)
        preempted_seq = self.scheduler.waiting[0]
        self.assertEqual(preempted_seq.status, SequenceStatus.WAITING)
        # 验证被抢占的序列释放了显存
        self.assertTrue(len(self.scheduler.blockManager.free_blocks) > 0)

if __name__ == '__main__':
    unittest.main()
