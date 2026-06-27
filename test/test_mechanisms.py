import importlib
import sys
import types
import unittest
from unittest import mock


try:
    import torch
except Exception:  # pragma: no cover - exercised only on machines without torch
    torch = None

HAS_CUDA = bool(torch is not None and torch.cuda.is_available())


class MockModelConfig:
    path = "/tmp/mock-model"
    eos_token_id = 0


class MockEngineConfig:
    block_size = 4
    max_num_running_seqs = 4
    max_num_kvcache_blocks = 16
    max_num_running_batched_tokens = 64
    max_model_len = 64
    gpu_memory_utilization = 0.9


class MockServerConfig:
    host = "127.0.0.1"
    port = 8000


class MockConfig:
    model = MockModelConfig()
    engine = MockEngineConfig()
    server = MockServerConfig()


def _install_mock_config():
    module = types.ModuleType("nanonona.utils.config")
    module.Config = MockConfig
    sys.modules["nanonona.utils.config"] = module


_install_mock_config()

sequence_mod = importlib.import_module("nanonona.engine.sequence")
block_manager_mod = importlib.import_module("nanonona.engine.blockManager")
scheduler_mod = importlib.import_module("nanonona.engine.scheduler")

sequence_mod = importlib.reload(sequence_mod)
block_manager_mod = importlib.reload(block_manager_mod)
scheduler_mod = importlib.reload(scheduler_mod)

Sequence = sequence_mod.Sequence
SequenceStatus = sequence_mod.SequenceStatus
SamplingParams = sequence_mod.SamplingParams
BlockManager = block_manager_mod.BlockManager
Scheduler = scheduler_mod.Scheduler

if torch is not None:
    context_mod = importlib.import_module("nanonona.utils.context")
    model_runner_mod = importlib.import_module("nanonona.engine.modelRunner")
    context_mod = importlib.reload(context_mod)
    model_runner_mod = importlib.reload(model_runner_mod)
    ModelRunner = model_runner_mod.ModelRunner
    get_context = context_mod.get_context
    reset_context = context_mod.reset_context
else:
    ModelRunner = None

    def get_context():
        raise unittest.SkipTest("torch is required for ModelRunner context tests")

    def reset_context():
        return None


def _configure_engine(*, block_size=4, max_seqs=4, max_blocks=16, max_tokens=64, eos=0):
    MockConfig.engine.block_size = block_size
    MockConfig.engine.max_num_running_seqs = max_seqs
    MockConfig.engine.max_num_kvcache_blocks = max_blocks
    MockConfig.engine.max_num_running_batched_tokens = max_tokens
    MockConfig.model.eos_token_id = eos
    Sequence.BLOCK_SIZE = block_size


def _params(max_tokens=8, ignore_eos=True):
    return SamplingParams(temperature=1.0, max_tokens=max_tokens, ignore_eos=ignore_eos)


class TestBlockManagerMechanisms(unittest.TestCase):
    def setUp(self):
        _configure_engine(block_size=4, max_blocks=8)

    def test_prefix_cache_reuses_full_blocks_after_deallocate(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seq1 = Sequence([10, 20, 30, 40, 50], _params())

        self.assertTrue(manager.can_allocate(seq1))
        manager.allocate(seq1)
        cached_block_id = seq1.block_table[0].block_id
        self.assertEqual(seq1.num_cached_blocks, 0)

        manager.deallocate(seq1)
        self.assertEqual(len(manager.free_blocks), 8)
        self.assertEqual(seq1.block_table, [])

        seq2 = Sequence([10, 20, 30, 40, 60], _params())
        self.assertTrue(manager.can_allocate(seq2))
        self.assertEqual(seq2.num_cached_blocks, 1)
        manager.allocate(seq2)

        self.assertEqual(seq2.block_table[0].block_id, cached_block_id)
        self.assertEqual(seq2.block_table[0].ref_count, 1)
        self.assertEqual(seq2.block_table[0].token_ids, [10, 20, 30, 40])
        self.assertEqual(len(manager.allocated_blocks), 2)

    def test_prefix_cache_shares_live_blocks_and_tracks_ref_counts(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seq1 = Sequence([1, 2, 3, 4, 100], _params())
        seq2 = Sequence([1, 2, 3, 4, 200], _params())

        self.assertTrue(manager.can_allocate(seq1))
        manager.allocate(seq1)
        self.assertTrue(manager.can_allocate(seq2))
        self.assertEqual(seq2.num_cached_blocks, 1)
        manager.allocate(seq2)

        self.assertIs(seq1.block_table[0], seq2.block_table[0])
        self.assertEqual(seq1.block_table[0].ref_count, 2)
        self.assertEqual(len(manager.allocated_blocks), 3)

        manager.deallocate(seq1)
        self.assertEqual(seq2.block_table[0].ref_count, 1)
        self.assertIn(seq2.block_table[0].block_id, manager.allocated_blocks)

    def test_partial_tail_blocks_are_not_prefix_cached(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seq1 = Sequence([1, 2, 3], _params())
        seq2 = Sequence([1, 2, 3], _params())

        self.assertTrue(manager.can_allocate(seq1))
        manager.allocate(seq1)
        self.assertTrue(manager.can_allocate(seq2))

        self.assertEqual(seq2.num_cached_blocks, 0)
        self.assertEqual(len(manager.hash_2_block_id), 0)

    def test_hash_collision_does_not_reuse_different_tokens(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seq1 = Sequence([1, 2, 3, 4], _params())
        seq2 = Sequence([9, 9, 9, 9], _params())

        with mock.patch.object(BlockManager, "compute_hash", classmethod(lambda cls, token_ids, prefix=-1: 123)):
            self.assertTrue(manager.can_allocate(seq1))
            manager.allocate(seq1)
            self.assertTrue(manager.can_allocate(seq2))

        self.assertEqual(seq2.num_cached_blocks, 0)

    @unittest.expectedFailure
    def test_cached_prefill_should_be_admitted_by_uncached_token_budget(self):
        """Documents a scheduler edge case worth fixing before final benchmarks.

        The scheduler currently checks len(seq) against max_num_running_batched_tokens
        before computing prefix-cache hits. A cached long prompt with a tiny suffix
        should be admitted based on uncached tokens, not total prompt length.
        """
        _configure_engine(block_size=4, max_blocks=8, max_tokens=5)
        scheduler = Scheduler()
        seed = Sequence([1, 2, 3, 4], _params())
        self.assertTrue(scheduler.blockManager.can_allocate(seed))
        scheduler.blockManager.allocate(seed)
        scheduler.blockManager.deallocate(seed)

        cached_request = Sequence([1, 2, 3, 4, 5, 6], _params())
        scheduler.add_sequence(cached_request)
        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [cached_request])
        self.assertEqual(cached_request.num_cached_blocks, 1)


class TestSchedulerMechanisms(unittest.TestCase):
    def setUp(self):
        _configure_engine(block_size=2, max_seqs=2, max_blocks=8, max_tokens=32, eos=0)

    def test_continuous_batching_prefills_new_waiting_request_before_decode(self):
        scheduler = Scheduler()
        seq1 = Sequence([11, 12], _params(max_tokens=4))
        scheduler.add_sequence(seq1)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [seq1])
        scheduler.postprocess(seqs, [91])
        self.assertEqual(seq1.num_completion_tokens, 1)

        seq2 = Sequence([21, 22], _params(max_tokens=4))
        scheduler.add_sequence(seq2)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [seq2])
        self.assertEqual(list(scheduler.running), [seq1, seq2])

    def test_decode_preempts_when_kv_blocks_are_exhausted(self):
        _configure_engine(block_size=2, max_seqs=2, max_blocks=3, max_tokens=32, eos=0)
        scheduler = Scheduler()
        seq1 = Sequence([1, 2, 3], _params(max_tokens=8))
        seq2 = Sequence([4, 5], _params(max_tokens=8))
        scheduler.add_sequence(seq1)
        scheduler.add_sequence(seq2)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(set(seqs), {seq1, seq2})
        self.assertEqual(len(scheduler.blockManager.free_blocks), 0)

        scheduler.postprocess(seqs, [31, 41])
        seqs, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [seq1])
        self.assertEqual(list(scheduler.waiting), [seq2])
        self.assertEqual(seq2.status, SequenceStatus.WAITING)
        self.assertGreater(len(scheduler.blockManager.free_blocks), 0)

    def test_postprocess_deallocates_completed_sequence(self):
        scheduler = Scheduler()
        seq = Sequence([7, 8], _params(max_tokens=1, ignore_eos=True))
        scheduler.add_sequence(seq)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(len(scheduler.blockManager.allocated_blocks), 1)
        scheduler.postprocess(seqs, [99])

        self.assertTrue(seq.is_completed)
        self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(len(scheduler.blockManager.allocated_blocks), 0)


@unittest.skipUnless(HAS_CUDA, "CUDA is required for ModelRunner context tests")
class TestModelRunnerContextPreparation(unittest.TestCase):
    def setUp(self):
        _configure_engine(block_size=4, max_seqs=4, max_blocks=8, max_tokens=64, eos=0)
        reset_context()

    def tearDown(self):
        reset_context()

    def _runner(self):
        runner = object.__new__(ModelRunner)
        runner.block_size = MockConfig.engine.block_size
        return runner

    def test_prepare_prefill_records_cached_suffix_context(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seed = Sequence([10, 20, 30, 40], _params())
        self.assertTrue(manager.can_allocate(seed))
        manager.allocate(seed)
        manager.deallocate(seed)

        seq = Sequence([10, 20, 30, 40, 50, 60], _params())
        self.assertTrue(manager.can_allocate(seq))
        self.assertEqual(seq.num_cached_blocks, 1)
        manager.allocate(seq)

        input_ids, positions = self._runner().prepare_prefill([seq])
        ctx = get_context()

        self.assertEqual(input_ids.cpu().tolist(), [50, 60])
        self.assertEqual(positions.cpu().tolist(), [4, 5])
        self.assertTrue(ctx.is_prefill)
        self.assertEqual(ctx.cu_seqlens_q.cpu().tolist(), [0, 2])
        self.assertEqual(ctx.cu_seqlens_k.cpu().tolist(), [0, 6])
        self.assertEqual(ctx.max_seqlen_q, 2)
        self.assertEqual(ctx.max_seqlen_k, 6)
        self.assertIsNotNone(ctx.block_tables)
        self.assertEqual(ctx.block_tables.shape[0], 1)
        self.assertEqual(ctx.block_tables.shape[1], 2)
        expected_slots = [
            seq.block_table[1].block_id * MockConfig.engine.block_size,
            seq.block_table[1].block_id * MockConfig.engine.block_size + 1,
        ]
        self.assertEqual(ctx.slot_mapping.cpu().tolist(), expected_slots)

    def test_prepare_decode_records_paged_kv_context(self):
        manager = BlockManager(num_blocks=8, block_size=4)
        seq = Sequence([1, 2, 3], _params())
        self.assertTrue(manager.can_allocate(seq))
        manager.allocate(seq)

        input_ids, positions = self._runner().prepare_decode([seq])
        ctx = get_context()

        self.assertEqual(input_ids.cpu().tolist(), [3])
        self.assertEqual(positions.cpu().tolist(), [2])
        self.assertFalse(ctx.is_prefill)
        self.assertEqual(ctx.context_lens.cpu().tolist(), [3])
        self.assertEqual(ctx.block_tables.cpu().tolist(), [[seq.block_table[0].block_id]])
        self.assertEqual(
            ctx.slot_mapping.cpu().tolist(),
            [seq.block_table[0].block_id * MockConfig.engine.block_size + 2],
        )


if __name__ == "__main__":
    unittest.main()
