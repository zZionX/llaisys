from copy import copy
from enum import Enum, auto
from itertools import count

from nanonona.utils.config import Config
from nanonona.engine.block import Block
from nanonona.utils.sample_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    COMPLETED = auto()

class Sequence:
    BLOCK_SIZE = Config.engine.block_size
    _ID_GEN = count()

    def __init__(self, token_ids: list[int], sample_parameters = SamplingParams()):
        self.seq_id = next(Sequence._ID_GEN)
        self.token_ids = copy(token_ids)
        self.request_len = len(token_ids)
        self.num_tokens = len(token_ids)
        self.block_table: list[Block] = []
        self.status = SequenceStatus.WAITING
        self.num_cached_blocks = 0
        self.temperature = sample_parameters.temperature
        self.max_tokens = sample_parameters.max_tokens
        self.ignore_eos = sample_parameters.ignore_eos
    
    def __len__(self):
        return self.num_tokens

    @property
    def blocks_num(self):
        return (self.num_tokens - 1) // Sequence.BLOCK_SIZE + 1

    @property
    def is_completed(self):
        return self.status == SequenceStatus.COMPLETED
    
    @property
    def generated_token_ids(self):
        return self.token_ids[self.request_len:]
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.request_len
    
    def blocked_slice(self, block_index: int):
        start = block_index * Sequence.BLOCK_SIZE
        end = min((block_index + 1) * Sequence.BLOCK_SIZE, self.num_tokens)
        return self.token_ids[start:end]
    
    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.num_tokens += 1