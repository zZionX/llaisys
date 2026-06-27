import os
from transformers import AutoTokenizer

from nanonona.utils.sample_params import SamplingParams
from nanonona.utils.config import Config
from nanonona.engine.sequence import Sequence
from nanonona.engine.scheduler import Scheduler
from nanonona.engine.modelRunner import ModelRunner

class llm_engine:
    def __init__(self, model_path: str | None = None):
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = Config.model.path
        assert self.model_path is not None, "Model path must be specified either in config or as an argument"
        assert os.path.exists(self.model_path), f"Model path does not exist: {self.model_path}"

        # Rust实现的快速分词器，性能更好
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        Config.model.eos_token_id = self.tokenizer.eos_token_id
        
        # 一定要先modelrunner再scheduler，因为modelrunner里会warmup以计算num_kvcache_blocks这个config
        self.modelrunner = ModelRunner(self.model_path)
        self.scheduler = Scheduler()


    def add_request(self, prompt: list[int], sample_parameters: SamplingParams):
        seq = Sequence(prompt, sample_parameters)
        self.scheduler.add_sequence(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.modelrunner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        completed_answers = [(seq.seq_id, seq.generated_token_ids) for seq in seqs if seq.is_completed]
        return completed_answers

    def generate(
        self, 
        prompts: list[str] | list[list[int]], 
        sample_parameters: SamplingParams | list[SamplingParams]
    ):
        # 加到调度器里
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            else:
                token_ids = prompt
            if isinstance(sample_parameters, list):
                params = sample_parameters[i]
            else:
                params = sample_parameters
            self.add_request(token_ids, params)

        # 按时间步调度执行，直到所有序列都完成
        outputs = {}
        while not self.scheduler.is_finished():
            requests_answer = self.step()
            for seq_id, token_ids in requests_answer:
                outputs[seq_id] = token_ids

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs