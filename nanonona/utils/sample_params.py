from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # FEATURE：暂时不考虑Top-k采样和Top-p采样

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"