import torch
from torch import nn


class Sampler(nn.Module):

    # TODO: 看到最近有个FlashSampling能优化采样效率，不过还融合了lm_head，后续可尝试优化，现在先不考虑，毕竟不是性能占比大头
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # LEARN：这里有优雅的数学证明
        # 1) 当 noise_i ~ Exponential(1)，则 noise_i / prob_i ~ Exponential(rate = prob_i)，指数函数的性质
        # 2) argmax(prob_i / noise_i) 等价于 argmin(noise_i / prob_i)
        # 3) noise_i / prob_i ~ Ex(prob_i)时，argmin(noise_i / prob_i)中，P(i最小) = prob_i，该定理被称为指数分布的“竞争性质”，可由概率密度函数的积分证明
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens