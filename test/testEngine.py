import os
from transformers import AutoTokenizer

from nanonona.utils.sample_params import SamplingParams
from nanonona.engine.engine import llm_engine

if __name__ == "__main__":

    path = os.path.expanduser("./DS-R1-Distill-Qwen-1.5B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = llm_engine()

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1000)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "Who are you?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")