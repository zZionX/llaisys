TRITON_PRINT_AUTOTUNING=1 TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 python test/testEngine.py 

TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 python -m unittest discover -s test -p "test_ops_correctness.py"

TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 python -m unittest discover -s test -p "test_mechanisms.py"

TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario fixed \
  --mode continuous \
  --num-requests 32 \
  --prompt-len 512 \
  --output-len 128 \
  --output-dir benchmarks/results/fixed_continuous

TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario fixed \
  --mode serial \
  --num-requests 32 \
  --prompt-len 512 \
  --output-len 128 \
  --output-dir benchmarks/results/fixed_serial

TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario shared-prefix \
  --mode continuous \
  --num-requests 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --output-len 128 \
  --output-dir benchmarks/results/prefix_on


TRITON_CACHE_DIR=./triton_cache_sm86 CUDA_VISIBLE_DEVICES=7 \
python benchmarks/bench_engine.py \
  --scenario shared-prefix \
  --mode continuous \
  --num-requests 32 \
  --prefix-len 1024 \
  --suffix-len 32 \
  --output-len 128 \
  --disable-prefix-cache \
  --output-dir benchmarks/results/prefix_off

