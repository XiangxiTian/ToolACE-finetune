# Finetuning LLMs using ToolACE

This repository uses decontainer for reproducibility.

## Training

**Finetune the model:**
```bash
python train.py
```

**Merge LoRA adapters:**
```bash
python merge_lora.py
```

**Quantize the finetuned model:**
```bash
activate-quantize  # separate venv needed (llmcompression conflicts)
python quantize.py
```

## Serving Models

### Original Model
```bash
vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

### Finetuned (BF16)
Local checkpoint:
```bash
vllm serve "outputs/qwen3_4b/checkpoint-1272_lora_merged" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Or from HuggingFace:
```bash
vllm serve "GlebBerjoskin/Qwen3-4B-Instruct-2507-Finetuned-BF16" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

### Finetuned W8A16
Local checkpoint:
```bash
vllm serve "outputs/qwen3_4b/checkpoint-1272_lora_merged_f8a16" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Or from HuggingFace:
```bash
vllm serve "GlebBerjoskin/Qwen3-4B-Instruct-2507-Finetuned-W8A16" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

### Finetuned W8A8
Local checkpoint:
```bash
vllm serve "outputs/qwen3_4b/checkpoint-1272_lora_merged_f8a8" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

Or from HuggingFace:
```bash
vllm serve "GlebBerjoskin/Qwen3-4B-Instruct-2507-Finetuned-W8A8" \
  --served-model-name "Qwen/Qwen3-4B-Instruct-2507" \
  --dtype auto \
  --port 8000 \
  --seed 42 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

## Evaluation

**Run BFCL evaluation:**
```bash
python bfcl.py
```
Note: Make sure the `.env` file from `./bfcl_dir` loads correctly via dotenv (adjust path if needed).

## Benchmarking

**Prepare ToolACE dataset:**
```bash
python prepare_toolace_benchmark.py
```

**Run load benchmark:**
```bash
vllm bench serve \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /workspaces/gleb_berjoskin/benchmark/toolace_benchmark.json \
  --num-prompts 5000 \
  --max-concurrency 32 \
  --request-rate 40 \
  --save-result \
  --result-dir /workspaces/gleb_berjoskin/benchmark \
  --percentile-metrics ttft,tpot,itl,e2el
```