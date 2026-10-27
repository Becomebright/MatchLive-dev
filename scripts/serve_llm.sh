#!/usr/bin/env bash
# Serve an open model with vLLM behind an OpenAI-compatible endpoint at http://localhost:$PORT/v1,
# for a run's model calls (the `api` section) or the commentary's scripts alone (commentary.api);
# docs/SETUP.md §2. The flags follow the vLLM recipes (recipes.vllm.ai/<model>) for H100 80GB.
#
#   bash scripts/serve_llm.sh qwen27b | flashnext
#
# Runs in the `vllm-serve` conda env (its own torch; docs/SETUP.md §2). GPUs: CUDA_VISIBLE_DEVICES
# (default below per model). The weights come from the Hugging Face cache (`hf download <model>`).
# Perception (detection + jersey, ~1.6 GiB) and Demucs (~0.6 GB) still fit beside the model on any
# of its GPUs: --gpu-memory-utilization 0.85 leaves ~12 GB per GPU.
set -euo pipefail
PORT=${PORT:-8000}
# The shell's default NCCL_NET=gIB plugin is for multi-node GPUDirect and fails within one node
# ("NCCL error: invalid usage"); the GPUs of one machine talk over NVLink either way.
export NCCL_NET=Socket
# Images per request: an analyzer window's 61 frames (events.window_seconds x sample_fps + 1) at most.
common=(--port "$PORT" --enable-prefix-caching --gpu-memory-utilization 0.85 --max-model-len 131072
        --limit-mm-per-prompt '{"image": 128, "video": 0}')

case "${1:?usage: serve_llm.sh qwen27b|flashnext}" in
  qwen27b)    # 27B dense, FP8 (31 GB): one GPU. Its linear-attention layers need a state block per
              # running sequence, which caps --max-num-seqs (617 fit at 0.85; the default 1024 does not)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
    exec vllm serve Qwen/Qwen3.8-27B-FP8 "${common[@]}" --tensor-parallel-size 1 --max-num-seqs 64 \
      --kv-cache-dtype fp8 --reasoning-parser qwen3 \
      --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' ;;
  flashnext)  # 180B / 6B active, FP8 (186 GB): 4 GPUs, tensor + expert parallel (33.5 GiB of weights
              # per GPU); the 51B n-gram table is offloaded to host memory (vLLM's default; needed on
              # 80 GB GPUs). Decoding one request is no faster on 8 (TEP8, the recipe's: 280 vs 266
              # tok/s), and two matches at once on one server lose ~10% each (241 tok/s). MTP with 3
              # draft tokens accepts 2.3 a step; a 4th adds nothing (its acceptance is 0.17)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
    exec vllm serve Qwen/Qwen3.8-Flash-Next-FP8 "${common[@]}" --tensor-parallel-size 4 --enable-expert-parallel \
      --moe-backend triton --max-num-seqs 256 --kv-cache-dtype fp8 --attention-config.indexer_kv_dtype fp8 \
      --reasoning-parser qwen3 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' ;;
  *) echo "unknown model: $1" >&2; exit 1 ;;
esac
