#!/usr/bin/env bash
# Serve the commentary's voice model, fishaudio/drama-3, with vLLM-Omni behind an OpenAI-compatible
# speech endpoint at http://localhost:$PORT/v1 (api.speech_endpoint in configs/default.yaml names
# the local file that points at it; docs/SETUP.md §2).
#
#   bash scripts/serve_tts.sh
#
# Runs in the `drama3` conda env (vLLM 0.32 + vLLM-Omni; docs/SETUP.md §2). One GPU: the backbone
# and the codec run on it together (37 GiB of weights; the server takes ~75 GB of the 80 GB).
# The first requests compile kernels (~10 s); later ones take ~0.3 s for a short phrase.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
exec vllm serve fishaudio/drama-3 --omni --port "${PORT:-8091}"
