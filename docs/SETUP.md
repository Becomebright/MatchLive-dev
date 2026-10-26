# SETUP.md

How to set up MatchLive on a new machine: code, environment, model weights and data. Paths are
relative to the repo root; see `CLAUDE.md` for the repo layout and day-to-day commands.

## 0. Prerequisites

- **Access.** The code repo (`github.com/Becomebright/MatchLive`) and the assets repo
  (`Becomebright/MatchLive-assets` on Hugging Face) are both private: ask the owner to add your
  GitHub account and your Hugging Face account, then create a Hugging Face token with read access
  (Settings → Access Tokens) for `hf auth login` (§3).
- **Platform.** Linux x86_64 with conda. The environment below pins CUDA 13 builds (torch `cu130`,
  TensorRT), which do not install on macOS or Windows.
  - Replaying the recorded streaming-QA runs from the shared caches (README) loads no model and
    needs no GPU.
  - Running perception (`scripts/run_perception.py`, SETUP §6) needs an NVIDIA GPU whose driver
    supports CUDA 13.
- **Disk.** About 7 GB for all assets (videos 5.3 GB, models 0.7 GB, caches 0.85 GB), plus
  ~30 MB of outputs per replayed match. The full download took about 9 minutes on a cloud VM.
- **API key.** Only for new model calls (a run from scratch, a changed setting): an
  OpenAI-compatible endpoint and key in `configs/api.local.yaml` (§2). Replays need none.

## 1. Code

```bash
git clone https://github.com/Becomebright/MatchLive.git
cd MatchLive                 # `main` is the stable implementation; `dev` carries ongoing work
```

## 2. Environment

conda env **`matchlive`** (Python 3.11):
```bash
conda create -n matchlive -c conda-forge --override-channels python=3.11 ffmpeg pip
conda activate matchlive
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "onnxruntime-gpu[cuda,cudnn]" "trackers==2.6.1" supervision opencv-python numpy scipy \
    transformers safetensors openai pyyaml tqdm pytest "tensorrt-cu13==10.16.*"
pip install -e . --no-deps    # makes `matchlive` importable from scripts/ and tests/
pip install onnx              # only for scripts/export_jersey_onnx.py (torch -> ONNX export)
pip install pyarrow datasets accelerate tensorboard   # only for retraining the jersey reader (§7)
pip install pycocotools       # only for scripts/eval_detection_gsr.py (detection AP)
# Only for scripts/render_stream.py: htdemucs takes the broadcast's commentators out of the sound
# under ours. --no-deps keeps pip from pulling another torch; torchaudio (which has no build for
# this torch) is not needed — render_stream.py reads and writes the audio with ffmpeg. The model
# weights (80 MB) download to ~/.cache/torch/hub on first use.
pip install --no-deps demucs==4.0.1 julius einops openunmix lameenc dora-search treetable submitit \
    retrying omegaconf antlr4-python3-runtime==4.9.3 cloudpickle
```
- **CUDA 13 everywhere:** PyPI `onnxruntime-gpu` 1.30 is a CUDA 13 build, so torch comes from the
  `cu130` index; both then share the pip `nvidia-*` CUDA/cuDNN wheels. Call
  `onnxruntime.preload_dlls()` before creating a CUDA session so ORT finds those libraries.
  The NVIDIA driver must support CUDA 13 (`nvidia-smi` shows the highest supported version).
- **TensorRT EP** (`tensorrt-cu13` 10.16) runs the detector and the jersey classifier. Keep the
  pin: ORT 1.30 loads `libnvinfer.so.10`, so with TensorRT 11 the TensorRT EP fails to load and
  ORT silently falls back to CUDA.
- Use `opencv-python` (not `-headless`): `trackers` depends on it, and two cv2 builds would clash.
- The LLM API is reached through OpenAI-compatible endpoints (the `openai` SDK), named in
  `configs/api.local.yaml`, which is gitignored so the keys never enter the repo. They are used
  in order: a failed request is retried on the next one at once (that request only), and the next
  takes over for good when one runs out of quota:
  ```yaml
  endpoints:
    - base_url: https://.../v1
      api_key: sk-...
    - base_url: https://.../v1        # backup
      api_key: sk-...
  ```
  The model is `api.model` in `configs/default.yaml` (`gemini-3.8-flash`). Cached API responses
  (`cache/api/`) replay without the file; without it, a request the cache lacks stops the run
  (`NoAPIKey`) instead of calling the API, so a replay either reproduces the recorded run or fails
  loudly. (Runs up to 2026-09-29 called Gemini directly through `google-genai`; the cache is keyed
  by model and request, not by endpoint, so their answers replay the same.)
- **Optional: an open model served locally** for a run's model calls, or the commentary's scripts
  alone (`commentary.api`). vLLM pins its own torch, so it has a conda env of its own,
  **`vllm-serve`** (Python 3.12):
  ```bash
  conda create -n vllm-serve -c conda-forge --override-channels python=3.12 pip
  ~/miniconda3/envs/vllm-serve/bin/pip install uv
  ~/miniconda3/envs/vllm-serve/bin/uv pip install --python ~/miniconda3/envs/vllm-serve/bin/python "vllm==0.32.0"
  hf download Qwen/Qwen3.8-Flash-Next-FP8      # or Qwen/Qwen3.8-27B-FP8
  PATH=~/miniconda3/envs/vllm-serve/bin:$PATH bash scripts/serve_llm.sh flashnext   # 4 GPUs | qwen27b: 1 GPU
  ```
  `scripts/serve_llm.sh` holds each model's flags (from the vLLM recipes, for H100 80GB). The
  endpoint goes in `configs/vllm.local.yaml` (`endpoints: [{base_url: "http://localhost:8000/v1",
  api_key: EMPTY}]`). A live run (and its QA, which reads the run's settings) picks the model with
  `--set api.endpoint=configs/vllm.local.yaml --set api.model=Qwen/Qwen3.8-Flash-Next-FP8 --set
  api.timeout=900 --set 'api.extra_body={reasoning_effort: medium}'` (Flash-Next thinks at `xhigh` by
  default: ~6,800 tokens, 27 s an analyzer window; results §35); the commentary's scripts alone with
  `--set 'commentary.api={endpoint: configs/vllm.local.yaml, model: Qwen/Qwen3.8-27B-FP8, extra_body:
  {chat_template_kwargs: {enable_thinking: false}}}' --set commentary.reasoning=null`.
- **The commentary's voice: fishaudio/drama-3, served locally** (`commentary.voice`,
  `api.speech_endpoint`). vLLM-Omni pins its own vLLM and torch, so it has a conda env of its own,
  **`drama3`** (Python 3.12). vLLM-Omni's main branch needs vLLM 0.32 (with the 0.31 of its recipe
  the server fails to start on an import error):
  ```bash
  conda create -n drama3 -c conda-forge --override-channels python=3.12
  ~/miniconda3/envs/drama3/bin/pip install uv
  ~/miniconda3/envs/drama3/bin/uv pip install --python ~/miniconda3/envs/drama3/bin/python vllm==0.32.0 --torch-backend=auto
  ~/miniconda3/envs/drama3/bin/uv pip install --python ~/miniconda3/envs/drama3/bin/python git+https://github.com/vllm-project/vllm-omni.git
  hf download fishaudio/drama-3                                  # 39 GB
  PATH=~/miniconda3/envs/drama3/bin:$PATH bash scripts/serve_tts.sh   # one GPU (CUDA_VISIBLE_DEVICES, default 4)
  ```
  The endpoint goes in `configs/tts.local.yaml` (`endpoints: [{base_url: "http://localhost:8091/v1",
  api_key: EMPTY}]`). The reference recordings it clones are `checkpoints/tts/` (§3). Cached speech
  (`cache/tts/`) replays without the server.
- **The broadcast commentary's transcript with names** (evaluation only:
  `scripts/transcribe_broadcast.py --names`, read by `scripts/compare_broadcast.py`) runs
  Qwen/Qwen3-ASR-1.7B-hf and Qwen/Qwen3-ForcedAligner-0.6B-hf on one GPU in the `matchlive` env
  (transformers ≥ 5.17 has both; no new package). Both download from the Hugging Face Hub on first use.

## 3. Model weights, match data and caches (private)

All three live in one private Hugging Face dataset repo, `Becomebright/MatchLive-assets`, laid out
like the repo root, so a single download drops every file in place:
```bash
hf auth login                 # a token with read access to the repo
hf download Becomebright/MatchLive-assets --repo-type dataset --local-dir . \
    --include "dataset/*" --include "checkpoints/*" --include "cache/*"
tar -xzf cache/api.tar.gz -C cache    # -> cache/api/<call id>.json
```
The caches make the results reproducible without a GPU or an API key (README, "Streaming QA"):
the raw perception cache replaces a GPU pass over each half, and the API cache holds every
model answer the recorded runs received. Leave `cache/*` out to start from scratch.
One pattern per `--include` (a second bare pattern is read as a filename). `--include` skips the
HF repo's own `.gitattributes`/`README.md`; the download also writes metadata to
`.cache/huggingface/` (gitignored).

| Path | Size | Used by |
|---|---|---|
| `dataset/organized_data/england_epl/<season>/<match>/{1,2}_720p.mkv`, `metadata.json` (3 matches) | 5.3G | every pipeline run (`paths.dataset`) |
| `dataset/organized_data/.../2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea/Labels-caption_with_gt.json` | 58K | ground-truth events (SoccerNet-Caption format, `gt_gameTime` = corrected time) |
| `dataset/organized_data/.../<match>/kb.json` (2 benchmark matches) | 86K | pre-match background for the commentary (`DESIGN.md` §6.8), built by `scripts/build_kb.py` from Wikipedia revisions before kick-off and football-data.co.uk results |
| `dataset/streamingqa/part1/` | 364K | streaming-QA benchmark: `benchmark-0926.jsonl` and the 0929 version (per-match files) |
| `dataset/streamingqa/part1_1001/` | 224K | streaming-QA benchmark, 1001 version (per-match files; `scripts/run_qa.py --benchmark`) |
| `checkpoints/rfdetr-2xl-1019.onnx` | 470M | detector (`perception.detection.model`), the retrained RF-DETR-2XL (`docs/results.md` §31) |
| `checkpoints/rfdetr-l.onnx` | 122M | the previous detector: the recorded runs and the shared perception caches were made with it (replay with `--set perception.detection.model=rfdetr-l.onnx perception.detection.trt_cache=trt/rfdetr-l_fp16`, one `--set` each) |
| `checkpoints/jersey_dinov3_convnext_tiny-20_matches_gsr.onnx` | 112M | jersey reader (`perception.jersey.model`), retrained on 20 matches + SoccerNet-GSR (`docs/results.md` §30) |
| `checkpoints/jersey_dinov3_convnext_tiny-20_matches.onnx` | 112M | the previous jersey reader: the recorded runs and the shared perception caches were made with it, so replaying them needs `--set perception.jersey.model=jersey_dinov3_convnext_tiny-20_matches.onnx` (README) |
| `checkpoints/tts/fenrir_{calm,lively,rising,excited}.{wav,txt}` | 2.9M | reference recordings the commentary's voice (drama-3) clones, one per energy, with their transcripts (`commentary.voice.tones`; made with gemini-3.8-flash-tts, `docs/results.md` §36) |
| `checkpoints/rfdetr-2xl.onnx` | 470M | v2 baseline detector in `scripts/compare_v2.py` |
| `cache/perception/<match>/half{1,2}_all_<key>.npz` (3 matches) | 830M | raw perception cache (`matchlive/common/cache.py`), written by `scripts/run_perception.py`, read by every later stage |
| `cache/api.tar.gz` → `cache/api/<call id>.json` | 2.3M (14M unpacked) | cached model answers: event analyzer, arbiter and QA calls of the recorded runs (and of other experiments). An archive because a Hugging Face folder holds at most 10,000 files; the `cache/api/` folder in the HF repo is an older subset of it. After new runs, re-pack with `tar -czf api.tar.gz --exclude='*.tmp' -C cache api` and upload it as `cache/api.tar.gz` |

The perception cache file name carries a hash of its inputs (video, both ONNX models, perception
settings), taken from file contents, so a download matches as long as those are unchanged; a
change to any of them makes the pipeline look for a new file. The shared perception caches were
written with the previous detector and jersey ONNX, so under the default config (`rfdetr-2xl-1019` and the `_gsr` model) they are not
found and the first run perceives from scratch on a GPU; the overrides above (jersey and detector) keep them usable. `cache/trt/` (TensorRT engines) is
specific to the GPU and is never shared.

Kept only on the development machine, not in the repo — the scripts that need them do not run
elsewhere:
- `checkpoints/jersey_dinov3_convnext_tiny-20_matches_gsr.safetensors` (and the previous
  `...-20_matches.safetensors`) and `checkpoints/dinov3-convnext-tiny/` (backbone config): the torch source for the jersey ONNX,
  needed by `scripts/export_jersey_onnx.py` and by v2's jersey reader in `scripts/compare_v2.py`.
- `checkpoints/rfdetr-l.pth`: the torch source of `rfdetr-l.onnx`.
- `checkpoints/rfdetr-2xl-1019.pth` (505M, on HF): the torch source of `rfdetr-2xl-1019.onnx`. Converted in a separate venv so the `matchlive` env keeps its package versions:
  `python3 -m venv ~/.venvs/rfdetr-export && ~/.venvs/rfdetr-export/bin/pip install "rfdetr[plus]==1.11.2" onnx onnxsim onnxscript`, then
  `RFDETR2XLarge(pretrain_weights="checkpoints/rfdetr-2xl-1019.pth", num_classes=4).export(output_dir=...)` writes `rfdetr-2xlarge.onnx`; copy it to `checkpoints/rfdetr-2xl-1019.onnx`.
- `checkpoints/jersey_dinov3_convnext_tiny-101.pt`: the superseded jersey checkpoint (`docs/results.md`).

The match videos come from SoccerNet (under its NDA) and DINOv3 weights are under Meta's license,
so the repo stays private.

## 4. SoccerNet labels and Game State Reconstruction (evaluation only)

**Event labels of our matches.** `scripts/ground_truth.py` (and the scripts built on it:
`eval_memory.py`, `inject_errors.py`, `eval_commentary.py`) reads each match's SoccerNet labels
from its folder: `Labels-v2.json` (action spotting: every goal, card, substitution, shot, corner,
foul, offside, free kick, with team and time) and `Labels-caption.json` (commentary naming the
players). They come from the `SoccerNet/SN-Labels` dataset repo on Hugging Face (accept its terms
on the website first), laid out as league/season/game, so they drop straight into place:
```bash
hf download SoccerNet/SN-Labels --repo-type dataset --local-dir dataset/organized_data \
    --include "england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United/Labels-*.json" \
    --include "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/Labels-*.json" \
    --include "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea/Labels-*.json"
```
The live system never reads them (`DESIGN.md` §6.8).

**SoccerNet-GSR.** Only the `scripts/eval_*_gsr.py` scripts need it. The test split is the public
`SoccerNet/SN-GSR-2024` dataset repo:
```bash
hf download SoccerNet/SN-GSR-2024 test.zip --repo-type dataset --local-dir dataset/SoccerNetGS/gamestate-2024
unzip -q dataset/SoccerNetGS/gamestate-2024/test.zip -d dataset/SoccerNetGS/test   # -> test/SNGS-*/
rm dataset/SoccerNetGS/gamestate-2024/test.zip                                  # optional, 8.3G
```
The train (9.8G) and valid (11G) splits are only needed to retrain the jersey reader (§7); download them
the same way (`train.zip`, `valid.zip`). The three splits cover different matches (`game_id` in each
`Labels-GameState.json`: train 4, 6, 9; valid 2, 3, 5; test 7, 8, 11). Use Python's `zipfile` if `unzip`
is not installed.

## 5. Optional: v2 reference implementation

`scripts/compare_v2.py` imports v2 from `paths.v2` (`../live-comment-v2`); clone it next to this
repo if needed. It also needs the jersey torch checkpoint (see §3), so it runs only on the
development machine as things stand.

## 6. First run

TensorRT engines are built on first use and cached under `cache/trt/`: the first run is slow
(minutes). Engines are specific to the GPU model and TensorRT version, so never copy `cache/trt/`
between machines. Check the setup end to end on a two-minute segment:
```bash
conda run -n matchlive python scripts/run_perception.py \
    --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1 --start 600 --seconds 120
```
It prints the cache path, detection/track counts and per-stage ms/frame; compare the timing with
`docs/results.md`.

## 7. Optional: retraining the jersey reader

Only needed to redo `docs/results.md` §30. Needs several GPUs for a reasonable time (2 x H100 with batch
256 each took 28 minutes for 3 epochs; 4 x H100 at 128 each should take about half), the SoccerNet-GSR train / valid / test splits (§4) and the training pack
(the 20 matches' crops with VLM labels, the DINOv3 backbone and the original training script):
```bash
hf download Becomebright/MatchLive-assets training/jersey_train_pack.tar --repo-type dataset --local-dir .
tar -xf training/jersey_train_pack.tar -C training        # -> training/jersey_train_pack/tmp/
```
`training/` is git-ignored. Then, from the repo root (every script prints what it wrote):
```bash
python scripts/prepare_jersey_labels.py                    # per-box label table of GSR train + valid
python scripts/read_jersey_labels.py                       # read the usable boxes with the current reader
python scripts/read_jersey_labels.py --reason null_track   # ... and those of tracks without a number
python scripts/build_jersey_training_data.py               # crops + the two 5000-row validation sets
PYTHONPATH=. NCCL_NET=Socket torchrun --nproc_per_node=4 scripts/train_jersey.py \
    --name e3_r2_aug --epochs 3 --gsr-repeat 2 --augment --batch-size 128
python scripts/export_jersey_onnx.py --checkpoint training/runs/e3_r2_aug/jersey.safetensors \
    --out training/runs/e3_r2_aug/jersey_e3_r2_aug.onnx
python scripts/eval_jersey_val.py --model training/runs/e3_r2_aug/jersey_e3_r2_aug.onnx
python scripts/eval_jersey_gsr.py --set perception.jersey.model=$PWD/training/runs/e3_r2_aug/jersey_e3_r2_aug.onnx
python scripts/eval_identity_gsr.py --set perception.jersey.model=$PWD/training/runs/e3_r2_aug/jersey_e3_r2_aug.onnx
```
Things that bite on this machine:
- **`NCCL_NET=Socket`** for multi-GPU training: the shell's default `NCCL_NET=gIB` plugin does not
  initialise here, and a single node needs no network plugin. `torchrun` also takes `--name`, not `--run`
  (it would abbreviate it to its own `--run-path`).
- **Give every exported ONNX its own file name.** The TensorRT engine cache (`cache/trt/`) is keyed by file
  name and graph, not by weights: two models both called `jersey.onnx` share one engine and silently
  give identical results.
- After a kernel upgrade `nvidia-smi` can fail until the matching driver module package is installed
  (`linux-modules-nvidia-580-server-open-<kernel version>`); `nvidia-fabricmanager` failing with
  "Nothing to..." is harmless on this VM (no NVSwitch is visible).
- `--set` on the GSR eval scripts takes an absolute ONNX path (`checkpoints/` is prepended to relative ones).
