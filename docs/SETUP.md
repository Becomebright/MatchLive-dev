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
```
- **CUDA 13 everywhere:** PyPI `onnxruntime-gpu` 1.30 is a CUDA 13 build, so torch comes from the
  `cu130` index; both then share the pip `nvidia-*` CUDA/cuDNN wheels. Call
  `onnxruntime.preload_dlls()` before creating a CUDA session so ORT finds those libraries.
  The NVIDIA driver must support CUDA 13 (`nvidia-smi` shows the highest supported version).
- **TensorRT EP** (`tensorrt-cu13` 10.16) runs the detector and the jersey classifier. Keep the
  pin: ORT 1.30 loads `libnvinfer.so.10`, so with TensorRT 11 the TensorRT EP fails to load and
  ORT silently falls back to CUDA.
- Use `opencv-python` (not `-headless`): `trackers` depends on it, and two cv2 builds would clash.
- The LLM API is reached through an OpenAI-compatible endpoint (the `openai` SDK), named in
  `configs/api.local.yaml`, which is gitignored so the key never enters the repo:
  ```yaml
  base_url: https://.../v1
  api_key: sk-...
  ```
  The model is `api.model` in `configs/default.yaml` (`gemini-3.8-flash`). Cached API responses
  (`cache/api/`) replay without the file; without it, a request the cache lacks stops the run
  (`NoAPIKey`) instead of calling the API, so a replay either reproduces the recorded run or fails
  loudly. (Runs up to 2026-09-29 called Gemini directly through `google-genai`; the cache is keyed
  by model and request, not by endpoint, so their answers replay the same.)

## 3. Model weights, match data and caches (private)

All three live in one private Hugging Face dataset repo, `Becomebright/MatchLive-assets`, laid out
like the repo root, so a single download drops every file in place:
```bash
hf auth login                 # a token with read access to the repo
hf download Becomebright/MatchLive-assets --repo-type dataset --local-dir . \
    --include "dataset/*" --include "checkpoints/*" --include "cache/*"
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
| `dataset/streamingqa/part1/` | 364K | streaming-QA benchmark |
| `checkpoints/rfdetr-l.onnx` | 122M | detector (`perception.detection.model`) |
| `checkpoints/jersey_dinov3_convnext_tiny-20_matches.onnx` | 112M | jersey reader (`perception.jersey.model`) |
| `checkpoints/rfdetr-2xl.onnx` | 470M | v2 baseline detector in `scripts/compare_v2.py` |
| `cache/perception/<match>/half{1,2}_all_<key>.npz` (3 matches) | 830M | raw perception cache (`matchlive/common/cache.py`), written by `scripts/run_perception.py`, read by every later stage |
| `cache/api/<call id>.json` | 21M | cached Gemini answers: event analyzer, arbiter and QA calls of the recorded runs |

The perception cache file name carries a hash of its inputs (video, both ONNX models, perception
settings), taken from file contents, so a download matches as long as those are unchanged; a
change to any of them makes the pipeline look for a new file. `cache/trt/` (TensorRT engines) is
specific to the GPU and is never shared.

Kept only on the development machine, not in the repo — the scripts that need them do not run
elsewhere:
- `checkpoints/jersey_dinov3_convnext_tiny-20_matches.safetensors` and
  `checkpoints/dinov3-convnext-tiny/` (backbone config): the torch source for the jersey ONNX,
  needed by `scripts/export_jersey_onnx.py` and by v2's jersey reader in `scripts/compare_v2.py`.
- `checkpoints/rfdetr-l.pth`: the torch source of `rfdetr-l.onnx`.
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
