# SETUP.md

How to set up MatchLive on a new machine: code, environment, model weights and data. Paths are
relative to the repo root; see `CLAUDE.md` for the repo layout and day-to-day commands.

## 1. Code

```bash
git clone https://github.com/Becomebright/MatchLive.git
cd MatchLive
git checkout dev
```

## 2. Environment

conda env **`matchstream`** (Python 3.11):
```bash
conda create -n matchstream -c conda-forge --override-channels python=3.11 ffmpeg pip
conda activate matchstream
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "onnxruntime-gpu[cuda,cudnn]" "trackers==2.6.1" supervision opencv-python numpy scipy \
    transformers safetensors openai python-dotenv pyyaml tqdm pytest tensorrt-cu13
pip install -e . --no-deps    # makes `matchlive` importable from scripts/ and tests/
pip install onnx              # only for scripts/export_jersey_onnx.py (torch -> ONNX export)
```
- **CUDA 13 everywhere:** PyPI `onnxruntime-gpu` 1.30 is a CUDA 13 build, so torch comes from the
  `cu130` index; both then share the pip `nvidia-*` CUDA/cuDNN wheels. Call
  `onnxruntime.preload_dlls()` before creating a CUDA session so ORT finds those libraries.
  The NVIDIA driver must support CUDA 13 (`nvidia-smi` shows the highest supported version).
- **TensorRT EP** (`tensorrt-cu13` 10.16) runs the detector and the jersey classifier.
- Use `opencv-python` (not `-headless`): `trackers` depends on it, and two cv2 builds would clash.
- Create `.env` in the repo root for API keys; no phase needs one yet.

## 3. Model weights and match data (private)

Both live in one private Hugging Face dataset repo, `Becomebright/MatchLive-assets`, laid out like
the repo root, so a single download drops every file in place:
```bash
hf auth login                 # a token with read access to the repo
hf download Becomebright/MatchLive-assets --repo-type dataset --local-dir . \
    --include "dataset/*" "checkpoints/*"
```
`--include` skips the HF repo's own `.gitattributes`/`README.md`; the download also writes
metadata to `.cache/huggingface/` (gitignored).

| Path | Size | Used by |
|---|---|---|
| `dataset/organized_data/england_epl/<season>/<match>/{1,2}_720p.mkv`, `metadata.json` | 3.3G | every pipeline run (`paths.dataset`) |
| `dataset/streamingqa/part1/` | 364K | streaming-QA benchmark |
| `checkpoints/rfdetr-l.onnx` | 122M | detector (`perception.detection.model`) |
| `checkpoints/jersey_dinov3_convnext_tiny-20_matches.onnx` | 112M | jersey reader (`perception.jersey.model`) |
| `checkpoints/rfdetr-2xl.onnx` | 470M | v2 baseline detector in `scripts/compare_v2.py` |

Kept only on the development machine, not in the repo — the scripts that need them do not run
elsewhere:
- `checkpoints/jersey_dinov3_convnext_tiny-20_matches.safetensors` and
  `checkpoints/dinov3-convnext-tiny/` (backbone config): the torch source for the jersey ONNX,
  needed by `scripts/export_jersey_onnx.py` and by v2's jersey reader in `scripts/compare_v2.py`.
- `checkpoints/rfdetr-l.pth`: the torch source of `rfdetr-l.onnx`.
- `checkpoints/jersey_dinov3_convnext_tiny-101.pt`: the superseded jersey checkpoint (`docs/results.md`).

The match videos come from SoccerNet (under its NDA) and DINOv3 weights are under Meta's license,
so the repo stays private.

## 4. SoccerNet Game State Reconstruction (public; evaluation only)

Only the `scripts/eval_*_gsr.py` scripts need it. The test split is the public
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
conda run -n matchstream python scripts/run_perception.py \
    --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1 --start 600 --seconds 120
```
It prints the cache path, detection/track counts and per-stage ms/frame; compare the timing with
`docs/results.md`.
