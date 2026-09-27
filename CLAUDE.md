# CLAUDE.md

**Role of this file:** *how to work in this repo* — layout, environment, commands, coding and
testing conventions, git rules. *What to build and why* (paper claim, architecture, data contracts,
module responsibilities, phase plan) lives only in `docs/DESIGN.md`; read it before any design or
implementation work, and don't restate its content here — link to its sections instead.

## Repo layout

| Path | Purpose | In git |
|---|---|---|
| `README.md` | project overview and pointers to the docs below | yes |
| `docs/DESIGN.md` | design source of truth (Chinese, user-reviewed); referred to as `DESIGN.md` | yes |
| `docs/SETUP.md` | new-machine setup: environment, model weights, data | yes |
| `docs/results.md` | measured results of each method (what was run, with which script, and the numbers) | yes |
| `matchlive/` | the Python package; subpackage split is `DESIGN.md` §8 | yes |
| `configs/` | YAML configs: paths (relative to repo root), models, loop switches (`DESIGN.md` §7) | yes |
| `scripts/` | entry points and visual checks that write videos/images for inspection | yes |
| `tests/` | pytest checks that a module does its job (e.g. memory cascade, rule firing) — see Testing | yes |
| `dataset/` | match videos + `metadata.json` (`organized_data/`), streaming QA (`streamingqa/`) | no |
| `checkpoints/` | model weights (RF-DETR ONNX, DINOv3 backbone, jersey head) | no |
| `cache/`, `outputs/` | perception/API caches, run outputs — safe to delete, costly to rebuild | no |
| `latex/` | paper draft; its **own** git repo synced with Overleaf — never add it here | no |

`~/live-comment-v2` is the reference implementation the perception modules are ported from.

## Environment & commands

conda env **`matchstream`** (Python 3.11). Install commands, CUDA/TensorRT notes, and where the
model weights and data come from: see `docs/SETUP.md`.
- Add further packages only when a phase actually needs them (e.g. `websocket-client` for TTS),
  and record them in `docs/SETUP.md`.
- Model weights and match data sync through the private HF dataset repo
  `Becomebright/MatchLive-assets` (`docs/SETUP.md` §3); upload any new file other machines need there.
- API keys are read from environment variables (loaded from the gitignored `.env`); never
  hardcode them.
- Behaviour tests: `conda run -n matchstream pytest tests/`. Visual checks: `conda run -n matchstream
  python scripts/<name>.py`, writing into `outputs/`.
- Diagrams: `mmdc` (mermaid-cli, via nvm Node) renders the Mermaid blocks in `DESIGN.md`; Chinese
  text needs `FONTCONFIG_FILE=~/.config/fontconfig/fonts.conf` (set in `~/.zshrc`).

## Conventions

- **Testing — check what a module achieves, not its plumbing.**
  - *Write* checks of a module's effect: the phase acceptance checks in `DESIGN.md` §10 and
    anything like them, e.g. perception overlay videos and per-frame timing, identity accuracy,
    memory cascade (retracting a yellow card removes the derived red card), rule firing counts.
    Visual/report checks go in `scripts/`, pass/fail checks in `tests/`.
  - *Don't write* tests for interfaces or I/O: function signatures and return types, config or
    file loading, cache read/write, serialization round-trips, API client wiring.
- Data conventions (time units, boxes, track and player keys) are a design contract — see
  `DESIGN.md` §4; follow them rather than restating or redefining them locally.
- No hardcoded absolute paths: everything resolves from `configs/` relative to the repo root.
- Comments: write explanatory English comments by default (heavily commented, like v2). Prefer
  inline literals over alias constants; avoid needless abstraction.
- If implementation forces a change to anything in `DESIGN.md`, stop and propose the change to
  the user first; update `DESIGN.md` only after they agree.

## Porting notes (v2 → matchlive)

Implementation fixes to carry over when porting v2's perception code (`DESIGN.md` §6.1 names them;
the details live here and, once ported, in code comments):
- **Jersey checkpoint loading:** under transformers 5.x, DINOv3's `stages.*` are nested under
  `model.` while `layer_norm` stays top-level. v2's `_remap_state_dict` prefixes *every* key with
  `model.`, so strict loading fails. Prefix only the keys the model actually nests (verified to load
  strictly on transformers 5.17).
- **Goalkeeper kit colours:** `metadata.json` `jersey_colors` also has `home_goalkeeper` /
  `away_goalkeeper` keys; v2 treats them as a 3rd and 4th "team". Match goalkeeper colours only
  against `role == "goalkeeper"` tracks, and resolve them back to `home` / `away`.

## Working guidelines

- **Git: never commit unasked.** Commit only when explicitly asked to. Put new features on their
  own feature branch — never directly on `dev`. The user reviews and merges into `dev`; `main` is
  the stable branch, updated from `dev` at milestones.
- **Think before coding.** State assumptions; if multiple interpretations exist, present them
  rather than picking silently. When unclear, stop and ask.
- **Simplicity first.** Minimum code that solves the problem; no speculative features or
  abstractions for single-use code.
- **Surgical changes.** Touch only what the request needs; match existing style. Flag pre-existing
  dead code instead of deleting it unasked.
- **Goal-driven execution.** Turn tasks into verifiable criteria and loop until they hold; each
  phase in `DESIGN.md` §10 has an acceptance check.
