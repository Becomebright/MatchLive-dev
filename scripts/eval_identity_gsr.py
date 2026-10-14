"""Identity accuracy (team + jersey number) on the SoccerNet-GSR test set.

Runs the perception chain (cached RF-DETR candidates, shot cuts, tracking, jersey reads,
kit colours) over every GSR test sequence at 720p — frames and boxes are scaled from
1080p so crops match our match videos — then the streaming IdentityResolver, and scores
each frame's identities against the ground truth. Identities at frame t use frames <= t
only, so this measures what the live system would show.

GSR gives no roster and no kit colours, so:
  * the roster is open: every number 0-99 of both teams, player_id "<team>#<number>";
  * kit colours come from the ground truth: the mean kit colour (identity.kit_color) of
    each kit (left / right outfield, left / right keeper) over every 15th frame. They name
    the outfield colour clusters and are matched directly for keepers; being truer than
    the metadata's crude colours, keeper team accuracy here is an upper bound.

Scored over our track boxes matched to a ground-truth box (IoU >= 0.5, per frame):
  team       coverage / accuracy of the team on player + keeper boxes
  number     coverage / precision of the number on outfield players with a labelled number
  player     precision of team + number together (the identity the resolver outputs)
  referee    share of referee boxes (detected as players) wrongly given a player
Variants: our configuration, and v2-style voting (a vote every 15 frames, any read counts,
one read is enough, no one-player-per-track assignment).

Per-sequence perception arrays are cached in outputs/gsr_eval/perception/, keyed like the
match caches (matchlive.common.cache.perception_key).

    python scripts/eval_identity_gsr.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers.eval import box_iou

from eval_tracking_gsr import LABEL_PATHS, load_raw_candidates   # sibling script (scripts/ is on sys.path)
from matchlive.common.cache import iter_frames, load_perception, perception_key, save_perception
from matchlive.common.config import load_config
from matchlive.perception.detection import to_detections
from matchlive.perception.identity import IdentityResolver, kit_color
from matchlive.perception.jersey import crop_upper_body
from matchlive.runtime.runner import MatchRunner

SCALE = 2 / 3                                    # 1080p -> 720p
SIDE = {"left": "home", "right": "away"}
# Open roster: any number of either team is a valid player.
OPEN_ROSTER = {(side, str(n)): f"{side}#{n}" for side in ("home", "away") for n in range(100)}


def frames_720p(label_path, images):
    for image in images:
        frame = cv2.imread(str(label_path.parent / "img1" / image["file_name"]))
        yield int(image["file_name"][:-4]), cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)


def ground_truth(labels):
    """frame -> list of (role, team, jersey, ltwh at 720p) for players, keepers, referees."""
    frame_of_image = {im["image_id"]: int(im["file_name"][:-4]) for im in labels["images"]}
    gt = defaultdict(list)
    for a in labels["annotations"]:
        if a.get("supercategory") == "object" and a["category_id"] in (1, 2, 3):
            b, attr = a["bbox_image"], a["attributes"]
            gt[frame_of_image[a["image_id"]]].append(
                (attr["role"], SIDE.get(attr.get("team")), attr.get("jersey") or None,
                 (b["x"] * SCALE, b["y"] * SCALE, b["w"] * SCALE, b["h"] * SCALE)))
    return gt


def kit_colours(label_path, images, gt):
    """Kit LAB colours from the ground truth of every 15th frame (see module docstring)."""
    samples = defaultdict(list)
    for frame_idx, frame in frames_720p(label_path, images[::15]):
        for role, team, _, ltwh in gt[frame_idx]:
            crop = crop_upper_body(frame, ltwh)
            if role in ("player", "goalkeeper") and crop is not None:
                color = kit_color(crop)
                if not np.isnan(color[0]):
                    samples[team + ("_goalkeeper" if role == "goalkeeper" else "")].append(color)
    return {kit: np.mean(v, axis=0).tolist() for kit, v in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting, e.g. perception.jersey.model=/path/to/jersey.onnx")
    args = ap.parse_args()

    config = load_config(overrides=args.set)
    runner = MatchRunner(config)
    confidence = config["perception"]["detection"]["confidence"]
    out_dir = config["paths"]["outputs"] / "gsr_eval"
    raw = load_raw_candidates(config, out_dir)
    raw = raw[raw[:, 5] > confidence]
    rows_of_image = defaultdict(list)
    for row in raw:
        rows_of_image[int(row[0])].append(row[1:])

    ours = config["perception"]["identity"]
    variants = {"ours": ours,
                "v2-style": dict(ours, vote_every=15, jersey_min_prob=0.0, min_number_votes=1,
                                 one_player_per_track=False)}
    counts = {name: defaultdict(int) for name in variants}

    for label_path in LABEL_PATHS:
        labels = json.load(open(label_path))
        name = labels["info"]["name"]
        images = sorted(labels["images"], key=lambda im: im["file_name"])
        gt = ground_truth(labels)

        key = perception_key(config, [label_path, out_dir / "rfdetr-l_raw.npy"])
        path = out_dir / "perception" / f"{name}_{key}.npz"
        if not path.exists():
            detections = {}
            for image in images:
                r = np.asarray(rows_of_image[int(image["image_id"])]).reshape(-1, 6)
                detections[int(image["file_name"][:-4])] = to_detections(
                    r[:, :4] * SCALE, r[:, 4], r[:, 5], confidence)
            arrays, timing = runner.perceive(frames_720p(label_path, images), 25.0, detections)
            save_perception(path, arrays, {"sequence": name, **timing,
                                           "kits": kit_colours(label_path, images, gt)})
        arrays, meta = load_perception(path)
        kits = {kit: np.asarray(v, dtype=np.float32) for kit, v in meta["kits"].items()}

        resolvers = {v: IdentityResolver(params, kits) for v, params in variants.items()}
        matched = 0
        for frame_idx, tracks, jersey, color, _ in iter_frames(arrays):
            # Match our track boxes to ground truth (Hungarian on IoU >= 0.5).
            truth = gt[frame_idx]
            match = {}
            if tracks and truth:
                sim = box_iou(np.array([t.bbox_ltwh for t in tracks]), np.array([t[3] for t in truth]), box_format="xywh")
                match = {r: truth[c] for r, c in zip(*linear_sum_assignment(-sim)) if sim[r, c] >= 0.5}
            matched += len(match)
            for variant, resolver in resolvers.items():
                c = counts[variant]
                for i, t in enumerate(resolver.update(frame_idx, tracks, jersey, color, OPEN_ROSTER, {})):
                    if i not in match:
                        continue
                    role, team, number, _ = match[i]
                    if role == "referee":
                        c["ref"] += 1
                        c["ref_as_player"] += t.player_id is not None
                        continue
                    c["team_rows"] += 1
                    if t.team is not None:
                        c["team_given"] += 1
                        c["team_ok"] += t.team == team
                    if role == "player" and number:
                        c["num_rows"] += 1
                        if t.player_id is not None:
                            side, read = t.player_id.split("#")
                            c["num_given"] += 1
                            c["num_ok"] += read == number
                            c["player_ok"] += read == number and side == team
        print(f"{name}: {len(arrays['track'])} track boxes, {matched} matched to GT, kits {sorted(kits)}", flush=True)

    print(f"\n{'variant':9s} {'team cov':>9s} {'team acc':>9s} {'num cov':>8s} {'num prec':>9s} "
          f"{'player prec':>12s} {'ref as player':>14s}")
    for variant, c in counts.items():
        print(f"{variant:9s} {c['team_given'] / c['team_rows']:9.3f} {c['team_ok'] / max(c['team_given'], 1):9.3f} "
              f"{c['num_given'] / c['num_rows']:8.3f} {c['num_ok'] / max(c['num_given'], 1):9.3f} "
              f"{c['player_ok'] / max(c['num_given'], 1):12.3f} {c['ref_as_player'] / max(c['ref'], 1):14.3f}")


if __name__ == "__main__":
    main()
