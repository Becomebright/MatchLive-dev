"""Offline match runner (DESIGN.md §6.10).

MatchRunner makes one pass over a half video and writes the raw perception cache
(matchlive.common.cache). run_live() then plays the whole match forward on one match clock:
identity voting frame by frame (fed by memory, loop.state_to_perception), the event analyzer
on every fixed window, the consistency check and the arbiter — each API answer taking effect
at its issue time + measured latency, as it would live. run_commentary() drives the commentary
over a live run's memory on the same clock. replay_events() is the older loop over
analyzer answers cached by scripts/run_events.py, kept for the subagent arbiter
(scripts/run_arbiter.py).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from matchlive.common.cache import ROLE_CODE, cache_path, iter_frames, save_perception
from matchlive.common.match import Match
from matchlive.common.types import Detection, Track
from matchlive.memory.store import MatchMemory
from matchlive.perception.detection import RFDETRDetector, to_detections
from matchlive.perception.identity import IdentityResolver, kit_color, kit_lab, roster_index
from matchlive.perception.jersey import NUM_LABELS, UNREADABLE, JerseyReader, crop_upper_body, label_to_number
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import PlayerTracker, select_ball
from matchlive.perception.visualization import nearest_to_ball


class MatchRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        paths, perception = config["paths"], config["perception"]
        # Model sessions are expensive to create (TensorRT engines load from disk), so
        # they live as long as the runner; per-sequence state is created in perceive().
        self.detector = RFDETRDetector(perception["detection"], paths["checkpoints"], paths["cache"])
        self.jersey = JerseyReader(perception["jersey"], paths["checkpoints"], paths["cache"])

    def run_perception(self, match: Match, half: int, start_frame: int = 0,
                       n_frames: int | None = None) -> Path:
        """Write the raw perception cache for frames [start_frame, start_frame + n_frames)
        of one half (the whole half when n_frames is None); return its path. An existing
        cache for the same inputs is reused."""
        video = match.video_path(half)
        path = cache_path(self.config, video, match.name, half, start_frame, n_frames)
        if path.exists():
            return path

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        def frames():
            frame_idx = start_frame
            while n_frames is None or frame_idx < start_frame + n_frames:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame_idx, frame
                frame_idx += 1

        arrays, timing = self.perceive(frames(), fps)
        meta = {"match": match.name, "half": half, "fps": fps, "start_frame": start_frame,
                "n_frames": len(arrays["shot_id"]), **timing}
        save_perception(path, arrays, meta)
        return path

    def perceive(self, frames: Iterable[tuple[int, np.ndarray]], fps: float,
                 detections: dict[int, list[Detection]] | None = None) -> tuple[dict, dict]:
        """Run the per-frame chain over (frame_idx, BGR frame) pairs and return the raw cache
        arrays (layout in matchlive.common.cache) and a timing summary: mean ms/frame per
        stage ("ms_per_frame") and percentiles of each frame's total ("total_ms").

        `detections` (frame_idx -> detections) replaces the detector, for data whose
        detections are already cached (SoccerNet-GSR evaluation).
        """
        perception = self.config["perception"]
        shots = ShotDetector(perception["shots"]["cut_threshold"])
        tracker = PlayerTracker(perception["tracking"], frame_rate=fps)

        shot_ids, det_rows, track_rows, jersey_rows, color_rows = [], [], [], [], []
        timing = {"decode": 0.0, "detect": 0.0, "shots": 0.0, "track": 0.0, "jersey": 0.0, "color": 0.0}
        totals = []
        t_prev = time.perf_counter()
        for frame_idx, frame in frames:
            t0 = time.perf_counter()
            dets = self.detector.predict(frame) if detections is None else detections.get(frame_idx, [])
            t1 = time.perf_counter()
            shot_id = shots.update(frame)
            t2 = time.perf_counter()
            tracks = tracker.update(dets, frame, shot_id)
            t3 = time.perf_counter()
            # Read every track on every frame; boxes too small to crop get NaN rows so the
            # jersey and colour arrays stay aligned with the track rows.
            crops = [crop_upper_body(frame, t.bbox_ltwh) for t in tracks]
            probs = np.full((len(tracks), NUM_LABELS), np.nan, dtype=np.float32)
            readable = [i for i, c in enumerate(crops) if c is not None]
            if readable:
                probs[readable] = self.jersey.predict_batch([crops[i] for i in readable])
            t4 = time.perf_counter()
            colors = np.full((len(tracks), 3), np.nan, dtype=np.float32)
            for i in readable:
                colors[i] = kit_color(crops[i])
            t5 = time.perf_counter()

            shot_ids.append(shot_id)
            det_rows += [(frame_idx, *d.bbox_ltwh, d.score, d.class_id) for d in dets]
            track_rows += [(frame_idx, shot_id, ROLE_CODE[t.role], t.track_id, *t.bbox_ltwh) for t in tracks]
            jersey_rows.append(probs)
            color_rows.append(colors)
            # "decode" is the time spent getting this frame from the iterator.
            for key, dt in zip(timing, (t0 - t_prev, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                timing[key] += dt
            totals.append(t5 - t_prev)
            t_prev = time.perf_counter()

        n = max(len(shot_ids), 1)
        arrays = {
            "shot_id": np.asarray(shot_ids, dtype=np.int32),
            "det": np.asarray(det_rows, dtype=np.float32).reshape(-1, 7),
            "track": np.asarray(track_rows, dtype=np.float32).reshape(-1, 8),
            "jersey": np.concatenate(jersey_rows or [np.zeros((0, NUM_LABELS))]).astype(np.float16),
            "color": np.concatenate(color_rows or [np.zeros((0, 3))]).astype(np.float32),
        }
        # Real time at 25 fps means every frame, not just the average, fits in 40 ms.
        ms = 1000 * np.asarray(totals or [0.0])
        total_ms = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
                    "p99": float(np.percentile(ms, 99)), "max": float(ms.max()),
                    "share_over_40": float((ms > 40).mean())}
        return arrays, {"ms_per_frame": {k: 1000 * v / n for k, v in timing.items()}, "total_ms": total_ms}


def resolve_frames(config: dict[str, Any], match: Match, arrays: dict[str, np.ndarray],
                   keep: set[int]) -> dict[int, tuple[list[Track], tuple | None, list[str | None]]]:
    """Replay identity voting over a half's raw perception cache, streaming in frame order
    (identities at frame t use only frames <= t), and keep frames `keep`:
    frame_idx -> (tracks with team / player_id, the key of the track nearest the ball or
    None, the number read clearly off each track in this frame or None). Frames without
    tracks or detections are absent.

    Votes over the whole squad without memory (the §7 loop.state_to_perception baseline);
    run_live() does the same with the on-pitch roster and identities from memory.
    """
    roster = roster_index(match.prematch["players"])
    resolver = IdentityResolver(config["perception"]["identity"], kit_lab(match.prematch["jersey_colors"]))
    out = {}
    for frame_idx, tracks, jersey, color, det in iter_frames(arrays):
        tracks = resolver.update(frame_idx, tracks, jersey, color, roster, confirmed={})
        if frame_idx in keep:
            out[frame_idx] = frame_view(config, tracks, jersey, det)
    return out


def frame_view(config: dict[str, Any], tracks: list[Track], jersey: np.ndarray,
               det: np.ndarray) -> tuple[list[Track], tuple | None, list[str | None]]:
    """One frame as the event analyzer is told about it: (tracks with identity, the key of the
    track nearest the ball or None, the number read clearly off each track or None)."""
    ball = select_ball(to_detections(det[:, :4], det[:, 4], det[:, 5], 0.0), config["perception"]["ball"]["min_score"])
    ball_point = None if ball is None else (ball.bbox_ltwh[0] + ball.bbox_ltwh[2] / 2,
                                            ball.bbox_ltwh[1] + ball.bbox_ltwh[3])
    carrier = nearest_to_ball(tracks, ball_point)
    # "Clearly read" uses the same bar as a jersey vote (identity.jersey_min_prob).
    min_prob = config["perception"]["identity"]["jersey_min_prob"]
    numbers = []
    for probs in jersey:
        label = -1 if np.isnan(probs[0]) else int(np.argmax(probs))
        clear = label not in (-1, UNREADABLE) and probs[label] >= min_prob
        numbers.append(label_to_number(label) if clear else None)
    return tracks, None if carrier is None else carrier.key, numbers


def observations(results: dict[int, list[dict]]) -> dict[str, dict]:
    """What produced each observation id ("obs:<call id>#<i>"): its analyzer window and candidate."""
    return {f"obs:{r['call_id']}#{i}": {"window": r["window"], "candidate": c}
            for rows in results.values() for r in rows for i, c in enumerate(r.get("candidates", []))}


def replay_events(config: dict[str, Any], match: Match, results: dict[int, list[dict]],
                  decide=None) -> tuple[MatchMemory, list[dict]]:
    """Offline run of the event loop (DESIGN.md §6.10) from cached analyzer answers.

    `results` maps half -> the windows written by scripts/run_events.py (window, call_id,
    latency, candidates with their voted players). Each window's answer takes effect at its
    window end + the measured latency, so ops enter memory in knowledge-time order as they
    would live; the missing-partner constraints are judged as time passes.

    Every case the consistency check opens goes to the arbiter at once (event-level
    conflicts escalate at k=1, DESIGN.md §6.4); its verdict takes effect at the case's
    opening time + the arbiter's latency, unless that is past the deadline. The arbiter is
    `decide(mem, case, packet, t) -> Decision | None` — by default the API arbiter; None
    leaves the case open (e.g. to evaluate the consistency check alone); scripts/run_arbiter.py
    passes cached subagent verdicts. With loop.arbiter off, the newest belief of each case is
    rejected instead. Returns memory and one record per case the arbiter decided.
    """
    import heapq
    import itertools

    from matchlive.arbiter.agent import Arbiter, build_packet
    from matchlive.arbiter.verdict import apply_verdict, reject_newest
    from matchlive.common.api import APIClient
    from matchlive.events.analyzer import Candidate
    from matchlive.events.consistency import ConsistencyChecker
    from matchlive.runtime.evidence import EvidenceTools

    mem = MatchMemory(match.prematch["players"], cascade=config["loop"]["cascade"])
    checker = ConsistencyChecker(config)
    tools = EvidenceTools(config, match)          # loads perception only when first asked
    if decide is None:
        client = APIClient(config["api"], config["paths"]["cache"])
        arbiter = Arbiter(config["arbiter"], lambda parts: client.generate(parts, json_output=True))

        def decide(mem, case, packet, t):
            return arbiter.decide(packet, lambda tool, args: tools.call(mem, t, tool, args))
    sources = observations(results)

    # One timeline: analyzer answers arriving, and verdicts taking effect.
    order = itertools.count()
    queue = [((half, r["window"][1] + r["latency"]), next(order), "answer", (half, r))
             for half, rows in results.items() for r in rows if "candidates" in r]
    queue.append(((3, 0.0), next(order), "whistle", None))       # after the final whistle
    heapq.heapify(queue)
    n_seen, records = 0, []
    while queue:
        t, _, kind, payload = heapq.heappop(queue)
        if kind == "verdict":
            record, decision, t_bound = payload
            if mem.cases[record["case"]].status == "open":
                record["void"] = apply_verdict(
                    mem, t, record["case"], decision.verdict, decision.evidence_ids, t_bound,
                    config["events"]["live_margin"],
                    lambda key: len(tools.track_frames(key, t_bound)) > 0,
                    lambda a, b: tools.cooccur(a, b, t_bound))
            continue
        checker.tick(mem, t)
        if kind == "answer":
            half, r = payload
            for i, c in enumerate(r["candidates"]):
                checker.propose(mem, t, half, tuple(r["window"]), Candidate.from_dict(c),
                                f"obs:{r['call_id']}#{i}", c["players"])
        # Case ids are handed out in order, so the new cases are the ones past n_seen.
        new, n_seen = list(mem.cases.values())[n_seen:], len(mem.cases)
        for case in new:
            if not config["loop"]["arbiter"]:
                reject_newest(mem, t, case.case_id)
                continue
            packet = build_packet(mem, case, t, sources, match.prematch, config["arbiter"]["context"])
            decision = decide(mem, case, packet, t)
            if decision is None:
                continue
            record = {"case": case.case_id, "constraint": case.constraint, "opened": t,
                      "calls": decision.calls, "latency": decision.latency, "verdict": decision.verdict,
                      "evidence": len(decision.evidence_ids), "void": decision.note or None}
            records.append(record)
            if decision.verdict is None:
                continue
            if decision.latency > config["arbiter"]["deadline"]:
                record["void"] = "past the deadline"
                continue
            heapq.heappush(queue, ((t[0], t[1] + decision.latency), next(order), "verdict", (record, decision, t)))
    return mem, records


def off_roster_obs(half: int, key: tuple, player: str) -> str:
    """Observation id of a track reading as a player memory has off the pitch
    (loop.state_to_perception): "read:<half>/<shot>/<track>:<player hash>"."""
    return f"read:{half}/{key[0]}/{key[1]}:{player}"


def run_live(config: dict[str, Any], match: Match, out_dir: Path, client, corrupt=None,
             until: tuple[int, float] | None = None) -> MatchMemory:
    """The whole match, live, from the raw perception cache (DESIGN.md §6.10 offline mode).

    One match clock drives everything, half by half and frame by frame:
      - identity voting (perception.identity) on every frame; with loop.state_to_perception it
        votes only among the players memory has on the pitch at that moment, keeps memory's
        confirmed identity entries, and reports a number worn by nobody on the pitch that names
        an off-pitch player as evidence (consistency.off_roster_read);
      - at the end of every fixed window, the event analyzer's request is built from the
        frames and identities seen so far and sent through `client` (APIClient); its answer
        takes effect at the window end + the measured latency;
      - answers go through the consistency check; every event-level case goes to the arbiter
        at once, a perception case once it has `off_roster_k` sources; the verdict takes effect
        at the case's opening + the arbiter's summed latency, unless past the deadline.
    The model calls are synchronous, so the wall time is the sum of their latencies; what memory
    holds at each match time is what a live run with the same latencies would have held.

    Writes to `out_dir`: ops.jsonl (the memory log — everything the QA reads), windows.jsonl
    (each window's call and candidates), cases.json (each arbitrated case), and
    perceived_half<h>.pkl (second -> the frame as the analyzer was told about it, at the
    sampled seconds, for the QA's current view). Returns memory.

    For the error-injection experiment (DESIGN.md §10 phase 4): `corrupt(half, window, candidates)`
    returns the candidates the analyzer is taken to have answered instead (e.g. without a
    substitution, or with a card given to the wrong player); `until` stops the match clock there.
    """
    import heapq
    import itertools
    import json
    import pickle

    from matchlive.arbiter.agent import Arbiter, Decision, build_packet
    from matchlive.arbiter.verdict import apply_verdict, reject_newest
    from matchlive.common.api import NoAPIKey
    from matchlive.common.cache import load_perception
    from matchlive.events.analyzer import Candidate, build_request, parse_candidates
    from matchlive.events.consistency import ConsistencyChecker
    from matchlive.events.windows import fixed_windows
    from matchlive.runtime.evidence import EvidenceTools

    loop, ev, prematch = config["loop"], config["events"], match.prematch
    feedback = loop["state_to_perception"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ops.jsonl").unlink(missing_ok=True)
    mem = MatchMemory(prematch["players"], log_path=out_dir / "ops.jsonl", cascade=loop["cascade"])
    checker = ConsistencyChecker(config)
    tools = EvidenceTools(config, match)
    arbiter = Arbiter(config["arbiter"], lambda parts: client.generate(parts, json_output=True))
    squad = roster_index(prematch["players"])
    sources: dict[str, dict] = {}      # observation id -> the analyzer window and candidate behind it
    records: list[dict] = []           # one per arbitrated case
    windows_log = open(out_dir / "windows.jsonl", "w")
    # One timeline of things taking effect later: analyzer answers arriving, verdicts landing.
    queue: list = []
    order = itertools.count()
    n_seen = 0                         # cases already handled (case ids are handed out in order)

    def arbitrate(case, t, opened=None, void=None):
        """Send `case` to the arbiter at match time t. `void` is why its previous verdict was void:
        the arbiter is told and rules once more (DESIGN.md §6.6), within the deadline counted from
        the case's opening (`opened`)."""
        if not loop["arbiter"]:
            reject_newest(mem, t, case.case_id)
            return
        opened = opened or t
        packet = build_packet(mem, case, t, sources, prematch, config["arbiter"]["context"])
        if void:
            packet += (f"\n\nYour previous verdict on this case was void: {void}. Rule again; every write must cite "
                       "evidence ids returned by the tools in this investigation.")
        try:
            decision = arbiter.decide(packet, lambda tool, args: tools.call(mem, t, tool, args))
        except NoAPIKey:
            raise
        except Exception as e:  # noqa: BLE001 — out of API retries: the case stays open, nothing blocks
            decision = Decision(None, [], 0, 0.0, f"API failure: {type(e).__name__}: {str(e)[:200]}")
        record = {"case": case.case_id, "constraint": case.constraint, "opened": opened, "calls": decision.calls,
                  "latency": decision.latency, "verdict": decision.verdict, "retry": bool(void),
                  "evidence": len(decision.evidence_ids), "void": decision.note or None}
        records.append(record)
        print(f"  case {case.case_id} {case.constraint} at {t[0]}H {t[1]:.0f}s{' (again)' if void else ''}: "
              f"{'verdict' if decision.verdict else decision.note} ({decision.calls} calls, {decision.latency:.0f}s)",
              flush=True)
        if decision.verdict is None:
            return
        elapsed = t[1] - opened[1] + decision.latency if t[0] == opened[0] else float("inf")
        if elapsed > config["arbiter"]["deadline"]:
            record["void"] = "past the deadline"
            return
        heapq.heappush(queue, ((t[0], t[1] + decision.latency), next(order), "verdict", (record, decision, t)))

    def settle(t):
        # Event-level conflicts go to the arbiter at once (k = 1, DESIGN.md §6.4); perception
        # ones (off_roster_read) only when off_roster_read() says they have enough sources.
        nonlocal n_seen
        new, n_seen = list(mem.cases.values())[n_seen:], len(mem.cases)
        for case in new:
            if case.constraint != "off_roster_read":
                arbitrate(case, t)

    def advance(t):
        """Apply everything that has taken effect by match time t, in time order."""
        while queue and queue[0][0] <= t:
            when, _, kind, payload = heapq.heappop(queue)
            if kind == "verdict":
                record, decision, t_bound = payload
                if mem.cases[record["case"]].status == "open":
                    record["void"] = apply_verdict(
                        mem, when, record["case"], decision.verdict, decision.evidence_ids, t_bound,
                        ev["live_margin"], lambda key: len(tools.track_frames(key, t_bound)) > 0,
                        lambda a, b: tools.cooccur(a, b, t_bound))
                    if record["void"] and not record["retry"]:
                        # A void verdict leaves the case open and its members pending for good;
                        # the arbiter gets one more try, told why (DESIGN.md §6.6).
                        arbitrate(mem.cases[record["case"]], when, record["opened"], record["void"])
                continue
            checker.tick(mem, when)
            half, r = payload
            for i, c in enumerate(r["candidates"]):
                checker.propose(mem, when, half, tuple(r["window"]), Candidate.from_dict(c),
                                f"obs:{r['call_id']}#{i}", c["players"])
            settle(when)

    for half in (1, 2):
        if until is not None and half > until[0]:
            break
        arrays, meta = load_perception(cache_path(config, match.video_path(half), match.name, half, 0, None))
        fps = meta["fps"]
        duration = meta["n_frames"] / fps
        windows = fixed_windows(duration, ev["window_seconds"], ev["window_overlap"])
        # Sample times on one grid for the whole half, so overlapping windows share frames.
        step = 1 / ev["sample_fps"]
        grid = [k * step for k in range(int(duration / step) + 1)]
        wanted = {int(round(s * fps)): s for s in grid}
        resolver = IdentityResolver(config["perception"]["identity"], kit_lab(prematch["jersey_colors"]))
        cap = cv2.VideoCapture(str(match.video_path(half)))
        video = {"pos": 0}
        images: dict[float, np.ndarray] = {}      # sampled frames a current or later window still needs
        perceived: dict[float, tuple] = {}        # sampled second -> frame_view()
        reads_seen: set[str] = set()
        roster, confirmed, version = squad, {}, None

        def fire(window):
            """Window over: build the analyzer's request from what has been seen, call the model,
            and queue the answer for when it arrives."""
            t = (half, window[1])
            advance(t)
            checker.tick(mem, t)
            settle(t)
            while video["pos"] <= int(round(window[1] * fps)):
                if not cap.grab():
                    break
                if video["pos"] in wanted:
                    ok, image = cap.retrieve()
                    if ok:
                        images[wanted[video["pos"]]] = image
                video["pos"] += 1
            shown = [(s, images[s]) for s in grid if window[0] <= s < window[1] and s in images]
            request = build_request(half, window, shown, perceived, prematch, ev["sample_fps"], ev["ask_confidence"])
            # The player behind each track as last seen in the window — what the model was told.
            players_of = {(half, *tr.key): tr.player_id
                          for s, _ in shown for tr in perceived.get(s, ((), None, ()))[0] if tr.player_id}
            try:
                result = client.generate(request.parts, json_output=True)
            except NoAPIKey:
                raise
            except Exception as e:  # noqa: BLE001 — out of retries: live, this window is simply lost
                windows_log.write(json.dumps({"half": half, "window": window, "error": f"{type(e).__name__}: {e}"}) + "\n")
                print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s failed: {e}", flush=True)
                return
            cands = [{**c.__dict__, "players": [players_of.get(key) for _, key in c.participants]}
                     for c in parse_candidates(result.text, request)]
            if corrupt is not None:
                cands = corrupt(half, window, cands)
            r = {"half": half, "window": window, "call_id": result.call_id, "model": result.model,
                 "latency": result.latency, "cached": result.cached, "candidates": cands}
            windows_log.write(json.dumps(r) + "\n")
            windows_log.flush()
            print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s {len(cands)} candidates, {result.latency:.1f}s"
                  f"{' cached' if result.cached else ''}", flush=True)
            for i, c in enumerate(cands):
                sources[f"obs:{result.call_id}#{i}"] = {"window": window, "candidate": c}
            heapq.heappush(queue, ((half, window[1] + result.latency), next(order), "answer", (half, r)))
            # The next window starts `window_overlap` s before this one ends.
            for s in [s for s in images if s < window[1] - ev["window_overlap"]]:
                del images[s]

        w = 0
        for frame_idx, tracks, jersey, color, det in iter_frames(arrays):
            second = frame_idx / fps
            if until is not None and (half, second) > until:
                break
            while w < len(windows) and windows[w][1] <= second:
                fire(windows[w])
                w += 1
            advance((half, second))
            if feedback and len(mem.ops) != version:
                # Memory changed: the players on the pitch now, and the confirmed identities of
                # this half's tracks (DESIGN.md §6.1 identity (a), (c)).
                version = len(mem.ops)
                view = mem.at((half, second))
                on = set(view.on_field["home"] + view.on_field["away"])
                roster = roster_index([p for p in prematch["players"] if p["hash"] in on])
                confirmed = {key[1:]: pid for key, pid in mem.track_players().items() if key[0] == half}
            tracks = resolver.update(frame_idx, tracks, jersey, color, roster, confirmed)
            if feedback and loop["consistency"]:
                for key, player in resolver.off_roster_reads({tr.key: tr.team for tr in tracks}, roster, squad):
                    obs = off_roster_obs(half, key, player)
                    if obs in reads_seen:
                        continue
                    reads_seen.add(obs)
                    case_id = checker.off_roster_read(mem, (half, second), obs, player)
                    settle((half, second))
                    if case_id:
                        arbitrate(mem.cases[case_id], (half, second))
            if frame_idx in wanted:
                perceived[wanted[frame_idx]] = frame_view(config, tracks, jersey, det)
        while w < len(windows) and (until is None or (half, windows[w][1]) <= until):
            fire(windows[w])
            w += 1
        cap.release()
        with open(out_dir / f"perceived_half{half}.pkl", "wb") as f:
            pickle.dump(perceived, f)

    # After the final whistle (or at `until`): the last answers and verdicts land.
    advance((3, 0.0) if until is None else until)
    windows_log.close()
    (out_dir / "cases.json").write_text(json.dumps(records, indent=1, default=str))
    return mem


def run_commentary(config: dict[str, Any], match: Match, run_dir: Path, client,
                   only: tuple[int, float, float] | None = None) -> list[dict]:
    """Live commentary (DESIGN.md §6.9) over a live run (run_live's output in `run_dir`).

    The broadcast output runs `output_delay` (D) s behind the live feed, and the commentary is laid
    out on the output's timeline. At match time t on the feed the viewer is at t - D: the policy
    plans the line that goes on air next — about what the viewer will be watching then — from
    memory as known at t, i.e. with D s of hindsight on that moment, and says nothing about what the
    viewer has not seen yet. A play line is written over the frames on air (the last `play_span` s
    up to it, labelled with who each tracked player is as far as known by t; runtime.qa_tools). The
    line is generated (one model call through `client`) and voiced (one text-to-speech call); it
    is ready at t + both latencies and goes on air at ready - D, or when the previous line has
    ended, and lasts as long as its audio. The next line is planned while this one is being spoken,
    so that it is ready as this one ends (+ `pause`).

    Commentary only reads memory, and memory as known at t is a prefix of the run's op log
    (DESIGN.md §5.4), so driving it from the log after the run gives the lines a live run would
    have given with the same latencies. A generation that fails after its retries loses the line,
    as it would live; a failed synthesis leaves it a subtitle only. `only` = (half, start, end)
    restricts the commentary to that stretch of video time (a trial).

    Returns one record per line: its kind, entry, text and tone, the entry's status and the players
    the line may name, the match times it was decided (t_decide), ready (t_ready) and of its event
    (t_event), when it is on air ("show": half, start, end in video time), the audio (path,
    duration), and the calls' ids and latencies.
    """
    import copy
    import json
    import pickle
    from collections import deque

    from matchlive.app.commentary import Commentator, CommentaryPolicy, speaking_time
    from matchlive.app.tts import Voice
    from matchlive.common.api import NoAPIKey
    from matchlive.events.analyzer import clock
    from matchlive.runtime.qa_tools import QATools

    cfg = config["commentary"]
    delay = cfg["output_delay"]
    mem = MatchMemory(match.prematch["players"], log_path=run_dir / "ops.jsonl", cascade=config["loop"]["cascade"])
    windows = [json.loads(line) for line in open(run_dir / "windows.jsonl")]
    sources = {f"obs:{w['call_id']}#{i}": {"window": w["window"], "candidate": c}
               for w in windows for i, c in enumerate(w.get("candidates", []))}
    half_end = {h: max(w["window"][1] for w in windows if w["half"] == h) for h in (1, 2)}
    # The frames on air for play lines, labelled as the QA's current view is.
    perceived = {h: pickle.load(open(run_dir / f"perceived_half{h}.pkl", "rb")) for h in (1, 2)}
    view_config = copy.deepcopy(config)
    view_config["qa"]["max_frames"] = cfg["play_frames"]
    pictures = QATools(view_config, match, mem, perceived, sources)
    policy = CommentaryPolicy(cfg, match.prematch, sources)
    commentator = Commentator(cfg, lambda parts: client.generate(parts, json_output=True))
    voice = Voice(cfg["voice"], client.speech)
    lines: list[dict] = []
    for half in (1, 2):
        if only and only[0] != half:
            continue
        first, last = (only[1], only[2]) if only else (0.0, half_end[half])
        ends = first                       # video time the previous line ends (+ pause)
        # How long a line takes to generate and voice: planned with the median of the last few + a
        # margin (one slow call must not throw the plan out); a line that is not ready in time is
        # dropped (before it is voiced, if possible).
        took = deque([10.0], maxlen=5)
        # Voicing is planned the same way, from the last few voiced lines: a line found too late is
        # dropped before it is voiced, so a single slow voicing taken as the estimate would make
        # every later line look too late, and never be replaced.
        spoke = deque([1.5], maxlen=5)

        def expected():
            return sorted(took)[len(took) // 2] + 2.0

        t = free = first + delay - expected()                    # match time on the feed
        while t - delay <= last:
            if t < free:
                t += cfg["tick"]
                continue
            # The moment on air: when this line will be ready — never past the feed (its frames).
            on_air = (half, min(max(t + expected() - delay, ends), t - 1.0))
            if on_air[1] > last:
                break
            item = policy.decide(mem.at((half, t)), on_air)
            frames = None
            if item.kind in ("play", "analysis"):
                frames = pictures.call((half, t), "get_frames", {"half": half, "start": item.t_event[1], "end": on_air[1]})
                if not frames.images:          # nothing to look at (e.g. no frames yet)
                    t += cfg["tick"]
                    continue
                # Pre-match background on a player the pictures identify, now and then.
                item.facts += policy.on_screen(frames.text, on_air)
            try:
                text, tone, latency, call_id, phrases = commentator.line(item, [line["text"] for line in lines], frames)
            except NoAPIKey:
                raise
            except Exception as e:  # noqa: BLE001 — out of API retries: this line is lost
                print(f"{half}H {t:.0f}s line lost: {e}", flush=True)
                t += cfg["tick"]
                continue

            def lateness(ready):
                start = max(ready - delay, ends)
                if item.t_event and item.t_event[0] == half and item.kind not in ("play", "analysis"):
                    start = max(start, item.t_event[1] + 1.0)          # never before its event is on screen
                return start, start - on_air[1] > cfg["late_event" if item.entry else "late"]

            speak_took = sorted(spoke)[len(spoke) // 2]
            start, late = lateness(t + latency + speak_took)
            if not late:
                try:
                    spoken = voice.say(text, tone, analyst=item.kind == "analysis")
                    audio, duration, tts_latency = str(spoken.path), spoken.duration, spoken.latency
                    spoke.append(tts_latency)
                except NoAPIKey:
                    raise
                except Exception as e:  # noqa: BLE001 — out of API retries: a subtitle without a voice
                    if "dry run" not in str(e):
                        print(f"{half}H {t:.0f}s no voice: {e}", flush=True)
                    audio, duration, tts_latency = None, speaking_time(text, cfg["words_per_second"]), 0.0
                start, late = lateness(t + latency + tts_latency)
            took.append(latency + speak_took)
            if late:
                # Ready too late for the moment it was written for (a slow call): live, it is dropped —
                # and not picked again (its moment has passed).
                print(f"{half}H {t:.0f}s [{item.kind}] too late by {start - on_air[1]:.0f} s: dropped", flush=True)
                policy.dropped(item, on_air)
                t += cfg["tick"]
                continue
            ready = t + latency + tts_latency
            policy.said(item, text, (half, start))
            lines.append({"kind": item.kind, "entry": item.entry, "status": item.status, "named": item.named,
                          "text": text, "tone": tone, "t_decide": (half, t), "t_ready": (half, ready),
                          "t_event": item.t_event, "show": (half, start, start + duration), "facts": item.facts,
                          "pictures": frames.text if frames is not None else None,   # who is who in its frames
                          "phrases": phrases,                                          # a play line's burst
                          "call_id": call_id, "latency": latency, "audio": audio, "duration": duration,
                          "tts_latency": tts_latency})
            print(f"{half}H {clock(start)} [{item.kind}/{tone}] {text}", flush=True)
            ends = start + duration + cfg["pause"]
            # Plan the next line so that it is ready as this one ends.
            free = max(t + cfg["tick"], ends + delay - expected())
            t += cfg["tick"]
    return lines
