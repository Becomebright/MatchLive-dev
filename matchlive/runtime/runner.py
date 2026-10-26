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


def resolve_frames(config: dict[str, Any], match: Match, arrays: dict[str, np.ndarray], fps: float,
                   frame_of: dict[float, int], windows: list[tuple[float, float]]
                   ) -> dict[tuple[float, float], dict[float, tuple[list[Track], tuple | None, list[str | None]]]]:
    """Replay identity voting over a half's raw perception cache and resolve it window by window
    (DESIGN.md §6.1 identity (f)): the votes are cast frame by frame, and when a window ends, the
    window's sampled frames (`frame_of`: sampled second -> frame, those with window start <=
    second < window end) are resolved with the votes up to that point, so a track is named from
    the whole window's reads and not from the reads before each frame.

    Returns window -> {second -> (tracks with team / player_id, the key of the track nearest the
    ball or None, the number read clearly off each track in this frame or None)}. Seconds
    shared by two windows appear in both, each as that window saw it; frames without tracks or
    detections are absent. Votes over the whole squad without memory (the §7
    loop.state_to_perception baseline); run_live() does the same with the on-pitch roster and
    identities from memory.
    """
    roster = roster_index(match.prematch["players"])
    resolver = IdentityResolver(config["perception"]["identity"], kit_lab(match.prematch["jersey_colors"]))
    sampled = set(frame_of.values())
    raw: dict[int, tuple] = {}            # sampled frame -> (tracks, jersey, detections) of it
    out = {}

    def close(window):
        out[window] = {}
        for s, f in frame_of.items():
            if window[0] <= s < window[1] and f in raw:
                tracks, jersey, det = raw[f]
                out[window][s] = frame_view(config, resolver.resolve(tracks, roster, {}), jersey, det)

    pending = list(windows)
    for frame_idx, tracks, jersey, color, det in iter_frames(arrays):
        while pending and pending[0][1] <= frame_idx / fps:
            close(pending.pop(0))
        resolver.vote(frame_idx, tracks, jersey, color)
        if frame_idx in sampled:
            raw[frame_idx] = (tracks, jersey, det)
    for window in pending:
        close(window)
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
    opening time + the arbiter's latency, however late (DESIGN.md §6.6). The arbiter is
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
    # A re-check's row (a run_live run with loop.recheck) is no window: it is not replayed here.
    queue = [((half, r["window"][1] + r["latency"]), next(order), "answer", (half, r))
             for half, rows in results.items() for r in rows if "candidates" in r and "recheck" not in r]
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
      - at the end of every fixed window, the window's frames are given their identities from
        the votes of the whole window (IdentityResolver.resolve, DESIGN.md §6.1 identity (f)),
        the event analyzer's request is built from them and sent through `client` (APIClient);
        its answer takes effect at the window end + the measured latency;
      - with loop.recheck, an entry still pending on one live sighting once the next window's
        answer has arrived is re-checked once: the analyzer, unchanged, on a window centred on
        that sighting (DESIGN.md §6.4), unless that window is one of the fixed ones; the last
        window of a half has no next one, so its lone sightings are not re-checked;
      - answers go through the consistency check; every event-level case goes to the arbiter
        at once, a perception case once it has `off_roster_k` sources; the verdict takes effect
        at the case's opening + the arbiter's summed latency, however late.
    The model calls are synchronous, so the wall time is the sum of their latencies; what memory
    holds at each match time is what a live run with the same latencies would have held.

    Writes to `out_dir`: ops.jsonl (the memory log — everything the QA reads), windows.jsonl
    (each window's call and candidates; a re-check's row has "recheck": <entry id> and "of":
    the sighting it re-checks), cases.json (each arbitrated case), and
    perceived_half<h>.pkl (second -> the frame as the analyzer was told about it, at the
    sampled seconds, for the QA's current view; a second in two windows as the later one saw
    it, which is known only from that window's end). Returns memory.

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
    from matchlive.memory.types import Event
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
    # Re-checks (loop.recheck, DESIGN.md §6.4): a lone live sighting is re-checked once the
    # window after its own has had its say.
    after: dict[tuple, tuple] = {}     # (half, window) -> (half, the window after it)
    done: set[tuple] = set()           # (half, window) whose answer has arrived, or that failed
    rechecked: set[str] = set()        # entries re-checked already (once each)
    halves: dict[int, dict] = {}       # half -> the sampled frames a re-check is built from

    def arbitrate(case, t, opened=None, void=None):
        """Send `case` to the arbiter at match time t. `void` is why its previous verdict was void:
        the arbiter is told and rules once more (DESIGN.md §6.6). `opened` is when the case opened.
        There is no deadline: a valid verdict takes effect when it arrives, however late."""
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
            # No usable answer (malformed JSON, out of rounds or API retries) is a void verdict
            # too: one more try, told why, once this one's time has passed (DESIGN.md §6.6).
            if not void:
                heapq.heappush(queue, ((t[0], t[1] + decision.latency), next(order), "retry",
                                       (case, opened, decision.note)))
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
            if kind == "retry":
                case, opened, note = payload
                if case.status == "open":
                    arbitrate(case, when, opened, note)
                continue
            checker.tick(mem, when)
            if kind == "recheck":
                half, eid, r = payload
                # The first candidate that is a sighting of the entry joins it (any() stops there);
                # the rest of the footage was analysed by the regular windows already.
                seen = any(checker.recheck(mem, when, eid, half, tuple(r["window"]), Candidate.from_dict(c),
                                           f"obs:{r['call_id']}#{i}", c["players"])
                           for i, c in enumerate(r["candidates"]))
                print(f"  re-check {eid}: {'seen again' if seen else 'not seen'}", flush=True)
                settle(when)
                continue
            half, r = payload
            for i, c in enumerate(r["candidates"]):
                checker.propose(mem, when, half, tuple(r["window"]), Candidate.from_dict(c),
                                f"obs:{r['call_id']}#{i}", c["players"])
            settle(when)
            done.add((half, tuple(r["window"])))
            if loop["recheck"] and loop["consistency"]:
                for eid in due():
                    recheck(eid, when)

    def due():
        """The entries to re-check now: pending on a single live sighting (not a replay), whose
        next window's answer has arrived (or failed) without seeing it again."""
        out = []
        for eid, e in mem.entries.items():
            if not isinstance(e, Event) or eid in rechecked:
                continue
            js = [j for _, j in mem.justifications[eid]]
            if (len(js) != 1 or js[0].kind != "observation" or js[0].sufficient or js[0].obs_id not in sources
                    or mem.status(eid) != "pending"):
                continue
            src = sources[js[0].obs_id]
            if src["candidate"]["replay"] and loop["replay"]:
                continue
            half = e.t_interval[0][0]
            if after.get((half, tuple(src["window"]))) in done:
                out.append(eid)
        return out

    def analyze(half, window, **extra):
        """One event-analysis call on `window` of `half` (DESIGN.md §6.3), its request built from the
        frames and identities seen so far. The candidates carry the player behind each participant
        track as the model was told (after `corrupt`, in the error-injection experiment). The call
        is logged to windows.jsonl with `extra` (a re-check's entry) and its observations are
        registered. Returns the logged row; out of retries, the row has "error" instead."""
        h = halves[half]
        shown = [(s, h["images"][s]) for s in h["grid"] if window[0] <= s < window[1] and s in h["images"]]
        request = build_request(half, window, shown, h["perceived"], prematch, ev["sample_fps"], ev["ask_confidence"])
        # The player behind each track as last seen in the window — what the model was told.
        players_of = {(half, *tr.key): tr.player_id
                      for s, _ in shown for tr in h["perceived"].get(s, ((), None, ()))[0] if tr.player_id}
        try:
            result = client.generate(request.parts, json_output=True)
        except NoAPIKey:
            raise
        except Exception as e:  # noqa: BLE001 — out of retries: live, this call is simply lost
            r = {"half": half, "window": window, **extra, "error": f"{type(e).__name__}: {e}"}
        else:
            cands = [{**c.__dict__, "players": [players_of.get(key) for _, key in c.participants]}
                     for c in parse_candidates(result.text, request)]
            if corrupt is not None:
                cands = corrupt(half, window, cands)
            r = {"half": half, "window": window, "call_id": result.call_id, "model": result.model,
                 "latency": result.latency, "cached": result.cached, **extra, "candidates": cands}
            for i, c in enumerate(cands):
                sources[f"obs:{result.call_id}#{i}"] = {"window": window, "candidate": c}
        windows_log.write(json.dumps(r) + "\n")
        windows_log.flush()
        return r

    def recheck(eid, t):
        """Re-check entry `eid` at match time t (DESIGN.md §6.4): the analyzer, unchanged, on a
        window of the same length centred on its sighting; the answer takes effect at t + the
        measured latency. A window that is one of the fixed ones was analysed already: asking
        again would be the same request, so the entry is not re-checked."""
        rechecked.add(eid)
        e = mem.entries[eid]
        half = e.t_interval[0][0]
        obs_id = mem.justifications[eid][0][1].obs_id
        second, h = sources[obs_id]["candidate"]["second"], halves[half]
        length = ev["window_seconds"]
        w0 = max(0.0, min(second - length / 2, h["grid"][-1] - length))
        window = (w0, w0 + length)
        if window in h["windows"]:
            print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s: skipped, a fixed window", flush=True)
            return
        # Logged whole: only a candidate that is a sighting of this entry will join it.
        r = analyze(half, window, recheck=eid, of=obs_id)
        if "error" in r:
            print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s failed: {r['error']}", flush=True)
            return
        print(f"  re-check {eid} {e.type} at {half}H {second:.0f}s: {len(r['candidates'])} candidates, "
              f"{r['latency']:.1f}s{' cached' if r['cached'] else ''}", flush=True)
        heapq.heappush(queue, ((t[0], t[1] + r["latency"]), next(order), "recheck", (half, eid, r)))

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
        perceived: dict[float, tuple] = {}        # sampled second -> frame_view(), as of the last window it was in
        raw: dict[float, tuple] = {}              # sampled second -> (tracks, jersey, detections), kept for the windows that hold it
        reads_seen: set[str] = set()
        roster, confirmed, version = squad, {}, None
        halves[half] = {"images": images, "perceived": perceived, "grid": grid, "windows": set(windows)}
        after.update({(half, a): (half, b) for a, b in zip(windows, windows[1:])})

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
            # Identity of the window's frames from the votes of the whole window (DESIGN.md §6.1
            # identity (f)); the analyzer is only shown them now. The next window shares its first
            # `window_overlap` s with this one and resolves them again with more votes.
            for s, (tracks, jersey, det) in raw.items():
                if window[0] <= s < window[1]:
                    perceived[s] = frame_view(config, resolver.resolve(tracks, roster, confirmed), jersey, det)
            r = analyze(half, window)
            if "error" in r:
                print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s failed: {r['error']}", flush=True)
                done.add((half, tuple(window)))
                return
            print(f"{half}H {window[0]:.0f}-{window[1]:.0f}s {len(r['candidates'])} candidates, {r['latency']:.1f}s"
                  f"{' cached' if r['cached'] else ''}", flush=True)
            heapq.heappush(queue, ((half, window[1] + r["latency"]), next(order), "answer", (half, r)))
            if loop["recheck"] and loop["consistency"]:
                # A re-check (DESIGN.md §6.4) is sent once a sighting's window and the one after it
                # have answered, on frames from half a window before the sighting's window: keep
                # them back to the earliest window still waiting for either answer.
                waiting = next(a for a in windows if not {(half, a), after.get((half, a))} <= done)
                horizon = waiting[0] - ev["window_seconds"] / 2
            else:
                # The next window starts `window_overlap` s before this one ends.
                horizon = window[1] - ev["window_overlap"]
            for s in [s for s in images if s < horizon]:
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
            if frame_idx in wanted:
                raw[wanted[frame_idx]] = (tracks, jersey, det)
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
        while w < len(windows) and (until is None or (half, windows[w][1]) <= until):
            fire(windows[w])
            w += 1
        cap.release()
        if until is None or until[0] > half:
            # Everything of this half lands before the next half starts ((half, s) < (half + 1, 0)),
            # re-checks included; then its frames are no longer needed.
            advance((half, float("inf")))
            images.clear()
            del halves[half]
        with open(out_dir / f"perceived_half{half}.pkl", "wb") as f:
            pickle.dump(perceived, f)

    # After the final whistle (or at `until`): the last answers and verdicts land.
    advance((3, 0.0) if until is None else until)
    windows_log.close()
    (out_dir / "cases.json").write_text(json.dumps(records, indent=1, default=str))
    return mem


def run_commentary(config: dict[str, Any], match: Match, run_dir: Path, client,
                   only: tuple[int, float, float] | None = None, writer=None) -> list[dict]:
    """Live commentary (DESIGN.md §6.9) over a live run (run_live's output in `run_dir`).

    The broadcast output runs `output_delay` (D) s behind the live feed, and the commentary is
    written on the output's timeline, `segment` s of video at a time. The script for the stretch
    [a, b) is asked for at feed time a + D - `budget`: the feed is then past b, so every sampled
    frame of the stretch is there (labelled with who each tracked player is as far as known then;
    runtime.qa_tools), and memory as known then has D - budget s of hindsight on its start. The
    policy gives the material (what must be said, from memory), the generator writes the
    script (one model call through `client`), each phrase at a frame, and every phrase is voiced
    (one text-to-speech call each, all at once). The stretch is ready at the asking time + both
    latencies, so it can go on air from ready - D on. A stretch is asked for every `segment` s and
    takes longer than that, so a few are under way at once: what one says is known to the next
    ones (the phrases they carry on from, what the policy has said) only once it is ready, and the
    items it was given are offered to no other stretch meanwhile. Each phrase goes on air at its frame — or
    when the previous phrase has ended (+ `gap`), or once the stretch is ready, if later — and
    lasts as long as its audio; a phrase that would start more than `late` s after its frame
    (`late_event` for a must-say item) is dropped, and an item none of whose phrases went on air is
    offered again with the next stretch while it is still news.

    Commentary only reads memory, and memory as known at t is a prefix of the run's op log
    (DESIGN.md §5.4), so driving it from the log after the run gives the phrases a live run would
    have given with the same latencies. A generation that fails after its retries loses the
    stretch, as it would live; a failed synthesis leaves the phrase a subtitle only. `only` =
    (half, start, end) restricts the commentary to that stretch of video time (a trial). `writer`,
    if given, writes the scripts in place of `client` (a model of its own, e.g. served locally;
    commentary.api), which then only voices them.

    Returns one record per phrase on air: its kind (the material item it says — correction,
    confirmation, opener, news — or background when it only tells a pre-match fact, else play;
    and the fact's id), entry, text and tone (its energy), the
    entry's status and the players it may name, the stretch it belongs to, the match times it was
    asked for (t_decide), ready (t_ready), of its event (t_event, or its own frame) and of its frame
    (at), when it is on air ("show": half, start, end in video time), the material's facts, who is
    who in the stretch's frames, the audio (path, duration), and the calls' ids and latencies.
    """
    import copy
    import json
    import pickle
    from concurrent.futures import ThreadPoolExecutor

    from matchlive.app.commentary import Commentator, CommentaryPolicy, speaking_time
    from matchlive.app.tts import Voice
    from matchlive.common.api import NoAPIKey
    from matchlive.events.analyzer import clock
    from matchlive.runtime.qa_tools import QATools

    cfg = config["commentary"]
    delay, segment = cfg["output_delay"], cfg["segment"]
    mem = MatchMemory(match.prematch["players"], log_path=run_dir / "ops.jsonl", cascade=config["loop"]["cascade"])
    windows = [json.loads(line) for line in open(run_dir / "windows.jsonl")]
    sources = {f"obs:{w['call_id']}#{i}": {"window": w["window"], "candidate": c}
               for w in windows for i, c in enumerate(w.get("candidates", []))}
    half_end = {h: max(w["window"][1] for w in windows if w["half"] == h) for h in (1, 2)}
    # The frames of a stretch, labelled as the QA's current view is: every sampled frame.
    perceived = {h: pickle.load(open(run_dir / f"perceived_half{h}.pkl", "rb")) for h in (1, 2)}
    view_config = copy.deepcopy(config)
    view_config["qa"]["max_frames"] = int(segment * config["events"]["sample_fps"]) + 1
    pictures = QATools(view_config, match, mem, perceived, sources)
    policy = CommentaryPolicy(cfg, match.prematch, sources)
    writer = writer or client
    commentator = Commentator(cfg, lambda parts: writer.generate(parts, json_output=True, reasoning=cfg["reasoning"]),
                              match.prematch["players"], policy.knowledge, policy.lineups)
    voice = Voice(cfg["voice"], client.speech)

    def speak(phrase):
        """(audio, duration, latency) of one phrase; a failed synthesis leaves it a subtitle only."""
        try:
            spoken = voice.say(phrase.text, phrase.energy)
            return str(spoken.path), spoken.duration, spoken.latency
        except NoAPIKey:
            raise
        except Exception as e:  # noqa: BLE001 — out of API retries: a subtitle without a voice
            if "dry run" not in str(e):
                print(f"no voice for {phrase.text!r}: {e}", flush=True)
            return None, speaking_time(phrase.text, cfg["words_per_second"]), 0.0

    def key(item):
        """What an item is about, to tell two offers of the same thing apart."""
        return ("entry", item.entry) if item.entry is not None else (item.kind,)

    lines: list[dict] = []
    with ThreadPoolExecutor(8) as pool:
        for half in (1, 2):
            if only and only[0] != half:
                continue
            first, last = (only[1], only[2]) if only else (0.0, half_end[half])
            ends = first                       # video time the previous phrase ends (+ gap)
            # Stretches asked for and not yet ready: one is asked every `segment` s and takes
            # longer than that, so a few are under way at once. What one says (the phrases the next
            # ones carry on from, the items the policy marks as said) is known once it is ready;
            # until then the items it was given are offered to no other stretch.
            under_way: list[dict] = []
            history: list[tuple[float, str]] = []  # (video second it began, text)

            def settle(now):
                """Take in every stretch ready by feed time `now`."""
                for done in [s for s in under_way if s["ready"] <= now]:
                    under_way.remove(done)
                    history.extend(done["texts"])
                    for item, (texts, start) in done["said"]:
                        policy.said(item, " ".join(texts), (half, start))
                    for fact, start in done["told"]:
                        policy.told(fact, (half, start))

            a = first
            while a < last:
                b = min(a + segment, last)
                t = max(a + delay - cfg["budget"], b)          # feed time the script is asked for
                settle(t)
                # Every sampled frame of [a, b) (the frame at b opens the next stretch).
                frames = pictures.call((half, t), "get_frames", {"half": half, "start": a, "end": b - 0.01})
                if not frames.images:          # nothing to look at
                    a = b
                    continue
                view = mem.at((half, t))
                taken = {key(item) for s in under_way for item in s["items"]}
                items = [item for item in policy.material(view, (half, a, b)) if key(item) not in taken]
                try:
                    before = [(s["span"][0], s["span"][1], s["pictures"], s["items"])
                              for s in sorted(under_way, key=lambda s: s["span"])]
                    # Facts already told, and those a stretch under way tells: none goes twice.
                    told = {**{f: (half, start) for s in under_way for f, start in s["told"]}, **policy.facts_told}
                    script, latency, call_id = commentator.script(items, frames, policy.where(view, (half, a, b)),
                                                                  history, before, told)
                except NoAPIKey:
                    raise
                except Exception as e:  # noqa: BLE001 — out of API retries: this stretch is lost
                    print(f"{half}H {clock(a)} stretch lost: {e}", flush=True)
                    a = b
                    continue
                spoken = list(pool.map(speak, script))
                tts_latency = max((s[2] for s in spoken), default=0.0)
                ready = t + latency + tts_latency
                said: dict[int, tuple[list[str], float]] = {}  # item -> (its phrases on air, when the first started)
                texts, told = [], []
                for p, (audio, duration, _) in zip(script, spoken):
                    start = max(p.at, ends, ready - delay)
                    if start - p.at > cfg["late_event" if p.item is not None else "late"]:
                        print(f"{half}H {clock(p.at)} [{p.item.kind if p.item else 'play'}] too late by "
                              f"{start - p.at:.1f} s: dropped {p.text!r}", flush=True)
                        continue
                    item = p.item
                    if item is not None:
                        said.setdefault(id(item), ([], start))[0].append(p.text)
                    texts.append((start, p.text))
                    if p.fact:
                        told.append((p.fact, start))
                    lines.append({"kind": item.kind if item else "background" if p.fact else "play",
                                  "fact": p.fact, "entry": item.entry if item else None,
                                  "status": item.status if item else None, "named": item.named if item else {},
                                  "text": p.text, "tone": p.energy, "segment": (half, a, b),
                                  "t_decide": (half, t), "t_ready": (half, ready),
                                  "t_event": item.t_event if item and item.t_event else (half, p.at),
                                  "at": (half, p.at), "show": (half, start, start + duration),
                                  "facts": item.facts if item else [policy.knowledge[p.fact]] if p.fact else [],
                                  "pictures": frames.text,
                                  "call_id": call_id, "latency": latency, "audio": audio, "duration": duration,
                                  "tts_latency": tts_latency})
                    print(f"{half}H {clock(start)} [{lines[-1]['kind']}/{p.energy}] {p.text}", flush=True)
                    ends = start + duration + cfg["gap"]
                under_way.append({"ready": ready, "items": items, "texts": texts, "told": told, "span": (a, b),
                                  "pictures": frames.text,
                                  "said": [(item, said[id(item)]) for item in items if id(item) in said]})
                a = b
            settle(float("inf"))
    return lines
