"""Pre-match background for the commentary (DESIGN.md §6.8): kb.json in each match folder.

    python scripts/build_kb.py [--match M ...]

Two sources, both frozen before kick-off (the metadata's `timestamp`, UTC):
  - the league's results (football-data.co.uk, this season and the last): each side's place,
    points and last results, and the last league meeting of the two, from the games played
    before the match day only — computed here, no model;
  - Wikipedia: each club's season article and each player's article as they stood just before
    kick-off (the last revision older than it). A model picks the facts a commentator could use,
    quoting the words each rests on; a fact is kept only if its quote is in the article and
    every number in it is in the quote.
Every fact carries its source (page, revision id and time), and the build fails if a revision
is not older than kick-off. Downloads and model calls are cached (paths.cache/kb, the API cache).

Output: {"kickoff", "sources": {id: {...}}, "teams": {side: {"manager", "captain", "facts"}},
"players": {hash: {"facts"}}}, each fact {"text", "source", "quote"}. Match.prematch reads it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import parse_json

MATCHES = ("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool")
AGENT = "MatchLive-research/0.1 (pre-match knowledge base; one request at a time)"
# Club names as football-data.co.uk writes them, where they differ from the metadata's.
RESULTS_NAMES = {"Manchester United": "Man United", "Manchester City": "Man City",
                 "Newcastle United": "Newcastle", "West Bromwich Albion": "West Brom",
                 "Tottenham Hotspur": "Tottenham", "Wolverhampton Wanderers": "Wolves"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()
    config = load_config()
    cache = config["paths"]["cache"] / "kb"
    cache.mkdir(parents=True, exist_ok=True)
    client = APIClient(config["api"], config["paths"]["cache"])
    for name in args.match or MATCHES:
        match = Match(config, name)
        meta = match.metadata
        kickoff = datetime.fromtimestamp(int(meta["timestamp"]), tz=timezone.utc)
        teams = meta["teams"]
        kb = {"kickoff": kickoff.isoformat(), "sources": {}, "teams": {}, "players": {}}

        # 1. The table and form, from the results of the games before the match day.
        season = meta["season"]                                   # "2016-2017"
        code = season[2:4] + season[7:9]                          # "1617"
        last_code = f"{int(code[:2]) - 1:02d}{code[:2]}"          # "1516"
        games = [g for c in (last_code, code) for g in results(cache, c)]
        before = [g for g in games if g["date"] < kickoff.date()]
        this_season = [g for g in before if g["season"] == code]
        kb["sources"]["results"] = {"url": f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv",
                                    "games_before": kickoff.date().isoformat(), "games": len(this_season)}
        for side in ("home", "away"):
            kb["teams"][side] = {"manager": None, "captain": None,
                                 "facts": standing_facts(teams[side], this_season, before)}
        meeting = last_meeting(teams["home"], teams["away"], before)
        if meeting:
            kb["teams"]["home"]["facts"].append(meeting)

        # 2. Each club's season article: manager, captain and a few facts.
        years = f"{season[:4]}–{season[7:9]}"                     # "2016–17"
        for side in ("home", "away"):
            title = f"{years} {teams[side]} F.C. season"
            page = revision(cache, title, kickoff)
            if page is None:
                print(f"no article {title!r} before kick-off")
                continue
            source = add_source(kb, page)
            answer = ask(client, CLUB_PROMPT.format(team=teams[side], date=page["timestamp"][:10],
                                                    match=f"{teams['home']} v {teams['away']}",
                                                    day=kickoff.date().isoformat()), page["text"])
            text = normalize(page["text"])
            for role in ("manager", "captain"):
                who = answer.get(role) or {}
                if who.get("name") and quoted(who.get("quote", ""), text) and _words(who["name"]) <= _words(who["quote"]):
                    kb["teams"][side][role] = {"text": who["name"], "source": source, "quote": who["quote"]}
            kb["teams"][side]["facts"] += checked(answer.get("facts", []), text, source)

        # 3. Each player's article: up to three facts.
        for p in meta["players"]:
            team = teams[p["team_side"]]
            page = player_page(cache, p["long_name"], team, kickoff)
            if page is None:
                print(f"  {p['long_name']}: no article found")
                continue
            source = add_source(kb, page)
            excerpt = player_excerpt(page["text"], team)
            answer = ask(client, PLAYER_PROMPT.format(name=p["long_name"], team=team, date=page["timestamp"][:10],
                                                      match=f"{teams['home']} v {teams['away']}",
                                                      day=kickoff.date().isoformat()), excerpt)
            facts = checked(answer.get("facts", []), normalize(page["text"]), source)[:3]
            kb["players"][p["hash"]] = {"facts": facts}
            print(f"  {p['short_name']}: {len(facts)} facts from {page['title']!r} ({page['timestamp'][:10]})")

        # Nothing from after kick-off.
        late = [s for s in kb["sources"].values() if "timestamp" in s and s["timestamp"] >= kickoff.isoformat()]
        if late:
            raise RuntimeError(f"revisions not older than kick-off: {late}")
        out = match.dir / "kb.json"
        out.write_text(json.dumps(kb, indent=1, ensure_ascii=False))
        n = sum(len(v["facts"]) for v in kb["players"].values())
        print(f"{name}: {out} — {len(kb['players'])} players with {n} facts; "
              f"{sum(len(t['facts']) for t in kb['teams'].values())} team facts")


# ---- the league's results ----

def results(cache: Path, code: str) -> list[dict]:
    """The Premier League games of one season: [{"season", "date", "home", "away", "hg", "ag"}]."""
    text = fetch(cache, f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv")
    games = []
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        if not row.get("HomeTeam") or not row.get("FTHG"):
            continue
        day = row["Date"]
        date = datetime.strptime(day, "%d/%m/%y" if len(day) == 8 else "%d/%m/%Y").date()
        games.append({"season": code, "date": date, "home": row["HomeTeam"], "away": row["AwayTeam"],
                      "hg": int(row["FTHG"]), "ag": int(row["FTAG"])})
    return games


def standing_facts(team: str, season: list[dict], before: list[dict]) -> list[dict]:
    """The side's place, points and last results going into the match."""
    name = RESULTS_NAMES.get(team, team)
    if not season:
        return [{"text": f"This is {team}'s first league game of the season.", "source": "results", "quote": ""}]
    table = {}
    for g in season:
        for side, other, gf, ga in ((g["home"], g["away"], g["hg"], g["ag"]), (g["away"], g["home"], g["ag"], g["hg"])):
            row = table.setdefault(side, [0, 0, 0, 0])            # played, points, goals for, goals against
            row[0] += 1
            row[1] += 3 if gf > ga else 1 if gf == ga else 0
            row[2] += gf
            row[3] += ga
    order = sorted(table, key=lambda t: (-table[t][1], -(table[t][2] - table[t][3]), -table[t][2], t))
    played, points, gf, ga = table[name]
    place = order.index(name) + 1
    facts = [{"text": f"Going into this game {team} are {_ordinal(place)} in the Premier League with {points} "
                      f"points from {played} games (scored {gf}, conceded {ga}).", "source": "results", "quote": ""}]
    last = [g for g in season if name in (g["home"], g["away"])][-3:]
    if last:
        said = []
        for g in last:
            home = g["home"] == name
            gf, ga = (g["hg"], g["ag"]) if home else (g["ag"], g["hg"])
            other = g["away"] if home else g["home"]
            result = "won" if gf > ga else "drew" if gf == ga else "lost"
            said.append(f"{result} {gf}-{ga} {'at home to' if home else 'away at'} {other}")
        facts.append({"text": f"{team}'s last league results, most recent last: {'; '.join(said)}.",
                      "source": "results", "quote": ""})
    return facts


def last_meeting(home: str, away: str, before: list[dict]) -> dict | None:
    names = {RESULTS_NAMES.get(home, home), RESULTS_NAMES.get(away, away)}
    met = [g for g in before if {g["home"], g["away"]} == names]
    if not met:
        return None
    g = met[-1]
    return {"text": f"Last league meeting: {g['home']} {g['hg']}-{g['ag']} {g['away']}, "
                    f"{g['date'].strftime('%B %Y')}.", "source": "results", "quote": ""}


# ---- Wikipedia as of kick-off ----

def revision(cache: Path, title: str, kickoff: datetime) -> dict | None:
    """The article's last revision older than kick-off: {"title", "revid", "timestamp", "text"}."""
    params = {"action": "query", "prop": "revisions", "titles": title, "redirects": 1, "rvlimit": 1,
              "rvstart": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "rvdir": "older",
              "rvprop": "ids|timestamp|content", "rvslots": "main", "format": "json", "formatversion": 2}
    data = json.loads(fetch(cache, "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    pages = data.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0]:
        return None
    r = pages[0]["revisions"][0]
    # rvstart is inclusive: a revision saved at the kick-off second itself is not before it.
    if r["timestamp"] >= kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"):
        return None
    return {"title": pages[0]["title"], "revid": r["revid"], "timestamp": r["timestamp"],
            "text": r["slots"]["main"]["content"]}


def player_page(cache: Path, name: str, team: str, kickoff: datetime) -> dict | None:
    """The player's article as of kick-off: the first search hit that is a footballer's article,
    whose title has one of the player's names and whose text of then names the club."""
    params = {"action": "query", "list": "search", "srsearch": f"{name} footballer", "srlimit": 5,
              "format": "json", "formatversion": 2}
    hits = json.loads(fetch(cache, "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))
    names = [_plain_letters(w) for w in name.split() if len(w) >= 4]
    for hit in hits.get("query", {}).get("search", []):
        if not any(n in _plain_letters(hit["title"]) for n in names):
            continue
        page = revision(cache, hit["title"], kickoff)
        if page and team.split()[0] in page["text"] and re.search(r"\{\{\s*Infobox football biography", page["text"], re.I):
            return page
    return None


def player_excerpt(wikitext: str, team: str, limit: int = 14000) -> str:
    """The infobox and lead, and the latest sections about the club (the recent news)."""
    sections = re.split(r"(?m)^(==+[^=\n]+==+)\s*$", wikitext)
    lead, rest = sections[0], list(zip(sections[1::2], sections[2::2]))
    about = [h + b for h, b in rest if team.split()[0] in h]
    return (strip_refs(lead)[:limit // 2] + "\n" + strip_refs("".join(about))[-(limit // 2):]).strip()


def add_source(kb: dict, page: dict) -> str:
    key = f"wiki:{page['title']}"
    kb["sources"][key] = {"title": page["title"], "revid": page["revid"], "timestamp": page["timestamp"],
                          "url": f"https://en.wikipedia.org/w/index.php?oldid={page['revid']}"}
    return key


# ---- the model's picks, checked against the text ----

CLUB_PROMPT = """Below is the Wikipedia article on {team}'s season as it stood on {date}, before the match {match} on
{day}. Using only this text:
- who is {team}'s manager (head coach), and who is the club captain (if the text says);
- up to three facts a TV commentator could use about {team} before this match: new signings or departures of
  the summer or winter, how the season has gone so far, a milestone. Nothing that is not in the text.
For each, quote the exact words of the text it rests on (a short span, copied character for character).
Answer with JSON only: {{"manager": {{"name": "...", "quote": "..."}} | null, "captain": {{"name": "...",
"quote": "..."}} | null, "facts": [{{"text": "one short sentence", "quote": "..."}}]}}"""

PLAYER_PROMPT = """Below is (part of) the Wikipedia article on {name} of {team}, as it stood on {date}, before the match
{match} on {day}. Using only this text, give up to three facts a TV commentator could use about him during that
match: when and from where he came to {team} (and for how much, if stated), his international career, his honours,
a notable record, his season so far. Short spoken sentences, as of the match day. Nothing that is not in the text.
For each fact, quote the exact words of the text it rests on (a short span, copied character for character).
Answer with JSON only: {{"facts": [{{"text": "one short sentence", "quote": "..."}}]}}"""


def ask(client: APIClient, prompt: str, text: str) -> dict:
    result = client.generate([prompt, strip_refs(text)[:30000]], json_output=True)
    answer = parse_json(result.text)
    return answer if isinstance(answer, dict) else {}


def checked(facts, text: str, source: str) -> list[dict]:
    """The facts whose quote is in the article's text and whose numbers are all in the quote."""
    out = []
    for f in facts if isinstance(facts, list) else []:
        if not isinstance(f, dict) or not f.get("text") or not f.get("quote"):
            continue
        if quoted(f["quote"], text) and set(re.findall(r"\d+", f["text"])) <= set(re.findall(r"\d+", f["quote"])):
            out.append({"text": f["text"].strip(), "source": source, "quote": f["quote"].strip()})
    return out


def quoted(quote: str, normalized_text: str) -> bool:
    q = normalize(quote)
    return len(q) >= 12 and q in normalized_text


def normalize(text: str) -> str:
    """Text for matching a quote: links and markup reduced to their words, lower case, letters and
    digits only."""
    text = strip_refs(text)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)          # [[target|label]] -> label
    text = re.sub(r"'{2,}", "", text)
    return " ".join(re.findall(r"[^\W_]+", text.lower()))


def strip_refs(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    return re.sub(r"\{\{\s*(?:cite|sfn|efn)[^{}]*\}\}", "", text, flags=re.I)


def _words(s: str) -> set[str]:
    return set(normalize(s).split())


def _plain_letters(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if c.isalpha())


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- downloads, cached ----

def fetch(cache: Path, url: str) -> str:
    path = cache / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".txt")
    if path.exists():
        return path.read_text()
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                text = r.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            # Rate-limited: wait as told (or back off), then try again.
            time.sleep(float(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt)
    path.write_text(text)
    time.sleep(1.0)                          # one request at a time, politely
    return text


if __name__ == "__main__":
    main()
