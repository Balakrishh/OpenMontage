#!/usr/bin/env python3
"""Upload every not-yet-published episode of a Backlot series to YouTube.

Goes through the running Backlot server's publish endpoint — the same path
as the board's PUBLISH button — using each episode's own suggested copy,
thumbnail, category and made-for-kids flag. Resumable: episodes already on
YouTube are skipped, and the run stops cleanly when the daily API quota
(~6 uploads/day) runs out, so just run it again the next day.

    python scripts/publish_series.py "Quick Science with Arjun & Dev" --skip kids-drawn-walking-water
    python scripts/publish_series.py "..." --dry-run      # build payloads only
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOARD = "http://127.0.0.1:4750"


def call(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(BOARD + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "x-backlot-publish": "1"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.code}: {e.read().decode()[:300]}"}


def episodes(series: str) -> list[Path]:
    out = []
    for pj in sorted((REPO / "projects").glob("*/project.json")):
        try:
            if json.loads(pj.read_text()).get("series") == series:
                out.append(pj.parent)
        except (OSError, ValueError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--privacy", default=None, help="public | unlisted | private (default: the board's default)")
    ap.add_argument("--max", type=int, default=99)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    done = uploaded = 0
    for proj in episodes(a.series):
        pid = proj.name
        if pid in a.skip:
            print(f"skip       {pid}")
            continue
        video = next(iter(sorted((proj / "export").glob("*.mp4"))), None)
        if not video:
            print(f"no video   {pid}")
            continue
        rel = video.relative_to(proj).as_posix()
        pre = call("GET", f"/api/project/{pid}/publish/preflight?path={urllib.parse.quote(rel)}")
        if pre.get("already_published"):
            print(f"on youtube {pid}: {pre['already_published']}")
            done += 1
            continue
        if not pre.get("credentials_ready"):
            sys.exit(f"YouTube credentials missing: {pre.get('missing_env')}")
        s = pre["suggested"]
        body = {
            "confirm": True, "path": rel, "title": s["title"], "description": s["description"],
            "tags": ", ".join(s.get("tags") or []), "chapters": s.get("chapters") or [],
            "thumbnail_path": s.get("thumbnail_path"), "contains_synthetic_media": s.get("contains_synthetic_media") is not False,
            "made_for_kids": bool(s.get("made_for_kids")), "category_id": s.get("category_id") or "27",
            "privacy": a.privacy or pre["defaults"]["privacy"], "short": "auto", "dry_run": a.dry_run, "force": False,
        }
        res = call("POST", f"/api/project/{pid}/publish", body)
        ok = res.get("success", res.get("status") == "published")
        text = json.dumps(res)[:400]
        if not ok and any(k in text.lower() for k in ("quota", "uploadlimitexceeded", "ratelimit", "429")):
            print(f"QUOTA      {pid}: daily YouTube limit reached — run again after midnight Pacific")
            break
        print(f"{'uploaded ' if ok else 'FAILED   '}  {pid}: {res.get('url') or res.get('video_id') or text}", flush=True)
        if ok and not a.dry_run:
            uploaded += 1
            if uploaded >= a.max:
                break
    print(f"\n{uploaded} uploaded this run; {done} were already on YouTube.")


if __name__ == "__main__":
    main()
