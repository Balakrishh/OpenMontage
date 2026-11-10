#!/usr/bin/env python3
"""Render a code-drawn Quick Science episode for one experiment (or all of them).

    python scripts/kids_science_drawn.py fizzy-volcano            # full episode
    python scripts/kids_science_drawn.py fizzy-volcano --preview  # contact sheet + thumbnail only
    python scripts/kids_science_drawn.py --all                    # every experiment in scenes.yaml

Output: projects/kids-drawn-<id>/export/<Title>.mp4 (+ _THUMBNAIL.png, metadata.json).
Export only — nothing is uploaded.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.kids_science.stage import Episode, scenes  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    times = []
    if "--preview" in argv and argv.index("--preview") + 1 < len(argv) and argv[argv.index("--preview") + 1][:1].isdigit():
        times = [float(x) for x in argv.pop(argv.index("--preview") + 1).split(",")]
    args = [a for a in argv if not a.startswith("--")]
    ids = list(scenes()) if "--all" in sys.argv else args
    for exp_id in ids:
        t = time.time()
        ep = Episode(exp_id, REPO / "projects" / f"kids-drawn-{exp_id}")
        out = ep.render(preview=times if "--preview" in sys.argv else None)
        print(f"{exp_id}: {out} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
