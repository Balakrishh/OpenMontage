#!/usr/bin/env python3
"""Run one kids-science-story episode end to end, unattended.

    .venv/bin/python scripts/kids_science_episode.py walking-water
    .venv/bin/python scripts/kids_science_episode.py --list
    .venv/bin/python scripts/kids_science_episode.py walking-water --project-id ep01-walking-water --resume

Why a runner, when OpenMontage is normally agent-driven: this pipeline is
built to need no decisions mid-run (pipeline_defs/kids-science-story.yaml has
no approval gates), so walking it stage by stage through an agent would spend
tokens re-reading context to make no choices. The runner calls the same
registered stage tools an agent would, writes the same checkpoints (so the
Backlot board shows the run live), and stops at the first failed check with
the reason in projects/<id>/run_report.json. --resume skips stages whose
checkpoint is already completed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib import kids_science as ks  # noqa: E402
from lib.checkpoint import init_project, read_checkpoint, write_checkpoint  # noqa: E402
from tools.tool_registry import registry  # noqa: E402

PIPELINE = "kids-science-story"
PROJECTS = REPO / "projects"


def _art(project: Path, name: str) -> dict:
    return json.loads((project / "artifacts" / f"{name}.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="experiment id from lib/kids_science/experiments.yaml")
    parser.add_argument("--project-id")
    parser.add_argument("--resume", action="store_true", help="skip stages already completed")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    args = parser.parse_args()

    if args.list or not args.experiment:
        for eid, e in ks.experiments().items():
            print(f"{eid:22} {e['title']}")
        return 0

    exp = ks.experiment(args.experiment)
    project_id = args.project_id or f"kids-{exp['id']}"
    project = PROJECTS / project_id
    if not (project / "project.json").exists():
        init_project(project_id, title=exp["title"], pipeline_type=PIPELINE)
    registry.discover()  # also loads .env
    report: dict = {"experiment": exp["id"], "project_id": project_id, "started": datetime.now(timezone.utc).isoformat(), "stages": []}

    def done(stage: str) -> bool:
        if not args.resume:
            return False
        try:
            cp = read_checkpoint(PROJECTS, project_id, stage)
        except Exception:
            return False
        return bool(cp and cp.get("status") == "completed")

    def checkpoint(stage: str, artifact: str) -> None:
        write_checkpoint(PROJECTS, project_id, stage, "completed", {artifact: _art(project, artifact)},
                         pipeline_type=PIPELINE, checkpoint_policy="auto_noncreative")

    def run(tool_name: str, stage: str, **inputs) -> dict:
        # Per-tool resume: a stage like "assets" runs several long tools; a run
        # stopped in the middle of one must not redo the ones that finished.
        marker = project / "artifacts" / f".done-{tool_name}.json"
        if args.resume and marker.is_file():
            print(f"    {tool_name}: already done, skipping (--resume)", flush=True)
            return json.loads(marker.read_text())
        t0 = time.time()
        print(f"[{datetime.now():%H:%M:%S}] {stage}: {tool_name} …", flush=True)
        write_checkpoint(PROJECTS, project_id, stage, "in_progress", {}, pipeline_type=PIPELINE,
                         checkpoint_policy="auto_noncreative")
        result = registry.get(tool_name).execute({"project_dir": str(project), **inputs})
        entry = {"stage": stage, "tool": tool_name, "ok": result.success, "seconds": round(time.time() - t0),
                 "data": {k: v for k, v in (result.data or {}).items() if k != "log"}, "error": result.error}
        report["stages"].append(entry)
        print(f"    {'ok' if result.success else 'FAILED'} in {entry['seconds']}s — "
              f"{result.error or json.dumps(entry['data'])[:300]}", flush=True)
        if not result.success:
            write_checkpoint(PROJECTS, project_id, stage, "failed", {}, pipeline_type=PIPELINE,
                             checkpoint_policy="auto_noncreative", error=result.error)
            raise SystemExit(_finish(project, report, ok=False))
        marker.write_text(json.dumps(entry["data"], default=str))
        return result.data or {}

    runtime = ks.series().get("render_runtime")
    if runtime not in {"remotion", "hyperframes"}:
        print("render_runtime is not set in lib/kids_science/series.yaml — choose remotion or hyperframes "
              "once for the series (see skills/pipelines/kids-science-story/director.md). "
              "Stages before edit will still run.", flush=True)
    if not (done("idea") and done("script") and done("scene_plan")):
        run("kids_story_writer", "script", experiment_id=exp["id"])
        checkpoint("idea", "brief")
        checkpoint("script", "script")
        checkpoint("scene_plan", "scene_plan")
    if not done("assets"):
        run("story_keyframes", "assets")
        run("story_motion", "assets")
        run("story_narration", "assets")
        checkpoint("assets", "asset_manifest")
    if runtime not in {"remotion", "hyperframes"}:
        print("\nStopped before edit: set render_runtime in lib/kids_science/series.yaml, then re-run with --resume.")
        return _finish(project, report, ok=False)
    if not done("edit"):
        run("story_assemble", "edit")
        s = ks.series()
        # The series-level runtime choice, logged on every episode's audit trail.
        write_checkpoint(PROJECTS, project_id, "edit", "completed", {
            "edit_decisions": _art(project, "edit_decisions"),
            "decision_log": {"version": "1.0", "project_id": project_id, "decisions": [{
                "decision_id": f"{project_id}-render-runtime", "stage": "edit",
                "category": "render_runtime_selection", "subject": "Series composition runtime",
                "options_considered": [
                    {"option_id": o, "label": o.title(), "score": 1.0 if o == s["render_runtime"] else 0.5,
                     "reason": "selected for the series" if o == s["render_runtime"]
                               else "offered; needs a new story_assemble render path before use"}
                    for o in s.get("render_runtime_options_considered", [])
                ],
                "selected": s["render_runtime"], "reason": " ".join(str(s.get("render_runtime_reason", "")).split()),
                "user_visible": True, "user_approved": True,
            }]},
        }, pipeline_type=PIPELINE, checkpoint_policy="auto_noncreative")
    if not (done("compose") and done("publish")):
        run("story_finalize", "compose")
        checkpoint("compose", "render_report")
        checkpoint("publish", "publish_log")
    return _finish(project, report, ok=True)


def _finish(project: Path, report: dict, ok: bool) -> int:
    report["finished"] = datetime.now(timezone.utc).isoformat()
    report["ok"] = ok
    report["total_minutes"] = round(sum(s["seconds"] for s in report["stages"]) / 60, 1)
    (project / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'DONE' if ok else 'STOPPED'} in {report['total_minutes']} min — {project / 'run_report.json'}")
    if ok:
        print(f"export: {project / 'export'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
