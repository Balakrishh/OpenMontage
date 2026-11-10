"""Quick Science planner: a 60-120 s experiment-first episode, written entirely by code.

No model is involved — the narration is the experiment library's vetted text
and the picture list is fixed by the library's steps (see lib/kids_science/quick.py).
Writes the same artifacts as kids_story_writer, so the image, narration,
assembly and final-check stages run unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)


class QuickEpisodePlanner(BaseTool):
    name = "quick_episode_planner"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "script_generation"
    provider = "local"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    install_instructions = "No dependencies — reads lib/kids_science."
    agent_skills: list[str] = []
    capabilities = ["script_generation", "shot_planning"]
    supports = {"offline": True, "no_model": True}
    best_for = ["60-120 s experiment-first episodes and Shorts with zero narrative drift"]
    not_good_for = ["character stories (use kids_story_writer)"]

    input_schema = {
        "type": "object",
        "required": ["experiment_id", "project_dir"],
        "properties": {"experiment_id": {"type": "string"}, "project_dir": {"type": "string"}},
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=1, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["experiment_id", "project_dir"]
    side_effects = ["writes artifacts/brief.json, story_bible.json, script.json, scene_plan.json"]
    user_visible_verification = ["Read artifacts/script.json — every line is the library's own text"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 1.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from lib import kids_science as ks
        from lib.kids_science import quick

        start = time.time()
        try:
            exp = ks.experiment(inputs["experiment_id"])
        except KeyError as exc:
            return ToolResult(success=False, error=str(exc))
        missing = [k for k in ["materials", "observe", *map(str, range(1, len(exp["steps"]) + 1))]
                   if k not in (exp.get("states") or {})]
        if missing:
            return ToolResult(success=False, error=f"states.yaml lacks {missing} for {exp['id']}")
        beats, shots, lines = quick.plan(exp)
        art = Path(inputs["project_dir"]) / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        wpm = ks.series()["series"]["words_per_minute"]
        t, sections, scenes = 0.0, [], []
        for b in beats:
            text = " ".join(l["text"] for l in lines[b["id"]])
            secs = max(2.5, ks.word_count(text) * 60 / wpm)
            sections.append({"id": b["id"], "label": b["title"], "text": text,
                             "start_seconds": round(t, 2), "end_seconds": round(t + secs, 2)})
            beat_shots = [s for s in shots if s["beat"] == b["id"]]
            for i, sh in enumerate(beat_shots):
                span = secs / len(beat_shots)
                scenes.append({"id": sh["id"], "type": "generated", "description": sh.get("description") or sh["kind"],
                               "start_seconds": round(t + i * span, 2), "end_seconds": round(t + (i + 1) * span, 2),
                               "script_section_id": b["id"]})
            t += secs
        title = exp["title"]
        (art / "brief.json").write_text(json.dumps({
            "version": "1.0", "title": title, "hook": exp["question"], "key_points": [exp["observe"], *exp["facts"][:2]],
            "tone": "bright, quick, curious", "style": "experiment-first close-ups on a clean backdrop, time-lapse result",
            "target_platform": "youtube", "target_duration_seconds": round(t),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "story_bible.json").write_text(json.dumps({
            "experiment_id": exp["id"], "episode_title": title, "format": "quick",
            "beats": {b["id"]: {"summary": b["title"]} for b in beats},
            "assigned_steps": {f"step{i}": [s] for i, s in enumerate(exp["steps"], 1)},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "script.json").write_text(json.dumps({
            "version": "1.0", "title": title, "total_duration_seconds": round(t, 1), "sections": sections,
            "metadata": {"lines": lines, "beats": beats, "format": "quick", "experiment_id": exp["id"],
                         "captions": False},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "scene_plan.json").write_text(json.dumps({
            "version": "1.0", "scenes": scenes,
            # The boys' hello and goodbye get short generated clips; the experiment
            # moves by code (quick_motion), never by generation.
            "metadata": {"shots": shots, "motion": ["hello-1", "bye-1"], "format": "quick", "captions": False},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return ToolResult(
            success=True,
            data={"episode_title": title, "beats": len(beats), "shots": len(shots),
                  "plates": sum(s["kind"] == "plate" for s in shots), "estimated_seconds": round(t)},
            duration_seconds=round(time.time() - start, 2),
        )
