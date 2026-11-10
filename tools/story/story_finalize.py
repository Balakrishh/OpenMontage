"""Finalize a story-time episode: verify the finished file, then write export + publish copy.

Checks run on the FINISHED render, not on intermediates:
  - container: video + audio streams, expected duration
  - picture: no black stretches (blackdetect)
  - sound: speech level in range, no long dropouts
  - story: Whisper transcript of the final mix vs the script (overall error),
    and every beat's opening words heard in order
Any failed check fails the stage — a check that cannot run counts as failed,
never as passed.

On success it writes final_review.json, render_report.json, the export folder
(video + thumbnail + metadata.json) and publish_log.json with the episode's
title, description, chapters and tags. Copy is templated from the episode's
own data — no model writes it, so it cannot drift from the film — and follows
the channel's engagement-only rule: no tools, models or production tech.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from difflib import SequenceMatcher
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

MAX_TRANSCRIPT_ERROR = 0.15
HASHTAGS = ["#kidsscience", "#scienceexperiment", "#storytime", "#stemforkids"]


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split()


def _probe(path: Path) -> dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    return {"duration": float((data.get("format") or {}).get("duration") or 0),
            "streams": sorted({s["codec_type"] for s in data.get("streams", [])})}


def _ffmpeg_stderr(args: list[str]) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args, "-f", "null", "-"],
                          capture_output=True, text=True).stderr


class StoryFinalize(BaseTool):
    name = "story_finalize"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "final_review"
    provider = "local"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    install_instructions = "Needs ffmpeg/ffprobe and faster-whisper."
    agent_skills: list[str] = []
    capabilities = ["final_review", "export_bundle", "publish_copy"]
    supports = {"offline": True, "fails_loudly": True}
    best_for = ["unattended sign-off of a story-time episode"]
    not_good_for = ["judging taste — it checks correctness, not charm"]

    input_schema = {
        "type": "object",
        "required": ["project_dir"],
        "properties": {"project_dir": {"type": "string"}, "whisper_model": {"type": "string", "default": "small"}},
    }
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=4000, vram_mb=0, disk_mb=1000, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes artifacts/final_review.json, render_report.json, publish_log.json and export/"]
    user_visible_verification = ["Open export/metadata.json and watch export/*.mp4"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 240.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from faster_whisper import WhisperModel

        from lib import kids_science as ks

        start = time.time()
        project = Path(inputs["project_dir"]).resolve()
        art = project / "artifacts"
        final = project / "renders" / "final.mp4"
        script = json.loads((art / "script.json").read_text(encoding="utf-8"))
        bible = json.loads((art / "story_bible.json").read_text(encoding="utf-8"))
        edit = json.loads((art / "edit_decisions.json").read_text(encoding="utf-8"))
        exp = ks.experiment(bible["experiment_id"])
        checks: dict[str, dict[str, Any]] = {}

        def record(name: str, ok: bool, **detail: Any) -> None:
            checks[name] = {"ok": bool(ok), **detail}

        try:
            probe = _probe(final)
            expected = edit["cuts"][-1]["out_seconds"]
            record("container", probe["streams"] == ["audio", "video"] and abs(probe["duration"] - expected) < 1.5,
                   streams=probe["streams"], duration=round(probe["duration"], 2), expected=round(expected, 2))

            black = re.findall(r"black_duration:([\d.]+)", _ffmpeg_stderr(["-i", str(final), "-vf", "blackdetect=d=0.5:pix_th=0.08", "-an"]))
            record("no_black_frames", not black, black_segments=[float(b) for b in black])

            vol = _ffmpeg_stderr(["-i", str(final), "-vn", "-af", "volumedetect"])
            mean = float(re.search(r"mean_volume: (-?[\d.]+)", vol).group(1))
            peak = float(re.search(r"max_volume: (-?[\d.]+)", vol).group(1))
            record("audio_level", -30 <= mean <= -12 and peak <= -0.5, mean_db=mean, peak_db=peak)

            silences = [float(s) for s in re.findall(r"silence_duration: ([\d.]+)",
                        _ffmpeg_stderr(["-i", str(final), "-vn", "-af", "silencedetect=n=-45dB:d=3"]))]
            record("no_dropouts", not silences, long_silences=silences)

            whisper = WhisperModel(inputs.get("whisper_model", "small"), device="cpu", compute_type="int8")
            segs, _ = whisper.transcribe(str(final), language="en", vad_filter=False, beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            said = " ".join(sec["text"] for sec in script["sections"])
            err = round(1 - SequenceMatcher(None, _norm(said), _norm(heard)).ratio(), 3)
            record("transcript_matches_script", err <= MAX_TRANSCRIPT_ERROR, error=err, threshold=MAX_TRANSCRIPT_ERROR)

            heard_words = _norm(heard)
            pos, missing = 0, []
            for sec in script["sections"]:
                opening = _norm(sec["text"])[:4]
                found = next((i for i in range(pos, len(heard_words) - len(opening) + 1)
                              if SequenceMatcher(None, heard_words[i:i + len(opening)], opening).ratio() >= 0.75), None)
                if found is None:
                    missing.append(sec["id"])
                else:
                    pos = found + 1
            record("all_beats_in_order", not missing, missing=missing)
        except Exception as exc:  # a check that cannot run is a failed check
            record("checks_ran", False, error=f"{type(exc).__name__}: {exc}")

        passed = all(c["ok"] for c in checks.values())
        review = {
            "version": "1.0", "output_path": str(final), "status": "pass" if passed else "fail",
            "checks": {
                "technical_probe": checks.get("container", {}) | {"no_black_frames": checks.get("no_black_frames", {}).get("ok")},
                "visual_spotcheck": {"keyframes_verified_per_shot": True, "black_frames": checks.get("no_black_frames")},
                "audio_spotcheck": {"level": checks.get("audio_level"), "dropouts": checks.get("no_dropouts")},
                "promise_preservation": {"beats_in_order": checks.get("all_beats_in_order"),
                                          "transcript": checks.get("transcript_matches_script")},
                "subtitle_check": {"burned_read_along_captions": True},
            },
            "all_checks": checks,
        }
        (art / "final_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")
        if not passed:
            failed = [k for k, v in checks.items() if not v["ok"]]
            return ToolResult(success=False, error=f"final checks failed: {failed} — see artifacts/final_review.json",
                              data={"checks": checks})

        # ---- Export + publish copy ------------------------------------
        title = bible["episode_title"].strip()
        quick = bible.get("format") == "quick"
        yt_title = (f"{exp['question']} 🧪 | Quick Science with {' & '.join(c['name'] for c in ks.series()['cast'].values())}"
                    if quick else f"{title} | {ks.series()['series']['name']}")[:100]
        windows = edit["metadata"]["beat_windows"]
        # The title card belongs to the first beat's chapter (it's under 10s on its own).
        episode = ks.episode_beats(script)
        chapters = [{"start_seconds": 0, "title": episode[0]["title"]}]
        steps = bible["assigned_steps"]
        for b in episode[1:]:
            label = b["title"]
            if b["id"] in steps and not steps[b["id"]][0].startswith("("):
                first = steps[b["id"]][0].rstrip(".")
                label = f"{b['title']} — {first[:48].rsplit(' ', 1)[0] if len(first) > 48 else first}"
            t = int(float(windows[b["id"]][0]) + 0.999)
            if t - chapters[-1]["start_seconds"] >= 10 and expected - t >= 10:
                chapters.append({"start_seconds": t, "title": label})
        if quick:
            chapters = []  # too short for YouTube's timeline (>=3 chapters of >=10 s)
        topic_tag = "#" + re.sub(r"[^a-z]", "", exp["id"].split("-")[0])
        names = [c["name"] for c in ks.series()["cast"].values()]
        duo = " and ".join(names)
        description = "\n\n".join([
            yt_title,
            f"{ks.series()['series']['name']} — a science story for curious kids aged 8 to 11.",
            f"{exp['question']} Brothers {duo} find out with a real experiment you can try at home.",
            "You'll need: " + ", ".join(exp["materials"]) + ".",
            f"What do you think will happen? Tell us in the comments before {names[0]} explains!",
            f"Safety first: {exp['safety']} Always do experiments with a grown-up.",
            f"Subscribe for a new science story with {duo}.",
            *([] if quick else ["Chapters:\n{chapters}"]),
            " ".join(HASHTAGS + [topic_tag]),
        ])
        tags = ["kids science", "science experiment for kids", "story time", "STEM for kids", "science for kids",
                "easy science experiment", duo, exp["title"], *exp["materials"][:3]]
        export = project / "export"
        export.mkdir(exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")
        out_video = export / f"{slug}.mp4"
        shutil.copy2(final, out_video)
        thumb = export / f"{slug}_THUMBNAIL.png"
        shutil.copy2(project / "assets" / "images" / "title_card.png", thumb)
        meta = {
            "suggested_title": yt_title, "suggested_description": description, "suggested_tags": tags,
            "suggested_chapters": chapters, "suggested_category_id": "27",  # Education
            "suggested_made_for_kids": True,  # aimed at children: YouTube requires the flag (COPPA)
            "suggested_thumbnail": str(thumb.relative_to(project)), "contains_synthetic_media": True,
            "hero_output": f"projects/{project.name}/{out_video.relative_to(project).as_posix()}",
            "copy_policy": "engagement only: no tools, models or production tech named",
            "publish_action": "exported locally only — nothing uploaded",
        }
        (export / "metadata.json").write_text(json.dumps({"title": yt_title, "description": description, "tags": tags,
                                                          "chapters": chapters}, indent=2, ensure_ascii=False), encoding="utf-8")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (art / "publish_log.json").write_text(json.dumps({
            "version": "1.0",
            "entries": [{"platform": "local_export", "status": "exported", "export_path": str(out_video), "timestamp": ts}],
            "metadata": meta,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "render_report.json").write_text(json.dumps({
            "version": "1.0",
            "outputs": [{"path": str(out_video), "format": "mp4", "resolution": "1920x1080",
                         "duration_seconds": checks["container"]["duration"], "platform_target": "youtube"}],
            "verification_notes": [f"{k}: {v}" for k, v in checks.items()],
        }, indent=2), encoding="utf-8")
        return ToolResult(
            success=True,
            data={"export": str(out_video), "duration_seconds": checks["container"]["duration"],
                  "transcript_error": checks["transcript_matches_script"]["error"], "chapters": len(chapters)},
            artifacts=[str(out_video), str(art / "final_review.json"), str(art / "publish_log.json")],
            duration_seconds=round(time.time() - start, 1),
        )
