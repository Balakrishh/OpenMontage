"""Narration for a story-time episode: synthesize, verify every line, assemble, time.

Reads the script's per-beat lines, voices each with the series voice for its
speaker (Kokoro), then transcribes every line with Whisper and compares it to
the script. A line whose transcript drifts too far is re-voiced once at a
slower speed; if it still drifts the stage fails — skipped or mangled words
never reach the final film unnoticed.

Outputs:
  assets/audio/narration.wav        — the full read, loudness-normalised
  artifacts/narration_timing.json   — every line's and beat's real start/end
  artifacts/captions.json           — word-level timings for read-along captions
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import wave
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

LINE_GAP = 0.35   # pause between lines inside a beat
BEAT_GAP = 0.9    # breath between beats
MAX_LINE_ERROR = 0.2


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).split()


def word_error(expected: str, heard: str) -> float:
    """1 - similarity over word sequences: 0 is perfect, 1 is unrelated."""
    a, b = _norm(expected), _norm(heard)
    if not a:
        return 0.0
    return round(1 - SequenceMatcher(None, a, b).ratio(), 3)


def _duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


class StoryNarration(BaseTool):
    name = "story_narration"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "narration"
    provider = "kokoro"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = "Needs kokoro_tts_mlx (mlx-audio venv) and faster-whisper in this venv."
    agent_skills: list[str] = []
    capabilities = ["narration_synthesis", "narration_verification", "word_timings"]
    supports = {"offline": True, "per_line_verification": True}
    best_for = ["story-time narration with per-line transcript verification"]
    not_good_for = ["lip-synced dialogue"]

    input_schema = {
        "type": "object",
        "required": ["project_dir"],
        "properties": {"project_dir": {"type": "string"}, "whisper_model": {"type": "string", "default": "small"}},
    }
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=4000, vram_mb=0, disk_mb=200, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes assets/audio/*.wav, artifacts/narration_timing.json, artifacts/captions.json"]
    user_visible_verification = ["Listen to the start of assets/audio/narration.wav"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 300.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from faster_whisper import WhisperModel

        from lib import kids_science as ks
        from tools.audio.kokoro_tts_mlx import KokoroTTSMLX

        start = time.time()
        project = Path(inputs["project_dir"])
        script = json.loads((project / "artifacts" / "script.json").read_text(encoding="utf-8"))
        lines_by_beat: dict[str, list[dict[str, str]]] = script["metadata"]["lines"]
        voices = ks.series()["voices"]
        line_dir = project / "assets" / "audio" / "lines"
        line_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        for beat in ks.episode_beats(script):
            for i, line in enumerate(lines_by_beat[beat["id"]], 1):
                v = voices[line["speaker"]]
                lines.append({
                    "beat": beat["id"], "speaker": line["speaker"], "text": line["text"].strip(),
                    "path": str(line_dir / f"{beat['id']}-{i:02d}.wav"),
                    "voice": v["kokoro_voice"], "lang_code": v["lang_code"], "speed": v["speed"],
                })

        tts = KokoroTTSMLX()
        jobs = [{"text": l["text"], "output_path": l["path"], "voice": l["voice"],
                 "lang_code": l["lang_code"], "speed": l["speed"]} for l in lines]
        res = tts.execute({"jobs": jobs})
        if not res.success:
            return ToolResult(success=False, error=f"voice synthesis failed: {res.error}")

        whisper = WhisperModel(inputs.get("whisper_model", "small"), device="cpu", compute_type="int8")

        def hear(path: str) -> str:
            segs, _ = whisper.transcribe(path, language="en", vad_filter=False, beam_size=5)
            return " ".join(s.text.strip() for s in segs)

        redo = []
        for l in lines:
            l["heard"] = hear(l["path"])
            l["error"] = word_error(l["text"], l["heard"])
            if l["error"] > MAX_LINE_ERROR:
                redo.append(l)
        if redo:  # one slower retake for every line that drifted
            res = tts.execute({"jobs": [{"text": l["text"], "output_path": l["path"], "voice": l["voice"],
                                         "lang_code": l["lang_code"], "speed": round(l["speed"] * 0.9, 2)} for l in redo]})
            if not res.success:
                return ToolResult(success=False, error=f"voice retake failed: {res.error}")
            for l in redo:
                l["heard"] = hear(l["path"])
                l["error"] = word_error(l["text"], l["heard"])
                l["retaken"] = True
        bad = [l for l in lines if l["error"] > MAX_LINE_ERROR]
        timing_path = project / "artifacts" / "narration_timing.json"
        if bad:
            timing_path.write_text(json.dumps({"lines": lines}, indent=2, ensure_ascii=False), encoding="utf-8")
            return ToolResult(
                success=False,
                error=f"{len(bad)} line(s) still don't match the script after a retake, e.g. "
                      f"{bad[0]['beat']}: expected {bad[0]['text']!r}, heard {bad[0]['heard']!r}",
            )

        # Assemble with fixed pauses; every timestamp below is in the final track.
        raw = project / "assets" / "audio" / "narration_raw.wav"
        with wave.open(lines[0]["path"]) as first:
            params = first.getparams()
        rate = params.framerate
        silence = lambda s: b"\x00\x00" * int(s * rate) * params.nchannels  # noqa: E731
        t = 0.0
        beat_spans: dict[str, dict[str, float]] = {}
        with wave.open(str(raw), "wb") as out:
            out.setparams(params)
            prev_beat = None
            for l in lines:
                if prev_beat is not None:
                    gap = BEAT_GAP if l["beat"] != prev_beat else LINE_GAP
                    out.writeframes(silence(gap))
                    t += gap
                with wave.open(l["path"]) as w:
                    out.writeframes(w.readframes(w.getnframes()))
                    dur = w.getnframes() / w.getframerate()
                l["start"], l["end"] = round(t, 3), round(t + dur, 3)
                span = beat_spans.setdefault(l["beat"], {"start": l["start"]})
                span["end"] = l["end"]
                t += dur
                prev_beat = l["beat"]
            out.writeframes(silence(0.6))

        final = project / "assets" / "audio" / "narration.wav"
        norm = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ar", "48000", "-ac", "1", str(final)], capture_output=True, text=True,
        )
        if norm.returncode != 0:
            return ToolResult(success=False, error=f"loudness normalisation failed: {norm.stderr[-400:]}")

        # Word timings from the finished track itself, so captions can't drift.
        segs, _ = whisper.transcribe(str(final), language="en", word_timestamps=True, vad_filter=False)
        words = [
            {"word": w.word.strip(), "startMs": int(w.start * 1000), "endMs": int(w.end * 1000)}
            for s in segs for w in (s.words or []) if w.word.strip()
        ]
        # Page breaks at beat boundaries keep a caption from straddling two scenes.
        bounds = sorted(v["end"] for v in beat_spans.values())
        for w in words:
            if any(abs(w["endMs"] / 1000 - b) < 0.25 for b in bounds):
                w["pageBreakAfter"] = True

        total = round(_duration(final), 3)
        timing_path.write_text(json.dumps({
            "duration_seconds": total, "beats": beat_spans,
            "lines": [{k: l[k] for k in ("beat", "speaker", "text", "heard", "error", "start", "end", "path")}
                      | {"retaken": l.get("retaken", False)} for l in lines],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (project / "artifacts" / "captions.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        return ToolResult(
            success=True,
            data={
                "duration_seconds": total, "lines": len(lines), "retaken": len(redo),
                "worst_line_error": max(l["error"] for l in lines), "caption_words": len(words),
            },
            artifacts=[str(final), str(timing_path), str(project / "artifacts" / "captions.json")],
            duration_seconds=round(time.time() - start, 1),
        )
