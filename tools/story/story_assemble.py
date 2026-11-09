"""Assemble a story-time episode: title card, timed stills with gentle motion, captions, mix.

Pictures follow the narration, never the plan: every beat's shots share the
beat's REAL spoken span (from narration_timing.json), so a beat that came out
longer or shorter than planned still lines up. Motion is a slow Ken Burns
family cycled per shot — enough life for story time, nothing that can drift.

Remotion renders picture + read-along captions; the soundtrack (narration +
a low music bed) is mixed in ffmpeg and muxed after, because the Remotion
path does not reliably carry external audio.
"""

from __future__ import annotations

import json
import subprocess
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

TITLE_SECONDS = 4.0
TAIL_SECONDS = 2.0
MOTIONS = ["ken-burns", "pan-right", "zoom-in", "pan-left", "zoom-out", "parallax"]


def title_card(background: Path, title: str, series_name: str, out: Path, size: tuple[int, int]) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    bg = Image.open(background).convert("RGB").resize(size).filter(ImageFilter.GaussianBlur(6))
    veil = Image.new("RGBA", size, (255, 246, 228, 120))
    img = Image.alpha_composite(bg.convert("RGBA"), veil)
    d = ImageDraw.Draw(img)
    w, h = size

    def font(px: int):
        for name in ("/System/Library/Fonts/Supplemental/Chalkboard.ttc", "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"):
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default()

    def centered(text: str, y: int, f, fill) -> int:
        words, lines, cur = text.split(), [], ""
        for word in words:  # wrap to ~80% width
            trial = f"{cur} {word}".strip()
            if d.textbbox((0, 0), trial, font=f)[2] > w * 0.8 and cur:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        for line in lines:
            tw, th = d.textbbox((0, 0), line, font=f)[2:]
            d.text(((w - tw) // 2, y), line, font=f, fill=fill)
            y += int(th * 1.3)
        return y

    y = centered(series_name, int(h * 0.22), font(h // 22), (150, 90, 50, 255))
    centered(title, y + h // 18, font(h // 11), (70, 40, 25, 255))
    img.convert("RGB").save(out)


class StoryAssemble(BaseTool):
    name = "story_assemble"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "video_post"
    provider = "remotion"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    install_instructions = "Needs video_compose (Remotion via npx) and ffmpeg."
    agent_skills = ["remotion-best-practices"]
    capabilities = ["story_timeline", "title_card", "caption_render", "audio_mix"]
    supports = {"offline": True, "captions": True}
    best_for = ["narration-led still-image story videos with read-along captions"]
    not_good_for = ["dialogue-synced motion footage"]

    input_schema = {
        "type": "object",
        "required": ["project_dir"],
        "properties": {"project_dir": {"type": "string"}, "playbook": {"type": "string", "default": "anime-ghibli"}},
    }
    resource_profile = ResourceProfile(cpu_cores=8, ram_mb=8000, vram_mb=0, disk_mb=3000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes renders/final.mp4, artifacts/edit_decisions.json, artifacts/render_report.json"]
    user_visible_verification = ["Watch the first 30 seconds: title, voice, captions in sync"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 600.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from lib import kids_science as ks
        from tools.video.video_compose import VideoCompose

        start = time.time()
        runtime = ks.series().get("render_runtime")
        if runtime != "remotion":
            # HyperFrames is installed and was offered for this series; this
            # stage only implements the Remotion Explainer path so far.
            return ToolResult(success=False, error=(
                f"render_runtime is {runtime!r} in lib/kids_science/series.yaml; story_assemble renders with "
                "remotion only — set render_runtime: remotion, or add the hyperframes path first"))
        project = Path(inputs["project_dir"]).resolve()
        art = project / "artifacts"
        timing = json.loads((art / "narration_timing.json").read_text(encoding="utf-8"))
        bible = json.loads((art / "story_bible.json").read_text(encoding="utf-8"))
        shots = json.loads((art / "scene_plan.json").read_text(encoding="utf-8"))["metadata"]["shots"]
        assets = json.loads((art / "asset_manifest.json").read_text())["assets"]
        images = {a["id"]: a["path"] for a in assets if a["type"] == "image"}
        clips = {a["scene_id"]: a["path"] for a in assets if a["type"] == "video"}
        captions = json.loads((art / "captions.json").read_text(encoding="utf-8"))
        s = ks.series()
        size = (s["style"]["width"], s["style"]["height"])

        renders = project / "renders"
        renders.mkdir(exist_ok=True)
        card = project / "assets" / "images" / "title_card.png"
        title_card(Path(images[shots[0]["id"]]), bible["episode_title"], s["series"]["name"], card, size)

        narr_total = float(timing["duration_seconds"])
        total = TITLE_SECONDS + narr_total + TAIL_SECONDS
        beat_ids = [b["id"] for b in ks.beats()]
        starts = [TITLE_SECONDS + timing["beats"][b]["start"] for b in beat_ids]
        # A beat's pictures run from its first word to the next beat's first word.
        windows = [(starts[i] if i else TITLE_SECONDS, starts[i + 1] if i + 1 < len(starts) else total)
                   for i in range(len(starts))]

        cuts = [{"id": "title", "source": str(card), "in_seconds": 0.0, "out_seconds": TITLE_SECONDS, "animation": "zoom-in"}]
        # Each picture switches in on the first word of ITS lines, so the
        # picture always shows what is being said right then.
        line_starts: dict[str, list[float]] = {}
        for line in timing["lines"]:
            line_starts.setdefault(line["beat"], []).append(TITLE_SECONDS + line["start"])
        k = 0
        for b, (w0, w1) in zip(beat_ids, windows):
            beat_shots = [sh for sh in shots if sh["beat"] == b]
            ins = [w0] + [line_starts[b][sh["lines"][0]] if sh.get("lines") else w0 for sh in beat_shots[1:]]
            outs = ins[1:] + [w1]
            for sh, t0, t1 in zip(beat_shots, ins, outs):
                if t1 - t0 < 0.5:
                    # A group whose words begin with the next one's: let the
                    # previous picture cover it, so there is never a gap.
                    cuts[-1]["out_seconds"] = round(t1, 3)
                    continue
                cut = {"id": sh["id"], "source": images[sh["id"]], "in_seconds": round(t0, 3),
                       "out_seconds": round(t1, 3), "animation": MOTIONS[k % len(MOTIONS)]}
                if sh["id"] in clips:
                    # Animated shot: hold the clip's last frame for the rest of the shot.
                    padded = Path(clips[sh["id"]]).with_name(f"{sh['id']}-cut.mp4")
                    # Clips may be trimmed to their clean prefix: always pad by the full shot length.
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", clips[sh["id"]], "-vf",
                                    f"tpad=stop_mode=clone:stop_duration={(t1 - t0) + 0.5:.2f}", "-an", str(padded)],
                                   check=True, capture_output=True)
                    cut.update(source=str(padded), animation="static")
                cuts.append(cut)
                k += 1
        offset = int(TITLE_SECONDS * 1000)
        words = [{**w, "startMs": w["startMs"] + offset, "endMs": w["endMs"] + offset} for w in captions]

        props = {
            "cuts": cuts, "captions": words, "renderer_family": "explainer-data",
            "playbook": inputs.get("playbook", "anime-ghibli"),
        }
        video_only = renders / "video_only.mp4"
        # A long still-image episode is thousands of frames; the tool's default
        # subprocess limit (max(600s, 15s/cut)) is sized for short explainers.
        res = VideoCompose().execute({"operation": "remotion_render", "edit_decisions": props,
                                      "output_path": str(video_only), "remotion_timeout_ms": 3_600_000})
        if not res.success:
            return ToolResult(success=False, error=f"render failed: {res.error}")

        # Soundtrack: narration after the title card, a quiet looped music bed with fades.
        mix = project / "assets" / "audio" / "mix.wav"
        music = (ks.REPO_ROOT / s["music"]["path"]).resolve()
        delay = int(TITLE_SECONDS * 1000)
        graph = (
            f"[0:a]adelay={delay}|{delay},apad[n];"
            f"[1:a]volume={s['music']['volume']},afade=t=in:d=2,afade=t=out:st={total - 3:.2f}:d=3[m];"
            f"[n][m]amix=inputs=2:duration=first:normalize=0,atrim=0:{total:.3f}[a]"
        )
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(project / "assets" / "audio" / "narration.wav"),
               "-stream_loop", "-1", "-i", str(music), "-filter_complex", graph, "-map", "[a]",
               "-ar", "48000", "-ac", "2", str(mix)]
        if not music.is_file():
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(project / "assets" / "audio" / "narration.wav"),
                   "-af", f"adelay={delay}|{delay},apad,atrim=0:{total:.3f}", "-ar", "48000", "-ac", "2", str(mix)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return ToolResult(success=False, error=f"audio mix failed: {p.stderr[-400:]}")
        final = renders / "final.mp4"
        p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_only), "-i", str(mix), "-map", "0:v:0",
                            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return ToolResult(success=False, error=f"mux failed: {p.stderr[-400:]}")

        (art / "edit_decisions.json").write_text(json.dumps({
            "version": "1.0", "render_runtime": "remotion", "renderer_family": "explainer-data",
            "cuts": [{k2: c[k2] for k2 in ("id", "source", "in_seconds", "out_seconds")} | {"transform": {"animation": c["animation"]}} for c in cuts],
            "metadata": {"title_seconds": TITLE_SECONDS, "beat_windows": dict(zip(beat_ids, windows)),
                         "music": str(music), "caption_words": len(words)},
        }, indent=2), encoding="utf-8")
        return ToolResult(
            success=True,
            data={"output": str(final), "duration_seconds": round(total, 2), "cuts": len(cuts)},
            artifacts=[str(final), str(art / "edit_decisions.json")],
            duration_seconds=round(time.time() - start, 1),
        )
