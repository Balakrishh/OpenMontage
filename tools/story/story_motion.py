"""Real motion for a story-time episode's key moments — LTX image-to-video from approved stills.

Only a handful of shots move (``ks.pick_motion``: the result, the hook, the
explanation, the send-off): animation is expensive (~4 min of GPU per output
second) and every moving shot is another chance to drift. Each clip starts
from a still that already passed both reviewers, uses a short, calm motion
prompt (long action lists drift), and is checked on its middle and last
frames with the same hard checks. A clip that drifts is discarded and the
shot keeps its still — motion is an upgrade, never a risk to the plot.
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

CLIP_SECONDS = 5
FPS = 24
# Structural similarity of a clip frame to the still it was animated from.
# Gentle motion stays well above this; a texture or pattern washing over the
# frame (a failure both vision reviewers passed as "clean") collapses below it.
MIN_SIMILARITY = 0.65
MIN_KEEP_SECONDS = 1.2


def similarity_to_still(still: Path, video: Path, at: float) -> float:
    """Blockwise SSIM (grayscale, 256x144) between a still and a clip frame — no model involved."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view as windows
    from PIL import Image

    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
                          "-vf", "scale=256:144,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    if len(raw) != 256 * 144:
        return 0.0
    b = np.frombuffer(raw, np.uint8).reshape(144, 256).astype(float)
    a = np.asarray(Image.open(still).convert("L").resize((256, 144)), float)
    A, B = windows(a, (8, 8))[::4, ::4], windows(b, (8, 8))[::4, ::4]
    ma, mb = A.mean((-1, -2)), B.mean((-1, -2))
    va, vb = A.var((-1, -2)), B.var((-1, -2))
    cov = ((A - ma[..., None, None]) * (B - mb[..., None, None])).mean((-1, -2))
    c1, c2 = 6.5, 58.5
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def motion_prompt(shot: Any, exp: dict[str, Any]) -> str:
    if shot.kind == "plate":
        return (f"Very slow, gentle change: {exp['observe']} The camera stays perfectly still. The cups, the "
                "table and everything else stay exactly where they are.")
    names = " and ".join(c["name"] for c in shot._cast()) or "The scene"
    # A still camera: the push-in blew faces up to fill the frame and smeared them.
    return (f"{names} move gently and naturally: blinking, small smiles, a slight turn of the head. The camera "
            "stays perfectly still. The kitchen stays still. No one new appears and nothing new appears.")


def frame_at(video: Path, seconds: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{seconds:.2f}", "-i", str(video), "-frames:v", "1",
                    str(out)], check=True, capture_output=True)
    return out


class StoryMotion(BaseTool):
    name = "story_motion"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "ltx-mlx"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = "Needs ltx_video_mlx (ltx-2-mlx venv) and the LM Studio vision models used by story_keyframes."
    agent_skills = ["ltx2"]
    capabilities = ["image_to_video", "drift_check"]
    supports = {"offline": True, "drift_fallback": True}
    best_for = ["a few animated key moments in an otherwise still storybook episode"]
    not_good_for = ["animating every shot (cost and drift)"]

    input_schema = {
        "type": "object",
        "required": ["project_dir"],
        "properties": {
            "project_dir": {"type": "string"},
            "quality": {"type": "string", "enum": ["draft", "final"], "default": "final"},
            "max_clips": {"type": "integer", "default": 5},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=8, ram_mb=60000, vram_mb=0, disk_mb=3000, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes assets/video/*.mp4, updates artifacts/asset_manifest.json, writes artifacts/motion_qa.json"]
    user_visible_verification = ["Watch the animated shots: same characters, same setup, gentle motion"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 240.0 * CLIP_SECONDS * int(inputs.get("max_clips", 5))

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import os

        from lib import kids_science as ks
        from lib import lmstudio
        from tools.story.story_keyframes import REVIEW_MODEL, VQA_MODEL, judge
        from tools.video.ltx_video_mlx import LTXVideoMLX

        start = time.time()
        project = Path(inputs["project_dir"]).resolve()
        art = project / "artifacts"
        plan = json.loads((art / "scene_plan.json").read_text(encoding="utf-8"))
        shots = {s["id"]: ks.ShotSpec(**s) for s in plan["metadata"]["shots"]}
        wanted = plan["metadata"].get("motion", [])[: int(inputs.get("max_clips", 5))]
        manifest = json.loads((art / "asset_manifest.json").read_text(encoding="utf-8"))
        stills = {a["id"]: Path(a["path"]) for a in manifest["assets"] if a["type"] == "image"}
        exp = ks.experiment(json.loads((art / "story_bible.json").read_text())["experiment_id"])
        vqa = os.environ.get("KIDS_VQA_MODEL", VQA_MODEL)
        reviewer = os.environ.get("KIDS_REVIEW_MODEL", REVIEW_MODEL)
        vdir = project / "assets" / "video"
        vdir.mkdir(parents=True, exist_ok=True)
        ltx = LTXVideoMLX()
        report: dict[str, Any] = {}
        kept = []

        for sid in wanted:
            shot, still = shots.get(sid), stills.get(sid)
            if shot is None or still is None:
                continue
            # LTX wants /64 dims; the composer scales 1024x576 to the frame like any still.
            ref = vdir / f"{sid}-ref.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(still), "-vf", "scale=1024:576", str(ref)],
                           check=True, capture_output=True)
            clip = vdir / f"{sid}.mp4"
            raw = vdir / f"{sid}-raw.mp4"
            t0 = time.time()
            quality = inputs.get("quality", "final")
            res = ltx.execute({
                "operation": "image_to_video", "reference_image_path": str(ref), "prompt": motion_prompt(shot, exp),
                "width": 1024, "height": 576, "num_frames": CLIP_SECONDS * FPS + 1, "fps": FPS,
                "quality": quality, "stage1_steps": 35, "stage2_steps": 3, "cfg_scale": 2.0,
                "teacache_thresh": 0.18, "seed": 7, "output_path": str(raw),
            })
            entry: dict[str, Any] = {"prompt": motion_prompt(shot, exp), "seconds": round(time.time() - t0)}
            if not res.success:
                entry.update(status="kept_still", reason=f"generation failed: {res.error[:200]}")
                report[sid] = entry
                continue
            # Silent clip: the narration is the soundtrack; LTX's own audio is dropped.
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-an", "-c:v", "copy", str(clip)],
                           check=True, capture_output=True)
            verdicts = []
            keep_until = None
            try:
                # Drift lives in a clip's tail; its opening seconds are nearly
                # always clean. Check in time order and keep the clean prefix.
                for label, at in (("first", 0.6), ("early", 1.2), ("middle", 2.6), ("end", CLIP_SECONDS - 0.2)):
                    sim = similarity_to_still(ref, clip, at)
                    if sim < MIN_SIMILARITY:
                        verdicts.append({"frame": label, "at": at, "hard_ok": False, "similarity": round(sim, 3),
                                         "failed": [f"drifted from the approved still (similarity {sim:.2f} < {MIN_SIMILARITY})"]})
                        break
                    frame = frame_at(clip, at, vdir / f"{sid}-{label}.png")
                    v = judge(vqa, frame, shot.checklist())
                    r = judge(reviewer, frame, [shot.review_checks(exp_does=exp["observe"])[-1],
                                                "Scene state: the image is clean and sharp, with no blotchy smears, "
                                                "smudges, pixelation or distorted faces."])
                    ok = v["hard_ok"] and r["hard_ok"]
                    verdicts.append({"frame": label, "at": at, "hard_ok": ok,
                                     "failed": [c["check"][:80] for c in v["checks"] + r["checks"] if c["hard"] and not c["yes"]]})
                    if not ok:
                        break
                    keep_until = at
            except lmstudio.LMStudioError as exc:
                verdicts.append({"frame": "?", "hard_ok": False, "failed": [str(exc)[:120]]})
            entry["checks"] = verdicts
            if keep_until is not None and keep_until < MIN_KEEP_SECONDS:
                keep_until = None  # a sub-second wiggle isn't motion worth keeping
            if keep_until is not None:
                if keep_until < CLIP_SECONDS - 0.2:
                    trimmed = vdir / f"{sid}-trim.mp4"
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-t", f"{keep_until:.2f}",
                                    "-an", str(trimmed)], check=True, capture_output=True)
                    trimmed.replace(clip)
                entry["status"] = "animated"
                entry["kept_seconds"] = keep_until
                kept.append(sid)
                manifest["assets"].append({
                    "id": f"{sid}-motion", "type": "video", "path": str(clip), "source_tool": "ltx_video_mlx",
                    "scene_id": sid, "duration_seconds": keep_until, "model": "ltx-2.3-mlx-q8",
                    "generation_summary": f"animated from the approved still; clean for {keep_until}s (later frames drifted or clip ended)",
                })
            else:
                entry.update(status="kept_still", reason="drifted: " + "; ".join(sum((v["failed"] for v in verdicts), [])[:3]))
            report[sid] = entry
        lmstudio.unload_all()

        (art / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (art / "motion_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return ToolResult(
            success=True,  # a drifted clip is not a failure: the still is the safe default
            data={"requested": len(wanted), "animated": kept,
                  "kept_still": [k for k, v in report.items() if v["status"] != "animated"]},
            artifacts=[str(art / "motion_qa.json")],
            duration_seconds=round(time.time() - start, 1),
        )
