"""Experiment motion for Quick Science, computed from approved pictures — nothing generated.

For each step, the plate before and the plate after are both state-checked.
Their pixel difference is exactly what the step changed (water appearing,
colour arriving, towels darkening). Only that region is animated:

  - rise:  the new state is revealed from the bottom of each changed region
           upward — liquid filling a cup, colour creeping up a towel
  - drops: when a step adds colour, drops of the new colour fall into each
           changed region first, then it fills

Everything outside the changed region stays perfectly still, and every frame
is a blend of two approved pictures, so the motion cannot show anything the
checks did not already pass. The text band is laid over every frame.
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

FPS = 24
SECONDS = 4.0
MIN_CHANGED = 0.006  # below this share of the frame, a step changed nothing visible
MAX_CHANGED = 0.25   # above it the picture was redrawn (e.g. cups became mugs) — not a step to animate


def match_colors(b, a):
    """b recoloured to a's per-channel mean/std: removes the global tint drift an
    edit introduces, so only real content changes register."""
    import numpy as np

    out = (b - b.mean(axis=(0, 1))) / (b.std(axis=(0, 1)) + 1e-6) * a.std(axis=(0, 1)) + a.mean(axis=(0, 1))
    return np.clip(out, 0, 255)


def change_mask(a, b):
    """Soft mask (0..1, HxW) of where picture b differs meaningfully from picture a."""
    import numpy as np
    from PIL import Image, ImageFilter

    diff = np.abs(a.astype(float) - b.astype(float)).max(axis=2)
    m = (diff > 40).astype(np.uint8) * 255
    img = Image.fromarray(m).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(5))
    img = img.filter(ImageFilter.GaussianBlur(6))
    return np.asarray(img, float) / 255.0


def blobs(mask, min_share: float = 0.002):
    """Bounding boxes (x0, y0, x1, y1) of the separate changed regions, left to right."""
    import numpy as np

    h, w = mask.shape
    cols = (mask > 0.5).sum(axis=0)
    boxes, start = [], None
    for x in range(w + 1):
        on = x < w and cols[x] > 0
        if on and start is None:
            start = x
        elif not on and start is not None:
            sub = mask[:, start:x] > 0.5
            if sub.sum() > min_share * h * w:
                ys = np.where(sub.any(axis=1))[0]
                boxes.append((start, int(ys[0]), x, int(ys[-1])))
            start = None
    return boxes


def animate(before: Path, after: Path, out: Path, band_text: str = "", drops: bool = False,
            seconds: float = SECONDS) -> bool:
    """Write an mp4 of before→after; False if the step changed nothing visible."""
    import numpy as np
    from PIL import Image, ImageDraw

    from tools.story.story_keyframes import text_band

    a = np.asarray(Image.open(before).convert("RGB"), float)
    b = np.asarray(Image.open(after).convert("RGB").resize((a.shape[1], a.shape[0])), float)
    bm = match_colors(b, a)
    tint = b - bm  # applied gradually, so the clip still ends exactly on the approved picture
    mask = change_mask(a, bm)
    share = (mask > 0.5).mean()
    if share < MIN_CHANGED or share > MAX_CHANGED:
        return False
    h, w = mask.shape
    boxes = blobs(mask)
    ys = np.arange(h, dtype=float)[:, None]
    n = int(seconds * FPS)
    drop_end = int(n * 0.3) if drops else 0
    reveal_end = int(n * 0.85)
    frames_dir = out.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    feather = h * 0.08
    for i in range(n):
        if i < drop_end:
            line = h * 2.0  # nothing revealed while the drops fall
        else:
            p = min(1.0, (i - drop_end) / max(1, reveal_end - drop_end))
            p = p * p * (3 - 2 * p)  # ease in/out
            line = h - p * (h + feather)  # reveal edge moves from the bottom up
        reveal = np.clip((ys - line) / feather, 0, 1) * mask
        g = i / max(1, n - 1)
        frame = np.clip(a * (1 - reveal[..., None]) + bm * reveal[..., None] + g * tint, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame)
        if i < drop_end and boxes:
            d = ImageDraw.Draw(img)
            t = i / max(1, drop_end)
            for (x0, y0, x1, y1) in boxes:
                region = b[y0:y1, x0:x1].reshape(-1, 3)
                col = tuple(int(c) for c in region.mean(axis=0))
                cx = (x0 + x1) // 2
                for k in range(3):  # three drops, staggered
                    tk = (t * 1.6 - k * 0.3)
                    if 0 <= tk <= 1:
                        y = int(y0 - h * 0.35 + tk * h * 0.35)
                        r = max(6, w // 110)
                        d.ellipse((cx - r, y - int(r * 1.3), cx + r, y + r), fill=col)
                        d.ellipse((cx - r // 3, y - r // 2, cx, y - r // 6), fill=(255, 255, 255))
        path = frames_dir / f"{i:04d}.png"
        img.save(path)
        if band_text:
            text_band(path, band_text, path)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%04d.png"),
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", str(out)], check=True, capture_output=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()
    return True


class QuickMotion(BaseTool):
    name = "quick_motion"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_post"
    provider = "local"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    install_instructions = "Needs ffmpeg, numpy and Pillow."
    agent_skills: list[str] = []
    capabilities = ["change_animation", "rise_reveal", "drops"]
    supports = {"offline": True, "no_model": True}
    best_for = ["animating an experiment's steps from approved before/after plates"]
    not_good_for = ["character motion (use story_motion)"]

    input_schema = {"type": "object", "required": ["project_dir"], "properties": {"project_dir": {"type": "string"}}}
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=2000, vram_mb=0, disk_mb=500, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes assets/video/*-change.mp4 and adds them to artifacts/asset_manifest.json"]
    user_visible_verification = ["Watch each step: only the changed part moves"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        project = Path(inputs["project_dir"]).resolve()
        art = project / "artifacts"
        shots = json.loads((art / "scene_plan.json").read_text())["metadata"]["shots"]
        qa = json.loads((art / "keyframe_qa.json").read_text())["shots"]
        manifest = json.loads((art / "asset_manifest.json").read_text())
        # Re-runs replace this tool's earlier clips instead of piling up duplicates.
        manifest["assets"] = [a for a in manifest["assets"] if a.get("source_tool") != "quick_motion"]
        vdir = project / "assets" / "video"
        vdir.mkdir(parents=True, exist_ok=True)

        def source(sid: str) -> Path | None:
            e = qa.get(sid) or {}
            if e.get("source_shot"):
                return source(e["source_shot"])
            return Path(e["source"]) if e.get("source") else None

        made, skipped = [], []
        prev_plate = None
        for sh in shots:
            sid = sh["id"]
            if sh["kind"] == "plate":
                cur = source(sid)
                if prev_plate and cur and qa.get(sid, {}).get("status") != "fallback":
                    adds_colour = any(w in (sh.get("action") or "").lower() for w in ("color", "colour", "drop"))
                    out = vdir / f"{sid}-change.mp4"
                    if animate(prev_plate, cur, out, sh.get("overlay", ""), drops=adds_colour):
                        made.append(sid)
                        manifest["assets"].append({"id": f"{sid}-change", "type": "video", "path": str(out),
                                                   "source_tool": "quick_motion", "scene_id": sid,
                                                   "generation_summary": "change animation between approved plates"})
                    else:
                        skipped.append(sid)
                prev_plate = cur or prev_plate
            elif sh["kind"] == "blend":
                a, b = source(sh["blend_from"]), source(sh["blend_to"])
                out = vdir / f"{sid}-change.mp4"
                if a and b and animate(a, b, out, "", drops=False, seconds=6.0):
                    made.append(sid)
                    manifest["assets"].append({"id": f"{sid}-change", "type": "video", "path": str(out),
                                               "source_tool": "quick_motion", "scene_id": sid,
                                               "generation_summary": "time-lapse: rise reveal between approved plates"})
        (art / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ToolResult(success=True, data={"animated": made, "no_visible_change": skipped},
                          duration_seconds=round(time.time() - start, 1))
