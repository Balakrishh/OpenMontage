"""Keyframes for a story-time episode: generate, verify against a checklist, retry, fall back.

Character consistency comes from series-level reference portraits
(projects/_series/kids-science/masters/<id>.png), minted once and reused by
every episode — the recurring cast looks the same in episode 1 and episode 40.

Every shot image is judged by a local vision model against a yes/no checklist
derived from the shot spec (who is in it, how many humans, no animals that
shouldn't be there, no text, normal hands, key props). Hard failures (extra
people or animals, text) are never accepted; soft ones (a prop not obvious)
are accepted only from the best of all attempts. A shot that never clears the
hard checks reuses an accepted image from the same beat, so the film stays
coherent instead of showing something wrong.

Exact text — "Step 2" badges — is drawn with PIL, never generated.
"""

from __future__ import annotations

import json
import random
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

VQA_MODEL = "qwen/qwen3-vl-30b"
# A second vision model from a different family reviews every picture against
# the words spoken over it. Two lineages rarely share a blind spot.
REVIEW_MODEL = "gemma-3-27b-it"
HARD_MARKERS = ("human", "no owl", "animal", "text", "watercolor")

CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answers"],
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["n", "yes", "note"],
                "properties": {"n": {"type": "integer"}, "yes": {"type": "boolean"}, "note": {"type": "string"}},
            },
        }
    },
}


def judge(model: str, image: Path, checks: list[str]) -> dict[str, Any]:
    """Ask the vision model each check; returns per-check answers and hard/soft verdicts."""
    from lib import lmstudio

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(checks, 1))
    out = lmstudio.chat_json(
        model,
        "You inspect illustrations for a children's book. Answer each statement strictly from what is "
        "visible. yes=true only if the statement is clearly true. Keep notes to a few words.",
        f"Statements about the attached image:\n{numbered}\n\nAnswer every statement number exactly once.",
        CHECK_SCHEMA,
        temperature=0.0,
        max_tokens=1200,
        images=[image],
        timeout=300,
    )
    answers = {a["n"]: a for a in out.get("answers", [])}
    results = []
    for i, c in enumerate(checks, 1):
        a = answers.get(i, {"yes": False, "note": "not answered"})
        hard = c.startswith("Scene state:") or (
            not c.startswith("Detail:") and any(m in c.lower() for m in HARD_MARKERS))
        results.append({"check": c, "yes": bool(a["yes"]), "note": a.get("note", ""), "hard": hard})
    return {
        "checks": results,
        "hard_ok": all(r["yes"] for r in results if r["hard"]),
        "score": round(sum(r["yes"] for r in results) / len(results), 3),
    }


def step_badge(image: Path, label: str, out: Path) -> None:
    """Paint a storybook-style step badge in the top-left corner."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image).convert("RGBA")
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    size = max(28, h // 16)
    font = None
    for name in ("/System/Library/Fonts/Supplemental/Chalkboard.ttc", "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    tw, th = d.textbbox((0, 0), label, font=font)[2:]
    pad = size // 2
    # Ken Burns zooms up to 1.22x (≈9% cropped per edge) and pans ~40px:
    # sit well inside that so the badge is never cut off.
    x, y = int(w * 0.12), int(h * 0.12)
    d.rounded_rectangle((x, y, x + tw + 2 * pad, y + th + 2 * pad), radius=(th + 2 * pad) // 2,
                        fill=(255, 248, 230, 235), outline=(214, 120, 64, 255), width=max(3, size // 10))
    d.text((x + pad, y + pad - size // 8), label, font=font, fill=(120, 60, 30, 255))
    Image.alpha_composite(img, layer).convert("RGB").save(out)


CROPS = [(0.5, 0.55), (0.3, 0.5), (0.7, 0.5)]  # centre, left, right — as fractions of the frame


def detail_crop(src: Path, out: Path, variant: int, zoom: float = 1.6) -> None:
    """A tighter view of an already-approved picture: variety without anything new to check."""
    from PIL import Image

    img = Image.open(src).convert("RGB")
    w, h = img.size
    cw, ch = int(w / zoom), int(h / zoom)
    fx, fy = CROPS[variant % len(CROPS)]
    x = min(max(int(fx * w - cw / 2), 0), w - cw)
    y = min(max(int(fy * h - ch / 2), 0), h - ch)
    img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.LANCZOS).save(out)


def asset_entry(shot_id: str, path: Path, qa: dict[str, Any]) -> dict[str, Any]:
    """A schema-valid asset_manifest row; the QA outcome rides in the allowed fields."""
    return {
        "id": shot_id, "type": "image", "path": str(path), "source_tool": "flux2_edit_mlx", "scene_id": shot_id,
        "quality_score": float(qa.get("score", 1.0 if qa.get("status") == "accepted" else 0.0)),
        "generation_summary": f"{qa['status']} after {len(qa.get('attempts', []))} attempt(s)",
    }


class StoryKeyframes(BaseTool):
    name = "story_keyframes"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "mflux"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = "Needs flux2_edit_mlx (mflux) and an LM Studio vision model (KIDS_VQA_MODEL)."
    agent_skills = ["flux-best-practices"]
    capabilities = ["keyframe_generation", "keyframe_verification", "series_character_masters"]
    supports = {"offline": True, "checklist_qa": True, "auto_retry": True}
    best_for = ["unattended storybook stills with recurring characters"]
    not_good_for = ["exact text or counts in the image (composite those)"]

    input_schema = {
        "type": "object",
        "required": ["project_dir"],
        "properties": {
            "project_dir": {"type": "string"},
            "base_model": {"type": "string", "default": "flux2-klein-9b"},
            "steps": {"type": "integer", "default": 14},
            "max_attempts": {"type": "integer", "default": 3},
            "vqa_model": {"type": "string"},
            "review_model": {"type": "string", "description": "Second, independent vision reviewer"},
            "seed": {"type": "integer", "description": "Base seed for reproducible runs"},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=60000, vram_mb=0, disk_mb=2000, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["project_dir"]
    side_effects = ["writes assets/images/*, artifacts/asset_manifest.json, artifacts/keyframe_qa.json",
                    "mints series masters on first run"]
    user_visible_verification = ["Skim assets/images/ — every shot on-model, no stray people or text"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 3000.0

    # ------------------------------------------------------------------

    def _generate(self, prompt: str, refs: list[Path], out: Path, seed: int, inputs: dict[str, Any]) -> bool:
        from lib import kids_science as ks
        from tools.graphics.flux2_edit_mlx import Flux2EditMLX

        style = ks.series()["style"]
        out.unlink(missing_ok=True)
        res = Flux2EditMLX().execute({
            "prompt": prompt,
            "reference_image_paths": [str(r) for r in refs] or None,
            "text_to_image": not refs,
            "base_model": inputs.get("base_model", "flux2-klein-9b"),
            "steps": int(inputs.get("steps", 14)),
            "width": style["width"], "height": style["height"],
            "seed": seed, "output_path": str(out),
        })
        return res.success and out.is_file()

    def ensure_masters(self, inputs: dict[str, Any], vqa: str, log: list[str]) -> dict[str, Path]:
        from lib import kids_science as ks

        mdir = ks.SERIES_ASSETS / "masters"
        mdir.mkdir(parents=True, exist_ok=True)
        masters: dict[str, Path] = {}
        for cid, c in ks.series()["cast"].items():
            path = mdir / f"{cid}.png"
            if path.is_file():
                masters[cid] = path
                continue
            prompt = (
                f"{ks.series()['style']['image']} Character reference portrait: {c['identity']}. Full body, "
                "standing, facing the viewer, friendly expression, centered, plain warm cream paper background, "
                "nothing else in the picture."
            )
            checks = [
                "The image is a soft watercolor children's book illustration (not a photo, not 3D render).",
                f"{c['identity']} is clearly visible.",
                "There is exactly one character in the image.",
                "There is no readable text, lettering, numbers, or signs anywhere in the image.",
                "The background is plain with nothing else in it.",
            ]
            best = None
            for k in range(4):
                cand = mdir / "candidates" / f"{cid}-{k}.png"
                cand.parent.mkdir(exist_ok=True)
                if not self._generate(prompt, [], cand, 1000 + k, {**inputs, "steps": 20}):
                    continue
                v = judge(vqa, cand, checks)
                log.append(f"master {cid} candidate {k}: score {v['score']}")
                if v["hard_ok"] and (best is None or v["score"] > best[0]):
                    best = (v["score"], cand)
                if best and best[0] == 1.0:
                    break
            if not best:
                raise RuntimeError(f"could not mint a clean reference portrait for {cid}")
            path.write_bytes(best[1].read_bytes())
            masters[cid] = path
            log.append(f"master {cid}: minted from {best[1].name} (score {best[0]})")
        return masters

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import os

        from lib import kids_science as ks
        from lib import lmstudio

        start = time.time()
        project = Path(inputs["project_dir"])
        plan = json.loads((project / "artifacts" / "scene_plan.json").read_text(encoding="utf-8"))
        shots = [ks.ShotSpec(**s) for s in plan["metadata"]["shots"]]
        vqa = inputs.get("vqa_model") or os.environ.get("KIDS_VQA_MODEL", VQA_MODEL)
        bible_path = project / "artifacts" / "story_bible.json"
        exp_does = ks.experiment(json.loads(bible_path.read_text())["experiment_id"])["observe"] if bible_path.is_file() else ""
        reviewer = inputs.get("review_model") or os.environ.get("KIDS_REVIEW_MODEL", REVIEW_MODEL)
        attempts = int(inputs.get("max_attempts", 3))
        rng = random.Random(inputs.get("seed", 7))
        img_dir = project / "assets" / "images"
        (img_dir / "raw").mkdir(parents=True, exist_ok=True)
        log: list[str] = []
        report: dict[str, Any] = {}

        try:
            masters = self.ensure_masters(inputs, vqa, log)
            accepted: dict[str, Path] = {}
            prev_plate: Path | None = None
            beat_plate: dict[str, str] = {}  # one plate image per beat, reused by its other plate shots
            for shot in shots:
                if shot.kind == "plate" and shot.beat in beat_plate:
                    accepted[shot.id] = accepted[beat_plate[shot.beat]]
                    report[shot.id] = {**report[beat_plate[shot.beat]], "reused": beat_plate[shot.beat]}
                    continue
                checks = shot.checklist()
                review = shot.review_checks(exp_does=exp_does)
                tries = []
                # How to make each attempt: plates edit the previous plate first
                # (layout and counts carry over), then fall back to drawing fresh.
                if shot.kind == "plate":
                    plan_ = [("edit", [prev_plate], shot.edit_prompt())] * attempts if prev_plate else []
                    # The chain's first plate has nothing to edit from: give it more tries.
                    plan_ += [("fresh", [], shot.prompt())] * (2 if prev_plate else attempts * 2)
                else:
                    refs = [masters[c] for c in shot.characters if c in masters]
                    plan_ = [("fresh", refs, shot.prompt())] * attempts
                for a, (how, refs, prompt) in enumerate(plan_):
                    seed = rng.randrange(1, 2**31)
                    raw = img_dir / "raw" / f"{shot.id}-a{a + 1}.png"
                    if not self._generate(prompt, refs, raw, seed, inputs):
                        tries.append({"attempt": a + 1, "how": how, "seed": seed, "error": "generation failed"})
                        continue
                    v = judge(vqa, raw, checks)
                    r = judge(reviewer, raw, review) if v["hard_ok"] else {"hard_ok": False, "score": 0.0, "checks": []}
                    both = {"hard_ok": v["hard_ok"] and r["hard_ok"], "score": round((v["score"] + r["score"]) / 2, 3),
                            "checks": v["checks"], "review": r["checks"]}
                    tries.append({"attempt": a + 1, "how": how, "seed": seed, "path": str(raw), **both})
                    if both["hard_ok"] and both["score"] == 1.0:
                        break
                good = [t for t in tries if t.get("hard_ok")]
                entry: dict[str, Any] = {"beat": shot.beat, "kind": shot.kind, "words": shot.words, "attempts": tries}
                if good:
                    best = max(good, key=lambda t: t["score"])
                    accepted[shot.id] = Path(best["path"])
                    entry.update(status="accepted" if best["score"] == 1.0 else "accepted_best", score=best["score"],
                                 source=best["path"])
                    if shot.kind == "plate":
                        prev_plate = accepted[shot.id]
                        beat_plate[shot.beat] = shot.id
                else:
                    entry.update(status="needs_fallback")
                report[shot.id] = entry
                log.append(f"{shot.id} ({shot.kind}): {entry['status']} after {len(tries)} attempt(s)")
        except (lmstudio.LMStudioError, RuntimeError) as exc:
            lmstudio.unload_all()
            (project / "artifacts" / "keyframe_qa.json").write_text(json.dumps({"log": log, "shots": report}, indent=2), encoding="utf-8")
            return ToolResult(success=False, error=f"keyframes failed: {exc}")

        # Second pass: a shot that failed gets fresh seeds before anything is borrowed.
        try:
            for shot in shots:
                if shot.id in accepted or report.get(shot.id, {}).get("status") != "needs_fallback" or shot.kind == "plate":
                    continue
                refs = [masters[c] for c in shot.characters if c in masters]
                for a in range(attempts):
                    seed = rng.randrange(1, 2**31)
                    raw = img_dir / "raw" / f"{shot.id}-b{a + 1}.png"
                    if not self._generate(shot.prompt(), refs, raw, seed, inputs):
                        continue
                    v = judge(vqa, raw, shot.checklist())
                    r = judge(reviewer, raw, shot.review_checks(exp_does=exp_does)) if v["hard_ok"] else {"hard_ok": False, "score": 0.0, "checks": []}
                    report[shot.id]["attempts"].append({"attempt": f"b{a + 1}", "seed": seed, "path": str(raw),
                                                        "hard_ok": v["hard_ok"] and r["hard_ok"],
                                                        "score": round((v["score"] + r["score"]) / 2, 3)})
                    if v["hard_ok"] and r["hard_ok"]:
                        accepted[shot.id] = raw
                        report[shot.id].update(status="accepted_second_pass", source=str(raw))
                        log.append(f"{shot.id}: accepted on second pass")
                        break
        except lmstudio.LMStudioError as exc:
            log.append(f"second pass stopped: {exc}")
        finally:
            lmstudio.unload_all()

        # Fallbacks only ever borrow pictures that cannot contradict the plot:
        # anything accepted in the same beat, else the nearest accepted
        # character shot elsewhere (reactions and scenes never show the setup's
        # state wrongly — reactions keep it out of frame, scenes predate it).
        order = {s.id: i for i, s in enumerate(shots)}
        for shot in shots:
            if shot.id in accepted:
                continue
            same = [s for s in shots if s.beat == shot.beat and s.id in accepted]
            pool = same or sorted(
                (s for s in shots if s.id in accepted and s.kind == "reaction"
                 or (s.id in accepted and s.kind == "scene" and shot.kind == "scene")),
                key=lambda s: abs(order[s.id] - order[shot.id]))
            if not pool:
                (project / "artifacts" / "keyframe_qa.json").write_text(json.dumps({"log": log, "shots": report}, indent=2), encoding="utf-8")
                return ToolResult(success=False, error=f"no safe picture available for '{shot.id}' — see artifacts/keyframe_qa.json")
            prev = next((accepted.get(s.id) for s in reversed(shots[: order[shot.id]]) if s.id in accepted), None)
            choice = next((p for p in pool if accepted[p.id] != prev), pool[0])  # avoid a repeated frame
            accepted[shot.id] = accepted[choice.id]
            report[shot.id].update(status="fallback", source=str(accepted[shot.id]), borrowed_from=choice.id)

        # Final stills: step beats get an exact, drawn badge.
        step_label = {b: f"Step {i}" for i, b in enumerate(ks.STEP_BEATS, 1)}
        assets = []
        seen: dict[str, int] = {}
        for shot in shots:
            final = img_dir / f"{shot.id}.png"
            src = accepted[shot.id]
            n = seen.get(str(src), 0)
            seen[str(src)] = n + 1
            if n:  # second+ use of the same approved picture: a different detail of it
                cropped = img_dir / "raw" / f"{shot.id}-detail.png"
                detail_crop(src, cropped, n - 1)
                src = cropped
                report[shot.id]["detail_crop"] = n - 1
            if shot.beat in step_label:
                step_badge(src, step_label[shot.beat], final)
            else:
                final.write_bytes(src.read_bytes())
            assets.append(asset_entry(shot.id, final, report[shot.id]))
        (project / "artifacts" / "asset_manifest.json").write_text(json.dumps({"version": "1.0", "assets": assets}, indent=2), encoding="utf-8")
        counts: dict[str, int] = {}
        for e in report.values():
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        (project / "artifacts" / "keyframe_qa.json").write_text(json.dumps({"counts": counts, "log": log, "shots": report}, indent=2), encoding="utf-8")
        return ToolResult(
            success=True,
            data={"shots": len(shots), "counts": counts,
                  "generations": sum(len(e["attempts"]) for e in report.values())},
            artifacts=[str(project / "artifacts" / "asset_manifest.json"), str(project / "artifacts" / "keyframe_qa.json")],
            duration_seconds=round(time.time() - start, 1),
        )
