"""Local ensemble visual QA via LM Studio (OpenAI-compatible API).

Fans one image + spec out to a panel of locally-served vision models
(lineage-diverse ensemble), collects forced-JSON verdicts at temperature 0,
majority-votes PASS/FAIL, and flags disagreements for agent escalation.
Text-only review mode routes to QA_TEXT_MODEL.

Env config:
  LMSTUDIO_BASE_URL  (default http://localhost:1234/v1)
  QA_VISION_MODELS   comma list (default qwen/qwen3-vl-30b,qwen/qwen2.5-vl-7b)
  QA_TEXT_MODEL      (default openai/gpt-oss-120b)
  QA_VOTE_THRESHOLD  min PASS votes to pass (default majority)
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolStability,
    ToolTier,
)

LENS_PROMPTS = {
    "anatomy": "Count and verify every physical attribute in the spec exactly (limbs, eyes, treads/wheels, antennae, duplicates). Judge only what is visible.",
    "scene_intent": "Judge whether the image matches the described scene, mood, lighting, and style. Ignore minor anatomy details.",
    "violations": "Hunt ONLY for hard violations: humans present, wrong/duplicate creatures, forbidden elements, wrong art style (photoreal/2D-anime when 3D is required).",
}


def _base_url() -> str:
    return os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")


def _chat(model: str, messages: list, timeout: int) -> str:
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 300,
                       "messages": messages}).encode()
    req = urllib.request.Request(f"{_base_url()}/chat/completions", body,
                                 {"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return resp["choices"][0]["message"]["content"]


def _parse_verdict(text: str) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        v = json.loads(text[start:end])
        return {"verdict": str(v.get("verdict", "FAIL")).upper(),
                "reason": str(v.get("reason", ""))[:300]}
    except Exception:
        up = text.upper()
        return {"verdict": "PASS" if "PASS" in up and "FAIL" not in up else "FAIL",
                "reason": text.strip()[:300]}


class LocalVQA(BaseTool):
    name = "local_vqa"
    version = "0.1.0"
    tier = ToolTier.ANALYZE
    capability = "analysis"
    provider = "lmstudio"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime_env = "local"

    install_instructions = (
        "Run LM Studio with its local server on :1234 and load vision models.\n"
        "Config: LMSTUDIO_BASE_URL, QA_VISION_MODELS (comma list), QA_TEXT_MODEL, QA_VOTE_THRESHOLD."
    )
    fallback_tools = ["video_understand"]
    agent_skills: list[str] = []
    capabilities = ["ensemble_image_qa", "text_review"]
    supports = {"offline": True, "ensemble_voting": True, "escalation_flag": True}
    best_for = [
        "zero-token repetitive frame/keyframe QA with lineage-diverse local VLM panel",
        "pre-generation prompt critique and text review via local LLM",
    ]
    not_good_for = ["final creative judgment (escalate disagreements to the agent)"]

    input_schema = {
        "type": "object",
        "required": ["spec"],
        "properties": {
            "image_path": {"type": "string", "description": "Image to QA (omit for text-only review)"},
            "spec": {"type": "string", "description": "The requirements to judge against"},
            "lens": {"type": "string", "enum": ["anatomy", "scene_intent", "violations", "all"], "default": "all"},
            "models": {"type": "array", "items": {"type": "string"}, "description": "Override QA_VISION_MODELS"},
            "text": {"type": "string", "description": "Text to review (text-only mode)"},
            "timeout_seconds": {"type": "integer", "default": 180},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=10, network_required=False)
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["image_path", "spec", "lens"]
    side_effects: list[str] = []
    user_visible_verification = ["Spot-check a sample of ensemble verdicts against your own judgment"]

    def get_status(self):
        from tools.base_tool import ToolStatus
        try:
            urllib.request.urlopen(f"{_base_url()}/models", timeout=3)
            return ToolStatus.AVAILABLE
        except Exception:
            return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 60.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        timeout = int(inputs.get("timeout_seconds", 180))
        spec = inputs["spec"]

        # text-only review mode
        if not inputs.get("image_path"):
            model = os.environ.get("QA_TEXT_MODEL", "openai/gpt-oss-120b")
            try:
                out = _chat(model, [{"role": "user", "content":
                    f"Review the following against this spec. Reply JSON {{\"verdict\":\"PASS|FAIL\",\"reason\":\"...\"}}.\nSPEC: {spec}\nCONTENT:\n{inputs.get('text','')}"}], timeout)
            except Exception as e:
                return ToolResult(success=False, error=f"text review failed: {e}")
            v = _parse_verdict(out)
            return ToolResult(success=True, data={"mode": "text", "model": model, **v},
                              duration_seconds=round(time.time() - start, 1))

        img_path = Path(inputs["image_path"])
        if not img_path.is_file():
            return ToolResult(success=False, error=f"image not found: {img_path}")
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        img_block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

        models = inputs.get("models") or [m.strip() for m in os.environ.get(
            "QA_VISION_MODELS", "qwen/qwen3-vl-30b,qwen/qwen2.5-vl-7b").split(",") if m.strip()]
        lens = inputs.get("lens", "all")
        lenses = list(LENS_PROMPTS) if lens == "all" else [lens]

        verdicts = []
        for i, model in enumerate(models):
            lens_key = lenses[i % len(lenses)]
            prompt = (f"{LENS_PROMPTS[lens_key]}\nSPEC: {spec}\n"
                      'Reply ONLY JSON: {"verdict":"PASS|FAIL","reason":"one short sentence"}')
            try:
                out = _chat(model, [{"role": "user", "content": [{"type": "text", "text": prompt}, img_block]}], timeout)
                v = _parse_verdict(out)
            except Exception as e:
                v = {"verdict": "ERROR", "reason": str(e)[:200]}
            verdicts.append({"model": model, "lens": lens_key, **v})

        valid = [v for v in verdicts if v["verdict"] in ("PASS", "FAIL")]
        passes = sum(1 for v in valid if v["verdict"] == "PASS")
        thresh = int(os.environ.get("QA_VOTE_THRESHOLD", 0)) or (len(valid) // 2 + 1)
        overall = "PASS" if valid and passes >= thresh else "FAIL"
        unanimous = len({v["verdict"] for v in valid}) <= 1
        return ToolResult(
            success=True,
            data={"mode": "ensemble_image", "overall": overall,
                  "votes": f"{passes}/{len(valid)} pass (threshold {thresh})",
                  "needs_escalation": (not unanimous) or not valid,
                  "verdicts": verdicts, "image": str(img_path)},
            duration_seconds=round(time.time() - start, 1),
        )
