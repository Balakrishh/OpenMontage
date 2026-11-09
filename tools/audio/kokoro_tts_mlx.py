"""Kokoro-82M text-to-speech on Apple Silicon via mlx-audio.

Natural, warm narration voices at ~2 s per line once the model is cached —
a clear step up from Piper for story-time narration. mlx-audio lives in its
own venv (default ~/mlx-audio/.venv) so its MLX/torch pins never touch
OpenMontage's environment; this tool shells out to it.

Batch mode synthesizes many lines with one model load.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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
    ToolStatus,
    ToolTier,
)

DEFAULT_PYTHON = os.path.expanduser("~/mlx-audio/.venv/bin/python")
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"
HELPER = Path(__file__).with_name("_kokoro_batch.py")


class KokoroTTSMLX(BaseTool):
    name = "kokoro_tts_mlx"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "kokoro"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = (
        "uv venv --python 3.12 ~/mlx-audio/.venv && "
        "uv pip install --python ~/mlx-audio/.venv/bin/python mlx-audio 'misaki[en]'. "
        "Override with KOKORO_PYTHON / KOKORO_MODEL."
    )
    fallback_tools = ["piper_tts"]
    agent_skills: list[str] = []
    capabilities = ["text_to_speech", "batch_text_to_speech"]
    supports = {"offline": True, "voice_selection": True, "speed": True, "batch": True}
    best_for = ["warm read-aloud narration, local and free", "many short lines in one call"]
    not_good_for = ["voice cloning", "child voices (Kokoro voices are adult)"]

    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "output_path": {"type": "string"},
            "voice": {"type": "string", "default": "af_heart"},
            "lang_code": {"type": "string", "default": "a", "description": "a = American, b = British English"},
            "speed": {"type": "number", "default": 1.0},
            "jobs": {
                "type": "array",
                "description": "Batch mode: [{text, output_path, voice, lang_code, speed}]",
                "items": {"type": "object", "required": ["text", "output_path"]},
            },
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=2000, vram_mb=0, disk_mb=400, network_required=False)
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["text", "voice", "speed"]
    side_effects = ["writes WAV files", "downloads the Kokoro model on first run"]
    user_visible_verification = ["Listen to a line; transcribe it to confirm no words were skipped"]

    def _python(self) -> str:
        return os.environ.get("KOKORO_PYTHON", DEFAULT_PYTHON)

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if Path(self._python()).is_file() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 10.0 + 2.5 * len(inputs.get("jobs") or [1])

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="mlx-audio venv not found. " + self.install_instructions)
        raw = inputs.get("jobs") or [inputs]
        jobs = []
        for job in raw:
            if not job.get("text") or not job.get("output_path"):
                return ToolResult(success=False, error="each job needs text and output_path")
            out = Path(job["output_path"]).resolve()
            out.unlink(missing_ok=True)  # freshness: a stale file must not pass as output
            jobs.append({
                "text": job["text"], "out": str(out),
                "voice": job.get("voice", "af_heart"), "lang_code": job.get("lang_code", "a"),
                "speed": job.get("speed", 1.0),
            })
        start = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"model": os.environ.get("KOKORO_MODEL", DEFAULT_MODEL), "jobs": jobs}, fh)
            spec = fh.name
        try:
            proc = subprocess.run(
                [self._python(), str(HELPER), spec], capture_output=True, text=True,
                timeout=120 + 30 * len(jobs),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Kokoro batch timed out")
        finally:
            Path(spec).unlink(missing_ok=True)
        missing = [j["out"] for j in jobs if not Path(j["out"]).is_file()]
        if proc.returncode != 0 or missing:
            tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
            return ToolResult(success=False, error=f"Kokoro failed ({len(missing)} missing): {tail}")
        return ToolResult(
            success=True,
            data={"outputs": [j["out"] for j in jobs], "count": len(jobs)},
            artifacts=[j["out"] for j in jobs],
            duration_seconds=round(time.time() - start, 2),
            model=os.environ.get("KOKORO_MODEL", DEFAULT_MODEL),
        )
