"""FLUX.2 Klein edit-model image generation via mflux (Apple Silicon).

Wraps the `mflux-generate-flux2-edit` CLI (~/.local/bin). Generates a new
image conditioned on one or more reference images — the character-consistency
keyframe mechanism: pass character identity-master frames as references and
describe the target composition in the prompt.

Models (cached locally): flux2-klein-9b (fast, 14 steps) and
flux2-klein-base-9b (detail, 28 steps). NOTE: FLUX.2 errors on
--negative-prompt — positive prompts only.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

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

DEFAULT_CLI = os.path.expanduser("~/.local/bin/mflux-generate-flux2-edit")


class Flux2EditMLX(BaseTool):
    name = "flux2_edit_mlx"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "mflux"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = (
        "Install mflux (Apple Silicon): pipx install mflux (or uv tool install mflux).\n"
        "Override CLI path with MFLUX_FLUX2_EDIT_CLI. Models download from HF on first run."
    )
    fallback = "local_diffusion"
    fallback_tools = ["local_diffusion"]
    agent_skills = ["flux-best-practices"]

    capabilities = ["image_edit", "reference_conditioned_generation"]
    supports = {"reference_image": True, "offline": True, "negative_prompt": False, "seed": True, "custom_size": True}
    best_for = [
        "character-consistent keyframes from identity-master reference frames",
        "scene composition while preserving referenced subjects",
    ]
    not_good_for = ["non-Apple-Silicon machines", "negative-prompt workflows"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "reference_image_paths"],
        "properties": {
            "prompt": {"type": "string"},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "base_model": {"type": "string", "enum": ["flux2-klein-9b", "flux2-klein-base-9b", "flux2-klein-4b"], "default": "flux2-klein-9b"},
            "width": {"type": "integer", "default": 768},
            "height": {"type": "integer", "default": 512},
            "steps": {"type": "integer", "default": 14},
            "guidance": {"type": "number"},
            "seed": {"type": "integer"},
            "timeout_seconds": {"type": "integer", "default": 900},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=24000, vram_mb=0, disk_mb=20000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["prompt", "seed", "width", "height", "base_model"]
    side_effects = ["writes image file to output_path", "may download model weights on first run"]
    user_visible_verification = ["Check the keyframe keeps the referenced character on-model"]

    def get_status(self) -> ToolStatus:
        cli = os.environ.get("MFLUX_FLUX2_EDIT_CLI", DEFAULT_CLI)
        return ToolStatus.AVAILABLE if Path(cli).is_file() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, object]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, object]) -> float:
        return 120.0

    def execute(self, inputs: dict[str, object]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="mflux-generate-flux2-edit not found. " + self.install_instructions)
        refs = [str(p) for p in inputs["reference_image_paths"]]
        for r in refs:
            if not Path(r).is_file():
                return ToolResult(success=False, error=f"reference image missing: {r}")
        output_path = Path(str(inputs.get("output_path", "flux2_edit_output.png")))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cli = os.environ.get("MFLUX_FLUX2_EDIT_CLI", DEFAULT_CLI)
        cmd = [
            cli, "--base-model", str(inputs.get("base_model", "flux2-klein-9b")),
            "--image-paths", *refs,
            "--prompt", str(inputs["prompt"]),
            "--width", str(inputs.get("width", 768)), "--height", str(inputs.get("height", 512)),
            "--steps", str(inputs.get("steps", 14)),
            "--output", str(output_path),
        ]
        if inputs.get("guidance") is not None:
            cmd += ["--guidance", str(inputs["guidance"])]
        if inputs.get("seed") is not None:
            cmd += ["--seed", str(inputs["seed"])]

        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=int(inputs.get("timeout_seconds", 900)))
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="mflux-generate-flux2-edit timed out")
        elapsed = round(time.time() - start, 2)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return ToolResult(success=False, error=f"mflux failed (exit {proc.returncode}): {tail}")
        # Freshness guard: mflux can exit 0 without writing when the output
        # already exists — require the file to have been (re)written this run.
        if not output_path.is_file() or output_path.stat().st_mtime < start:
            return ToolResult(success=False, error=f"mflux exited 0 but did not write {output_path} (stale or missing output — delete any pre-existing file and retry)")

        result = ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "base_model": inputs.get("base_model", "flux2-klein-9b"),
                "prompt": inputs["prompt"],
                "references": refs,
                "output": str(output_path),
                "generation_wall_seconds": elapsed,
            },
            artifacts=[str(output_path)],
            seed=inputs.get("seed"),
            model=str(inputs.get("base_model", "flux2-klein-9b")),
        )
        result.duration_seconds = elapsed
        return result
