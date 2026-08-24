"""LTX-2.3 local video generation via MLX (Apple Silicon).

Wraps the `ltx-2-mlx` CLI (installed in its own venv, default
~/ltx-2-mlx/.venv/bin/ltx-2-mlx) running the quantized
`dgrauet/ltx-2.3-mlx-q8` build. Far lighter and faster than the diffusers
LTX-2 path on Apple Silicon (q8 quantization + distilled transformer).

CLI contract (positive prompts only — the runner has no negative-prompt flag):
  ltx-2-mlx generate --model dgrauet/ltx-2.3-mlx-q8
      --width W --height H --frames N --frame-rate FPS
      [--image keyframe.png] --output out.mp4 --prompt "..." --seed S
      draft:  --distilled
      final:  --two-stage --enable-teacache --teacache-thresh 0.05
              --stage1-steps A --stage2-steps B --cfg-scale C

Dims must be multiples of 64 (two-stage snaps to /64); frames = 8n+1.
The model renders in-clip audio (often speech-like gibberish) — downstream
compose should strip or duck it; this tool reports native_audio=True so the
edit stage knows the track exists.
"""

from __future__ import annotations

import os
import shutil
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

DEFAULT_CLI = os.path.expanduser("~/ltx-2-mlx/.venv/bin/ltx-2-mlx")
DEFAULT_MODEL = "dgrauet/ltx-2.3-mlx-q8"


def _cli_path() -> str:
    return os.environ.get("LTX_MLX_CLI", DEFAULT_CLI)


class LTXVideoMLX(BaseTool):
    name = "ltx_video_mlx"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "ltx-mlx"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = (
        "Install the ltx-2-mlx CLI in its own venv (Apple Silicon only):\n"
        "  git clone https://github.com/dgrauet/ltx-2-mlx ~/ltx-2-mlx\n"
        "  cd ~/ltx-2-mlx && python -m venv .venv && .venv/bin/pip install -e .\n"
        "Override the CLI location with LTX_MLX_CLI. The model "
        f"({DEFAULT_MODEL}) downloads from HuggingFace on first run."
    )
    fallback = "ltx_video_local"
    fallback_tools = ["ltx_video_local", "wan_video", "hunyuan_video"]
    agent_skills = ["ltx2"]

    capabilities = ["text_to_video", "image_to_video"]
    supports = {
        "reference_image": True,
        "offline": True,
        "native_audio": True,
        "local_gpu": True,
        "negative_prompt": False,
    }
    best_for = [
        "LTX-2.3 quality on Apple Silicon at workable speed (q8 + distilled)",
        "image-to-video with a character reference keyframe",
        "long clips (up to 481 frames / 20s) that the diffusers path cannot fit",
    ]
    not_good_for = ["non-Apple-Silicon machines", "prompts that rely on negative prompting"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {"type": "string", "enum": ["text_to_video", "image_to_video"], "default": "text_to_video"},
            "reference_image_path": {"type": "string"},
            "width": {"type": "integer", "default": 768, "description": "multiple of 64"},
            "height": {"type": "integer", "default": 512, "description": "multiple of 64"},
            "num_frames": {"type": "integer", "default": 121, "description": "8n+1"},
            "fps": {"type": "integer", "default": 24},
            "quality": {"type": "string", "enum": ["draft", "final"], "default": "final",
                        "description": "draft=--distilled single pass; final=--two-stage + teacache"},
            "stage1_steps": {"type": "integer", "default": 12},
            "stage2_steps": {"type": "integer", "default": 8},
            "cfg_scale": {"type": "number", "default": 4.0},
            "teacache_thresh": {"type": "number", "default": 0.05},
            "seed": {"type": "integer"},
            "timeout_seconds": {"type": "integer", "default": 5400},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=4, ram_mb=32000, vram_mb=0, disk_mb=30000, network_required=False)
    retry_policy = RetryPolicy(max_retries=1)
    idempotency_key_fields = ["prompt", "operation", "seed", "width", "height", "num_frames", "quality"]
    side_effects = ["writes video file to output_path", "may download model weights on first run"]
    user_visible_verification = [
        "Watch generated clip for motion coherence and artifacts",
        "In-clip audio is model-generated gibberish — mute or duck it at compose",
    ]

    def get_status(self) -> ToolStatus:
        if shutil.which(_cli_path()) or Path(_cli_path()).is_file():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, object]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, object]) -> float:
        return 600.0 if inputs.get("quality", "final") == "final" else 240.0

    def execute(self, inputs: dict[str, object]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="ltx-2-mlx CLI not found. " + self.install_instructions)

        width = int(inputs.get("width", 768))
        height = int(inputs.get("height", 512))
        if width % 64 or height % 64:
            return ToolResult(success=False, error=f"width/height must be multiples of 64, got {width}x{height}")
        num_frames = int(inputs.get("num_frames", 121))
        if (num_frames - 1) % 8:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
        fps = int(inputs.get("fps", 24))

        output_path = Path(str(inputs.get("output_path", "ltx_mlx_output.mp4")))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            _cli_path(), "generate", "--model", os.environ.get("LTX_MLX_MODEL", DEFAULT_MODEL),
            "--width", str(width), "--height", str(height),
            "--frames", str(num_frames), "--frame-rate", str(fps),
            "--output", str(output_path), "--prompt", str(inputs["prompt"]),
        ]

        operation = inputs.get("operation", "text_to_video")
        if operation == "image_to_video":
            ref = inputs.get("reference_image_path")
            if not ref or not Path(str(ref)).is_file():
                return ToolResult(success=False, error="image_to_video requires an existing reference_image_path")
            cmd += ["--image", str(ref)]

        quality = inputs.get("quality", "final")
        if quality == "draft":
            cmd += ["--distilled"]
        else:
            cmd += [
                "--two-stage", "--enable-teacache",
                "--teacache-thresh", str(inputs.get("teacache_thresh", 0.05)),
                "--stage1-steps", str(inputs.get("stage1_steps", 12)),
                "--stage2-steps", str(inputs.get("stage2_steps", 8)),
                "--cfg-scale", str(inputs.get("cfg_scale", 4.0)),
            ]
        if inputs.get("seed") is not None:
            cmd += ["--seed", str(inputs["seed"])]

        start = time.time()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=int(inputs.get("timeout_seconds", 5400)),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"ltx-2-mlx timed out after {inputs.get('timeout_seconds', 5400)}s")
        elapsed = round(time.time() - start, 2)

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return ToolResult(success=False, error=f"ltx-2-mlx failed (exit {proc.returncode}): {tail}")
        # Freshness guard: require the output to have been (re)written this run,
        # not merely to exist from an earlier attempt.
        if not output_path.is_file() or output_path.stat().st_mtime < start:
            return ToolResult(success=False, error=f"ltx-2-mlx exited 0 but did not write {output_path} (stale or missing output)")

        result = ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "provider_name": "LTX-2.3 (MLX q8, local)",
                "mode": "local_mlx",
                "model": os.environ.get("LTX_MLX_MODEL", DEFAULT_MODEL),
                "operation": operation,
                "quality": quality,
                "prompt": inputs["prompt"],
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "duration_seconds": round(num_frames / fps, 2),
                "generation_wall_seconds": elapsed,
                "output": str(output_path),
                "format": "mp4",
                "has_native_audio": True,
                "license": "LTX-2-Community",
            },
            artifacts=[str(output_path)],
            seed=inputs.get("seed"),
            model=os.environ.get("LTX_MLX_MODEL", DEFAULT_MODEL),
        )
        result.duration_seconds = elapsed
        return result
