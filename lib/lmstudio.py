"""Minimal LM Studio client: schema-constrained JSON chat, image input, model residency.

LM Studio serves MLX models behind an OpenAI-compatible API. Two things make
it the right backend for unattended production on Apple Silicon:

- ``response_format: json_schema`` is enforced by constrained decoding, so a
  reply is always parseable and enum fields can only hold allowed values.
- Models load on demand (JIT) and can be unloaded between stages, so a 63 GB
  critic never sits in memory while FLUX needs the GPU.

Config: LMSTUDIO_BASE_URL (default http://localhost:1234/v1), LMS_CLI
(default ~/.lmstudio/bin/lms).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


class LMStudioError(RuntimeError):
    pass


def base_url() -> str:
    return os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")


def lms_cli() -> str:
    return os.environ.get("LMS_CLI", os.path.expanduser("~/.lmstudio/bin/lms"))


def available() -> bool:
    try:
        urllib.request.urlopen(f"{base_url()}/models", timeout=3)
        return True
    except Exception:
        return False


def ensure_server() -> None:
    if available():
        return
    cli = lms_cli()
    if not Path(cli).is_file():
        raise LMStudioError(f"LM Studio server is not running and {cli} was not found")
    subprocess.run([cli, "server", "start"], capture_output=True, text=True, timeout=60)
    for _ in range(30):
        if available():
            return
        time.sleep(1)
    raise LMStudioError("LM Studio server did not start on " + base_url())


def unload_all() -> None:
    """Free unified memory for the next stage (FLUX, Whisper, Remotion)."""
    cli = lms_cli()
    if Path(cli).is_file():
        subprocess.run([cli, "unload", "--all"], capture_output=True, text=True, timeout=120)


def _image_block(path: Path) -> dict[str, Any]:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def chat_json(
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    temperature: float = 0.4,
    max_tokens: int = 4000,
    images: Optional[list[Path]] = None,
    timeout: int = 900,
    retries: int = 2,
) -> dict[str, Any]:
    """One schema-constrained completion, parsed. Raises LMStudioError on failure."""
    ensure_server()
    content: Any = user
    if images:
        content = [{"type": "text", "text": user}, *[_image_block(Path(p)) for p in images]]
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}},
    }
    last = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{base_url()}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
            )
            resp = json.load(urllib.request.urlopen(req, timeout=timeout))
            text = resp["choices"][0]["message"]["content"] or ""
            return json.loads(text[text.index("{"): text.rindex("}") + 1])
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read()[:300]!r}"
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * (attempt + 1))
    raise LMStudioError(f"{model} failed after {retries + 1} attempts — {last}")
