"""Batch Kokoro synthesis — runs INSIDE the mlx-audio venv, not OpenMontage's.

Usage: <mlx-audio python> _kokoro_batch.py jobs.json
jobs.json: {"model": "...", "jobs": [{"text", "voice", "lang_code", "speed", "out"}]}

Loads the model once for every line (a fresh CLI call per line reloads it).
Prints one JSON object per finished job so the caller can stream progress.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model

    model = load_model(model_path=spec["model"])
    for job in spec["jobs"]:
        out = Path(job["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        generate_audio(
            text=job["text"],
            model=model,
            voice=job["voice"],
            lang_code=job["lang_code"],
            speed=float(job["speed"]),
            output_path=str(out.parent),
            file_prefix=out.stem,
            audio_format="wav",
            join_audio=True,
            verbose=False,
        )
        print(json.dumps({"out": str(out), "ok": out.is_file()}), flush=True)


if __name__ == "__main__":
    main()
