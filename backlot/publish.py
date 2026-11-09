"""Publish action — the board's one deliberate write path.

Backlot's contract is observation: "all state derives from files the pipeline
already writes", and the server "never writes to project directories". This
module is the single, scoped exception, and it is written to bend that rule as
little as possible:

- The server still writes nothing itself. It invokes the `youtube_upload` tool,
  and the TOOL writes `artifacts/publish_log.json` / `checkpoint_publish.json` —
  the same files the publish stage always produced. The shape of project state
  is unchanged; only the trigger moved from the agent to a human's click.
- The board learns the result the same way it learns everything else: the
  watcher sees those files change and pushes an SSE `change`. No push channel,
  no server-held UI state.
- The publish stage is `human_approval_default: true`. The button does not
  bypass that gate — it *is* the gate, moved to where the human already is.
  `confirm: true` in the body is required, so a stray request cannot publish.

Everything here is synchronous and in-process: one board, one machine, one
upload at a time.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

# One upload per project at a time. A second click while an upload is running
# gets a clear 409 instead of a duplicate video on the channel.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class PublishError(Exception):
    """Bad request from the board (mapped to 4xx). Message is user-facing."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _lock_for(project_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(project_id, threading.Lock())


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def resolve_render(project_dir: Path, rel_path: str) -> Path:
    """Project-relative video path → absolute, refusing anything outside."""
    if not rel_path:
        raise PublishError("no render path given")
    target = (project_dir / rel_path).resolve()
    try:
        target.relative_to(project_dir.resolve())
    except ValueError:
        raise PublishError("path escapes project", status_code=403) from None
    if not target.is_file():
        raise PublishError(f"render not found: {rel_path}", status_code=404)
    if target.suffix.lower() not in {".mp4", ".mov", ".webm"}:
        raise PublishError(f"not a video file: {rel_path}")
    return target


def resolve_thumbnail(project_dir: Path, rel_path: str) -> Path:
    """Project-relative thumbnail → absolute, holding it to YouTube's rules."""
    target = (project_dir / rel_path).resolve()
    try:
        target.relative_to(project_dir.resolve())
    except ValueError:
        raise PublishError("thumbnail path escapes project", status_code=403) from None
    if not target.is_file():
        raise PublishError(f"thumbnail not found: {rel_path}", status_code=404)
    if target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise PublishError(f"thumbnail must be PNG or JPEG: {rel_path}")
    if target.stat().st_size > 2 * 1024 * 1024:
        raise PublishError(f"thumbnail is over YouTube's 2 MB limit: {rel_path}")
    return target


def set_thumbnail(project_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Put a thumbnail on a video that is already on YouTube. Requires explicit confirmation."""
    from tools.publishers.youtube_upload import YouTubeUpload

    if not isinstance(body, dict):
        raise PublishError("expected a JSON object")
    if body.get("confirm") is not True:
        raise PublishError("setting a thumbnail requires an explicit confirmation")
    thumbnail = str(body.get("thumbnail_path") or "").strip()
    if not thumbnail:
        raise PublishError("a thumbnail is required")
    inputs: dict[str, Any] = {
        "operation": "set_thumbnail",
        "project_dir": str(project_dir),
        "thumbnail_path": str(resolve_thumbnail(project_dir, thumbnail)),
    }
    if body.get("video_id"):
        inputs["video_id"] = str(body["video_id"]).strip()
    elif body.get("path"):
        inputs["video_path"] = str(resolve_render(project_dir, str(body["path"])))
    else:
        raise PublishError("give the render path or a video_id")
    _load_env()
    lock = _lock_for(project_dir.name)
    if not lock.acquire(blocking=False):
        raise PublishError("a publish is already running for this project", status_code=409)
    try:
        result = YouTubeUpload().execute(inputs)
    finally:
        lock.release()
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, **(result.data or {})}


def suggest_metadata(project_dir: Path, video_path: Path) -> dict[str, Any]:
    """Best available publish copy for THIS render, in order of how considered it is.

    A hero film and its trailer are different videos: each tier is matched to
    the render first, so a trailer never pre-fills with the film's chapters.
    """
    from tools.publishers.youtube_upload import YouTubeUpload

    rel = YouTubeUpload._rel(project_dir, video_path)
    log = YouTubeUpload.read_publish_log(project_dir)
    meta = log.get("metadata") or {}

    # 1. A previous publish of this render — the human signed off on it once.
    uploads = (meta.get("youtube") or {}).get("uploads") or []
    ids = {u.get("video_id") for u in uploads if isinstance(u, dict) and u.get("source") == rel}
    for entry in reversed(log.get("entries") or []):
        used = (entry or {}).get("metadata_used") or {}
        if entry.get("video_id") in ids and used.get("title"):
            return {
                "title": used.get("title", ""),
                "description": used.get("description", ""),
                "tags": [t.lstrip("#") for t in (used.get("hashtags") or [])],
                "chapters": used.get("chapters") or [],
                "thumbnail_path": meta.get("suggested_thumbnail"),
                "contains_synthetic_media": meta.get("contains_synthetic_media"),
                "category_id": str(meta.get("suggested_category_id") or "24"),
                "source": "previous publish",
            }

    # 2. Copy the publish stage wrote: per render, then for the hero film.
    per_render = (meta.get("suggested_by_render") or {}).get(rel)
    is_hero = video_path.stem.lower() in {"final", "output", "master"} or str(
        meta.get("hero_output") or ""
    ).endswith(rel)
    if isinstance(per_render, dict) and per_render.get("title"):
        return _suggestion(per_render, meta, "publish stage suggestion (this render)")
    if meta.get("suggested_title"):
        suggestion = _suggestion(
            {
                "title": meta["suggested_title"],
                "description": meta.get("suggested_description"),
                "tags": meta.get("suggested_tags"),
                "chapters": meta.get("suggested_chapters") if is_hero else None,
                "thumbnail_path": meta.get("suggested_thumbnail") if is_hero else None,
            },
            meta,
            "publish stage suggestion",
        )
        if not is_hero:
            suggestion["title"] = f"{suggestion['title']} — {video_path.stem.replace('_', ' ')}"
        return suggestion

    # 3. An export bundle written by export_bundle (or by hand next to it).
    candidates = [
        project_dir / "exports" / "metadata" / "metadata.json",
        project_dir / "export" / "metadata" / "metadata.json",
        project_dir / "exports" / "metadata.json",
        project_dir / "export" / "metadata.json",
    ]
    for candidate in candidates:
        data = _read_json(candidate)
        if data and data.get("title"):
            return _suggestion(data, meta, f"export bundle ({candidate.name})")

    # 4. The project marker plus whatever logline the proposal carries.
    marker = _read_json(project_dir / "project.json") or {}
    proposal = _read_json(project_dir / "artifacts" / "proposal_packet.json") or {}
    logline = ""
    for key in ("logline", "premise", "synopsis", "summary"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            logline = value.strip()
            break
    title = marker.get("title") or project_dir.name.replace("-", " ").title()
    if video_path.stem.lower() not in {"final", "output", "master"}:
        title = f"{title} — {video_path.stem.replace('_', ' ')}"
    return _suggestion({"title": title, "description": logline}, meta, "project marker")


def _suggestion(data: dict[str, Any], meta: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "title": str(data.get("title") or ""),
        "description": str(data.get("description") or ""),
        "tags": [str(t) for t in (data.get("tags") or [])][:20],
        "chapters": data.get("chapters") or [],
        "thumbnail_path": data.get("thumbnail_path"),
        "contains_synthetic_media": meta.get("contains_synthetic_media"),
        "category_id": str(data.get("category_id") or meta.get("suggested_category_id") or "24"),
        "source": source,
    }


def preflight(project_dir: Path, rel_path: str) -> dict[str, Any]:
    """Everything the confirmation dialog needs — and no network calls.

    Deliberately never raises on a missing credential: the dialog has to be able
    to render "not configured yet, here is how" rather than an error toast.
    """
    from tools.publishers.youtube_upload import YouTubeUpload, YouTubeUploadError

    video_path = resolve_render(project_dir, rel_path)
    _load_env()
    tool = YouTubeUpload()

    missing_env = [
        dep[4:] for dep in tool.dependencies
        if dep.startswith("env:") and not _env_set(dep[4:])
    ]

    probe: dict[str, Any] = {}
    probe_error = None
    try:
        probe = tool.probe(video_path)
    except YouTubeUploadError as exc:
        probe_error = str(exc)

    already = None
    try:
        already = tool.find_existing_upload(project_dir, tool.fingerprint(video_path))
    except OSError:
        pass

    suggested = suggest_metadata(project_dir, video_path)
    hero = str((tool.read_publish_log(project_dir).get("metadata") or {}).get("hero_output") or "")
    hero_rel = hero.split(f"{project_dir.name}/", 1)[-1] if hero else None
    return {
        # The finished cut, when the publish stage named one. The dialog warns
        # when a different file (an older final_vN, a draft) is about to go up.
        "hero_render": hero_rel,
        "render": {
            "path": rel_path,
            "size_bytes": video_path.stat().st_size,
        },
        "probe": probe,
        "probe_error": probe_error,
        "is_short": tool.is_short(probe) if probe else False,
        "short_max_seconds": 60,
        "credentials_ready": not missing_env,
        "missing_env": missing_env,
        "setup_hint": tool.install_instructions,
        "estimated_runtime_seconds": tool.estimate_runtime({"video_path": str(video_path)}),
        "already_published": already,
        "suggested": suggested,
        "defaults": {"privacy": _default_privacy(), "category_id": "24", "made_for_kids": False},
    }


def _env_set(name: str) -> bool:
    import os

    return bool(os.environ.get(name, "").strip())


def _default_privacy() -> str:
    """The dialog's starting visibility: YOUTUBE_DEFAULT_PRIVACY, else unlisted.

    Unlisted stays the safe default for anyone who has not chosen; a channel
    owner who always launches public sets it once in .env.
    """
    import os

    value = os.environ.get("YOUTUBE_DEFAULT_PRIVACY", "").strip().lower()
    return value if value in {"public", "private", "unlisted"} else "unlisted"


def _load_env() -> None:
    """Pick up credentials from .env without a server restart.

    The server never runs registry discovery (which is what loads .env for
    agents), so without this the board reports "not configured" even after
    scripts/get_youtube_token.py has written the keys. Re-read on each call so
    a freshly minted token goes live on the next dialog open. Existing
    environment values win, matching the registry's loader.
    """
    from lib.env_loader import load_env

    load_env()


def publish(project_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Run the upload. Raises PublishError for anything the caller can fix."""
    from tools.publishers.youtube_upload import YouTubeUpload

    if not isinstance(body, dict):
        raise PublishError("expected a JSON object")
    if body.get("confirm") is not True:
        # The dialog sets this. Its absence means something other than a
        # human's confirmation reached this endpoint.
        raise PublishError("publish requires an explicit confirmation")

    video_path = resolve_render(project_dir, str(body.get("path") or ""))
    title = str(body.get("title") or "").strip()
    if not title:
        raise PublishError("a title is required")

    privacy = str(body.get("privacy") or "unlisted")
    if privacy not in {"public", "private", "unlisted"}:
        raise PublishError(f"invalid privacy: {privacy}")

    short = body.get("short", "auto")
    if short not in (True, False, "auto"):
        raise PublishError("short must be true, false, or 'auto'")

    tags = body.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    inputs = {
        "project_dir": str(project_dir),
        "video_path": str(video_path),
        "title": title,
        "description": str(body.get("description") or ""),
        "tags": [str(t) for t in tags][:20],
        "privacy": privacy,
        "category_id": str(body.get("category_id") or "24"),
        "short": short,
        "made_for_kids": bool(body.get("made_for_kids", False)),
        "chapters": body.get("chapters") or [],
        "dry_run": bool(body.get("dry_run", False)),
        "force": bool(body.get("force", False)),
        # A publish triggered from the board IS the human approval, so the
        # stage checkpoint should say so.
        "write_checkpoint": not bool(body.get("dry_run", False)),
    }
    if body.get("contains_synthetic_media") is not None:
        inputs["contains_synthetic_media"] = bool(body["contains_synthetic_media"])
    thumbnail = str(body.get("thumbnail_path") or "").strip()
    if thumbnail:
        inputs["thumbnail_path"] = str(resolve_thumbnail(project_dir, thumbnail))

    _load_env()
    lock = _lock_for(project_dir.name)
    if not lock.acquire(blocking=False):
        raise PublishError("a publish is already running for this project", status_code=409)
    try:
        result = YouTubeUpload().execute(inputs)
    finally:
        lock.release()

    if not result.success:
        # A failed upload is a real outcome, not a server fault: report it as
        # data so the dialog can show the message and offer a retry.
        return {"success": False, "error": result.error}
    return {
        "success": True,
        "duration_seconds": result.duration_seconds,
        **(result.data or {}),
    }
