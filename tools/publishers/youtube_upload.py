"""YouTube uploader — the first networked PUBLISH-tier tool.

`export_bundle` packages a finished render for a human to upload by hand; its
docstring says the obvious next step out loud: "A networked publisher (e.g. a
YouTube uploader) can be added later as a separate `provider` under the same
`publish` capability." This is that provider.

The upload itself is the YouTube Data API v3 **resumable** protocol driven with
plain `requests` — the same shape socialsmith's `engine/publish.py` uses (open a
session, PUT the bytes, best-effort thumbnail), ported here and hardened for
feature films rather than 60-second social clips:

- **Resume, not restart.** socialsmith PUTs the file once; a dropped connection
  190 MB into a 217-second hero film loses the whole upload. Here a failed PUT
  queries the session for its committed offset (`Content-Range: bytes */<size>`
  → `308` + `Range`) and continues from there, with bounded backoff.
- **Idempotency.** Every completed upload is recorded in the publish_log's
  free-form `metadata` block, keyed by content fingerprint (size + mtime). A
  second publish of the same bytes is refused unless `force=true`, so a
  double-clicked button cannot put the same film on the channel twice.
- **Shorts by measurement.** The render is probed with ffprobe; portrait and
  <= 60s (YouTube's actual Shorts ceiling, not socialsmith's 180s) means the
  `#Shorts` tag is appended. The caller can override the verdict either way.

Nothing here decides *whether* to publish. That is a human gate: the publish
stage is `human_approval_default: true`, and the board's button asks for an
explicit confirmation before it ever calls this tool.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ResumeSupport,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
THUMBNAIL_URL = "https://www.googleapis.com/upload/youtube/v3/thumbnails/set"
WATCH_URL = "https://www.youtube.com/watch?v="
SHORTS_URL = "https://www.youtube.com/shorts/"

# YouTube's real Shorts ceiling (socialsmith uses 180s; that is its own
# convention, not the platform's rule for the Shorts shelf).
SHORT_MAX_SECONDS = 60.0

TITLE_LIMIT = 100
SHORT_TITLE_LIMIT = 88  # leaves room for " #Shorts"
DESCRIPTION_LIMIT = 4900  # YouTube's cap is 5000; keep headroom
TAG_LIMIT = 20
TAG_CHARS_LIMIT = 500  # YouTube rejects the whole upload past this

# YouTube only renders chapters (the progress-bar timeline) when the
# description lists at least three, starting at 0:00, each >= 10s long.
CHAPTER_MIN_COUNT = 3
# Put this token in a description to place the timeline there instead of at the end.
CHAPTERS_PLACEHOLDER = "{chapters}"
CHAPTER_MIN_SECONDS = 10.0

UPLOAD_CHUNK_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3.0
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class YouTubeUploadError(RuntimeError):
    """Upload failed. The message is user-facing — it reaches the board."""


class YouTubeUpload(BaseTool):
    name = "youtube_upload"
    version = "0.1.0"
    tier = ToolTier.PUBLISH
    capability = "publish"
    provider = "youtube"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = [
        "env:YOUTUBE_CLIENT_ID",
        "env:YOUTUBE_CLIENT_SECRET",
        "env:YOUTUBE_REFRESH_TOKEN",
        "python:requests",
        "cmd:ffprobe",
    ]
    install_instructions = (
        "Create an OAuth 2.0 Desktop client in Google Cloud (YouTube Data API v3 enabled), "
        "then run `python scripts/get_youtube_token.py` once — it opens the consent screen, "
        "mints a refresh token, and writes YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET / "
        "YOUTUBE_REFRESH_TOKEN into .env."
    )

    agent_skills = []

    capabilities = ["upload_video", "upload_short", "set_thumbnail", "write_publish_log"]
    supports = {
        "local_offline": False,
        "free": True,  # no per-call charge; consumes daily API quota instead
        "uploads": True,
        "resumable": True,
    }
    best_for = [
        "publishing a finished render straight to a YouTube channel",
        "posting a vertical cut as a YouTube Short",
        "a one-click publish from the Backlot board, behind a human confirmation",
    ]
    not_good_for = [
        "cropping or reformatting a video (render a 9:16 cut first — this tool never re-encodes)",
        "writing SEO metadata (the publish-director skill prepares title/description/tags)",
        "bulk uploads — the default API quota allows roughly 6 uploads per day",
    ]

    input_schema = {
        "type": "object",
        "required": ["video_path", "title"],
        "properties": {
            "video_path": {"type": "string", "description": "Render to upload (must live inside project_dir)."},
            "title": {"type": "string", "description": f"Video title; truncated to {TITLE_LIMIT} chars."},
            "project_dir": {
                "type": "string",
                "description": "Project root. Enables Backlot events and publish_log writing.",
            },
            "description": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "privacy": {"type": "string", "enum": ["public", "private", "unlisted"], "default": "unlisted"},
            "category_id": {"type": "string", "description": "YouTube category id. Default 24 (Entertainment)."},
            "short": {
                "description": "true / false to force, or 'auto' (default) to decide by probing the file.",
                "anyOf": [{"type": "boolean"}, {"type": "string", "enum": ["auto"]}],
            },
            "chapters": {
                "description": (
                    "Timeline chapters appended to the description: a list of "
                    "{start_seconds, title}, or text with one 'M:SS Title' per line. "
                    "Ignored for Shorts."
                ),
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["start_seconds", "title"],
                            "properties": {"start_seconds": {"type": "number"}, "title": {"type": "string"}},
                        },
                    },
                ],
            },
            "thumbnail_path": {"type": "string", "description": "Custom thumbnail; best-effort, needs a verified channel."},
            "made_for_kids": {"type": "boolean", "default": False},
            "contains_synthetic_media": {
                "type": "boolean",
                "description": "YouTube's altered-or-synthetic-content disclosure. True for AI-generated footage.",
            },
            "operation": {
                "type": "string", "enum": ["upload", "set_thumbnail"], "default": "upload",
                "description": "set_thumbnail: put thumbnail_path on an ALREADY uploaded video (video_id, or the "
                               "upload recorded for video_path in the project's publish_log). No video is uploaded.",
            },
            "video_id": {"type": "string", "description": "set_thumbnail: the YouTube video id (else looked up)"},
            "dry_run": {"type": "boolean", "description": "Probe and build the payload, make no network calls."},
            "force": {"type": "boolean", "description": "Upload even if these exact bytes were already published."},
            "write_checkpoint": {
                "type": "boolean",
                "description": "Also write checkpoint_publish.json (the board sets this; agents keep their own protocol).",
            },
            "timestamp": {"type": "string", "description": "Override the ISO-8601 timestamp (for deterministic tests)."},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "video_id": {"type": "string"},
            "url": {"type": "string"},
            "short": {"type": "boolean"},
            "privacy": {"type": "string"},
            "publish_log": {"type": "object"},
            "probe": {"type": "object"},
            "dry_run": {"type": "boolean"},
            "already_published": {"type": "boolean"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=0, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=UPLOAD_CHUNK_RETRIES,
        backoff_seconds=RETRY_BACKOFF_SECONDS,
        retryable_errors=["timeout", "connection", "429", "5xx"],
    )
    resume_support = ResumeSupport.FROM_CHECKPOINT
    idempotency_key_fields = ["video_path", "title", "privacy"]

    side_effects = [
        "uploads a video to the authenticated YouTube channel — undoing it means deleting it on YouTube",
        "writes artifacts/publish_log.json (and checkpoint_publish.json when write_checkpoint is set)",
    ]
    user_visible_verification = [
        "Open the returned URL and confirm the video plays, is the right cut, and has the intended visibility",
        "For a Short: confirm YouTube shows it in the Shorts player (vertical, <= 60s)",
    ]

    # ---- Probing -------------------------------------------------------

    @staticmethod
    def probe(video_path: Path) -> dict[str, Any]:
        """width / height / duration via ffprobe. Raises on an unreadable file."""
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height:format=duration",
                    "-of", "json", str(video_path),
                ],
                capture_output=True, text=True, timeout=60, check=True,
            ).stdout
            data = json.loads(out or "{}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            raise YouTubeUploadError(f"could not probe {video_path.name}: {exc}") from exc
        stream = (data.get("streams") or [{}])[0]
        duration = (data.get("format") or {}).get("duration")
        try:
            duration = float(duration) if duration is not None else 0.0
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration_seconds": round(duration, 2),
        }

    @staticmethod
    def is_short(probe: dict[str, Any]) -> bool:
        """Portrait and <= 60s. Square (1:1) is not a Short."""
        height, width = probe.get("height") or 0, probe.get("width") or 0
        duration = probe.get("duration_seconds") or 0
        return bool(height > width and 0 < duration <= SHORT_MAX_SECONDS)

    @staticmethod
    def fingerprint(video_path: Path) -> str:
        """Cheap content identity: size + mtime. Re-rendering changes both.

        A full hash of a 500 MB film costs seconds on every board refresh;
        this is only guarding against a repeated click, not a hostile actor.
        """
        stat = video_path.stat()
        return f"{stat.st_size}:{int(stat.st_mtime)}"

    # ---- Publish log ---------------------------------------------------

    @staticmethod
    def _publish_log_path(project_dir: Path) -> Path:
        return project_dir / "artifacts" / "publish_log.json"

    @classmethod
    def read_publish_log(cls, project_dir: Path) -> dict[str, Any]:
        """The project's publish_log, wherever the publish stage left it.

        Older runs embedded it only in checkpoint_publish.json; starting a fresh
        log there would drop their earlier export entries on the first upload.
        """
        candidates = (
            (cls._publish_log_path(project_dir), None),
            (project_dir / "checkpoint_publish.json", "publish_log"),
        )
        for path, key in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if key and isinstance(data, dict):
                data = (data.get("artifacts") or {}).get(key)
            if isinstance(data, dict) and isinstance(data.get("entries"), list):
                return data
        return {"version": "1.0", "entries": []}

    @classmethod
    def find_existing_upload(cls, project_dir: Path, fingerprint: str) -> Optional[dict[str, Any]]:
        """A previous successful upload of these exact bytes, if any.

        The publish_log schema pins `entries` to a closed set of properties, so
        the per-upload bookkeeping (which file, which fingerprint) lives in the
        artifact's free-form top-level `metadata` block instead.
        """
        log = cls.read_publish_log(project_dir)
        uploads = ((log.get("metadata") or {}).get("youtube") or {}).get("uploads") or []
        for record in reversed(uploads):
            if isinstance(record, dict) and record.get("fingerprint") == fingerprint and record.get("video_id"):
                return record
        return None

    # ---- Auth + HTTP ---------------------------------------------------

    @staticmethod
    def _credentials() -> tuple[str, str, str]:
        client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()
        refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
        missing = [
            name for name, value in (
                ("YOUTUBE_CLIENT_ID", client_id),
                ("YOUTUBE_CLIENT_SECRET", client_secret),
                ("YOUTUBE_REFRESH_TOKEN", refresh_token),
            ) if not value
        ]
        if missing:
            raise YouTubeUploadError(
                f"missing credentials: {', '.join(missing)}. "
                "Run `python scripts/get_youtube_token.py` to mint them."
            )
        return client_id, client_secret, refresh_token

    @staticmethod
    def _access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
        """Exchange the long-lived refresh token for a short-lived access token."""
        import requests

        response = requests.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if not response.ok:
            hint = ""
            body = response.text or ""
            if response.status_code == 401 or "unauthorized_client" in body or "invalid_grant" in body:
                hint = (
                    " — the refresh token was minted by a different OAuth client than "
                    "YOUTUBE_CLIENT_ID/SECRET, or it was revoked. Re-run "
                    "`python scripts/get_youtube_token.py`."
                )
            raise YouTubeUploadError(f"YouTube OAuth failed ({response.status_code}): {body[:200]}{hint}")
        token = (response.json() or {}).get("access_token")
        if not token:
            raise YouTubeUploadError("YouTube OAuth returned no access_token")
        return token

    @staticmethod
    def _open_session(token: str, metadata: dict[str, Any], size: int) -> str:
        import requests

        response = requests.post(
            f"{UPLOAD_URL}?uploadType=resumable&part=snippet,status",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": "video/*",
                "X-Upload-Content-Length": str(size),
            },
            data=json.dumps(metadata),
            timeout=30,
        )
        if not response.ok:
            raise YouTubeUploadError(
                f"YouTube rejected the upload session ({response.status_code}): {response.text[:200]}"
            )
        session_url = response.headers.get("Location")
        if not session_url:
            raise YouTubeUploadError("YouTube opened no upload session (no Location header)")
        return session_url

    @staticmethod
    def _committed_offset(session_url: str, size: int) -> int:
        """Ask the session how many bytes it already has (resumable protocol).

        A `308 Resume Incomplete` carries `Range: bytes=0-<last>`; no Range at
        all means nothing landed yet.
        """
        import requests

        response = requests.put(
            session_url,
            headers={"Content-Range": f"bytes */{size}", "Content-Length": "0"},
            timeout=60,
        )
        if response.status_code in (200, 201):
            return size  # already complete
        if response.status_code != 308:
            return 0
        rng = response.headers.get("Range") or ""
        if "-" not in rng:
            return 0
        try:
            return int(rng.rsplit("-", 1)[1]) + 1
        except (ValueError, IndexError):
            return 0

    @classmethod
    def _upload_bytes(cls, session_url: str, video_path: Path, size: int) -> dict[str, Any]:
        """PUT the file, resuming from the committed offset after a failure."""
        import requests

        offset = 0
        last_error = ""
        for attempt in range(UPLOAD_CHUNK_RETRIES + 1):
            if attempt:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
                try:
                    offset = cls._committed_offset(session_url, size)
                except Exception as exc:  # session itself unreachable
                    last_error = f"could not query upload offset: {exc}"
                    continue
                if offset >= size:
                    # Bytes are all there but we never saw the final response;
                    # one more probe returns the video resource.
                    offset = size
            headers = {"Content-Type": "video/*"}
            if offset:
                headers["Content-Range"] = f"bytes {offset}-{size - 1}/{size}"
            try:
                with open(video_path, "rb") as handle:
                    handle.seek(offset)
                    response = requests.put(session_url, headers=headers, data=handle, timeout=3600)
            except Exception as exc:  # network drop mid-PUT
                last_error = f"connection lost after {offset} bytes: {exc}"
                continue
            if response.ok:
                body = response.json() or {}
                video_id = body.get("id")
                if not video_id:
                    raise YouTubeUploadError("YouTube accepted the upload but returned no video id")
                return body
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRYABLE_STATUS:
                raise YouTubeUploadError(f"YouTube upload failed ({last_error})")
        raise YouTubeUploadError(
            f"YouTube upload failed after {UPLOAD_CHUNK_RETRIES + 1} attempts — {last_error}"
        )

    @staticmethod
    def _set_thumbnail(video_id: str, thumbnail: Path, token: str) -> None:
        import requests

        content_type = "image/jpeg" if thumbnail.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        with open(thumbnail, "rb") as handle:
            response = requests.post(
                f"{THUMBNAIL_URL}?videoId={video_id}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
                data=handle,
                timeout=120,
            )
        if not response.ok:
            raise YouTubeUploadError(f"thumbnail rejected ({response.status_code}): {response.text[:150]}")

    # ---- Chapters ------------------------------------------------------

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        seconds = int(seconds)
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def parse_chapters(value: Any) -> list[dict[str, Any]]:
        """Chapters as [{start_seconds, title}], from a list or 'M:SS Title' lines."""
        if not value:
            return []
        if isinstance(value, str):
            chapters = []
            for line in value.splitlines():
                line = line.strip()
                if not line:
                    continue
                stamp, _, title = line.partition(" ")
                try:
                    parts = [int(p) for p in stamp.split(":")]
                except ValueError:
                    raise YouTubeUploadError(f"chapter line needs a 'M:SS Title' timestamp: {line!r}") from None
                if not 2 <= len(parts) <= 3:
                    raise YouTubeUploadError(f"chapter line needs a 'M:SS Title' timestamp: {line!r}")
                start = 0
                for part in parts:
                    start = start * 60 + part
                chapters.append({"start_seconds": start, "title": title.strip()})
            return chapters
        if isinstance(value, list):
            try:
                return [
                    {"start_seconds": float(c["start_seconds"]), "title": str(c.get("title") or "").strip()}
                    for c in value
                ]
            except (KeyError, TypeError, ValueError):
                raise YouTubeUploadError("chapters must be {start_seconds, title} objects") from None
        raise YouTubeUploadError("chapters must be a list or 'M:SS Title' lines")

    @classmethod
    def validate_chapters(cls, chapters: list[dict[str, Any]], duration: Optional[float]) -> None:
        """Enforce the rules YouTube silently applies — a violation means no timeline at all."""
        if not chapters:
            return
        if len(chapters) < CHAPTER_MIN_COUNT:
            raise YouTubeUploadError(
                f"YouTube shows chapters only when there are at least {CHAPTER_MIN_COUNT} (got {len(chapters)})"
            )
        if chapters[0]["start_seconds"] != 0:
            raise YouTubeUploadError("the first chapter must start at 00:00")
        for chapter in chapters:
            if not chapter["title"]:
                raise YouTubeUploadError(f"chapter at {cls.format_timestamp(chapter['start_seconds'])} has no title")
        ends = [c["start_seconds"] for c in chapters[1:]] + ([duration] if duration else [])
        for chapter, end in zip(chapters, ends):
            if end - chapter["start_seconds"] < CHAPTER_MIN_SECONDS:
                raise YouTubeUploadError(
                    f"chapter '{chapter['title']}' at {cls.format_timestamp(chapter['start_seconds'])} "
                    f"is shorter than {int(CHAPTER_MIN_SECONDS)}s — YouTube would drop the whole timeline"
                )

    # ---- Payload -------------------------------------------------------

    @classmethod
    def build_metadata(
        cls, inputs: dict[str, Any], short: bool, duration: Optional[float] = None
    ) -> dict[str, Any]:
        title = (inputs.get("title") or "Untitled").strip()
        description = (inputs.get("description") or "").strip()
        tags = cls._cap_tags(inputs.get("tags") or [])
        # Shorts have no progress-bar timeline, so chapters would only be noise.
        chapters = [] if short else cls.parse_chapters(inputs.get("chapters"))
        cls.validate_chapters(chapters, duration)
        chapter_block = "\n".join(
            f"{cls.format_timestamp(c['start_seconds'])} {c['title']}" for c in chapters
        )
        # Re-publishing pre-fills copy that was already sent, so every suffix
        # below is added only if it is not there yet.
        if CHAPTERS_PLACEHOLDER in description:
            # The copy says where the timeline goes (e.g. above the hashtags).
            description = description.replace(CHAPTERS_PLACEHOLDER, chapter_block)
            description = re.sub(r"\n{3,}", "\n\n", description).strip()
            if len(description) > DESCRIPTION_LIMIT and chapter_block:
                raise YouTubeUploadError(
                    f"description with chapters is {len(description)} chars; YouTube allows "
                    f"{DESCRIPTION_LIMIT} — shorten the prose or drop some chapters"
                )
        elif chapter_block and chapter_block not in description:
            # Trim the prose, never the timeline: a cut-off chapter list breaks it.
            room = DESCRIPTION_LIMIT - len(chapter_block) - 2
            description = f"{description[:max(room, 0)].rstrip()}\n\n{chapter_block}".strip()
        if short:
            if not title.endswith("#Shorts"):
                title = f"{title[:SHORT_TITLE_LIMIT].rstrip()} #Shorts"
            if not description.endswith("#Shorts"):
                description = f"{description}\n\n#Shorts".strip()
        status: dict[str, Any] = {
            "privacyStatus": inputs.get("privacy") or "unlisted",
            "selfDeclaredMadeForKids": bool(inputs.get("made_for_kids", False)),
        }
        if inputs.get("contains_synthetic_media") is not None:
            status["containsSyntheticMedia"] = bool(inputs["contains_synthetic_media"])
        return {
            "snippet": {
                "title": title[:TITLE_LIMIT] or "Untitled",
                "description": description[:DESCRIPTION_LIMIT],
                "tags": tags,
                "categoryId": str(inputs.get("category_id") or "24"),
            },
            "status": status,
            "chapters": chapters,  # bookkeeping only; stripped before sending
        }

    @staticmethod
    def _cap_tags(raw: list[Any]) -> list[str]:
        """At most 20 tags and 500 characters, counted the way YouTube does
        (a tag containing a space costs two extra characters for its quotes)."""
        tags: list[str] = []
        used = 0
        for tag in raw:
            tag = str(tag).strip().lstrip("#").replace("<", "").replace(">", "")
            if not tag or tag in tags:
                continue
            cost = len(tag) + (2 if " " in tag else 0) + (1 if tags else 0)
            if len(tags) >= TAG_LIMIT or used + cost > TAG_CHARS_LIMIT:
                break
            tags.append(tag)
            used += cost
        return tags

    @staticmethod
    def _wire(metadata: dict[str, Any]) -> dict[str, Any]:
        """The videos.insert body: everything but local bookkeeping."""
        return {k: v for k, v in metadata.items() if k in ("snippet", "status")}

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0  # quota, not currency

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        try:
            size_mb = Path(inputs["video_path"]).stat().st_size / 1_048_576
        except (KeyError, OSError):
            return 60.0
        return round(15.0 + size_mb / 2.0, 1)  # ~2 MB/s plus handshake

    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Everything except the network: probe, Shorts verdict, exact payload."""
        video_path = Path(inputs["video_path"]).expanduser()
        info: dict[str, Any] = {
            "tool": self.name,
            "estimated_cost_usd": 0.0,
            "estimated_runtime_seconds": self.estimate_runtime(inputs),
            "status": self.get_status().value,
            "would_execute": True,
        }
        if not video_path.is_file():
            info.update(would_execute=False, error=f"video_path not found: {video_path}")
            return info
        probe = self.probe(video_path)
        short = self.is_short(probe) if inputs.get("short", "auto") == "auto" else bool(inputs.get("short"))
        info.update(probe=probe, short=short, size_bytes=video_path.stat().st_size)
        try:
            info["payload"] = self._wire(self.build_metadata(inputs, short, probe.get("duration_seconds")))
        except YouTubeUploadError as exc:
            info.update(would_execute=False, error=str(exc))
        return info

    # ---- Execution -----------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        try:
            if inputs.get("operation") == "set_thumbnail":
                return self._set_thumbnail_only(inputs)
            return self._execute(inputs)
        except YouTubeUploadError as exc:
            return ToolResult(success=False, error=str(exc))

    def _set_thumbnail_only(self, inputs: dict[str, Any]) -> ToolResult:
        """Thumbnail on an existing upload — e.g. after the channel is verified for custom thumbnails."""
        thumb = Path(str(inputs.get("thumbnail_path") or "")).expanduser()
        if not thumb.is_file():
            return ToolResult(success=False, error=f"thumbnail_path not found: {thumb}")
        if thumb.suffix.lower() not in (".png", ".jpg", ".jpeg") or thumb.stat().st_size > 2 * 1024 * 1024:
            return ToolResult(success=False, error="thumbnail must be a PNG or JPEG of at most 2 MB")
        project_dir = Path(inputs["project_dir"]).expanduser() if inputs.get("project_dir") else None
        video_id = str(inputs.get("video_id") or "").strip()
        if not video_id and project_dir is not None and inputs.get("video_path"):
            video_path = Path(inputs["video_path"]).expanduser()
            record = None
            if video_path.is_file():
                record = self.find_existing_upload(project_dir, self.fingerprint(video_path))
            if record is None:  # re-rendered since upload: fall back to the recorded source path
                rel = self._rel(project_dir, video_path)
                uploads = ((self.read_publish_log(project_dir).get("metadata") or {}).get("youtube") or {}).get("uploads") or []
                record = next((u for u in reversed(uploads) if u.get("source") == rel and u.get("video_id")), None)
            video_id = (record or {}).get("video_id", "")
        if not video_id:
            return ToolResult(success=False, error="no video_id given and no recorded upload found for this render")

        started = time.monotonic()
        token = self._access_token(*self._credentials())
        self._set_thumbnail(video_id, thumb, token)  # raises YouTubeUploadError with YouTube's reason
        timestamp = inputs.get("timestamp") or datetime.now(timezone.utc).isoformat()
        if project_dir is not None:
            log = self.read_publish_log(project_dir)
            youtube_meta = log.setdefault("metadata", {}).setdefault("youtube", {})
            youtube_meta.setdefault("thumbnails", []).append({
                "video_id": video_id, "thumbnail": self._rel(project_dir, thumb), "set_at": timestamp,
            })
            path = self._publish_log_path(project_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        return ToolResult(
            success=True,
            data={"video_id": video_id, "url": f"{WATCH_URL}{video_id}", "thumbnail": str(thumb), "thumbnail_set": True},
            duration_seconds=round(time.monotonic() - started, 2),
        )

    def _execute(self, inputs: dict[str, Any]) -> ToolResult:
        video_path = Path(inputs["video_path"]).expanduser()
        if not video_path.is_file():
            return ToolResult(success=False, error=f"video_path not found: {video_path}")

        project_dir = Path(inputs["project_dir"]).expanduser() if inputs.get("project_dir") else None
        if project_dir is not None:
            # The render must belong to the project whose log we are about to
            # write — publishing a file from elsewhere under this project's
            # name would put a wrong record on the board.
            try:
                video_path.resolve().relative_to(project_dir.resolve())
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"video_path is outside project_dir: {video_path}",
                )

        probe = self.probe(video_path)
        requested_short = inputs.get("short", "auto")
        short = self.is_short(probe) if requested_short == "auto" else bool(requested_short)
        size = video_path.stat().st_size
        fingerprint = self.fingerprint(video_path)

        if inputs.get("dry_run"):
            payload = self._wire(self.build_metadata(inputs, short, probe.get("duration_seconds")))
            return ToolResult(
                success=True,
                data={
                    "dry_run": True, "short": short, "probe": probe,
                    "privacy": payload["status"]["privacyStatus"],
                    "payload": payload, "video_id": None, "url": None,
                },
            )

        if project_dir is not None and not inputs.get("force"):
            existing = self.find_existing_upload(project_dir, fingerprint)
            if existing:
                return ToolResult(
                    success=True,
                    data={
                        "already_published": True,
                        "video_id": existing.get("video_id"),
                        "url": existing.get("url"),
                        "short": bool(existing.get("short")),
                        "probe": probe,
                        "uploaded_at": existing.get("uploaded_at"),
                    },
                )

        # Built before any network call, so a bad chapter list fails fast
        # instead of after the OAuth round-trip.
        metadata = self.build_metadata(inputs, short, probe.get("duration_seconds"))
        client_id, client_secret, refresh_token = self._credentials()
        token = self._access_token(client_id, client_secret, refresh_token)
        started = time.monotonic()
        session_url = self._open_session(token, self._wire(metadata), size)
        body = self._upload_bytes(session_url, video_path, size)
        video_id = body.get("id")
        url = f"{SHORTS_URL}{video_id}" if short else f"{WATCH_URL}{video_id}"

        thumbnail_warning = None
        thumbnail = inputs.get("thumbnail_path")
        if thumbnail and Path(thumbnail).expanduser().is_file():
            try:
                self._set_thumbnail(video_id, Path(thumbnail).expanduser(), token)
            except Exception as exc:
                # Custom thumbnails need a phone-verified channel. The video is
                # up; that is the deliverable — this is a note, not a failure.
                thumbnail_warning = f"thumbnail not set ({exc})"

        timestamp = inputs.get("timestamp") or datetime.now(timezone.utc).isoformat()
        publish_log = None
        if project_dir is not None:
            publish_log = self._record(
                project_dir=project_dir,
                video_path=video_path,
                video_id=video_id,
                url=url,
                short=short,
                fingerprint=fingerprint,
                metadata=metadata,
                timestamp=timestamp,
                write_checkpoint=bool(inputs.get("write_checkpoint")),
            )

        return ToolResult(
            success=True,
            data={
                "video_id": video_id,
                "url": url,
                "short": short,
                "privacy": metadata["status"]["privacyStatus"],
                "probe": probe,
                "publish_log": publish_log,
                "warning": thumbnail_warning,
                "already_published": False,
            },
            artifacts=[str(self._publish_log_path(project_dir))] if project_dir else [],
            duration_seconds=round(time.monotonic() - started, 2),
        )

    # ---- Recording -----------------------------------------------------

    def _record(
        self,
        *,
        project_dir: Path,
        video_path: Path,
        video_id: str,
        url: str,
        short: bool,
        fingerprint: str,
        metadata: dict[str, Any],
        timestamp: str,
        write_checkpoint: bool,
    ) -> dict[str, Any]:
        """Append to publish_log.json (and optionally checkpoint_publish.json).

        Both files are the ones the pipeline already writes for this stage, so
        the board picks the publish up through its normal watcher — no new
        state channel, no UI push.
        """
        log = self.read_publish_log(project_dir)
        entry = {
            "platform": "youtube",
            "status": "published",
            "url": url,
            "video_id": video_id,
            "visibility": metadata["status"]["privacyStatus"],
            "timestamp": timestamp,
            "metadata_used": {
                "title": metadata["snippet"]["title"],
                "description": metadata["snippet"]["description"],
                "hashtags": [f"#{t}" for t in metadata["snippet"]["tags"]],
                "chapters": metadata.get("chapters") or [],
            },
        }
        log.setdefault("entries", []).append(entry)
        # `entries` is a closed schema — the file/fingerprint bookkeeping that
        # makes a repeat click idempotent lives in the free-form metadata block.
        meta = log.setdefault("metadata", {})
        youtube_meta = meta.setdefault("youtube", {})
        youtube_meta.setdefault("uploads", []).append({
            "video_id": video_id,
            "url": url,
            "source": self._rel(project_dir, video_path),
            "fingerprint": fingerprint,
            "short": short,
            "uploaded_at": timestamp,
        })

        try:
            from schemas.artifacts import validate_artifact

            validate_artifact("publish_log", log)
        except ImportError:
            pass
        except Exception as exc:
            raise YouTubeUploadError(
                f"published as {video_id} but the publish_log failed validation: {exc}"
            ) from exc

        path = self._publish_log_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(log, indent=2), encoding="utf-8")

        if write_checkpoint:
            self._write_checkpoint(project_dir, timestamp, log)
        return log

    @staticmethod
    def _rel(project_dir: Path, path: Path) -> str:
        try:
            return path.resolve().relative_to(project_dir.resolve()).as_posix()
        except (ValueError, OSError):
            return path.name

    @staticmethod
    def _write_checkpoint(project_dir: Path, timestamp: str, log: dict[str, Any]) -> None:
        """Mark the publish stage completed, preserving whatever is already there.

        The publish_log is embedded, not referenced by path: lib/checkpoint
        validates every canonical artifact in a checkpoint as a JSON object.
        """
        path = project_dir / "checkpoint_publish.json"
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        marker = {}
        try:
            marker = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        existing.update({
            "version": existing.get("version", "1.0"),
            "project_id": existing.get("project_id") or project_dir.name,
            "pipeline_type": existing.get("pipeline_type") or marker.get("pipeline_type") or "unknown",
            "stage": "publish",
            "status": "completed",
            "timestamp": timestamp,
            "human_approved": True,
            "artifacts": {**(existing.get("artifacts") or {}), "publish_log": log},
        })
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
