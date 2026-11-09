"""Tests for the YouTube publisher.

No test here touches the network: `tests/conftest.py` blocks non-loopback
sockets for the whole session, and every HTTP call is faked by patching
`requests.post` / `requests.put` (the tool imports `requests` inside its
functions, so patching the module's attributes covers it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import requests

from tools.base_tool import ToolRuntime, ToolTier
from tools.publishers.youtube_upload import YouTubeUpload, YouTubeUploadError

CREDS = {
    "YOUTUBE_CLIENT_ID": "cid",
    "YOUTUBE_CLIENT_SECRET": "csecret",
    "YOUTUBE_REFRESH_TOKEN": "rtoken",
}


# --------------------------------------------------------------------------
# fixtures / fakes
# --------------------------------------------------------------------------

class _Response:
    def __init__(self, *, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeYouTube:
    """Records calls and plays a scripted YouTube API back at the tool."""

    def __init__(self, *, video_id="vid123", put_results=None, offset=0):
        self.calls: list[str] = []
        self.video_id = video_id
        self.put_results = list(put_results or [])
        self.offset = offset
        self.content_ranges: list[str | None] = []

    def post(self, url, **kwargs):
        if "oauth2" in url:
            self.calls.append("token")
            return _Response(payload={"access_token": "at"})
        if "upload/youtube/v3/videos" in url:
            self.calls.append("session")
            return _Response(headers={"Location": "https://upload.example/session"})
        if "thumbnails/set" in url:
            self.calls.append("thumbnail")
            return _Response(payload={"kind": "youtube#thumbnailSetResponse"})
        raise AssertionError(f"unexpected POST {url}")

    def put(self, url, **kwargs):
        headers = kwargs.get("headers") or {}
        content_range = headers.get("Content-Range")
        if content_range and content_range.startswith("bytes */"):
            self.calls.append("offset-probe")
            return _Response(status=308, headers={"Range": f"bytes=0-{self.offset - 1}"})
        self.calls.append("upload")
        self.content_ranges.append(content_range)
        if self.put_results:
            outcome = self.put_results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, int):
                return _Response(status=outcome, text="upstream said no")
        return _Response(payload={"id": self.video_id})


@pytest.fixture
def creds(monkeypatch):
    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def project(tmp_path) -> Path:
    project_dir = tmp_path / "projects" / "film"
    (project_dir / "renders").mkdir(parents=True)
    (project_dir / "artifacts").mkdir(parents=True)
    (project_dir / "project.json").write_text(
        json.dumps({"project_id": "film", "title": "The Film", "pipeline_type": "cinematic"}),
        encoding="utf-8",
    )
    (project_dir / "renders" / "final.mp4").write_bytes(b"\x00" * 2048)
    return project_dir


@pytest.fixture
def no_probe(monkeypatch):
    """Skip ffmpeg: a landscape 3-minute film unless a test says otherwise."""
    monkeypatch.setattr(
        YouTubeUpload, "probe",
        staticmethod(lambda path: {"width": 1920, "height": 1080, "duration_seconds": 180.0}),
    )


def _inputs(project: Path, **overrides):
    base = {
        "project_dir": str(project),
        "video_path": str(project / "renders" / "final.mp4"),
        "title": "The Film",
        "description": "A film.",
        "tags": ["cinema", "#ai"],
        "timestamp": "2026-08-24T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def _install(monkeypatch, fake: FakeYouTube) -> FakeYouTube:
    monkeypatch.setattr(requests, "post", fake.post)
    monkeypatch.setattr(requests, "put", fake.put)
    return fake


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------

class TestContract:
    def test_declares_a_networked_publish_tool(self):
        tool = YouTubeUpload()
        assert tool.name == "youtube_upload"
        assert tool.tier is ToolTier.PUBLISH
        assert tool.capability == "publish"
        assert tool.provider == "youtube"
        assert tool.runtime is ToolRuntime.API
        assert tool.resource_profile.network_required is True
        # An irreversible side effect must be declared, not discovered.
        assert any("upload" in effect.lower() for effect in tool.side_effects)
        assert "env:YOUTUBE_REFRESH_TOKEN" in tool.dependencies

    def test_is_discoverable_as_a_second_publish_provider(self):
        """Dropping the file in tools/publishers/ is the whole registration."""
        from tools.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.discover()

        assert "youtube_upload" in registry.list_all()
        providers = {tool.provider for tool in registry.get_by_capability("publish")}
        assert {"local", "youtube"} <= providers  # export_bundle still there

    def test_status_is_unavailable_without_credentials(self, monkeypatch):
        for key in CREDS:
            monkeypatch.delenv(key, raising=False)
        assert YouTubeUpload().get_status().value == "unavailable"


# --------------------------------------------------------------------------
# shorts detection + payload
# --------------------------------------------------------------------------

class TestShortsAndPayload:
    @pytest.mark.parametrize(
        ("probe", "expected"),
        [
            ({"width": 1080, "height": 1920, "duration_seconds": 21.0}, True),   # the real 9x16 trailer
            ({"width": 1080, "height": 1920, "duration_seconds": 61.0}, False),  # portrait but too long
            ({"width": 1920, "height": 1080, "duration_seconds": 30.0}, False),  # landscape
            ({"width": 1080, "height": 1080, "duration_seconds": 20.0}, False),  # square is not a Short
            ({"width": 1080, "height": 1920, "duration_seconds": 0.0}, False),   # unprobeable
        ],
    )
    def test_short_verdict(self, probe, expected):
        assert YouTubeUpload.is_short(probe) is expected

    def test_short_payload_is_tagged_and_truncated(self):
        payload = YouTubeUpload.build_metadata(
            {"title": "T" * 120, "description": "d", "tags": ["#a", "b"]}, short=True
        )
        assert payload["snippet"]["title"].endswith(" #Shorts")
        assert len(payload["snippet"]["title"]) <= 100
        assert payload["snippet"]["description"].endswith("#Shorts")
        assert payload["snippet"]["tags"] == ["a", "b"]  # leading # stripped

    def test_full_video_payload_is_untagged(self):
        payload = YouTubeUpload.build_metadata({"title": "T", "description": "d"}, short=False)
        assert payload["snippet"]["title"] == "T"
        assert "#Shorts" not in payload["snippet"]["description"]
        assert payload["status"]["privacyStatus"] == "unlisted"  # conservative default
        assert payload["status"]["selfDeclaredMadeForKids"] is False

    def test_description_and_tags_are_capped(self):
        payload = YouTubeUpload.build_metadata(
            {"title": "T", "description": "x" * 6000, "tags": [f"t{i}" for i in range(40)]}, short=False
        )
        assert len(payload["snippet"]["description"]) == 4900
        assert len(payload["snippet"]["tags"]) == 20

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_probe_reads_real_dimensions(self, tmp_path):
        video = tmp_path / "vertical.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=black:s=180x320:d=1", "-pix_fmt", "yuv420p", str(video)],
            check=True, capture_output=True, timeout=60,
        )
        probe = YouTubeUpload.probe(video)
        assert (probe["width"], probe["height"]) == (180, 320)
        assert 0 < probe["duration_seconds"] < 3
        assert YouTubeUpload.is_short(probe) is True

    def test_probe_raises_on_a_non_video(self, tmp_path):
        junk = tmp_path / "not-a-video.mp4"
        junk.write_bytes(b"nope")
        with pytest.raises(YouTubeUploadError):
            YouTubeUpload.probe(junk)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

class TestExecute:
    def test_dry_run_makes_no_http_calls(self, project, creds, no_probe, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("dry run must not touch the network")

        monkeypatch.setattr(requests, "post", explode)
        monkeypatch.setattr(requests, "put", explode)

        result = YouTubeUpload().execute(_inputs(project, dry_run=True))

        assert result.success and result.data["dry_run"] is True
        assert result.data["payload"]["snippet"]["title"] == "The Film"
        assert not (project / "artifacts" / "publish_log.json").exists()

    def test_upload_records_a_schema_valid_publish_log(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube())

        result = YouTubeUpload().execute(_inputs(project, write_checkpoint=True))

        assert result.success, result.error
        assert result.data["video_id"] == "vid123"
        assert result.data["url"] == "https://www.youtube.com/watch?v=vid123"
        assert fake.calls == ["token", "session", "upload"]

        log = json.loads((project / "artifacts" / "publish_log.json").read_text())
        from schemas.artifacts import validate_artifact

        validate_artifact("publish_log", log)  # raises if the shape drifted
        entry = log["entries"][-1]
        assert entry["platform"] == "youtube"
        assert entry["status"] == "published"
        assert entry["video_id"] == "vid123"
        assert entry["visibility"] == "unlisted"
        # Per-file bookkeeping lives in metadata (entries are a closed schema).
        upload = log["metadata"]["youtube"]["uploads"][-1]
        assert upload["source"] == "renders/final.mp4"

        checkpoint = json.loads((project / "checkpoint_publish.json").read_text())
        assert checkpoint["status"] == "completed"
        assert checkpoint["human_approved"] is True
        assert checkpoint["artifacts"]["publish_log"] == log
        from lib.checkpoint import validate_checkpoint

        validate_checkpoint(checkpoint)  # resume/get_completed_stages read it back

    def test_upload_keeps_a_log_that_lived_only_in_the_checkpoint(self, project, creds, no_probe, monkeypatch):
        # Older runs embedded publish_log in checkpoint_publish.json and never
        # wrote artifacts/publish_log.json; their export entries must survive.
        prior = {
            "version": "1.0",
            "entries": [{"platform": "local_export", "status": "exported", "timestamp": "2026-08-17T18:30:00Z"}],
        }
        (project / "checkpoint_publish.json").write_text(
            json.dumps({"stage": "publish", "status": "completed", "artifacts": {"publish_log": prior}})
        )
        _install(monkeypatch, FakeYouTube())

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success, result.error
        log = json.loads((project / "artifacts" / "publish_log.json").read_text())
        assert [e["platform"] for e in log["entries"]] == ["local_export", "youtube"]

    def test_short_render_gets_the_shorts_url(self, project, creds, monkeypatch):
        monkeypatch.setattr(
            YouTubeUpload, "probe",
            staticmethod(lambda path: {"width": 1080, "height": 1920, "duration_seconds": 21.0}),
        )
        _install(monkeypatch, FakeYouTube())

        result = YouTubeUpload().execute(_inputs(project))

        assert result.data["short"] is True
        assert result.data["url"] == "https://www.youtube.com/shorts/vid123"

    def test_short_can_be_forced_off(self, project, creds, monkeypatch):
        monkeypatch.setattr(
            YouTubeUpload, "probe",
            staticmethod(lambda path: {"width": 1080, "height": 1920, "duration_seconds": 21.0}),
        )
        _install(monkeypatch, FakeYouTube())

        result = YouTubeUpload().execute(_inputs(project, short=False))

        assert result.data["short"] is False
        assert "watch?v=" in result.data["url"]

    def test_second_publish_of_the_same_bytes_is_refused(self, project, creds, no_probe, monkeypatch):
        _install(monkeypatch, FakeYouTube())
        YouTubeUpload().execute(_inputs(project))

        fake2 = _install(monkeypatch, FakeYouTube(video_id="SHOULD-NOT-HAPPEN"))
        again = YouTubeUpload().execute(_inputs(project))

        assert again.success
        assert again.data["already_published"] is True
        assert again.data["video_id"] == "vid123"
        assert fake2.calls == []  # nothing was uploaded a second time

    def test_force_publishes_a_duplicate_on_purpose(self, project, creds, no_probe, monkeypatch):
        _install(monkeypatch, FakeYouTube())
        YouTubeUpload().execute(_inputs(project))

        fake2 = _install(monkeypatch, FakeYouTube(video_id="vid456"))
        again = YouTubeUpload().execute(_inputs(project, force=True))

        assert again.data["video_id"] == "vid456"
        assert "upload" in fake2.calls
        log = json.loads((project / "artifacts" / "publish_log.json").read_text())
        assert len(log["entries"]) == 2

    def test_a_re_rendered_file_is_not_treated_as_already_published(
        self, project, creds, no_probe, monkeypatch
    ):
        _install(monkeypatch, FakeYouTube())
        YouTubeUpload().execute(_inputs(project))

        # Same path, new bytes and mtime — a genuinely different cut.
        render = project / "renders" / "final.mp4"
        render.write_bytes(b"\x01" * 4096)
        import os
        os.utime(render, (1_800_000_000, 1_800_000_000))

        fake2 = _install(monkeypatch, FakeYouTube(video_id="vid789"))
        result = YouTubeUpload().execute(_inputs(project))

        assert result.data["video_id"] == "vid789"
        assert "upload" in fake2.calls

    def test_upload_resumes_after_a_dropped_connection(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube(
            put_results=[ConnectionError("network went away")], offset=1024,
        ))
        monkeypatch.setattr("tools.publishers.youtube_upload.RETRY_BACKOFF_SECONDS", 0)

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success, result.error
        assert fake.calls == ["token", "session", "upload", "offset-probe", "upload"]
        # The retry continued from the committed offset instead of restarting.
        assert fake.content_ranges[-1] == "bytes 1024-2047/2048"

    def test_retryable_status_is_retried(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube(put_results=[503], offset=0))
        monkeypatch.setattr("tools.publishers.youtube_upload.RETRY_BACKOFF_SECONDS", 0)

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success
        assert fake.calls.count("upload") == 2

    def test_permanent_rejection_fails_fast(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube(put_results=[403]))

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success is False
        assert "403" in result.error
        assert fake.calls.count("upload") == 1  # not retried
        assert not (project / "artifacts" / "publish_log.json").exists()

    def test_thumbnail_failure_does_not_fail_the_publish(self, project, creds, no_probe, monkeypatch):
        thumb = project / "renders" / "thumb.png"
        thumb.write_bytes(b"\x89PNG")

        fake = FakeYouTube()

        def post(url, **kwargs):
            if "thumbnails/set" in url:
                fake.calls.append("thumbnail")
                return _Response(status=403, text="channel not verified")
            return fake.post(url, **kwargs)

        monkeypatch.setattr(requests, "post", post)
        monkeypatch.setattr(requests, "put", fake.put)

        result = YouTubeUpload().execute(_inputs(project, thumbnail_path=str(thumb)))

        assert result.success
        assert result.data["video_id"] == "vid123"
        assert "thumbnail" in (result.data["warning"] or "")

    def test_missing_credentials_points_at_the_bootstrap_script(self, project, no_probe, monkeypatch):
        for key in CREDS:
            monkeypatch.delenv(key, raising=False)

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success is False
        assert "get_youtube_token.py" in result.error

    def test_oauth_failure_is_explained(self, project, creds, no_probe, monkeypatch):
        monkeypatch.setattr(requests, "post", lambda url, **kw: _Response(
            status=401, text='{"error":"unauthorized_client"}'))

        result = YouTubeUpload().execute(_inputs(project))

        assert result.success is False
        assert "OAuth" in result.error and "get_youtube_token.py" in result.error

    def test_render_outside_the_project_is_refused(self, project, creds, no_probe, tmp_path, monkeypatch):
        stray = tmp_path / "elsewhere.mp4"
        stray.write_bytes(b"\x00" * 16)
        _install(monkeypatch, FakeYouTube())

        result = YouTubeUpload().execute(_inputs(project, video_path=str(stray)))

        assert result.success is False
        assert "outside project_dir" in result.error

    def test_missing_file_is_reported_not_raised(self, project, creds, no_probe):
        result = YouTubeUpload().execute(_inputs(project, video_path=str(project / "renders" / "gone.mp4")))
        assert result.success is False
        assert "not found" in result.error

    def test_publish_emits_backlot_events(self, project, creds, no_probe, monkeypatch):
        """The board's activity ticker should show the upload with no extra code."""
        _install(monkeypatch, FakeYouTube())
        import lib.events as events_mod

        monkeypatch.setattr(events_mod, "PROJECTS_DIR", project.parent)
        monkeypatch.setattr("lib.events.PROJECTS_DIR", project.parent)

        YouTubeUpload().execute(_inputs(project))

        lines = (project / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        assert [e["event"] for e in events] == ["start", "finish"]
        assert all(e["tool"] == "youtube_upload" for e in events)


class TestChaptersAndDisclosure:
    CHAPTERS = [
        {"start_seconds": 0, "title": "Prologue"},
        {"start_seconds": 14, "title": "I. The Awakening"},
        {"start_seconds": 76, "title": "II. The Whisper"},
    ]

    def test_chapters_are_appended_as_a_youtube_timeline(self):
        meta = YouTubeUpload.build_metadata(
            {"title": "T", "description": "Logline.", "chapters": self.CHAPTERS}, short=False, duration=180.0
        )
        assert meta["snippet"]["description"] == (
            "Logline.\n\n00:00 Prologue\n00:14 I. The Awakening\n01:16 II. The Whisper"
        )
        assert meta["chapters"] == [dict(c, start_seconds=float(c["start_seconds"])) for c in self.CHAPTERS]

    def test_dialog_text_parses_to_the_same_chapters(self):
        text = "00:00 Prologue\n0:14 I. The Awakening\n\n1:16 II. The Whisper\n"
        assert [c["start_seconds"] for c in YouTubeUpload.parse_chapters(text)] == [0, 14, 76]
        assert YouTubeUpload.parse_chapters("1:02:05 Late")[0]["start_seconds"] == 3725

    def test_republishing_does_not_duplicate_the_timeline_or_shorts_tag(self):
        first = YouTubeUpload.build_metadata(
            {"title": "T", "description": "Logline.", "chapters": self.CHAPTERS}, short=False, duration=180.0
        )
        again = YouTubeUpload.build_metadata(
            {"title": "T", "description": first["snippet"]["description"], "chapters": self.CHAPTERS},
            short=False, duration=180.0,
        )
        assert again["snippet"]["description"] == first["snippet"]["description"]
        short = YouTubeUpload.build_metadata({"title": "T #Shorts", "description": "d\n\n#Shorts"}, short=True)
        assert short["snippet"]["title"] == "T #Shorts"
        assert short["snippet"]["description"].count("#Shorts") == 1

    def test_shorts_drop_chapters(self):
        meta = YouTubeUpload.build_metadata({"title": "T", "chapters": self.CHAPTERS}, short=True, duration=21.0)
        assert "00:14" not in meta["snippet"]["description"]
        assert meta["chapters"] == []

    @pytest.mark.parametrize(
        ("chapters", "duration", "message"),
        [
            ([{"start_seconds": 0, "title": "a"}, {"start_seconds": 20, "title": "b"}], 60.0, "at least 3"),
            ([{"start_seconds": 5, "title": "a"}, {"start_seconds": 20, "title": "b"},
              {"start_seconds": 40, "title": "c"}], 60.0, "00:00"),
            ([{"start_seconds": 0, "title": "a"}, {"start_seconds": 5, "title": "b"},
              {"start_seconds": 40, "title": "c"}], 60.0, "shorter than 10s"),
            ([{"start_seconds": 0, "title": "a"}, {"start_seconds": 20, "title": "b"},
              {"start_seconds": 55, "title": "c"}], 60.0, "shorter than 10s"),
            ("0:00 a\nnot-a-time b", None, "M:SS"),
        ],
    )
    def test_rule_violations_fail_before_any_upload(self, chapters, duration, message):
        with pytest.raises(YouTubeUploadError, match=message):
            YouTubeUpload.build_metadata({"title": "T", "chapters": chapters}, short=False, duration=duration)

    def test_bad_chapters_fail_without_touching_the_network(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube())
        result = YouTubeUpload().execute(_inputs(project, chapters="0:00 only one"))
        assert not result.success and "at least 3" in result.error
        assert fake.calls == []

    def test_tags_are_capped_at_youtubes_500_characters(self):
        tags = [f"tag number {i:03d} " + "x" * 30 for i in range(30)]
        capped = YouTubeUpload.build_metadata({"title": "T", "tags": tags}, short=False)["snippet"]["tags"]
        cost = sum(len(t) + 2 for t in capped) + len(capped) - 1
        assert 0 < len(capped) < 20 and cost <= 500

    def test_upload_sends_disclosure_and_records_chapters(self, project, creds, no_probe, monkeypatch):
        fake = FakeYouTube()
        sent = {}
        post = fake.post

        def capture(url, **kwargs):
            if "upload/youtube/v3/videos" in url:
                sent.update(json.loads(kwargs["data"]))
            return post(url, **kwargs)

        _install(monkeypatch, fake)
        monkeypatch.setattr(requests, "post", capture)

        result = YouTubeUpload().execute(
            _inputs(project, chapters=self.CHAPTERS, contains_synthetic_media=True)
        )

        assert result.success, result.error
        assert set(sent) == {"snippet", "status"}  # bookkeeping never goes on the wire
        assert sent["status"]["containsSyntheticMedia"] is True
        assert sent["snippet"]["description"].endswith("01:16 II. The Whisper")
        log = json.loads((project / "artifacts" / "publish_log.json").read_text())
        assert [c["title"] for c in log["entries"][-1]["metadata_used"]["chapters"]] == [
            "Prologue", "I. The Awakening", "II. The Whisper",
        ]


class TestChapterPlacement:
    CHAPTERS = TestChaptersAndDisclosure.CHAPTERS

    def test_placeholder_puts_the_timeline_above_the_hashtags(self):
        meta = YouTubeUpload.build_metadata(
            {"title": "T", "description": "Story.\n\nChapters:\n{chapters}\n\n#a #b", "chapters": self.CHAPTERS},
            short=False, duration=180.0,
        )
        assert meta["snippet"]["description"] == (
            "Story.\n\nChapters:\n00:00 Prologue\n00:14 I. The Awakening\n01:16 II. The Whisper\n\n#a #b"
        )

    def test_republishing_placed_chapters_does_not_append_them_again(self):
        first = YouTubeUpload.build_metadata(
            {"title": "T", "description": "S\n\n{chapters}\n\n#a", "chapters": self.CHAPTERS}, short=False, duration=180.0
        )["snippet"]["description"]
        again = YouTubeUpload.build_metadata(
            {"title": "T", "description": first, "chapters": self.CHAPTERS}, short=False, duration=180.0
        )["snippet"]["description"]
        assert again == first

    def test_placeholder_without_chapters_leaves_no_gap(self):
        meta = YouTubeUpload.build_metadata({"title": "T", "description": "S\n\n{chapters}\n\n#a"}, short=False)
        assert meta["snippet"]["description"] == "S\n\n#a"



class TestSetThumbnailOnly:
    def test_finds_the_recorded_upload_and_logs_the_thumbnail(self, project, creds, no_probe, monkeypatch):
        fake = _install(monkeypatch, FakeYouTube())
        assert YouTubeUpload().execute(_inputs(project)).success  # the earlier upload
        thumb = project / "renders" / "poster.png"
        thumb.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        fake.calls.clear()

        result = YouTubeUpload().execute({"operation": "set_thumbnail", "project_dir": str(project),
                                          "video_path": str(project / "renders" / "final.mp4"),
                                          "thumbnail_path": str(thumb)})

        assert result.success, result.error
        assert result.data["video_id"] == "vid123"
        assert fake.calls == ["token", "thumbnail"]  # no upload, no new video
        log = json.loads((project / "artifacts" / "publish_log.json").read_text())
        assert log["metadata"]["youtube"]["thumbnails"][-1]["video_id"] == "vid123"
        assert len([e for e in log["entries"] if e["status"] == "published"]) == 1

    def test_without_a_recorded_upload_it_refuses(self, project, creds, no_probe, monkeypatch):
        _install(monkeypatch, FakeYouTube())
        thumb = project / "renders" / "poster.png"
        thumb.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        result = YouTubeUpload().execute({"operation": "set_thumbnail", "project_dir": str(project),
                                          "video_path": str(project / "renders" / "final.mp4"),
                                          "thumbnail_path": str(thumb)})
        assert not result.success and "no recorded upload" in result.error
