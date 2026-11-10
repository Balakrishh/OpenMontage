"""Tests for the board's publish endpoint — Backlot's one write path.

The upload itself is covered in tests/tools/test_youtube_upload.py; here the
subject is the HTTP surface: who is allowed to trigger a publish, what a
malformed request does, and that a real upload is never reachable by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backlot import publish as publish_mod
from backlot import server as server_mod
from backlot import state as state_mod
from tools.base_tool import ToolResult

HEADERS = {"X-Backlot-Publish": "1"}


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(state_mod, "PROJECTS_DIR", root)
    monkeypatch.setattr(server_mod, "PROJECTS_DIR", root)
    monkeypatch.setattr(server_mod, "_summary_cache", {})
    monkeypatch.setattr(
        server_mod, "_PROJECTS_ROOT_STR", __import__("os").path.normcase(str(root.resolve()))
    )
    monkeypatch.setattr(server_mod, "THUMB_CACHE_DIR", tmp_path / "thumbs")
    return root


@pytest.fixture
def client(projects_root, monkeypatch):
    async def no_watch():
        return None

    monkeypatch.setattr(server_mod, "_watch_projects", no_watch)
    # Keep the developer's real .env (and its YouTube token) out of the tests.
    monkeypatch.setattr(publish_mod, "_load_env", lambda: None)
    # Registry discovery elsewhere in the suite may already have loaded .env.
    monkeypatch.delenv("YOUTUBE_DEFAULT_PRIVACY", raising=False)
    with TestClient(server_mod.create_app(), base_url="http://127.0.0.1:4750") as c:
        yield c


@pytest.fixture
def project(projects_root) -> Path:
    project_dir = projects_root / "film"
    (project_dir / "renders").mkdir(parents=True)
    (project_dir / "artifacts").mkdir(parents=True)
    (project_dir / "project.json").write_text(
        json.dumps({"project_id": "film", "title": "The Film", "pipeline_type": "cinematic"}),
        encoding="utf-8",
    )
    (project_dir / "renders" / "final.mp4").write_bytes(b"\x00" * 1024)
    return project_dir


@pytest.fixture
def no_probe(monkeypatch):
    from tools.publishers.youtube_upload import YouTubeUpload

    monkeypatch.setattr(
        YouTubeUpload, "probe",
        staticmethod(lambda path: {"width": 1080, "height": 1920, "duration_seconds": 20.0}),
    )


@pytest.fixture
def fake_upload(monkeypatch):
    """Replace the real upload with a recorder — no network, no channel."""
    from tools.publishers.youtube_upload import YouTubeUpload

    calls: list[dict] = []

    def execute(self, inputs):
        calls.append(inputs)
        return ToolResult(success=True, data={
            "video_id": "vid123",
            "url": "https://www.youtube.com/shorts/vid123",
            "short": True,
            "privacy": inputs.get("privacy"),
            "already_published": False,
        })

    monkeypatch.setattr(YouTubeUpload, "execute", execute)
    return calls


def _body(**overrides):
    body = {"confirm": True, "path": "renders/final.mp4", "title": "The Film"}
    body.update(overrides)
    return body


class TestPreflight:
    def test_reports_render_facts_and_shorts_verdict(self, client, project, no_probe):
        response = client.get("/api/project/film/publish/preflight", params={"path": "renders/final.mp4"})

        assert response.status_code == 200
        body = response.json()
        assert body["render"]["size_bytes"] == 1024
        assert body["is_short"] is True
        assert body["short_max_seconds"] == 60
        assert body["defaults"]["privacy"] == "unlisted"

    def test_reports_missing_credentials_instead_of_failing(self, client, project, no_probe, monkeypatch):
        for key in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
            monkeypatch.delenv(key, raising=False)

        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()

        assert body["credentials_ready"] is False
        assert "YOUTUBE_REFRESH_TOKEN" in body["missing_env"]
        assert "get_youtube_token.py" in body["setup_hint"]

    def test_suggests_metadata_from_a_previous_publish(self, client, project, no_probe):
        (project / "artifacts" / "publish_log.json").write_text(json.dumps({
            "version": "1.0",
            "entries": [{
                "platform": "youtube", "status": "published", "timestamp": "2026-08-01T00:00:00Z",
                "video_id": "v1",
                "metadata_used": {"title": "Accepted Title", "description": "d", "hashtags": ["#a"]},
            }],
            # Matched by render: the tool records which file each upload was.
            "metadata": {"youtube": {"uploads": [{"video_id": "v1", "source": "renders/final.mp4"}]}},
        }), encoding="utf-8")

        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()

        assert body["suggested"]["title"] == "Accepted Title"
        assert body["suggested"]["tags"] == ["a"]

    def test_suggests_the_publish_stage_copy_for_a_local_export(self, client, project, no_probe):
        # The publish director's SEO copy for a run that was only exported.
        (project / "artifacts" / "publish_log.json").write_text(json.dumps({
            "version": "1.0",
            "entries": [{"platform": "youtube", "status": "exported", "timestamp": "2026-08-01T00:00:00Z"}],
            "metadata": {
                "suggested_title": "The Film — A Story",
                "suggested_description": "logline",
                "suggested_tags": ["short film"],
            },
        }), encoding="utf-8")

        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()

        assert body["suggested"]["title"] == "The Film — A Story"
        assert body["suggested"]["description"] == "logline"
        assert body["suggested"]["tags"] == ["short film"]

    def test_a_trailer_gets_its_own_copy_not_the_films_chapters(self, client, project, no_probe):
        (project / "renders" / "trailer.mp4").write_bytes(b"\x00" * 512)
        (project / "artifacts" / "publish_log.json").write_text(json.dumps({
            "version": "1.0",
            "entries": [],
            "metadata": {
                "suggested_title": "The Film",
                "suggested_chapters": [{"start_seconds": 0, "title": "Opening"}],
                "suggested_category_id": "1",
                "contains_synthetic_media": True,
                "suggested_by_render": {"renders/trailer.mp4": {"title": "The Film — Trailer"}},
            },
        }), encoding="utf-8")

        film = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()["suggested"]
        trailer = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/trailer.mp4"}
        ).json()["suggested"]

        assert film["chapters"] == [{"start_seconds": 0, "title": "Opening"}]
        assert film["category_id"] == "1" and film["contains_synthetic_media"] is True
        assert trailer["title"] == "The Film — Trailer"
        assert trailer["chapters"] == []

    def test_names_the_finished_cut_so_the_dialog_can_flag_an_old_version(self, client, project, no_probe):
        (project / "renders" / "final_v3.mp4").write_bytes(b"\x00" * 256)
        (project / "artifacts" / "publish_log.json").write_text(json.dumps({
            "version": "1.0", "entries": [],
            "metadata": {"hero_output": "projects/film/renders/final.mp4", "suggested_title": "The Film"},
        }), encoding="utf-8")

        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final_v3.mp4"}
        ).json()

        assert body["hero_render"] == "renders/final.mp4"

    @pytest.mark.parametrize(("env", "expected"), [(None, "unlisted"), ("public", "public"), ("bogus", "unlisted")])
    def test_starting_visibility_follows_the_owner_setting(self, client, project, no_probe, monkeypatch, env, expected):
        if env is None:
            monkeypatch.delenv("YOUTUBE_DEFAULT_PRIVACY", raising=False)
        else:
            monkeypatch.setenv("YOUTUBE_DEFAULT_PRIVACY", env)
        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()
        assert body["defaults"]["privacy"] == expected

    def test_falls_back_to_the_project_marker(self, client, project, no_probe):
        body = client.get(
            "/api/project/film/publish/preflight", params={"path": "renders/final.mp4"}
        ).json()
        assert body["suggested"]["title"].startswith("The Film")

    @pytest.mark.parametrize(
        ("path", "status"),
        [
            ("../../etc/passwd", 403),
            ("renders/missing.mp4", 404),
            ("project.json", 400),  # not a video
        ],
    )
    def test_rejects_bad_paths(self, client, project, path, status):
        response = client.get("/api/project/film/publish/preflight", params={"path": path})
        assert response.status_code == status


class TestPublishGuards:
    def test_requires_the_publish_header(self, client, project, fake_upload):
        response = client.post("/api/project/film/publish", json=_body())
        assert response.status_code == 403
        assert fake_upload == []

    def test_requires_an_explicit_confirmation(self, client, project, fake_upload):
        response = client.post("/api/project/film/publish", json=_body(confirm=False), headers=HEADERS)
        assert response.status_code == 400
        assert "confirmation" in response.json()["detail"]
        assert fake_upload == []

    def test_requires_a_title(self, client, project, fake_upload):
        response = client.post("/api/project/film/publish", json=_body(title="  "), headers=HEADERS)
        assert response.status_code == 400
        assert fake_upload == []

    def test_rejects_a_path_outside_the_project(self, client, project, fake_upload):
        response = client.post(
            "/api/project/film/publish", json=_body(path="../other/final.mp4"), headers=HEADERS)
        assert response.status_code == 403
        assert fake_upload == []

    def test_rejects_an_unknown_privacy_value(self, client, project, fake_upload):
        response = client.post(
            "/api/project/film/publish", json=_body(privacy="everyone"), headers=HEADERS)
        assert response.status_code == 400
        assert fake_upload == []

    def test_rejects_an_unknown_project(self, client, project, fake_upload):
        response = client.post("/api/project/nope/publish", json=_body(), headers=HEADERS)
        assert response.status_code == 404
        assert fake_upload == []

    def test_a_second_publish_while_one_runs_is_refused(self, client, project, fake_upload, monkeypatch):
        lock = publish_mod._lock_for("film")
        assert lock.acquire(blocking=False)
        try:
            response = client.post("/api/project/film/publish", json=_body(), headers=HEADERS)
        finally:
            lock.release()

        assert response.status_code == 409
        assert fake_upload == []


class TestThumbnailGuard:
    def test_rejects_a_thumbnail_outside_the_project(self, client, project, fake_upload):
        res = client.post("/api/project/film/publish", headers=HEADERS, json={
            "confirm": True, "path": "renders/final.mp4", "title": "T", "thumbnail_path": "../../etc/hosts",
        })
        assert res.status_code == 403
        assert fake_upload == []


class TestPublishHappyPath:
    def test_passes_the_dialog_choices_through_to_the_tool(self, client, project, fake_upload):
        response = client.post("/api/project/film/publish", headers=HEADERS, json=_body(
            description="A film.", tags="cinema, ai", privacy="private", short=True,
        ))

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["video_id"] == "vid123"

        inputs = fake_upload[0]
        assert inputs["title"] == "The Film"
        assert inputs["tags"] == ["cinema", "ai"]
        assert inputs["privacy"] == "private"
        assert inputs["short"] is True
        assert inputs["video_path"].endswith("renders/final.mp4")
        # A board-triggered publish is the human approval for the gated stage.
        assert inputs["write_checkpoint"] is True

    def test_dry_run_does_not_claim_the_stage(self, client, project, fake_upload):
        client.post("/api/project/film/publish", headers=HEADERS, json=_body(dry_run=True))
        assert fake_upload[0]["dry_run"] is True
        assert fake_upload[0]["write_checkpoint"] is False

    def test_a_failed_upload_is_an_outcome_not_a_500(self, client, project, monkeypatch):
        from tools.publishers.youtube_upload import YouTubeUpload

        monkeypatch.setattr(
            YouTubeUpload, "execute",
            lambda self, inputs: ToolResult(success=False, error="YouTube upload failed (503)"),
        )

        response = client.post("/api/project/film/publish", json=_body(), headers=HEADERS)

        assert response.status_code == 200
        assert response.json() == {"success": False, "error": "YouTube upload failed (503)"}

    def test_publish_wakes_the_board(self, client, project, fake_upload, monkeypatch):
        published: list[str] = []
        monkeypatch.setattr(server_mod.hub, "publish", published.append)

        client.post("/api/project/film/publish", json=_body(), headers=HEADERS)

        assert published == ["film"]


class TestReadOnlyElsewhere:
    def test_no_other_route_accepts_a_write(self, client):
        """Publishing (upload + thumbnail on an existing upload) is the *only* write on the board."""
        app = server_mod.create_app()
        writable = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
            if method not in {"GET", "HEAD", "OPTIONS"}
        }
        assert writable == {("/api/project/{project_id}/publish", "POST"),
                            ("/api/project/{project_id}/publish/thumbnail", "POST")}


class TestSetThumbnail:
    def _thumb(self, project):
        (project / "renders" / "poster.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        return "renders/poster.png"

    def test_requires_header_and_confirmation(self, client, project):
        thumb = self._thumb(project)
        assert client.post("/api/project/film/publish/thumbnail",
                           json={"confirm": True, "video_id": "v1", "thumbnail_path": thumb}).status_code == 403
        res = client.post("/api/project/film/publish/thumbnail", headers=HEADERS,
                          json={"video_id": "v1", "thumbnail_path": thumb})
        assert res.status_code == 400

    def test_rejects_a_thumbnail_outside_the_project(self, client, project):
        res = client.post("/api/project/film/publish/thumbnail", headers=HEADERS,
                          json={"confirm": True, "video_id": "v1", "thumbnail_path": "../../etc/hosts"})
        assert res.status_code == 403

    def test_passes_the_existing_video_to_the_tool(self, client, project, monkeypatch):
        from tools.publishers.youtube_upload import YouTubeUpload

        seen = {}

        def execute(self, inputs):
            seen.update(inputs)
            return ToolResult(success=True, data={"video_id": "v1", "thumbnail_set": True})

        monkeypatch.setattr(YouTubeUpload, "execute", execute)
        res = client.post("/api/project/film/publish/thumbnail", headers=HEADERS,
                          json={"confirm": True, "path": "renders/final.mp4", "thumbnail_path": self._thumb(project)})
        assert res.status_code == 200 and res.json()["success"] is True
        assert seen["operation"] == "set_thumbnail" and seen["video_path"].endswith("renders/final.mp4")
