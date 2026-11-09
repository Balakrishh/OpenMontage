"""Browser test for the publish button — the click path, end to end.

The unit tests cover the tool and the endpoint. What they cannot show is that a
human can actually reach them: that the button renders in the render meta row,
that the dialog opens with the render's real facts, and that nothing uploads
until a second, deliberate click.

This runs the whole stack — real server subprocess, real board JS, real Chromium
— and drives it in dry-run mode, so it proves the path without a channel, a
credential, or a byte of network traffic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PORT = 4899


@pytest.fixture(scope="module")
def publish_board(tmp_path_factory):
    """A one-project board whose render is a real (tiny) vertical video."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to build the render fixture")

    root = tmp_path_factory.mktemp("publish-projects")
    project = root / "vertical-cut"
    (project / "renders").mkdir(parents=True)
    (project / "artifacts").mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({
        "version": "1.0", "project_id": "vertical-cut",
        "title": "Vertical Cut", "pipeline_type": "cinematic",
    }), encoding="utf-8")
    (project / "checkpoint_compose.json").write_text(json.dumps({
        "version": "1.0", "project_id": "vertical-cut", "pipeline_type": "cinematic",
        "stage": "compose", "status": "completed", "timestamp": "2026-08-24T00:00:00Z",
        "artifacts": {},
    }), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=#1b2b3b:s=180x320:d=2", "-pix_fmt", "yuv420p",
         str(project / "renders" / "teaser_9x16.mp4")],
        check=True, capture_output=True, timeout=60,
    )

    env = dict(os.environ)
    env["OPENMONTAGE_PROJECTS_DIR"] = str(root)
    # No credentials: the dialog must degrade to "dry run only", not break.
    for key in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
        env.pop(key, None)
    server = subprocess.Popen(
        [sys.executable, "-m", "backlot", "serve", "--port", str(PORT)],
        cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=1):
                break
        except Exception:
            time.sleep(0.2)
    else:
        server.terminate()
        raise RuntimeError("Backlot server did not become healthy")

    try:
        yield f"http://127.0.0.1:{PORT}", project
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


@pytest.fixture(scope="module")
def page(publish_board):
    base, _project = publish_board
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base}/p/vertical-cut?static=1", wait_until="networkidle")
        yield page
        browser.close()


def _open_dialog(page):
    if not page.locator(".publish-modal").is_visible():
        page.locator(".pub-btn").click()
        page.wait_for_selector(".publish-modal .pub-field", timeout=10_000)
    return page.locator(".publish-modal")


class TestPublishDialog:
    def test_button_sits_with_the_render(self, page):
        button = page.locator(".pub-btn")
        assert button.count() == 1
        assert "PUBLISH TO YOUTUBE" in button.inner_text()

    def test_dialog_opens_with_the_render_facts(self, page):
        dialog = _open_dialog(page)
        facts = dialog.locator(".pub-facts").inner_text()
        assert "teaser_9x16.mp4" in facts
        assert "180×320" in facts  # probed, not guessed
        # Portrait and 2s — the format select should have decided "Short".
        assert "DETECTED SHORT" in dialog.locator("select").first.inner_text() or \
               "DETECTED SHORT" in dialog.inner_text()
        # Title falls back to the project marker, qualified by which cut this
        # is — a teaser and the hero film must not both suggest the same title.
        assert dialog.locator("input[type=text]").first.input_value() == "Vertical Cut — teaser 9x16"

        # Conditional sections must not leak "null" text nodes into the dialog
        # (Node.append(null) renders the literal string), and a sub-megabyte
        # render must not read as "0.0 MB".
        text = dialog.inner_text()
        assert "null" not in text.lower()
        assert "0.0 MB" not in text
        assert "KB" in facts

    def test_uncredentialed_board_offers_a_dry_run_not_an_upload(self, page):
        dialog = _open_dialog(page)
        assert "NOT CONNECTED TO A CHANNEL" in dialog.inner_text()
        confirm = dialog.locator(".pub-go")
        assert confirm.is_disabled()

        dialog.locator(".pub-checks input[type=checkbox]").first.check()
        assert confirm.is_enabled()
        assert "DRY RUN" in confirm.inner_text()

    def test_dry_run_shows_the_payload_and_uploads_nothing(self, page, publish_board):
        _base, project = publish_board
        dialog = _open_dialog(page)
        dialog.locator(".pub-checks input[type=checkbox]").first.check()
        dialog.locator(".pub-go").click()

        page.wait_for_selector(".pub-payload", timeout=15_000)
        payload = json.loads(page.locator(".pub-payload").inner_text())
        assert payload["snippet"]["title"].endswith("#Shorts")
        assert payload["status"]["privacyStatus"] == "unlisted"
        assert "NOTHING UPLOADED" in page.locator(".pub-title").inner_text()

        # The gated stage must not have been claimed by a dry run.
        assert not (project / "artifacts" / "publish_log.json").exists()
        assert not (project / "checkpoint_publish.json").exists()

    def test_escape_closes_the_dialog(self, page):
        _open_dialog(page)
        page.keyboard.press("Escape")
        assert not page.locator(".publish-modal").is_visible()
