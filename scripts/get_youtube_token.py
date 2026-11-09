#!/usr/bin/env python3
"""One-time OAuth bootstrap for the YouTube publisher.

Ported from socialsmith's `scripts/get_youtube_token.py` (local-loopback consent
instead of OAuth-Playground copy-paste) and adapted to OpenMontage's single
top-level `.env`: no tenants, and the client id/secret are written back too, so
one run leaves the repo fully configured.

SETUP (once, in Google Cloud → APIs & Services):
  1. Enable **YouTube Data API v3**.
  2. Create an OAuth 2.0 Client ID of type **Desktop app** (it allows the
     localhost redirect automatically). If you use a "Web application" client
     instead, add this exact Authorized redirect URI: http://localhost:8765/
  3. While the app is in "Testing", add your own Google account as a test user.

USAGE:
    python scripts/get_youtube_token.py                     # prompts for client id/secret if not in .env
    python scripts/get_youtube_token.py --client-id X --client-secret Y
    python scripts/get_youtube_token.py --print-only        # show the token, write nothing

The refresh token is long-lived: the uploader exchanges it for a short-lived
access token on every publish. Re-run this only if you revoke access, rotate the
OAuth client, or want to point OpenMontage at a different channel.

NOTE ON SCOPE: this asks for `youtube.upload` only — enough to upload a video
and set its thumbnail, and NOT enough to read, edit, or delete anything already
on the channel.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8765
REDIRECT = f"http://localhost:{PORT}/"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

_received: dict[str, str | None] = {}

_PAGE = b"""<!doctype html><meta charset="utf-8">
<body style="font:16px/1.5 system-ui;padding:3rem;background:#111;color:#eee">
<h2>Authorized.</h2><p>You can close this tab and return to your terminal.</p>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _received["code"] = query.get("code", [None])[0]
        _received["error"] = query.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_PAGE)

    def log_message(self, *_args) -> None:  # keep the terminal clean
        pass


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _write_env(path: Path, updates: dict[str, str]) -> None:
    """Set keys in .env, replacing any existing line for each key."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for key, value in updates.items():
        lines = [ln for ln in lines if not ln.startswith(f"{key}=")]
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a YouTube refresh token for OpenMontage.")
    parser.add_argument("--client-id")
    parser.add_argument("--client-secret")
    parser.add_argument("--print-only", action="store_true", help="print the values instead of writing .env")
    args = parser.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("requests is not installed — run `make install-dev` (or `pip install requests`) first.")

    env = _read_env(ENV_PATH)
    client_id = args.client_id or os.environ.get("YOUTUBE_CLIENT_ID") or env.get("YOUTUBE_CLIENT_ID")
    client_secret = (
        args.client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET") or env.get("YOUTUBE_CLIENT_SECRET")
    )
    if not client_id:
        client_id = input("OAuth client ID: ").strip()
    if not client_secret:
        # getpass so the secret does not land in shell history or scrollback.
        client_secret = getpass.getpass("OAuth client secret (hidden): ").strip()
    if not client_id or not client_secret:
        sys.exit("Need both a client id and a client secret.")

    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",  # forces a refresh_token even on re-consent
    })
    print(
        "\nOpening your browser. Sign in as the CHANNEL OWNER and pick the channel "
        "OpenMontage should publish to.\nIf the browser does not open, paste this URL:\n"
        f"{url}\n"
    )
    try:
        server = HTTPServer(("localhost", PORT), _Handler)
    except OSError as exc:
        sys.exit(f"Cannot listen on {REDIRECT} ({exc}). Close whatever is using port {PORT} and re-run.")
    webbrowser.open(url)
    server.handle_request()  # blocks until Google redirects back
    server.server_close()

    if _received.get("error"):
        sys.exit(f"Authorization denied: {_received['error']}")
    code = _received.get("code")
    if not code:
        sys.exit(f"No authorization code received. Confirm the redirect URI is exactly {REDIRECT}")

    response = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT,
    }, timeout=30)
    if not response.ok:
        sys.exit(f"Token exchange failed ({response.status_code}): {response.text[:300]}")
    token = response.json()
    refresh = token.get("refresh_token")
    if not refresh:
        sys.exit(
            "No refresh_token returned. Google only issues one on first consent — revoke "
            "OpenMontage at myaccount.google.com (Security → Third-party access) and re-run."
        )

    # Which channel does this token actually control? An account can own
    # several, and the token is bound to whichever was picked at consent.
    try:
        channels = requests.get(
            CHANNELS_URL,
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {token.get('access_token')}"},
            timeout=30,
        ).json()
        items = channels.get("items") or []
        if items:
            print(f"\n>>> This token uploads to: \"{items[0]['snippet']['title']}\" (id: {items[0]['id']})")
            print(">>> Wrong channel? Re-run and choose the right one at the 'Choose a channel' step.")
        else:
            print("\n[!] No channel found for this account — create one before publishing.")
    except Exception as exc:  # verification is a courtesy, never a blocker
        print(f"\n[!] Channel verification skipped ({exc}) — confirm the channel before publishing.")

    if args.print_only:
        print("\nAdd these to .env:\n")
        print(f"YOUTUBE_CLIENT_ID={client_id}")
        print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
        print(f"YOUTUBE_REFRESH_TOKEN={refresh}")
        return

    _write_env(ENV_PATH, {
        "YOUTUBE_CLIENT_ID": client_id,
        "YOUTUBE_CLIENT_SECRET": client_secret,
        "YOUTUBE_REFRESH_TOKEN": refresh,
    })
    print(f"\n[+] Wrote YOUTUBE_CLIENT_ID / _SECRET / _REFRESH_TOKEN into {ENV_PATH}")
    print("[+] Done — restart the Backlot server and the Publish button goes live.")


if __name__ == "__main__":
    main()
