# Backlot — the living storyboard

A read-only local board that shows a production happening: pipeline stages
lighting up, the script as a screenplay page, the scene plan as a filmstrip
that fills in as assets generate, decisions, spend, and activity — all
derived from what the pipeline already writes to `projects/<id>/`.

```bash
python -m backlot open <project-id>   # start server if needed + open browser
python -m backlot open                # library view (all projects)
python -m backlot serve --port 4750   # run the server in the foreground
```

## How it stays live

No agent involvement. A `watchfiles` watcher on `projects/` publishes change
notifications over SSE; the browser refetches board state. State sources:

| Board element | Disk source |
|---|---|
| identity / rail order | `project.json` + `pipeline_defs/<type>.yaml` |
| stage states, gates, versions | `checkpoint_<stage>.json` + `history/` |
| script card / modal | `artifacts/script.json` |
| filmstrip cards | `scene_plan × script × asset_manifest` join |
| generating shimmer, activity | `events.jsonl` (written by `BaseTool` instrumentation) |
| cost meter | checkpoint `cost_snapshot` |
| renders | `renders/*.mp4` (+ root-level mp4 heuristic) |
| publish result | `artifacts/publish_log.json` (written by `youtube_upload`) |

Projects without checkpoints degrade gracefully to a "what the watcher
found" view — media, snapshots, renders.

## Publishing (the one write path)

Backlot observes; it does not act. The single exception is the **PUBLISH TO
YOUTUBE** button in the Renders section, added because the publish stage is
human-gated (`human_approval_default: true`) and the human is already looking
at the board.

The exception is kept as narrow as possible:

- The server writes nothing itself. It calls the `youtube_upload` tool, and the
  tool writes `artifacts/publish_log.json` and `checkpoint_publish.json` — the
  same files the publish stage has always produced. Project state keeps its
  shape; only the trigger moved.
- The result reaches the browser through the ordinary watcher → SSE path. There
  is no push channel and no server-held UI state.
- `POST /api/project/<id>/publish` requires an `X-Backlot-Publish: 1` header and
  `"confirm": true` in the body. The header cannot be sent cross-origin without a
  CORS preflight this app never grants, so no other page in the browser can
  reach the endpoint; the flag means a stray or replayed request cannot publish.
- One upload per project at a time (409 otherwise), and an upload of bytes that
  already went up is refused unless you tick "upload anyway".

Setup is one command — `python scripts/get_youtube_token.py` (see the
`# --- YouTube / Publish ---` block in `.env.example`). Without credentials the
dialog still opens and offers a **dry run**, which shows the exact payload and
uploads nothing.

Format is measured, not assumed: the render is probed with ffprobe, and portrait
+ ≤ 60s is published as a Short (`#Shorts` appended). The dialog shows the
verdict and lets you override it.

**Replay**: a completed run can be scrubbed end-to-end (▶ REPLAY RUN on the
board) — reconstructed from checkpoint history and event timestamps.

Try it without a real production:

```bash
python scripts/backlot_simulate_run.py          # live demo run (~1 min)
python -m backlot open backlot-demo-run
```

Design doc: `internal/design/LIVING_STORYBOARD.md`.
