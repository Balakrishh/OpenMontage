# Director — Kids Science Story Time

## When To Use

The user asks for a kids science story / experiment episode for the Maya & Newton
series (ages 8-11, watercolor stills, narration, ~5 minutes), or names an
experiment id from `lib/kids_science/experiments.yaml`.

## The Contract

This pipeline is **unattended by design**. It has no approval gates; every stage
verifies its own output and stops with a written reason if a check fails. Do not
add gates, do not improvise science, do not hand-edit generated artifacts to make
a check pass.

| Source of truth | Holds | Who may change it |
|---|---|---|
| `lib/kids_science/series.yaml` | cast, identity blocks, voices, style, beats, music | the user |
| `lib/kids_science/experiments.yaml` | materials, steps, real result, approved facts, misconceptions, safety | the user (vetted entries only) |
| `projects/_series/kids-science/masters/` | recurring character reference portraits | minted once by `story_keyframes` |

The local writer can only phrase what these files contain: characters and props
are schema enums, steps and facts are handed to it per beat, and an independent
critic model rejects anything unsupported.

## Composition Runtime (series-level, decided once)

`render_runtime` is a **series** decision stored in `lib/kids_science/series.yaml`, not
a per-episode one — every episode must look the same. When it is unset, Present both
runtimes to the user before anything renders (AGENT_GUIDE "Present Both Composition
Runtimes"), with a recommendation tied to this format:

- **remotion** — the stock Explainer composition already does what story time needs:
  stills with slow Ken Burns moves, word-by-word read-along captions, crossfades.
  Tradeoff: the stock look is shared with other Explainer videos.
- **hyperframes** — HTML/GSAP compositions, good for bespoke page-turn transitions and
  storybook typography. Tradeoff: `story_assemble` has no hyperframes path yet; choosing
  it means building that path first.

Record the choice as a `render_runtime_selection` decision in the project's decision log
with both runtimes in `options_considered`, and write it to `series.yaml`. The runner
stops before the edit stage while it is unset; `story_assemble` refuses a runtime it
does not implement rather than silently using Remotion.

## Process

Run the episode with one command (it calls the registered stage tools and writes
normal checkpoints, so the Backlot board follows it live):

```
.venv/bin/python scripts/kids_science_episode.py <experiment-id>
.venv/bin/python scripts/kids_science_episode.py --list
```

Stages and their tools:

1. **script + scene_plan** — `kids_story_writer` (LM Studio: writer `KIDS_WRITER_MODEL`, critic `KIDS_CRITIC_MODEL`). Output `story_bible.json`, `script.json`, `scene_plan.json`, `story_review.json`.
2. **assets** — `story_keyframes` (FLUX.2 + vision checklist via `KIDS_VQA_MODEL`), then `story_narration` (Kokoro + per-line Whisper check).
3. **edit** — `story_assemble` (title card, stills on the real narration timing, Ken Burns motion, read-along captions, music bed).
4. **compose + publish** — `story_finalize` (checks the finished file; writes `export/` and `publish_log.json`). Export only — never upload.

## When A Stage Fails

Read `projects/<id>/run_report.json` and the stage's report artifact
(`story_review.json`, `keyframe_qa.json`, `narration_timing.json`,
`final_review.json`). Fix the cause, not the symptom:

- critic keeps rejecting a beat → the experiment entry is ambiguous; tighten its `facts`/`steps` in `experiments.yaml`, then re-run.
- a shot never clears hard checks → the shot description invites the failure (crowds, signs, "boy/girl" without the identity block); rerun the stage — seeds change — or simplify the prop list.
- a narration line keeps mis-transcribing → usually a tricky word; rerun from script.

Then `--resume` to skip completed stages.

## Publish Copy Rules

Copy is templated by `story_finalize` from the episode's own data and follows the
channel rule: engagement only — no tools, models or production tech named. AI
disclosure is the YouTube synthetic-media flag set at upload time.
