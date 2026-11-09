"""Story writer for kids science story time — local LLM, locked plot, independent critic.

One call produces an episode's story bible, script and shot list from a
vetted experiment, with no human in the loop:

1. Bible: the writer model fills one fixed slot per series beat. Characters
   and props are schema enums, so nothing outside the bible can be emitted.
2. Script: written beat by beat, each beat given exactly its job, its word
   budget, its assigned steps / approved facts, and the beats before it.
   Code checks every beat (length, unknown names, banned claims) and
   rewrites with precise feedback.
3. Critic: a different model family checks the whole script against the
   experiment entry — science, step order, materials, continuity, age fit.
   Failing beats are rewritten with the critic's notes. If anything still
   fails after the last round, the tool fails loudly with the report; it
   never ships a script it could not verify.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

WRITER_MODEL = "qwen/qwen3-32b"
CRITIC_MODEL = "openai/gpt-oss-120b"


def _writer_system(model: str) -> str:
    from lib import kids_science as ks

    s = ks.series()
    cast = "; ".join(f"{c['name']} — {c['role']}" for c in s["cast"].values())
    # Qwen3 thinks by default; for constrained JSON the think block only costs time.
    prefix = "/no_think\n" if "qwen3" in model.lower() else ""
    return prefix + (
        f"You write a read-aloud story for the series \"{s['series']['name']}\" for {s['series']['audience']}. "
        f"The only characters are: {cast}. Never add another character, person, pet or named thing. "
        f"Every episode happens in {s['setting']['description'].strip()}.\n"
        "Style: warm, vivid and concrete; short sentences a child can follow by ear; wonder and gentle humor; "
        "speak to the viewer sometimes (\"What do you think will happen?\").\n"
        "Speaker: 'narrator' tells the whole story and reads the boys' words inside quotation marks "
        "(for example: \"Look!\" said Dev.). Keep the boys' voices distinct: the 6-year-old is excited and "
        "full of questions, the 9-year-old is patient and explains.\n"
        "Science: use ONLY the facts you are given, in simpler words if you like. Never add a scientific claim, "
        "saying, number, material or step that you were not given. If a beat runs short of facts, fill it with "
        "story — reactions, wonder, a question to the viewer — never with more science."
    )


def _cast_names() -> str:
    from lib import kids_science as ks

    return " and ".join(f"{c['name']} ({c['role'].split(' who ')[0]})" for c in ks.series()["cast"].values())


def materials_lines(exp: dict[str, Any]) -> list[dict[str, str]]:
    """The materials beat, written by code from the library — nothing to drift.

    The model kept adding actions or results here; the list is fully known, so
    a model adds nothing but risk.
    """
    from lib import kids_science as ks

    a, d = [c["name"] for c in ks.series()["cast"].values()][:2]
    items = list(exp["materials"])
    listed = ", ".join(items[:-1]) + f", and {items[-1]}" if len(items) > 1 else items[0]
    return [
        {"speaker": "narrator", "text": f"Here is everything {a} and {d} need for today's experiment: {listed}."},
        {"speaker": "narrator", "text": f"\"Let's count them together,\" said {a}. {d} counted each one out loud, "
                                        "grinning from ear to ear."},
        {"speaker": "narrator", "text": "Can you find all of them at home? Gather yours with a grown-up, and let's begin!"},
    ]


def hook_moment(exp: dict[str, Any]) -> str:
    """The hook opens on an APPROVED everyday example, never an invented phenomenon."""
    examples = exp.get("real_world") or [exp["question"]]
    return examples[-1]


_QUOTED = re.compile(r"[\"“”']([^\"“”]{12,}?)[\"“”']")


def remove_quoted(lines: list[dict[str, str]], problems: list[str]) -> list[dict[str, str]] | None:
    """Delete sentences the critic quoted for removal, if they appear verbatim. None if nothing applied."""
    targets = [m.group(1).strip() for p in problems if re.search(r"\b(remove|delete|cut|drop)\b", p, re.I)
               for m in _QUOTED.finditer(p)]
    if not targets:
        return None
    out, hit = [], False
    for line in lines:
        text = line["text"]
        for t in targets:
            if t in text:
                text, hit = text.replace(t, " "), True
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append({**line, "text": text})
    return out if hit else None


class KidsStoryWriter(BaseTool):
    name = "kids_story_writer"
    version = "0.1.0"
    tier = ToolTier.CORE
    capability = "script_generation"
    provider = "lmstudio"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    install_instructions = (
        "Install LM Studio, load (or allow JIT loading of) the writer and critic models. "
        "Config: KIDS_WRITER_MODEL (default qwen/qwen3-32b), KIDS_CRITIC_MODEL (default openai/gpt-oss-120b)."
    )
    agent_skills: list[str] = []
    capabilities = ["story_bible", "script_generation", "script_critique"]
    supports = {"offline": True, "schema_constrained": True, "independent_critic": True}
    best_for = ["kids science story-time episodes from the vetted experiment library, unattended"]
    not_good_for = ["experiments that are not in lib/kids_science/experiments.yaml", "open-ended creative writing"]

    input_schema = {
        "type": "object",
        "required": ["experiment_id", "project_dir"],
        "properties": {
            "experiment_id": {"type": "string"},
            "project_dir": {"type": "string"},
            "writer_model": {"type": "string"},
            "critic_model": {"type": "string"},
            "critic_rounds": {"type": "integer", "default": 4},
        },
    }
    resource_profile = ResourceProfile(cpu_cores=2, ram_mb=100000, vram_mb=0, disk_mb=10, network_required=False)
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["experiment_id", "project_dir"]
    side_effects = [
        "writes artifacts/story_bible.json, script.json, scene_plan.json, story_review.json",
        "loads and then unloads LM Studio models",
    ]
    user_visible_verification = ["Read artifacts/story_review.json: every beat ok, no open problems"]

    def get_status(self) -> ToolStatus:
        from lib import lmstudio

        return ToolStatus.AVAILABLE if lmstudio.available() or Path(lmstudio.lms_cli()).is_file() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 900.0

    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        from lib import kids_science as ks
        from lib import lmstudio

        start = time.time()
        try:
            exp = ks.experiment(inputs["experiment_id"])
        except KeyError as exc:
            return ToolResult(success=False, error=str(exc))
        project = Path(inputs["project_dir"])
        (project / "artifacts").mkdir(parents=True, exist_ok=True)
        writer = inputs.get("writer_model") or os.environ.get("KIDS_WRITER_MODEL", WRITER_MODEL)
        critic = inputs.get("critic_model") or os.environ.get("KIDS_CRITIC_MODEL", CRITIC_MODEL)
        rounds = int(inputs.get("critic_rounds", 4))
        log: list[str] = []

        try:
            bible = self._bible(writer, exp, log)
            steps = ks.assign_steps(exp)
            script: dict[str, list[dict[str, str]]] = {}
            for beat in ks.beats():
                if beat["id"] == "materials":
                    script["materials"] = materials_lines(exp)
                    continue
                script[beat["id"]] = self._write_beat(writer, exp, bible, beat, steps, script, [], log)

            review: dict[str, list[str]] = {}
            for rnd in range(1, rounds + 1):
                review = self._critique(critic, exp, bible, steps, script)
                failing = {k: v for k, v in review.items() if v}
                log.append(f"critic round {rnd}: {len(failing)} beat(s) need work: {sorted(failing)}")
                if not failing:
                    break
                if rnd == rounds:
                    break
                for beat in ks.beats():
                    if beat["id"] in failing:
                        script[beat["id"]] = self._write_beat(
                            writer, exp, bible, beat, steps, script, failing[beat["id"]], log
                        )
            failing = {k: v for k, v in review.items() if v}
            if failing:
                # Last chance, no model involved: apply quoted removals exactly
                # and let the critic look once more.
                applied = False
                for bid, notes in failing.items():
                    trimmed = remove_quoted(script[bid], notes)
                    beat = next(b for b in ks.beats() if b["id"] == bid)
                    if trimmed is not None and not ks.check_beat_lines(beat, trimmed):
                        script[bid], applied = trimmed, True
                        log.append(f"{bid}: final pass applied the critic's quoted removal in code")
                if applied:
                    review = self._critique(critic, exp, bible, steps, script)
                    log.append(f"critic final check: {sum(bool(v) for v in review.values())} beat(s) still need work")
            shots: list = []
            if not any(review.values()):
                # Pictures come last: planned from the exact words, then held to the step timeline.
                shots = self._plan_pictures(writer, exp, bible, script, log)
                self._review_shots(critic, exp, shots, log)
        except lmstudio.LMStudioError as exc:
            return ToolResult(success=False, error=f"local model failed: {exc}")
        finally:
            lmstudio.unload_all()

        open_problems = {k: v for k, v in review.items() if v}
        self._write_artifacts(project, exp, bible, steps, script, review, log, writer, critic, shots)
        if open_problems:
            return ToolResult(
                success=False,
                error=(
                    f"script still has problems after {rounds} critic rounds: "
                    + "; ".join(f"{k}: {v[0]}" for k, v in open_problems.items())
                    + " — see artifacts/story_review.json"
                ),
                artifacts=[str(project / "artifacts" / "story_review.json")],
            )
        words = sum(ks.word_count(l["text"]) for lines in script.values() for l in lines)
        return ToolResult(
            success=True,
            data={
                "episode_title": bible["episode_title"],
                "words": words,
                "beats": len(script),
                "shots": len(shots),
                "motion_shots": ks.pick_motion(shots),
                "log": log,
            },
            artifacts=[str(project / "artifacts" / n) for n in ("story_bible.json", "script.json", "scene_plan.json", "story_review.json")],
            duration_seconds=round(time.time() - start, 1),
            model=f"{writer} + critic {critic}",
        )

    # ------------------------------------------------------------------

    def _bible(self, model: str, exp: dict[str, Any], log: list[str]) -> dict[str, Any]:
        from lib import kids_science as ks
        from lib import lmstudio

        steps = ks.assign_steps(exp)
        beat_lines = []
        for b in ks.beats():
            extra = ""
            if b["id"] in steps:
                extra = " Steps: " + " ".join(steps[b["id"]])
            elif b["id"] == "materials":
                extra = " Materials: " + ", ".join(exp["materials"])
            elif b["id"] == "observe":
                extra = " What happens: " + exp["observe"]
            elif b["id"] == "hook":
                extra = " Everyday moment: " + hook_moment(exp)
            beat_lines.append(f"- {b['id']} ({b['seconds']}s): {b['purpose']}{extra}")
        user = (
            f"Plan one episode around this experiment.\nTitle idea: {exp['title']}\nQuestion: {exp['question']}\n"
            f"Materials: {', '.join(exp['materials'])}\n\nBeats, in order:\n" + "\n".join(beat_lines) + "\n\n"
            "For each beat write a one-sentence summary of what happens in it — only that beat's job, nothing "
            "from later beats. episode_title: a catchy title for kids, max 8 words."
        )
        bible = lmstudio.chat_json(model, _writer_system(model), user, ks.bible_schema(exp), temperature=0.5, max_tokens=3000)
        log.append(f"bible: '{bible['episode_title']}'")
        return bible

    def _plan_pictures(self, model, exp, bible, script, log):
        """One picture per group of narration lines, described from those exact words."""
        from lib import kids_science as ks
        from lib import lmstudio

        timeline = dict(line[2:].split(": ", 1) for line in ks.picture_timeline(exp))
        pictures: dict[str, list[dict]] = {}
        for b in ks.beats():
            lines = script[b["id"]]
            groups = ks.group_lines(lines)
            kinds = ks.shot_kinds(b["id"], groups, lines, exp)
            slots = []
            for n, (g, kind) in enumerate(zip(groups, kinds), 1):
                words = " ".join(lines[i]["text"] for i in g)
                how = {
                    "plate": "EXPERIMENT CLOSE-UP (painted separately from the step facts — leave description empty, characters and props empty)",
                    "reaction": "REACTION CLOSE-UP of the boys' faces — description = ONLY their expression and gesture (e.g. 'grinning, Dev pointing'), no objects",
                    "scene": "PICTURE — description = what an illustrator draws so a child sees exactly what these words say",
                }[kind]
                slots.append(f"{n}. [{how}] Words spoken: \"{words}\"")
            user = (
                f"Episode: {bible['episode_title']} — beat '{b['id']}' ({b['title']}).\n"
                f"What the scene may show at this point: {timeline.get(b['id'], '')}\n\n"
                "Plan one picture for each numbered slot, in order. Each picture must show what ITS words say — "
                "not other beats, not the result early. Use only the given characters and props. Natural poses: "
                "people stand on the floor or sit on chairs, never on tables; nothing floats. must_show: up to 2 "
                "short visual facts (poses, expressions, positions) — never names, text or sounds.\n\n"
                + "\n".join(slots)
            )
            out = lmstudio.chat_json(model, _writer_system(model), user, ks.shot_plan_schema(exp, len(groups)),
                                     temperature=0.4, max_tokens=2500)
            pictures[b["id"]] = out["pictures"]
        shots = ks.plan_shots(exp, script, pictures)
        log.append(f"pictures: {len(shots)} planned from the narration "
                   f"({sum(s.kind == 'plate' for s in shots)} experiment close-ups, "
                   f"{sum(s.kind == 'reaction' for s in shots)} reactions)")
        return shots

    def _review_shots(self, model: str, exp: dict[str, Any], shots: list, log: list[str]) -> None:
        """Hold every drawn (scene) picture to the step timeline; apply the critic's corrections."""
        from lib import kids_science as ks
        from lib import lmstudio

        scene = [s for s in shots if s.kind == "scene"]
        if not scene:
            return
        listing = "\n".join(f"{s.id} [{s.beat}] props={s.props} | {s.description} | must_show={s.must_show} | "
                            f"words: {s.words}" for s in scene)
        user = (
            f"EXPERIMENT: {exp['title']}\nSteps in order: {' | '.join(exp['steps'])}\nReal result: {exp['observe']}\n\n"
            "What the scene may show in each beat:\n" + "\n".join(ks.picture_timeline(exp)) + "\n\n"
            "PLANNED PICTURES (id [beat] props | description | must_show | words spoken over it):\n" + listing + "\n\n"
            "The words come first: a picture must show what its words say. "
            "For every picture: ok=false if it does not show what its words say, shows something that has not "
            "happened yet at that beat, shows the result too early, contradicts the steps, or asks for anything "
            "physically odd (people on tables, floating objects). When ok=false give the corrected description "
            "and must_show. When ok=true repeat them unchanged and leave problem empty. Return every id once."
        )
        out = lmstudio.chat_json(
            model, "You check a children's picture book's illustration plan against its words and step order. "
            "You only fix real mismatches.", user, ks.SHOT_REVIEW_SCHEMA, temperature=0.0, max_tokens=6000,
        )
        fixes = {r["id"]: r for r in out.get("shots", []) if not r["ok"] and r["description"].strip()}
        for sh in scene:
            fix = fixes.get(sh.id)
            if fix:
                sh.description = fix["description"].strip()
                sh.must_show = [m for m in fix["must_show"] if m.strip()][:2] or sh.must_show
                log.append(f"picture {sh.id} corrected: {fix['problem'][:160]}")
        log.append(f"picture review: {len(fixes)} of {len(scene)} drawn pictures corrected")

    def _write_beat(
        self,
        model: str,
        exp: dict[str, Any],
        bible: dict[str, Any],
        beat: dict[str, Any],
        steps: dict[str, list[str]],
        script: dict[str, list[dict[str, str]]],
        feedback: list[str],
        log: list[str],
    ) -> list[dict[str, str]]:
        from lib import kids_science as ks
        from lib import lmstudio

        lo, target, hi = ks.word_budget(beat)
        content = {
            "materials": ("Materials (say exactly these, nothing more): " + "; ".join(exp["materials"])
                          + ". They are only laid out on the table — no pouring, adding, mixing or folding yet."),
            "observe": "What happens (describe only this): " + exp["observe"],
            "explain": ("Approved facts (Arjun explains only these to Dev, simply; Dev may ask one question):\n- "
                        + "\n- ".join(exp["facts"])),
            "real_world": "Pick ONE approved real-world example:\n- " + "\n- ".join(exp["real_world"]),
            "recap": f"Include this safety note in full, every part of it: {exp['safety']}",
            "hypothesis": (f"The question: {exp['question']} Do NOT reveal what really happens, and nobody touches, "
                           "pours, folds or sets up anything yet."),
            "hook": (f"Open on this everyday moment, seen by the boys as a little mystery: {hook_moment(exp)} "
                     f"Do not explain it and do not invent any other strange happening. The episode's question "
                     f"(do not answer it yet): {exp['question']}"),
            "setup": f"The episode's question: {exp['question']}",
        }.get(beat["id"], "")
        if beat["id"] in steps:
            content = "Do exactly these steps, in this order, nothing else:\n- " + "\n- ".join(steps[beat["id"]])
            content += ("\nKeep every amount and detail exactly as written — never add numbers, drop counts or "
                        "quantities that are not in the step. Do not describe the final result yet.")
        # Only beats BEFORE this one: on a rewrite the script also holds later
        # beats, and showing those makes the model continue from the ending.
        order = [b["id"] for b in ks.beats()]
        earlier = [
            " ".join(l["text"] for l in script[b]) for b in order[: order.index(beat["id"])] if b in script
        ][-3:]
        previous = script.get(beat["id"]) if feedback else None
        if previous:
            # Notes that quote a sentence to remove are applied exactly, in code —
            # the cheapest fix there is, and it can't introduce anything new.
            trimmed = remove_quoted(previous, feedback)
            if trimmed is not None and not ks.check_beat_lines(beat, trimmed):
                log.append(f"{beat['id']}: applied the critic's quoted removal in code")
                return trimmed
        best: tuple[int, int, list[dict[str, str]]] | None = None
        notes = list(feedback)
        for attempt in range(3):
            revise = ""
            if previous:
                # A critic note means "fix this", not "start over": a fresh draft
                # trades one problem for new ones and never converges.
                revise = ("\n\nYour previous version of this beat:\n"
                          + "\n".join(f"{l['speaker']}: {l['text']}" for l in previous)
                          + "\nRevise it: fix ONLY the problems below and keep everything else the same.")
            user = (
                f"Episode: {bible['episode_title']}\nBeat '{beat['id']}' — {beat['title']}: {beat['purpose']}\n"
                f"Plan for this beat (the job and any review notes below override it): "
                f"{bible['beats'][beat['id']]['summary']}\n{content}\n\n"
                + ("Story so far (last beats):\n" + "\n".join(earlier) + "\n\n" if earlier else "")
                + f"Write this beat's lines: about {target} words in total — at least {lo} and NEVER more than {hi}. "
                "Continue naturally from the story so far without repeating it."
                + revise
                + ("\n\nFix these problems from review:\n- " + "\n- ".join(notes) if notes else "")
            )
            out = lmstudio.chat_json(model, _writer_system(model), user, ks.beat_lines_schema(), temperature=0.6, max_tokens=1500)
            lines = [l for l in out["lines"] if l["text"].strip()]
            text = " ".join(l["text"] for l in lines)
            problems = ks.check_beat_lines(beat, lines) + ks.check_banned_claims(text, exp)
            if not problems:
                return lines
            log.append(f"{beat['id']} attempt {attempt + 1}: {problems}")
            rank = (len(problems), abs(ks.word_count(text) - target))
            if best is None or rank < best[:2]:
                best = (*rank, lines)
            previous = lines
            notes = list(feedback) + problems
        # Keep the closest attempt; the critic gets the final say and
        # persistent problems fail the run there.
        return best[2] if best else lines

    def _critique(
        self,
        model: str,
        exp: dict[str, Any],
        bible: dict[str, Any],
        steps: dict[str, list[str]],
        script: dict[str, list[dict[str, str]]],
    ) -> dict[str, list[str]]:
        from lib import kids_science as ks
        from lib import lmstudio

        body = []
        for b in ks.beats():
            assigned = f"  [assigned steps: {' | '.join(steps[b['id']])}]" if b["id"] in steps else ""
            text = "\n".join(f"    {l['speaker']}: {l['text']}" for l in script[b["id"]])
            body.append(f"## {b['id']} ({b['title']}) — job: {b['purpose']}{assigned}\n{text}")
        user = (
            f"EXPERIMENT (the only allowed science)\nQuestion: {exp['question']}\n"
            f"Materials: {', '.join(exp['materials'])}\nSteps in order: {' | '.join(exp['steps'])}\n"
            f"Real result: {exp['observe']}\nApproved facts:\n- " + "\n- ".join(exp["facts"]) +
            "\nMisconceptions that must NOT appear:\n- " + "\n- ".join(exp["must_not_claim"]) +
            f"\nSafety note that must appear in recap: {exp['safety']}\n"
            f"Allowed characters: {_cast_names()}. Audience: kids 6-11.\n\n"
            "SCRIPT\n" + "\n\n".join(body) + "\n\n"
            "Check every beat. A beat is NOT ok if it: states any science claim or natural phenomenon not "
            "supported by the approved facts (including in the hook); contains a misconception; performs any "
            "action that belongs to another beat (materials and hypothesis must not pour, add, mix, fold or set "
            "up anything; a step beat may only do its assigned steps); changes, skips, reorders or adds steps or "
            "materials; reveals or hints at the result before the 'observe' beat; describes a result different "
            "from the real result; contradicts an earlier beat; introduces any character, person or pet besides "
            f"{_cast_names()}; is confusing for a 6-11 year old; or misses its job. The recap must include the "
            "safety note. Write each "
            "problem as a short instruction for fixing it. Beats with no problems: ok=true, problems=[]. "
            "Return every beat id exactly once."
        )
        out = lmstudio.chat_json(
            model,
            "You are a strict, fair checker of a children's science story. You do not rewrite; you find real "
            "problems only, never style preferences.",
            user,
            ks.CRITIC_SCHEMA,
            temperature=0.0,
            max_tokens=4000,
        )
        review = {b["id"]: [] for b in ks.beats()}
        for item in out.get("beats", []):
            if item["id"] == "materials":
                continue  # code-written from the library; nothing for a model to fix
            if item["id"] in review and not item["ok"]:
                review[item["id"]] = item["problems"] or ["marked not ok without a reason"]
        for b in ks.beats():  # code checks run every round too
            text = " ".join(l["text"] for l in script[b["id"]])
            review[b["id"]] += ks.check_banned_claims(text, exp)
        return review

    def _write_artifacts(self, project, exp, bible, steps, script, review, log, writer, critic, shots) -> None:
        from lib import kids_science as ks

        art = project / "artifacts"
        t = 0.0
        sections, scenes = [], []
        for b in ks.beats():
            text = " ".join(l["text"] for l in script[b["id"]])
            sections.append({
                "id": b["id"], "label": b["title"], "text": text,
                "start_seconds": t, "end_seconds": t + b["seconds"],
                "speaker_directions": "; ".join(f"{l['speaker']}" for l in script[b["id"]]),
            })
            beat_shots = [s for s in shots if s.beat == b["id"]]
            span = b["seconds"] / max(len(beat_shots), 1)
            for i, s in enumerate(beat_shots):
                scenes.append({
                    "id": s.id, "type": "generated", "description": s.description,
                    "start_seconds": round(t + i * span, 2), "end_seconds": round(t + (i + 1) * span, 2),
                    "script_section_id": b["id"],
                    "required_assets": [{"type": "image", "description": s.prompt()[:400], "source": "generate"}],
                })
            t += b["seconds"]
        (art / "brief.json").write_text(json.dumps({
            "version": "1.0", "title": bible["episode_title"], "hook": exp["question"],
            "key_points": [exp["observe"], *exp["facts"][:3]],
            "tone": "warm, curious, gently funny read-aloud story time",
            "style": "watercolor storybook stills with slow camera moves and read-along captions",
            "target_platform": "youtube", "target_duration_seconds": t,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "story_bible.json").write_text(json.dumps({
            "experiment_id": exp["id"], "episode_title": bible["episode_title"], "beats": bible["beats"],
            "assigned_steps": steps,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "script.json").write_text(json.dumps({
            "version": "1.0", "title": bible["episode_title"], "total_duration_seconds": t,
            "sections": sections,
            "metadata": {"lines": script, "experiment_id": exp["id"], "writer_model": writer, "critic_model": critic},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "scene_plan.json").write_text(json.dumps({
            "version": "1.0", "scenes": scenes,
            "metadata": {"shots": [s.__dict__ for s in shots], "motion": ks.pick_motion(shots)},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (art / "story_review.json").write_text(json.dumps({
            "ok": not any(review.values()), "open_problems": {k: v for k, v in review.items() if v},
            "log": log,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
