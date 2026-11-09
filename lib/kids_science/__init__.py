"""Kids science story time — series bible, experiment library, and plot guards.

The hallucination strategy lives here, not in prompts:

- The plot skeleton (beats, order, durations) is fixed by ``series.yaml``.
- The science is fixed by ``experiments.yaml``; the writer only phrases it.
- Characters and props are JSON-schema enums, so the local model cannot emit
  one that is not in the bible — constrained decoding refuses the token.
- Everything the schema cannot express (word budgets, step order, banned
  claims) is checked by code here, and failures are returned as precise,
  per-beat feedback the writer can act on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SERIES_ASSETS = REPO_ROOT / "projects" / "_series" / "kids-science"

STEP_BEATS = ("step1", "step2", "step3", "step4")
# Beats whose pictures are of the experiment itself — always held to its state.
APPARATUS_BEATS = ("materials", *STEP_BEATS, "observe")


@lru_cache(maxsize=1)
def series() -> dict[str, Any]:
    return yaml.safe_load((HERE / "series.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def experiments() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load((HERE / "experiments.yaml").read_text(encoding="utf-8"))
    states = yaml.safe_load((HERE / "states.yaml").read_text(encoding="utf-8")) or {}
    out = {}
    for e in data["experiments"]:
        out[e["id"]] = {**e, "states": {str(k): v for k, v in (states.get(e["id"]) or {}).items()}}
    return out


def beat_state(exp: dict[str, Any], beat_id: str) -> list[str]:
    """Authored facts the experiment's setup must show in this beat (empty if the beat has none).

    A step beat shows the state after its LAST step: pictures there may be
    mid-action, but they must never contradict where the beat ends up.
    """
    states = exp.get("states") or {}
    if beat_id in ("materials", "observe"):
        return list(states.get(beat_id, []))
    if beat_id in STEP_BEATS:
        assigned = [s for s in assign_steps(exp)[beat_id] if not s.startswith("(")]
        if not assigned:  # a "watch closely" beat keeps the previous state
            idx = len(exp["steps"])
        else:
            idx = exp["steps"].index(assigned[-1]) + 1
        return list(states.get(str(idx), []))
    return []


def shows_experiment(exp: dict[str, Any], props: list[str]) -> bool:
    return bool(set(props) & (set(exp["props"]) | set(exp["materials"])))


def experiment(experiment_id: str) -> dict[str, Any]:
    try:
        return experiments()[experiment_id]
    except KeyError:
        raise KeyError(
            f"unknown experiment {experiment_id!r}; choose from: {', '.join(sorted(experiments()))}"
        ) from None


def beats() -> list[dict[str, Any]]:
    return list(series()["beats"])


# Beats whose content is capped by the library (facts, materials, steps). A
# firm minimum there pressures the writer to pad with invented science — the
# exact failure this pipeline exists to prevent — so they may run short.
FACT_LIMITED_BEATS = ("materials", "step1", "step2", "step3", "step4", "explain")


def word_budget(beat: dict[str, Any]) -> tuple[int, int, int]:
    """(min, target, max) narration words for a beat at the series pace."""
    target = round(beat["seconds"] * series()["series"]["words_per_minute"] / 60)
    low = 0.5 if beat.get("id") in FACT_LIMITED_BEATS else 0.75
    return round(target * low), target, round(target * 1.25)


def assign_steps(exp: dict[str, Any]) -> dict[str, list[str]]:
    """Distribute the experiment's steps over step1..step4, in order, as evenly as possible."""
    steps = list(exp["steps"])
    out: dict[str, list[str]] = {b: [] for b in STEP_BEATS}
    for i, step in enumerate(steps):
        out[STEP_BEATS[min(i * len(STEP_BEATS) // len(steps), len(STEP_BEATS) - 1)]].append(step)
    # With fewer than four steps the last beats would be empty; let the final
    # step's beat be "watch closely" rather than invent a new step.
    for beat in STEP_BEATS:
        if not out[beat]:
            out[beat] = ["(No new step: watch closely and describe the waiting, without saying the result yet.)"]
    return out


def allowed_props(exp: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(list(exp["props"]) + list(exp["materials"]) + list(series()["setting"]["props"])))


def cast_ids() -> list[str]:
    return list(series()["cast"])


# ---------------------------------------------------------------------------
# Schemas (LM Studio json_schema / constrained decoding)
# ---------------------------------------------------------------------------


def bible_schema(exp: dict[str, Any]) -> dict[str, Any]:
    """Story bible: one fixed slot per beat, so beats cannot repeat or reorder.

    Pictures are NOT planned here any more: they are planned from the finished
    narration (see shot_plan_schema), so every picture illustrates the exact
    words spoken over it.
    """
    beat_props = {
        b["id"]: {
            "type": "object", "additionalProperties": False, "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
        }
        for b in beats()
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["episode_title", "beats"],
        "properties": {
            "episode_title": {"type": "string"},
            "beats": {"type": "object", "additionalProperties": False,
                      "required": [b["id"] for b in beats()], "properties": beat_props},
        },
    }


def group_lines(lines: list[dict[str, str]], lo: int = 12, hi: int = 30) -> list[list[int]]:
    """Consecutive line indices grouped into picture-sized chunks (~5-12 s of speech each)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    words = 0
    for i, line in enumerate(lines):
        n = word_count(line["text"])
        if cur and words >= lo and words + n > hi:
            groups.append(cur)
            cur, words = [], 0
        cur.append(i)
        words += n
    if cur:
        if groups and words < lo // 2:
            groups[-1] += cur  # a tiny tail joins the previous picture
        else:
            groups.append(cur)
    return groups


def shot_kinds(beat_id: str, groups: list[list[int]], lines: list[dict[str, str]], exp: dict[str, Any]) -> list[str]:
    """Which kind of picture each group gets. Experiment beats: words about the
    materials get the state-checked plate, everything else a reaction close-up."""
    if beat_id not in APPARATUS_BEATS:
        return ["scene"] * len(groups)
    nouns = {w for m in exp["materials"] for w in re.findall(r"[a-z]+", m.lower()) if len(w) > 3}
    kinds = []
    for g in groups:
        text = " ".join(lines[i]["text"] for i in g).lower()
        kinds.append("plate" if any(n in text for n in nouns) else "reaction")
    if "plate" not in kinds:
        kinds[0] = "plate"  # every experiment beat shows its setup at least once
    return kinds


def shot_plan_schema(exp: dict[str, Any], n: int) -> dict[str, Any]:
    """Exactly one picture per narration group, characters and props from closed sets."""
    pic = {
        "type": "object",
        "additionalProperties": False,
        "required": ["description", "characters", "props", "must_show"],
        "properties": {
            "description": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string", "enum": cast_ids()}, "maxItems": 2},
            "props": {"type": "array", "items": {"type": "string", "enum": allowed_props(exp)}, "maxItems": 3},
            "must_show": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        },
    }
    return {"type": "object", "additionalProperties": False, "required": ["pictures"],
            "properties": {"pictures": {"type": "array", "items": pic, "minItems": n, "maxItems": n}}}


def beat_lines_schema() -> dict[str, Any]:
    speakers = sorted({"narrator", *(c["voice"] for c in series()["cast"].values())})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["lines"],
        "properties": {
            "lines": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["speaker", "text"],
                    "properties": {
                        "speaker": {"type": "string", "enum": speakers},
                        "text": {"type": "string"},
                    },
                },
            }
        },
    }


def picture_timeline(exp: dict[str, Any]) -> list[str]:
    """What the scene may show in each beat — the rule pictures are checked against."""
    steps = assign_steps(exp)
    done: list[str] = []
    out = []
    for b in beats():
        bid = b["id"]
        if bid in ("hook", "setup", "hypothesis"):
            # Everyday objects the words mention (a towel wiping a spill) are
            # fine — only the experiment's SETUP must not exist yet.
            rule = ("the experiment has not been set up yet: no cups lined up for it, no colored water, nothing "
                    "connected; everyday objects the words mention are fine and should be shown as the words say")
        elif bid == "materials":
            rule = "the materials are laid out on the table, untouched — nothing poured, mixed, folded or changed yet"
        elif bid in steps:
            now = [s for s in steps[bid] if not s.startswith("(")]
            rule = (f"doing now: {' '.join(now) or 'waiting and watching'} "
                    f"Already done: {' '.join(done) or 'nothing'} "
                    "Nothing from later steps, and NOT the final result")
            done += now
        elif bid == "observe":
            rule = f"the real result is visible: {exp['observe']}"
        else:
            rule = f"after the experiment; the result may be visible: {exp['observe']}"
        facts = beat_state(exp, bid)
        if facts:
            rule += ". The setup must look like: " + "; ".join(facts)
        out.append(f"- {bid}: {rule}")
    return out


SHOT_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shots"],
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "ok", "problem", "description", "must_show"],
                "properties": {
                    "id": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "problem": {"type": "string"},
                    "description": {"type": "string"},
                    "must_show": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
            },
        }
    },
}

HARD_CHECK_PREFIXES = ("Scene state:",)


CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["beats"],
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "ok", "problems"],
                "properties": {
                    "id": {"type": "string"},
                    "ok": {"type": "boolean"},
                    "problems": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                },
            },
        }
    },
}


# ---------------------------------------------------------------------------
# Code-level checks (what a schema cannot express)
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def allowed_names() -> set[str]:
    names = {c["name"].lower() for c in series()["cast"].values()}
    # Capitalised words that are fine anywhere: sentence starts are handled by
    # the checker; these are proper nouns the library itself uses.
    return names | {"newton", "earth", "dead", "sea"}


_SENTENCE_START = set('.!?…"“”‘’\'(—–:;')


def unknown_names(text: str) -> list[str]:
    """Capitalised words mid-sentence that are not series names — likely invented characters.

    A capital after sentence punctuation, an opening quote, a dash or a line
    start is just a sentence start ("“What a mess!”", "— And then"), not a name.
    """
    found: list[str] = []
    for line in text.splitlines():
        for m in _WORD.finditer(line):
            tok = m.group(0)
            if not tok[0].isupper():
                continue
            before = line[: m.start()].rstrip()
            if not before or before[-1] in _SENTENCE_START:
                continue
            base = re.sub(r"['’]s$", "", tok).lower()
            if base in allowed_names() or base == "i" or base.startswith(("i'", "i’")) or tok == "OK":
                continue
            found.append(tok)
    return found


def check_beat_lines(beat: dict[str, Any], lines: list[dict[str, str]]) -> list[str]:
    """Problems with one beat's narration that code can see for certain."""
    problems: list[str] = []
    text = " ".join(line["text"] for line in lines)
    lo, target, hi = word_budget(beat)
    words = word_count(text)
    if words < lo:
        problems.append(f"too short: {words} words, write about {target} (at least {lo})")
    if words > hi:
        problems.append(f"too long: {words} words, cut to about {target} (at most {hi})")
    for line in lines:
        names = unknown_names(line["text"])
        if names:
            cast = " and ".join(c["name"] for c in series()["cast"].values())
            problems.append(f"names '{names[0]}', who/what is not in the series bible — use only {cast}")
            break
    return problems


def check_banned_claims(text: str, exp: dict[str, Any]) -> list[str]:
    """Literal hits on the experiment's known misconceptions (the critic catches paraphrases)."""
    low = text.lower()
    return [f"repeats the misconception: '{c}'" for c in exp.get("must_not_claim", []) if c.lower() in low]


# Reaction shots are framed by code: a model asked to show "Arjun pours the
# water" draws the cups — in whatever state it likes. Chest-up close-ups can't
# contradict the experiment; the planner only picks the expression.
DEFAULT_REACTIONS = {
    "materials": "looking at the viewer with an excited, ready-to-start smile",
    "step1": "concentrating carefully", "step2": "watching closely with wide curious eyes",
    "step3": "looking hopeful", "step4": "waiting patiently, chin on hands",
    "observe": "gasping in amazement",
}


@dataclass
class ShotSpec:
    """One picture, tied to the narration lines spoken while it is on screen.

    ``kind`` decides how it is made and how strictly it is checked:
    - ``plate``    — the experiment setup alone, close up, no people. Held to the
                     authored state (hard). One per beat, made by editing the
                     previous beat's plate so layout and counts carry over.
    - ``reaction`` — the boys in an experiment beat, chest-up, setup out of frame.
    - ``scene``    — every other beat, drawn from the planner's description.
    """

    id: str
    beat: str
    description: str
    characters: list[str]
    props: list[str]
    must_show: list[str] = field(default_factory=list)
    state: list[str] = field(default_factory=list)  # authored, from states.yaml
    kind: str = "scene"
    action: str = ""  # plate: what this beat's steps change
    lines: list[int] = field(default_factory=list)  # indices into the beat's narration lines
    words: str = ""  # the narration spoken over this picture
    motion: bool = False

    def _hands_allowed(self) -> bool:
        return any("hand" in f for f in self.state)

    def _cast(self) -> list[dict[str, Any]]:
        return [series()["cast"][c] for c in self.characters]

    def prompt(self) -> str:
        s = series()
        parts = [s["style"]["image"]]
        if self.kind == "plate":
            parts.append(
                "Close-up of a science experiment on a wooden kitchen table in a sunlit kitchen, seen from the "
                "front at table height, everything clearly visible and evenly spaced."
            )
            if not self._hands_allowed():
                parts.append("No people, no hands, no animals.")
            # Only the state for THIS moment — an experiment's display props
            # describe its finished look and would paint the ending too early.
            parts.append("Exactly as follows, and nothing more: " + "; ".join(self.state) + ".")
            if any("empty" in f for f in self.state):
                parts.append("Empty containers are completely dry and transparent, with nothing inside them.")
            return " ".join(parts)
        parts.append(f"Setting: {s['setting']['description']}.")
        who = [c["identity"] for c in self._cast()]
        if who:
            parts.append("Characters: " + "; ".join(who) + ".")
            parts.append(f"Only these {len(who)} people appear — nobody else, no animals."
                         if len(who) > 1 else "Only this one person appears — nobody else, no animals.")
        else:
            parts.append("No people and no animals in this picture.")
        if len(who) == 2:
            parts.append("The 6-year-old is clearly smaller and younger than the 9-year-old.")
        if self.kind == "reaction":
            names = " and ".join(c["name"] for c in self._cast()) or "The boys"
            mood = self.description or DEFAULT_REACTIONS.get(self.beat, "smiling")
            parts.append(
                f"Chest-up close-up: {names} {mood}. The camera is close, the kitchen behind is softly blurred, "
                "and no table top or objects in front of them are in frame. Both feet on the floor, natural poses."
            )
            return " ".join(parts)
        if self.props:
            parts.append("Showing: " + ", ".join(self.props) + ".")
        parts.append(self.description)
        parts.append("Everyone stands or sits naturally on the floor or chairs; everything obeys gravity.")
        if self.must_show:
            parts.append("Make sure: " + "; ".join(self.must_show) + ".")
        return " ".join(parts)

    def edit_prompt(self) -> str:
        """Plate N from plate N-1: same table, same objects, same camera — change only this step."""
        s = series()
        keep = ("Keep the same table, the same objects in the same places, the same containers with the same "
                "shapes, the same camera angle and the same watercolor style.")
        parts = [keep, f"Change only this: {self.action}" if self.action else "Change nothing about the layout.",
                 "The picture must look exactly like this: " + "; ".join(self.state) + "."]
        if not self._hands_allowed():
            parts.append("No people, no hands, no animals.")
        parts.append(s["style"]["image"])
        return " ".join(parts)

    def checklist(self) -> list[str]:
        """Yes/no statements for the first vision checker. Hard ones are marked by wording/prefix."""
        checks = ["The image is a soft watercolor children's book illustration (not a photo, not 3D render)."]
        if self.kind == "plate":
            checks.append(
                "There are no people in the image, except possibly a grown-up's hands."
                if self._hands_allowed() else "There are no humans and no human hands in the image."
            )
            checks.append("There is no animal in the image.")
            checks.append("There is no readable text, lettering, numbers, or signs anywhere in the image.")
            checks += [f"Scene state: {f.rstrip('.')}." for f in self.state]
            return checks
        for c in self._cast():
            checks.append(f"{c.get('traits') or c['identity']} is clearly visible.")
        n = len(self.characters)
        checks.append(f"There {'is exactly 1 human' if n == 1 else f'are exactly {n} humans'} in the image."
                      if n else "There are no humans in the image.")
        checks.append("There is no animal in the image.")
        checks.append("There is no readable text, lettering, numbers, or signs anywhere in the image.")
        checks.append("Hands and faces look normal (no extra fingers, no merged or duplicated characters).")
        if self.kind == "reaction":
            # Jars on the kitchen shelf are the series' set — only experiment ware counts.
            checks.append("Scene state: there are no drinking cups or drinking glasses in the picture (jars on shelves are fine).")
        else:
            for p in self.props[:2]:
                checks.append(f"The image shows: {p}.")
        # The planner's own notes help, but they are not authored facts: soft.
        checks += [f"Detail: {m.rstrip('.')}." for m in self.must_show]
        return checks

    def review_checks(self, exp_does: str = "") -> list[str]:
        """Statements for the SECOND, independent reviewer: does it fit the words, is it sane?

        ``exp_does`` is what the experiment itself makes happen (water climbing a
        towel): that is the science, not an oddity, and must not be rejected as one.
        """
        allowed = f" (what the experiment itself does is expected and fine: {exp_does})" if exp_does else ""
        checks = [
            "Scene state: nothing in the picture is odd or physically impossible — no one standing or sitting on "
            "a table, nothing floating in the air, no liquid pouring from nowhere, no melted or merged objects"
            + allowed + ".",
        ]
        if self.words:
            fits = ("shows the experiment setup these words talk about (the people in the words are not expected "
                    "in this close-up of the setup)" if self.kind == "plate" else
                    "fits these words (a close-up of the boys reacting to them also fits)" if self.kind == "reaction"
                    else "shows what these words describe, or the boys saying them")
            checks.insert(0, f"Scene state: the picture {fits}: \"{self.words}\"")
        return checks


def plan_shots(exp: dict[str, Any], script_lines: dict[str, list[dict[str, str]]],
               pictures: dict[str, list[dict[str, Any]]]) -> list[ShotSpec]:
    """Shots for the whole episode from the narration groups and the planner's pictures."""
    out: list[ShotSpec] = []
    steps = assign_steps(exp)
    for b in beats():
        bid = b["id"]
        lines = script_lines[bid]
        groups = group_lines(lines)
        kinds = shot_kinds(bid, groups, lines, exp)
        action = {"materials": "", "observe": exp["observe"]}.get(
            bid, " ".join(x for x in steps.get(bid, []) if not x.startswith("(")))
        for i, (g, kind, pic) in enumerate(zip(groups, kinds, pictures[bid]), 1):
            words = " ".join(lines[j]["text"] for j in g)
            if kind == "plate":
                out.append(ShotSpec(id=f"{bid}-{i}", beat=bid, kind="plate", description="", characters=[],
                                    props=[], state=beat_state(exp, bid), action=action, lines=g, words=words))
            else:
                chars = list(dict.fromkeys(pic["characters"])) or (cast_ids() if kind == "reaction" else [])
                out.append(ShotSpec(
                    id=f"{bid}-{i}", beat=bid, kind=kind, description=pic["description"].strip(),
                    characters=chars, props=[] if kind == "reaction" else list(dict.fromkeys(pic["props"])),
                    must_show=[m.strip() for m in pic.get("must_show", []) if m.strip()], lines=g, words=words,
                ))
    return out


def pick_motion(shots: list[ShotSpec], limit: int = 5) -> list[str]:
    """The few moments worth real animation: the result, the hook, the explanation, the send-off.

    Step beats are skipped — their exact drawn badges would have to be re-burned onto video.
    """
    by_beat: dict[str, list[ShotSpec]] = {}
    for sh in shots:
        by_beat.setdefault(sh.beat, []).append(sh)
    wanted = []
    for beat, kind in (("observe", "plate"), ("hook", None), ("observe", "reaction"), ("explain", None),
                       ("real_world", None), ("recap", None)):
        for sh in by_beat.get(beat, []):
            if (kind is None or sh.kind == kind) and sh.id not in wanted:
                wanted.append(sh.id)
                break
    return wanted[:limit]
