"""Quick Science — a 60-120 s experiment-first episode, planned entirely by code.

Modelled on what makes short experiment videos work (BeardedScienceGuy's
walking-water video, analysed 2026-11-09): the finished result is on screen in
the first seconds, every picture after that is the experiment itself on a
clean backdrop, and the change happens as a time-lapse.

Nothing here is written by a model. The narration is the library's own vetted
text — the question, the materials, each step, the real result, the approved
facts, the safety note — joined with a few fixed series phrases. So there is
no story to drift and no claim to check: the plot lock is total.

Pictures:
  - one state-checked plate per library step (edit chain, like story episodes)
  - the hook and the "why" beat reuse the approved result plate (detail crops)
  - the time-lapse is a dissolve between the last step plate and the result
  - the boys appear only in a 2-second hello and the send-off
"""

from __future__ import annotations

from typing import Any

from lib import kids_science as ks

BACKDROP = ("on a clean white tabletop against a plain soft lavender studio background, bright even light, "
            "nothing else in the picture")
ORDINALS = ["First", "Next", "Then", "Now", "After that", "Finally"]


def _cast_names() -> tuple[str, str]:
    names = [c["name"] for c in ks.series()["cast"].values()]
    return names[0], names[1]


def plan(exp: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    """(beats, shots, lines) for one Quick Science episode."""
    big, small = _cast_names()
    states = exp.get("states") or {}
    always = list(states.get("always", []))  # the containers themselves, held on every plate
    n = len(exp["steps"])
    say = lambda text: [{"speaker": "narrator", "text": text}]  # noqa: E731

    beats = [{"id": "hook", "title": "The Big Question"}, {"id": "hello", "title": f"{big} & {small}"},
             {"id": "materials", "title": "You'll Need"}]
    lines: dict[str, list[dict[str, str]]] = {
        "hook": say(f"{exp['question']} Watch this!"),
        "hello": say(f"Hi! It's {big} and {small}. Let's find out together!"),
        "materials": say("You'll need " + ", ".join(exp["materials"][:-1]) + f", and {exp['materials'][-1]}."),
    }
    shots = [
        dict(id="hook-1", beat="hook", kind="reuse", reuse_of="observe-1", overlay=exp["question"]),
        dict(id="hello-1", beat="hello", kind="scene", characters=list(ks.series()["cast"]),
             description="Both boys stand side by side in the family kitchen, smiling and waving at the viewer.",
             props=[], must_show=["both boys are waving"]),
        dict(id="materials-1", beat="materials", kind="plate", state=always + list(states.get("materials", [])),
             overlay="You'll need:\n" + "\n".join(f"• {m}" for m in exp["materials"])),
    ]
    prev_step = "materials-1"
    for i, step in enumerate(exp["steps"], 1):
        bid = f"step{i}"
        beats.append({"id": bid, "title": f"Step {i}"})
        lead = ORDINALS[min(i - 1, len(ORDINALS) - 1)] if i < n else ORDINALS[-1]
        lines[bid] = say(f"{lead}: {step[0].lower()}{step[1:]}")
        shots.append(dict(id=f"{bid}-1", beat=bid, kind="plate", state=always + list(states.get(str(i), [])),
                          action=step, overlay=f"Step {i}\n{step}"))
        prev_step = f"{bid}-1"
    beats += [{"id": "watch", "title": "Watch It Happen"}, {"id": "observe", "title": "Whoa!"},
              {"id": "explain", "title": "Why It Works"}, {"id": "bye", "title": "Try It Yourself"}]
    lines["watch"] = say("Now watch closely…")
    lines["observe"] = say(f"Whoa! {exp['observe']}")
    facts = exp["facts"][:3]
    # Two lines so the "why" gets two pictures, each switching in on its own words.
    lines["explain"] = say(" ".join(facts[:2])) + (say(facts[2]) if len(facts) > 2 else [])
    lines["bye"] = say(f"{exp['real_world'][0]} {exp['safety']} Try it with a grown-up, and see you next time!")
    shots += [
        dict(id="watch-1", beat="watch", kind="blend", blend_from=prev_step, blend_to="observe-1"),
        dict(id="observe-1", beat="observe", kind="plate", state=always + list(states.get("observe", [])),
             action=exp["observe"]),
        dict(id="explain-1", beat="explain", kind="reuse", reuse_of="observe-1"),
        *([dict(id="explain-2", beat="explain", kind="reuse", reuse_of="observe-1")] if len(facts) > 2 else []),
        dict(id="bye-1", beat="bye", kind="scene", characters=list(ks.series()["cast"]),
             description="Both boys grin and give a thumbs-up to the viewer in the family kitchen.",
             props=[], must_show=["both boys give a thumbs-up"]),  # no drawn text: this shot is animated
    ]
    for sh in shots:  # every plate shares the clean backdrop; lines map 1:1 to beats here
        if sh["kind"] == "plate":
            sh["backdrop"] = BACKDROP
        sh.setdefault("description", "")
        sh.setdefault("characters", [])
        sh.setdefault("props", [])
        idx = 1 if sh["id"] == "explain-2" else 0
        sh["lines"] = [idx]
        # The hello and goodbye are greetings: no picture can "show" plants or a
        # safety note, so they are only held to the plausibility check.
        sh["words"] = "" if sh["beat"] in ("hello", "bye") else lines[sh["beat"]][idx]["text"]
    return beats, shots, lines
