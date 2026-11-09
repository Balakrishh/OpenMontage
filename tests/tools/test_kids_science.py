"""Kids science story time — the plot guards, without any model running.

The writer, keyframe and narration tools call local models; these tests pin the
parts that keep those models on the plot: the library and series contracts, the
schema enums, the code checks, and the writer's rewrite-on-critic loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib import kids_science as ks


class TestLibrary:
    def test_every_experiment_is_complete(self):
        required = {"id", "title", "question", "materials", "steps", "observe", "facts", "must_not_claim",
                    "real_world", "safety", "props"}
        assert len(ks.experiments()) >= 20
        for eid, exp in ks.experiments().items():
            assert required <= set(exp), f"{eid} missing {required - set(exp)}"
            assert 3 <= len(exp["steps"]) <= 8, eid
            assert exp["facts"] and exp["must_not_claim"], eid

    def test_beats_fill_the_target_runtime(self):
        assert sum(b["seconds"] for b in ks.beats()) == ks.series()["series"]["target_seconds"]
        assert [b["id"] for b in ks.beats() if b["id"].startswith("step")] == list(ks.STEP_BEATS)

    def test_every_speaker_has_a_voice(self):
        voices = ks.series()["voices"]
        speakers = ks.beat_lines_schema()["properties"]["lines"]["items"]["properties"]["speaker"]["enum"]
        assert set(speakers) <= set(voices)


class TestSteps:
    def test_steps_keep_their_order_across_step_beats(self):
        exp = ks.experiment("walking-water")
        assigned = ks.assign_steps(exp)
        flat = [s for b in ks.STEP_BEATS for s in assigned[b]]
        assert flat == exp["steps"]

    def test_short_experiments_never_invent_a_step(self):
        exp = ks.experiment("dancing-raisins")  # three steps for four step beats
        assigned = ks.assign_steps(exp)
        real = [s for b in ks.STEP_BEATS for s in assigned[b] if not s.startswith("(")]
        assert real == exp["steps"]
        assert all(assigned[b] for b in ks.STEP_BEATS)


class TestSchemas:
    def test_characters_and_props_are_closed_sets(self):
        exp = ks.experiment("static-balloon")
        pic = ks.shot_plan_schema(exp, 2)["properties"]["pictures"]["items"]
        assert pic["properties"]["characters"]["items"]["enum"] == ["arjun", "dev"]
        props = pic["properties"]["props"]["items"]["enum"]
        assert "round red balloon" in props and "wooden kitchen table" in props
        assert pic["additionalProperties"] is False
        assert ks.shot_plan_schema(exp, 3)["properties"]["pictures"]["minItems"] == 3  # one per narration group

    def test_every_beat_has_exactly_one_slot(self):
        beats = ks.bible_schema(ks.experiment("oobleck"))["properties"]["beats"]
        assert beats["required"] == [b["id"] for b in ks.beats()]
        assert beats["additionalProperties"] is False


class TestChecks:
    BEAT = {"id": "hook", "seconds": 20}

    def test_word_budget_is_enforced_both_ways(self):
        lo, target, hi = ks.word_budget(self.BEAT)
        assert lo < target < hi
        assert any("too short" in p for p in ks.check_beat_lines(self.BEAT, [{"text": "Hi."}]))
        long = [{"text": " ".join(["word"] * (hi + 5))}]
        assert any("too long" in p for p in ks.check_beat_lines(self.BEAT, long))

    def test_an_invented_character_is_caught(self):
        text = "Maya looked up as her neighbor Priya walked in with a cat. " + "word " * 40
        problems = ks.check_beat_lines(self.BEAT, [{"text": text}])
        assert any("Priya" in p for p in problems)

    def test_series_names_and_sentence_starts_are_fine(self):
        text = "Arjun smiled. Dev blinked. Then Arjun said hi to Dev. " + "word " * 40
        assert not [p for p in ks.check_beat_lines(self.BEAT, [{"text": text}]) if p.startswith("names")]

    def test_listed_misconceptions_are_caught(self):
        exp = ks.experiment("static-balloon")
        assert ks.check_banned_claims("Now the balloon becomes a magnet!", exp)

    def test_checklist_forbids_extra_people_animals_and_text(self):
        shot = ks.ShotSpec(id="x", beat="hook", description="Dev at the table", characters=["dev", "arjun"], props=["tray"])
        checks = " ".join(shot.checklist()).lower()
        assert "exactly 2 humans" in checks and "no animal" in checks and "no readable text" in checks
        assert "smaller and younger" in shot.prompt()

class TestNarrationMatch:
    def test_word_error(self):
        from tools.story.story_narration import word_error

        assert word_error("Hello there, Maya!", "hello there maya") == 0.0
        assert word_error("The balloon sticks to the wall.", "the balloon sticks to a wall") < 0.2
        assert word_error("The balloon sticks to the wall.", "completely different words here") > 0.5


class TestWriterLoop:
    def test_a_beat_the_critic_rejects_is_rewritten_with_its_notes(self, tmp_path, monkeypatch):
        from lib import lmstudio
        from tools.story import kids_story_writer as w

        exp = ks.experiment("walking-water")
        calls = {"critic": 0, "rewrites_with_notes": 0}

        def fake_chat(model, system, user, schema, **kw):
            if "beats" in schema["properties"] and "episode_title" in schema["properties"]:
                return {"episode_title": "The Climbing Water", "beats": {b["id"]: {"summary": "s"} for b in ks.beats()}}
            if "pictures" in schema["properties"]:
                n = schema["properties"]["pictures"]["minItems"]
                return {"pictures": [{"description": "Dev at the table", "characters": ["dev"], "props": ["tray"],
                                      "must_show": []}] * n}
            if "shots" in schema["properties"]:
                return {"shots": []}
            if "lines" in schema["properties"]:
                if "Fix these problems" in user and "say orange" in user:
                    calls["rewrites_with_notes"] += 1
                beat_id = user.split("Beat '")[1].split("'")[0]
                target = ks.word_budget(next(b for b in ks.beats() if b["id"] == beat_id))[1]
                return {"lines": [{"speaker": "narrator", "text": " ".join(["water"] * target)}]}
            calls["critic"] += 1
            first = calls["critic"] == 1
            return {"beats": [{"id": b["id"], "ok": not (first and b["id"] == "observe"),
                               "problems": ["say orange"] if first and b["id"] == "observe" else []} for b in ks.beats()]}

        monkeypatch.setattr(lmstudio, "chat_json", fake_chat)
        monkeypatch.setattr(lmstudio, "unload_all", lambda: None)
        result = w.KidsStoryWriter().execute({"experiment_id": exp["id"], "project_dir": str(tmp_path)})

        assert result.success, result.error
        assert calls == {"critic": 2, "rewrites_with_notes": 1}
        plan = json.loads((tmp_path / "artifacts" / "scene_plan.json").read_text())
        assert plan["metadata"]["shots"] and all(sh["words"] for sh in plan["metadata"]["shots"])
        review = json.loads((tmp_path / "artifacts" / "story_review.json").read_text())
        assert review["ok"] is True
        script = json.loads((tmp_path / "artifacts" / "script.json").read_text())
        from schemas.artifacts import validate_artifact

        validate_artifact("script", script)
        validate_artifact("scene_plan", json.loads((tmp_path / "artifacts" / "scene_plan.json").read_text()))

    def test_unresolved_problems_fail_the_run(self, tmp_path, monkeypatch):
        from lib import lmstudio
        from tools.story import kids_story_writer as w

        def fake_chat(model, system, user, schema, **kw):
            if "episode_title" in schema["properties"]:
                return {"episode_title": "T", "beats": {b["id"]: {"summary": "s"} for b in ks.beats()}}
            if "shots" in schema["properties"]:
                return {"shots": []}
            if "lines" in schema["properties"]:
                beat_id = user.split("Beat '")[1].split("'")[0]
                target = ks.word_budget(next(b for b in ks.beats() if b["id"] == beat_id))[1]
                return {"lines": [{"speaker": "narrator", "text": " ".join(["water"] * target)}]}
            return {"beats": [{"id": "explain", "ok": False, "problems": ["claims something not in the facts"]}]}

        monkeypatch.setattr(lmstudio, "chat_json", fake_chat)
        monkeypatch.setattr(lmstudio, "unload_all", lambda: None)
        result = w.KidsStoryWriter().execute({"experiment_id": "walking-water", "project_dir": str(tmp_path), "critic_rounds": 2})

        assert not result.success and "explain" in result.error
        assert json.loads((tmp_path / "artifacts" / "story_review.json").read_text())["ok"] is False


def test_step_badge_draws_exact_text(tmp_path):
    from PIL import Image

    from tools.story.story_keyframes import step_badge

    src = tmp_path / "in.png"
    Image.new("RGB", (640, 360), (200, 220, 240)).save(src)
    out = tmp_path / "out.png"
    step_badge(src, "Step 2", out)
    assert Image.open(out).size == (640, 360)
    from PIL import ImageChops

    img = Image.open(out).convert("RGB")
    painted = ImageChops.difference(img, Image.open(src).convert("RGB")).getbbox()
    assert painted is not None
    left, top, _, _ = painted
    assert left >= int(640 * 0.12) - 2 and top >= int(360 * 0.12) - 2  # clear of the zoom-cropped margin


def test_picture_timeline_keeps_later_steps_out_of_earlier_pictures():
    exp = ks.experiment("walking-water")
    rules = dict(line[2:].split(": ", 1) for line in ks.picture_timeline(exp))
    assert "untouched" in rules["materials"]
    assert "Add red food coloring" in rules["step2"] and "Fill the two outside cups" in rules["step2"]
    assert "Put one end of each strip" not in rules["step2"]
    assert "NOT the final result" in rules["step3"]
    assert exp["observe"] in rules["observe"]


def test_writer_notes_reach_the_prompt_but_are_only_soft_checks():
    shot = ks.ShotSpec(id="s", beat="hook", description="d", characters=[], props=[],
                       must_show=["the window is open"])
    assert "Detail: the window is open." in shot.checklist()
    assert not any(c.startswith("Scene state: the window") for c in shot.checklist())
    assert "the window is open" in shot.prompt()


def test_keyframe_manifest_rows_are_schema_valid(tmp_path):
    from schemas.artifacts import validate_artifact
    from tools.story.story_keyframes import asset_entry

    rows = [asset_entry("hook-1", tmp_path / "a.png", {"status": "accepted", "score": 1.0, "attempts": [{}]}),
            asset_entry("hook-2", tmp_path / "b.png", {"status": "fallback", "attempts": [{}, {}, {}]})]
    validate_artifact("asset_manifest", {"version": "1.0", "assets": rows})


def test_every_experiment_has_authored_states_for_every_point():
    for eid, exp in ks.experiments().items():
        keys = set(exp["states"])
        need = {"materials", "observe"} | {str(i) for i in range(1, len(exp["steps"]) + 1)}
        assert need <= keys, f"{eid} missing states for {sorted(need - keys)}"
        assert all(1 <= len(v) <= 3 for v in exp["states"].values()), eid


def test_step_beats_carry_the_authored_state_of_where_they_end():
    exp = ks.experiment("walking-water")
    # step2 adds the coloring (library step 3): red left, yellow right, middle still empty.
    assert any("middle cup is empty" in f for f in ks.beat_state(exp, "step2"))
    assert any("red" in f for f in ks.beat_state(exp, "step2"))
    assert any("empty" in f for f in ks.beat_state(exp, "materials"))
    assert ks.beat_state(exp, "hook") == []


def test_experiment_beats_get_a_state_checked_plate_and_reaction_shots():
    exp = ks.experiment("walking-water")
    lines = {b["id"]: [{"speaker": "narrator", "text": "Arjun carefully lined up three clear cups in a neat row on the "
                                                      "wooden table, one next to the other, ready for the next part."},
                       {"speaker": "narrator", "text": "\"Wow,\" said Dev, his eyes going wide with excitement as he "
                                                      "bounced on his toes and clapped his small hands together."}]
             for b in ks.beats()}
    pics = {b["id"]: [{"description": "grinning", "characters": ["dev"], "props": [], "must_show": ["Dev grins"]}] * 2
            for b in ks.beats()}
    shots = {s.id: s for s in ks.plan_shots(exp, lines, pics)}
    plate = shots["step2-1"]
    assert plate.kind == "plate" and plate.characters == [] and "three clear cups" in plate.words
    assert "Scene state: the middle cup is empty." in plate.checklist()
    assert "Add red food coloring" in plate.edit_prompt()
    assert "bridging" not in plate.prompt() and "yellow-colored water" not in plate.prompt()
    reaction = shots["step2-2"]
    assert reaction.kind == "reaction" and reaction.props == [] and reaction.state == []
    assert any(c.startswith("Scene state: there are no drinking cups") for c in reaction.checklist())
    assert "Chest-up close-up: Dev grinning" in reaction.prompt()
    assert "Detail: Dev grins." in reaction.checklist()  # the planner's note is soft
    assert shots["hook-1"].kind == "scene"
    assert any("wow" in c.lower() for c in reaction.review_checks())  # reviewer sees the words

def test_only_authored_facts_and_safety_checks_are_hard(monkeypatch):
    from tools.story import story_keyframes as kf

    checks = ["Detail: Maya's face.", "Scene state: the middle cup is empty.",
              "There is exactly 1 human in the image.", "The image shows: tray."]
    monkeypatch.setattr("lib.lmstudio.chat_json", lambda *a, **k: {"answers": [
        {"n": 1, "yes": False, "note": ""}, {"n": 2, "yes": True, "note": ""},
        {"n": 3, "yes": True, "note": ""}, {"n": 4, "yes": False, "note": ""}]})
    v = kf.judge("m", Path("x.png"), checks)
    assert v["hard_ok"] is True and v["score"] == 0.5
    assert [c["hard"] for c in v["checks"]] == [False, True, True, False]


def test_pictures_follow_the_narration_groups():
    lines = [{"text": "One two three four five six seven eight."}, {"text": "Nine ten eleven twelve thirteen."},
             {"text": "A brand new idea starts here with plenty of words to fill it up nicely."},
             {"text": "Short end."}]
    groups = ks.group_lines(lines, lo=12, hi=20)
    assert [i for g in groups for i in g] == [0, 1, 2, 3]  # every line covered, in order
    assert groups[0] == [0, 1] and groups[-1][-1] == 3

def test_character_check_uses_stable_traits_not_expression():
    shot = ks.ShotSpec(id="s", beat="hook", description="d", characters=["dev"], props=[])
    checks = " ".join(shot.checklist())
    assert "red t-shirt" in checks and "round cheeks" not in checks
    assert "round cheeks" in shot.prompt()  # the generator still gets the full identity

def test_fact_limited_beats_may_run_short_instead_of_padding():
    explain = next(b for b in ks.beats() if b["id"] == "explain")
    hook = next(b for b in ks.beats() if b["id"] == "hook")
    lo, target, _ = ks.word_budget(explain)
    assert lo == round(target * 0.5)
    lo_h, target_h, _ = ks.word_budget(hook)
    assert lo_h == round(target_h * 0.75)


def test_quoted_removals_are_applied_exactly():
    from tools.story.kids_story_writer import remove_quoted

    lines = [{"speaker": "narrator", "text": "Maya gathered three cups. Maya watched the colors climb into the middle cup."}]
    out = remove_quoted(lines, ['Remove the sentence "Maya watched the colors climb into the middle cup." — no results yet'])
    assert out == [{"speaker": "narrator", "text": "Maya gathered three cups."}]
    assert remove_quoted(lines, ["The tone is a bit flat"]) is None


def test_materials_beat_is_written_from_the_library():
    from tools.story.kids_story_writer import materials_lines

    exp = ks.experiment("walking-water")
    text = " ".join(l["text"] for l in materials_lines(exp))
    assert all(m in text for m in exp["materials"])
    assert not ks.check_beat_lines(next(b for b in ks.beats() if b["id"] == "materials"), materials_lines(exp))
    assert not any(w in text.lower() for w in ("pour", "fold", "mix", "climb", "orange"))


def test_detail_crop_gives_a_different_view_of_the_same_picture(tmp_path):
    from PIL import Image, ImageChops

    from tools.story.story_keyframes import detail_crop

    src = tmp_path / "a.png"
    img = Image.new("RGB", (640, 360), "white")
    img.paste((255, 0, 0), (0, 0, 320, 360))
    img.save(src)
    out = tmp_path / "b.png"
    detail_crop(src, out, 1)  # left detail
    assert Image.open(out).size == (640, 360)
    assert ImageChops.difference(Image.open(src).convert("RGB"), Image.open(out).convert("RGB")).getbbox()


def test_similarity_gate_catches_a_pattern_washing_over_the_frame(tmp_path):
    import subprocess

    import numpy as np
    from PIL import Image, ImageDraw

    from tools.story.story_motion import MIN_SIMILARITY, similarity_to_still

    still = tmp_path / "still.png"
    img = Image.new("RGB", (512, 288), (240, 230, 210))
    d = ImageDraw.Draw(img)
    for x in range(40, 480, 120):
        d.rectangle((x, 120, x + 70, 240), fill=(200, 60, 60))
    img.save(still)
    broken = img.copy()
    d2 = ImageDraw.Draw(broken)
    for x in range(0, 512, 24):  # a lattice over everything, like the observe-1 failure
        d2.line((x, 0, x - 150, 288), fill=(180, 90, 30), width=5)
        d2.line((x, 0, x + 150, 288), fill=(180, 90, 30), width=5)
    broken_png = tmp_path / "broken.png"
    broken.save(broken_png)
    for name, src in (("same.mp4", still), ("broken.mp4", broken_png)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(src), "-t", "1", "-r", "24",
                        "-pix_fmt", "yuv420p", str(tmp_path / name)], check=True)
    assert similarity_to_still(still, tmp_path / "same.mp4", 0.5) > 0.9
    assert similarity_to_still(still, tmp_path / "broken.mp4", 0.5) < MIN_SIMILARITY
