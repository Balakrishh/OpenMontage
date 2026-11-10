#!/usr/bin/env python3
"""Walking water, fully code-drawn Quick Science episode (~1.5 min).

Every visual change is triggered by the narration word that describes it
(Whisper word timings of each verified line), so picture and words cannot
disagree. The experiment is drawn from numbers — nothing is generated. The
boys appear only in their already-approved hello/goodbye pictures and a
reaction badge cut from them.

Output: projects/proto-code-drawn/export/walking-water-code-drawn.mp4
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lib import kids_science as ks  # noqa: E402
from lib.kids_science import draw as dr  # noqa: E402
from lib.kids_science import sfx  # noqa: E402
from lib.kids_science.draw import ORANGE, RED, WATER, YELLOW, Cup, Strip, ease, mix  # noqa: E402
from lib.kids_science.quick import plan  # noqa: E402
from tools.audio.kokoro_tts_mlx import KokoroTTSMLX  # noqa: E402
from tools.story.story_keyframes import _font  # noqa: E402
from tools.story.story_narration import MAX_LINE_ERROR, word_error  # noqa: E402

W, H, FPS = 1280, 720, 24
OUT = REPO / "projects" / "proto-code-drawn"
BOYS = REPO / "projects" / "kids-quick-walking-water-v3" / "assets" / "images"
exp = ks.experiment("walking-water")
steps = exp["steps"]
_, _, LINES = plan(exp)
ORDER = ["hook", "hello", "materials", "step1", "step2", "step3", "step4", "step5", "step6", "watch", "observe",
         "explain", "bye"]
HOLD = {"hello": 0.6, "step1": 0.6, "step2": 1.2, "step3": 1.2, "step4": 1.2, "step5": 0.8, "step6": 1.2,
        "watch": 0.8, "observe": 1.6, "explain": 0.8, "bye": 1.5}
GAP = 0.45
CUPS_BEATS = {"hook", "step1", "step2", "step3", "step5", "step6", "watch", "observe"}

# ---------------------------------------------------------------- narration


def narrate():
    adir = OUT / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    voice = ks.series()["voices"]["narrator"]
    items = [(b, i, l["text"]) for b in ORDER for i, l in enumerate(LINES[b])]
    jobs = [{"text": t, "output_path": str(adir / f"{b}-{i}.wav"), "voice": voice["kokoro_voice"],
             "lang_code": voice["lang_code"], "speed": voice["speed"]} for b, i, t in items]
    r = KokoroTTSMLX().execute({"jobs": jobs})
    assert r.success, r.error
    from faster_whisper import WhisperModel
    wm = WhisperModel("small", device="cpu", compute_type="int8")
    spans, words, t = {}, [], 0.5
    with wave.open(jobs[0]["output_path"]) as w0:
        params = w0.getparams()
    rate = params.framerate
    with wave.open(str(adir / "narration_raw.wav"), "wb") as out:
        out.setparams(params)
        out.writeframes(b"\x00\x00" * int(0.5 * rate))
        prev = None
        for (b, i, text), j in zip(items, jobs):
            if prev is not None:
                pause = GAP + (HOLD.get(prev, 0) if b != prev else 0)
                out.writeframes(b"\x00\x00" * int(pause * rate))
                t += pause
            segs, _ = wm.transcribe(j["output_path"], language="en", word_timestamps=True, vad_filter=False)
            segs = list(segs)
            heard = " ".join(s.text.strip() for s in segs)
            err = word_error(text, heard)
            assert err <= MAX_LINE_ERROR, f"{b}: heard {heard!r}"
            with wave.open(j["output_path"]) as w:
                out.writeframes(w.readframes(w.getnframes()))
                dur = w.getnframes() / w.getframerate()
            for s in segs:
                for wd in s.words or []:
                    words.append((b, re.sub(r"[^a-z]", "", wd.word.lower()), t + wd.start))
            sp = spans.setdefault(b, [t, t])
            sp[1] = t + dur
            t += dur
            prev = b
        out.writeframes(b"\x00\x00" * int(HOLD["bye"] * rate))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(adir / "narration_raw.wav"), "-af",
                    "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "1", str(adir / "narration.wav")], check=True)
    for k, b in enumerate(ORDER):  # each beat's scene runs until the next beat starts
        spans[b].append(spans[ORDER[k + 1]][0] - 0.25 if k + 1 < len(ORDER) else spans[b][1] + HOLD["bye"])
    return spans, words, t + HOLD["bye"]


SPANS: dict = {}
WORDS: list = []


def at(beat: str, word: str, nth: int = 1, default: float | None = None) -> float:
    """Time the narrator says `word` (its nth occurrence) in `beat`."""
    hits = [t for b, w, t in WORDS if b == beat and w.startswith(word)]
    if len(hits) >= nth:
        return hits[nth - 1]
    s, e, _ = SPANS[beat]
    return default if default is not None else s


def prog(t, a, b):
    return ease((t - a) / max(1e-3, b - a))


# ---------------------------------------------------------------- scenes

CX = (360, 640, 920)
LEFT_PATH = lambda: [(CX[0] + 35, 565), (CX[0] + 60, 295), (CX[0] + 140, 220), (CX[1] - 55, 295), (CX[1] - 25, 565)]  # noqa: E731
RIGHT_PATH = lambda: [(CX[2] - 35, 565), (CX[2] - 60, 295), (CX[2] - 140, 220), (CX[1] + 55, 295), (CX[1] + 25, 565)]  # noqa: E731


def cups(xs=CX):
    return [Cup(cx=x, top=300, bottom=590, w_top=210, w_bottom=165) for x in xs]


def experiment(t: float):
    """The experiment's state at time t — the single source of truth for the cups scenes."""
    cs = cups()
    sl, sr = Strip(LEFT_PATH(), shown=0, color=RED), Strip(RIGHT_PATH(), shown=0, color=YELLOW)
    front, back = [], []
    s2, e2, _ = SPANS["step2"]
    # step 1: cups slide in from the left, one at a time
    s1, e1, n1 = SPANS["step1"]
    if t < n1:
        k0 = at("step1", "line", default=s1)
        xs = []
        for k, x in enumerate(CX):
            p = prog(t, k0 + k * 0.45, k0 + k * 0.45 + 0.9)
            xs.append(-150 + (x + 150) * p)
        cs = cups(xs)
        return cs, [], front, back
    # step 2: the jug pours into the left then the right cup
    p2a = prog(t, at("step2", "fill"), at("step2", "outside", default=s2 + 1.2) + 0.6)
    p2b = prog(t, at("step2", "outside", default=s2 + 1.2) + 0.7, at("step2", "leave", default=e2 - 1) + 0.9)
    cs[0].level, cs[2].level = 0.7 * p2a, 0.7 * p2b
    if s2 - 0.3 < t < SPANS["step2"][2]:
        k = 0 if p2a < 1 else 2
        pouring = 0 < (p2a if k == 0 else p2b) < 1
        tilt = -38 if pouring else -8
        jx = CX[k] + 150
        front.append(lambda d, jx=jx, tilt=tilt, k=k, pouring=pouring, cs=cs: _pour(d, jx, tilt, cs[k], pouring))
    # step 3: a bottle squeezes drops into each outside cup; the colour spreads
    s3, e3, n3 = SPANS["step3"]
    for k, (cup, col, word) in enumerate(((cs[0], RED, "red"), (cs[2], YELLOW, "yellow"))):
        d0 = at("step3", word, default=s3 + k * 2)
        surf = dr.lerp(cup.bottom - 6, cup.top + 10, cup.level)
        if s3 - 0.2 < t < n3:
            front.append(lambda d, cup=cup, col=col, t=t, d0=d0: _bottle_over(d, cup, col, t, d0))
        for j in range(3):
            tj = d0 + 0.25 + j * 0.35
            f = (t - tj) / 0.4
            if 0 <= f < 1:
                cup.drops.append((cup.cx, 290 + f * (surf - 290), col))
            if t >= tj + 0.4:
                cup.ripples.append((t - tj - 0.4) / 0.8)
        cp = prog(t, d0 + 0.65, d0 + 2.6)
        if t >= d0 + 0.65:
            cup.color = mix(WATER, col, cp)
            if cp < 1:
                cup.cloud = (col, cp)
    # step 5: strips laid across, left then right
    s5, e5, _ = SPANS["step5"]
    sl.shown = prog(t, at("step5", "end", default=s5), at("step5", "colored", default=s5 + 1.5) + 0.6)
    sr.shown = prog(t, at("step5", "other", default=s5 + 2.5), at("step5", "middle", default=e5) + 0.4)
    # step 6 + watch: colour climbs the strips (an hour passes on the clock)
    s6, e6, n6 = SPANS["step6"]
    sw, ew, nw = SPANS["watch"]
    so, eo, no = SPANS["observe"]
    climb = 0.55 * prog(t, at("step6", "wait", default=s6), n6) + 0.45 * prog(t, sw, so + 1.2)
    sl.wick = sr.wick = climb
    if s6 - 0.2 < t < nw:
        mins = 60 * prog(t, at("step6", "wait", default=s6), n6)
        front.append(lambda d, mins=mins: dr.clock(d, 1150, 140, 62, mins))
    # observe: drips into the middle cup, which fills orange; outside cups go down a little
    fm = prog(t, at("observe", "drips", default=so + 2), at("observe", "orange", default=eo) + 0.5)
    cs[1].level, cs[1].color = 0.32 * fm, ORANGE
    if fm > 0:
        cs[0].level = cs[2].level = 0.7 - 0.08 * fm
        cs[1].color = mix((236, 80, 50), ORANGE, prog(t, at("observe", "red", default=eo - 2), at("observe", "orange", default=eo)))
    if 0 < fm < 1 or (at("observe", "drips", default=so + 2) - 0.6 < t < at("observe", "drips", default=so + 2)):
        ph = (t * 1.4) % 1
        cs[1].drops = [(CX[1] - 25, 300 + ph * (560 - 300 - 160 * 0.32 * fm), RED),
                       (CX[1] + 25, 300 + ((ph + 0.5) % 1) * (560 - 300 - 160 * 0.32 * fm), YELLOW)]
    return cs, [sl, sr], front, back


def _pour(d, jx, tilt, cup, pouring):
    tip = dr.jug(d, jx, 175, s=0.85, tilt=tilt, level=0.55)
    if pouring:
        surf = dr.lerp(cup.bottom - 6, cup.top + 10, cup.level)
        d.line(dr._sc([tip, (tip[0] - 8, tip[1] + 30), (cup.cx + 30, surf)]), fill=WATER + (230,), width=9 * dr.SS,
               joint="curve")


def _bottle_over(d, cup, col, t, d0):
    if t < d0 - 0.3:
        return
    squeeze = 1 - 0.08 * math.sin(max(0, t - d0) * 9) if d0 < t < d0 + 1.3 else 1
    dr.bottle(d, cup.cx, 205, col, s=0.9 * squeeze, flip=True)


# ---------------------------------------------------------------- frame


def materials_frame(t, bg):
    s, e, _ = SPANS["materials"]
    items = [("cups", "3 clear cups"), ("water", "water"), ("red", "red coloring"), ("yellow", "yellow coloring"),
             ("paper", "2 paper towels")]
    pops = [at("materials", w, default=s + i * 1.2) for i, (w, _) in enumerate(items)]
    sc = [(0 if t < p else min(1.15, (t - p) / 0.25) if t < p + 0.25 else 1.15 - 0.15 * prog(t, p + 0.25, p + 0.45))
          for p in pops]
    small = []
    for k, x in enumerate((120, 240, 360)):
        if sc[0] > 0:
            small.append(Cup(cx=x, top=590 - 175 * sc[0], bottom=590, w_top=112 * sc[0], w_bottom=88 * sc[0]))
    front = []
    if sc[1] > 0:
        front.append(lambda d: dr.jug(d, 530, 590 - 85 * 1.05 * sc[1], s=1.05 * sc[1], level=0.6))
    if sc[2] > 0:
        front.append(lambda d: dr.bottle(d, 700, 590 - 55 * 1.25 * sc[2], RED, s=1.25 * sc[2]))
    if sc[3] > 0:
        front.append(lambda d: dr.bottle(d, 880, 590 - 55 * 1.25 * sc[3], YELLOW, s=1.25 * sc[3]))
    if sc[4] > 0:
        front.append(lambda d: (dr.towel_flat(d, 1090, 560, sc[4]), dr.towel_flat(d, 1110, 610, sc[4])))
    img = dr.render(bg, small, [], front=front)
    d = ImageDraw.Draw(img)
    f = _font(26)
    for (_, label), x, p in zip(items, (240, 530, 700, 880, 1100), pops):
        if t >= p + 0.2:
            tw = d.textbbox((0, 0), label, font=f)[2]
            d.text((x - tw // 2, 645), label, font=f, fill=(90, 60, 40))
    return img


def fold_frame(t, bg):
    s, e, _ = SPANS["step4"]
    t0 = at("step4", "fold", default=s)
    widths = [240, 120, 60, 30]
    k = min(3, max(0, int((t - t0) / 0.8)))
    p = ((t - t0) / 0.8) - k if t >= t0 else 0
    if t < t0:
        w_now, fp = 240, 0
    elif k >= 3 or t - t0 >= 2.4:
        w_now, fp = 30, 0
    else:
        w_now, fp = widths[k] - (widths[k] - widths[k + 1]) * ease(p) * 0.5, ease(p)
        if fp >= 0.99:
            w_now = widths[k + 1]
    h_now = 240 + 60 * prog(t, t0, t0 + 2.4)
    front = [lambda d: (dr.towel_fold(d, 470, 420, w_now, h_now, fp), dr.towel_fold(d, 810, 420, w_now, h_now, fp))]
    return dr.render(bg, [], [], front=front)


def plant_frame(t, bg):
    s = at("bye", "plants", default=SPANS["bye"][0])
    e = at("bye", "food", default=s + 4)
    return dr.render(bg, [], [], front=[lambda d: dr.plant(d, 640, 640, prog(t, s + 0.5, e))])


def tray_frame(t, bg):
    cs, sts, front, back = experiment(1e9)
    cs = [Cup(cx=c.cx, top=c.top - 20, bottom=c.bottom - 20, w_top=c.w_top, w_bottom=c.w_bottom, level=c.level,
              color=c.color) for c in cs]
    sts = [Strip([(x, y - 20) for x, y in s.points], 1, 1, s.color) for s in sts]
    return dr.render(bg, cs, sts, back=[lambda d: dr.tray(d, 190, 1090, 585)])


def picture_frame(path: Path, t, s, e, zoom=0.06):
    im = Image.open(path).convert("RGB").resize((W, H))
    z = 1 + zoom * prog(t, s, e)
    cw, ch = W / z, H / z
    return im.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.LANCZOS)


def camera(img, z, fx=640, fy=420):
    if z <= 1.001:
        return img
    cw, ch = W / z, H / z
    x0 = min(max(0, fx - cw / 2), W - cw)
    y0 = min(max(0, fy - ch / 2), H - ch)
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)


def wrap(d, text, font, maxw):
    out, cur = [], ""
    for w in text.split():
        nxt = (cur + " " + w).strip()
        if d.textbbox((0, 0), nxt, font=font)[2] > maxw and cur:
            out.append(cur)
            cur = w
        else:
            cur = nxt
    return out + [cur] if cur else out


def caption(img, head, sub="", panel=False):
    d = ImageDraw.Draw(img, "RGBA")
    fh, fs = _font(50), _font(30)
    subs = wrap(d, sub, fs, W - 200) if sub else []
    hgt = 70 + 40 * len(subs)
    if panel:
        d.rounded_rectangle((120, 22, W - 120, 22 + hgt + 10), radius=24, fill=(255, 252, 245, 215))
    tw = d.textbbox((0, 0), head, font=fh)[2]
    d.text(((W - tw) // 2, 30), head, font=fh, fill=(120, 55, 25))
    for i, line in enumerate(subs):
        sw = d.textbbox((0, 0), line, font=fs)[2]
        d.text(((W - sw) // 2, 95 + i * 40), line, font=fs, fill=(70, 45, 30))
    return img


_BADGE = None


def badge(img, p):
    """The boys' faces (cut from their approved hello picture) popping in at the payoff."""
    global _BADGE
    if _BADGE is None:
        src = Image.open(BOYS / "hello-1.png").convert("RGB").resize((W, H))
        crop = src.crop((390, 70, 930, 430)).resize((240, 160), Image.LANCZOS)
        m = Image.new("L", crop.size, 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, crop.width - 1, crop.height - 1), radius=40, fill=255)
        _BADGE = (crop, m)
    crop, m = _BADGE
    s = min(1.12, p / 0.6 * 1.12) if p < 0.6 else 1.12 - 0.12 * min(1, (p - 0.6) / 0.4)
    if s <= 0.02:
        return img
    c2, m2 = crop.resize((int(240 * s), int(160 * s))), m.resize((int(240 * s), int(160 * s)))
    x, y = W - 18 - c2.width, H - 18 - c2.height
    frame = Image.new("RGB", (c2.width + 12, c2.height + 12), (255, 255, 255))
    fm = Image.new("L", frame.size, 0)
    ImageDraw.Draw(fm).rounded_rectangle((0, 0, frame.width - 1, frame.height - 1), radius=44, fill=255)
    img.paste(frame, (x - 6, y - 6), fm)
    img.paste(c2, (x, y), m2)
    return img


def frame(t, bg):
    def scene_of(t):
        for b in ORDER:
            s, e, n = SPANS[b]
            if t < n:
                return b
        return "bye"

    b = scene_of(t)
    s, e, n = SPANS[b]
    cups_beats = CUPS_BEATS
    if b == "hello":
        img = picture_frame(BOYS / "hello-1.png", t, s - 0.3, n)
        d = ImageDraw.Draw(img, "RGBA")
        f = _font(34)
        for name, x, p in (("Arjun", 500, at("hello", "arjun", default=s + 1)), ("Dev", 820, at("hello", "dev", default=s + 1.5))):
            if t >= p:
                tw = d.textbbox((0, 0), name, font=f)[2]
                d.rounded_rectangle((x - tw // 2 - 18, 650, x + tw // 2 + 18, 700), radius=22, fill=(255, 250, 240, 230))
                d.text((x - tw // 2, 656), name, font=f, fill=(120, 55, 25))
        return img
    if b == "materials":
        return caption(materials_frame(t, bg), "You'll need")
    if b == "step4":
        return caption(fold_frame(t, bg), "Step 4", steps[3])
    if b in cups_beats:
        cs, sts, front, back = experiment(1e9 if b == "hook" else t)
        img = dr.render(bg, cs, sts, front=front, back=back)
        zoom = {"hook": 1 + 0.05 * prog(t, s, n), "observe": 1 + 0.08 * prog(t, s, n),
                "watch": 1 + 0.04 * prog(t, s, n)}.get(b, 1 + 0.025 * prog(t, s, n))
        img = camera(img, zoom, 640, 430)
        if b == "hook":
            return caption(img, exp["question"])
        if b.startswith("step"):
            i = int(b[4:])
            return caption(img, f"Step {i}", steps[i - 1])
        if b == "watch":
            return caption(img, "Now watch closely…")
        whoa = at("observe", "whoa", default=s)
        img = caption(img, "Whoa!", "Red and yellow meet in the middle and make orange!")
        return badge(img, (t - whoa) / 0.5 if t < whoa + 3.5 else max(0, 1 - (t - whoa - 3.5) / 0.3) * 1.0)
    if b == "explain":
        res = dr.render(bg, *experiment(1e9)[:2])
        img = bg.copy()
        img.paste(res.resize((int(W * 0.6), int(H * 0.6)), Image.LANCZOS), (10, 190))
        lv = prog(t, at("explain", "water", default=s + 3), at("explain", "gaps", nth=2, default=s + 8))
        p_in = prog(t, s, s + 0.5)
        img = dr.lens(img, 930, 400, 230 * (0.3 + 0.7 * p_in), lv, anchor=(10 + 500 * 0.6, 190 + 222 * 0.6))
        cap = at("explain", "capillary", default=e - 3)
        if t >= cap:
            return caption(img, "Capillary action!", "Water climbs through tiny spaces.", panel=True)
        return caption(img, "Tiny fibers, tiny gaps", "Water sticks to the fibers and to itself.", panel=True)
    # bye: plant → tray → the boys
    tf, tt = at("bye", "food", default=s + 4), at("bye", "try", default=e - 3)
    if t < tf - 0.2:
        return caption(plant_frame(t, bg), "Plants do it too!", "Water climbs up thin tubes in their stems.")
    if t < tt - 0.2:
        return caption(tray_frame(t, bg), "Use a tray!", "Food coloring can stain clothes and tables.")
    return caption(picture_frame(BOYS / "bye-1.png", t, tt - 0.2, n), "Try it with a grown-up!", panel=True)


# ---------------------------------------------------------------- sound


def sound_events():
    ev = []
    s, e, n = SPANS["hook"]
    ev.append((s - 0.3, sfx.chime() * 0.6))
    ev += [(at("hello", w, default=SPANS["hello"][0] + 1), sfx.pop(600)) for w in ("arjun", "dev")]
    ev += [(at("materials", w, default=SPANS["materials"][0]), sfx.pop(480 + 60 * i))
           for i, w in enumerate(("cups", "water", "red", "yellow", "paper"))]
    k0 = at("step1", "line", default=SPANS["step1"][0])
    ev += [(k0 + k * 0.45, sfx.whoosh(0.7) * 0.6) for k in range(3)]
    s2 = SPANS["step2"][0]
    ev.append((at("step2", "fill", default=s2), sfx.bubbles(1.6)))
    ev.append((at("step2", "outside", default=s2 + 1.2) + 0.7, sfx.bubbles(1.6)))
    for w in ("red", "yellow"):
        d0 = at("step3", w, default=SPANS["step3"][0])
        ev += [(d0 + 0.25 + j * 0.35 + 0.4, sfx.plink(1150 - 120 * j)) for j in range(3)]
    t0 = at("step4", "fold", default=SPANS["step4"][0])
    ev += [(t0 + k * 0.8, sfx.whoosh(0.5) * 0.7) for k in range(3)]
    ev += [(at("step5", "end", default=SPANS["step5"][0]), sfx.whoosh(0.9) * 0.6),
           (at("step5", "other", default=SPANS["step5"][0] + 2.5), sfx.whoosh(0.9) * 0.6)]
    s6, e6, n6 = SPANS["step6"]
    tt = at("step6", "wait", default=s6)
    while tt < n6:
        ev.append((tt, sfx.tick()))
        tt += 0.25
    so, eo, no = SPANS["observe"]
    d0 = at("observe", "drips", default=so + 2)
    tt = d0
    while tt < at("observe", "orange", default=eo):
        ev.append((tt, sfx.plink(900) * 0.35))
        tt += 0.7
    ev.append((at("observe", "whoa", default=so), sfx.chime()))
    ev.append((at("observe", "whoa", default=so) + 0.1, sfx.pop(700)))
    ev.append((SPANS["explain"][0], sfx.pop(420)))
    ev.append((at("explain", "capillary", default=SPANS["explain"][1] - 3), sfx.chime() * 0.7))
    ev.append((at("bye", "plants", default=SPANS["bye"][0]), sfx.pop(500)))
    ev.append((at("bye", "food", default=SPANS["bye"][0] + 4), sfx.pop(560)))
    return ev


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sfx.RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


# ---------------------------------------------------------------- main


def main() -> None:
    global SPANS, WORDS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "project.json").write_text(json.dumps({"version": "1.0", "project_id": OUT.name,
                                                  "title": "Code-drawn walking water (prototype)",
                                                  "pipeline_type": "kids-science-story"}))
    tj = OUT / "audio" / "timing.json"
    if "--preview" in sys.argv and tj.exists():
        d = json.loads(tj.read_text())
        SPANS, WORDS, total = d["spans"], [tuple(w) for w in d["words"]], d["total"]
    else:
        SPANS, WORDS, total = narrate()
        tj.write_text(json.dumps({"spans": SPANS, "words": WORDS, "total": total}, indent=1))
    if "--preview" in sys.argv:
        bg = dr.background(W, H)
        times = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")] if len(sys.argv) > 2 else \
            [SPANS[b][0] + f * (SPANS[b][2] - SPANS[b][0]) for b in ORDER for f in (0.15, 0.6, 0.95)]
        cw, chh, cols = 400, 225, 6
        sheet = Image.new("RGB", (cw * cols, chh * ((len(times) + cols - 1) // cols)), "white")
        dd = ImageDraw.Draw(sheet)
        for k, t in enumerate(times):
            x, y = (k % cols) * cw, (k // cols) * chh
            sheet.paste(frame(t, bg).resize((cw, chh)), (x, y))
            dd.rectangle((x, y, x + 50, y + 14), fill="black")
            dd.text((x + 2, y + 1), f"{t:.1f}s", fill="white")
        sheet.save(OUT / "preview.jpg", quality=85)
        print(OUT / "preview.jpg")
        return
    write_wav(OUT / "audio" / "sfx.wav", sfx.mix_track(sound_events(), total))

    bg = dr.background(W, H)
    video = OUT / "audio" / "video.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           str(video)], stdin=subprocess.PIPE)
    prev_scene, last, fade_from = None, None, None
    n_frames = int(total * FPS)
    for i in range(n_frames):
        t = i / FPS
        img = frame(t, bg)
        scene = next((b for b in ORDER if t < SPANS[b][2]), "bye")
        scene = "cups" if scene in CUPS_BEATS else scene
        if prev_scene is not None and scene != prev_scene:
            fade_from = (last, i)
        if fade_from and i - fade_from[1] < 8:
            img = Image.blend(fade_from[0], img, (i - fade_from[1] + 1) / 8)
        prev_scene, last = scene, img
        ff.stdin.write(img.tobytes())
        if i % 240 == 0:
            print(f"  frame {i}/{n_frames}", flush=True)
    ff.stdin.close()
    ff.wait()

    music = REPO / ks.series()["music"]["path"]
    export = OUT / "export"
    export.mkdir(exist_ok=True)
    out = export / "walking-water-code-drawn.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(OUT / "audio" / "narration.wav"),
                    "-i", str(OUT / "audio" / "sfx.wav"), "-stream_loop", "-1", "-i", str(music),
                    "-filter_complex",
                    "[2:a]volume=0.55[s];[3:a]volume=0.10,afade=t=out:st=%.2f:d=2[m];"
                    "[1:a][s][m]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.9[a]" % (total - 2),
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}",
                    str(out)], check=True)
    print(out, f"{total:.1f}s")


if __name__ == "__main__":
    main()
