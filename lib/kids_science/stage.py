"""Code-drawn Quick Science episodes for any experiment in the library.

An experiment's scene lives in scenes.yaml as data: the parts on the table
and, per beat, actions such as `pour`, `drops`, `sprinkle`, `show`, `move`
and plain `set`s of part numbers. Each action starts on the narration word
it names (Whisper word timings of the verified line), so a picture can only
change when the narrator says it does. Sound effects follow the actions.

Pacing is deliberately slow for young viewers: a slower voice, a long pause
after every step with its picture held, and unhurried animations.
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

from lib import kids_science as ks
from lib.kids_science import parts as P
from lib.kids_science import sfx
from lib.kids_science.draw import SS, _texture, background, ease, lerp

REPO = Path(__file__).resolve().parents[2]
W, H, FPS = 1280, 720, 24
BOYS = Path(__file__).parent / "assets"
VOICE_SPEED = 0.85        # slower than the series default, for young viewers
GAP = 0.6                 # breath between lines
HOLD = {"hook": 1.0, "hello": 0.8, "materials": 2.0, "watch": 1.5, "observe": 3.0, "explain": 2.0, "bye": 2.0}
SERIES = "Quick Science with Arjun & Dev"  # groups the episodes into one card on the Backlot board
STEP_HOLD = 2.6           # after every step: the change stays on screen while kids take it in
MAX_LINE_ERROR = 0.2


def scenes() -> dict:
    return yaml.safe_load((Path(__file__).parent / "scenes.yaml").read_text())


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"] if bold else []) + [
        "/System/Library/Fonts/Supplemental/Chalkboard.ttc", "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _blend(a, b, u):
    if _num(a) and _num(b):
        return lerp(a, b, u)
    if isinstance(b, (str, list, tuple)) and isinstance(a, (str, list, tuple)):
        try:
            ca, cb = P.color(a), P.color(b)
            return tuple(int(lerp(x, y, u)) for x, y in zip(ca, cb))
        except (KeyError, TypeError, ValueError):
            pass
    return b if u >= 1 or u > 0 else a


class Stage:
    def __init__(self, exp_id: str, spans: dict, words: list):
        self.exp = ks.experiment(exp_id)
        self.spec = scenes()[exp_id]
        self.spans, self.words = spans, words
        self.parts = {}
        for d in self.spec["parts"]:
            p = dict(d)
            p.setdefault("show", 0 if p.pop("hidden", False) else 1)
            p.setdefault("s", 1)
            p.setdefault("rot", 0)
            p["_home"] = (p.get("x"), p.get("y"))
            self.parts[p["id"]] = p
        self.order = [d["id"] for d in self.spec["parts"]]
        self.keys: list = []        # (start, end, part, param, value)
        self.fx: list = []          # transient drawings: (start, end, kind, data)
        self.sounds: list = []      # (time, sound)
        self.clock_windows: list = []
        self._compile()

    # ------------------------------------------------------------ timing
    def win(self, beat):
        s, e, n = self.spans[beat]
        return s, e, n

    def at(self, beat, ref, default_frac=0.0):
        s, e, n = self.win(beat)
        if ref is None:
            return s + default_frac * (n - s)
        if _num(ref):
            return s + ref * (n - s)
        if isinstance(ref, str) and ref.startswith("+"):
            return e + float(ref[1:])
        if ref == "start":
            return s
        if ref == "end":
            return e
        if isinstance(ref, str) and "+" in ref:
            base, off = ref.split("+", 1)
            return self.at(beat, base, default_frac) + float(off)
        word, nth = (ref.split("#") + ["1"])[:2]
        hits = [t for b, w, t in self.words if b == beat and w.startswith(word.lower())]
        if len(hits) >= int(nth):
            return hits[int(nth) - 1]
        return s + default_frac * (n - s)

    # ------------------------------------------------------------ compile
    def key(self, a, b, pid, param, value):
        self.keys.append((a, max(b, a + 1e-3), pid, param, value))

    def _sets(self, a, b, sets):
        for k, v in (sets or {}).items():
            pid, param = k.split(".", 1)
            self.key(a, b, pid, param, v)

    def _compile(self):
        exp_steps = self.exp["steps"]
        beats = ["materials"] + [f"step{i}" for i in range(1, len(exp_steps) + 1)] + ["watch", "observe", "explain"]
        for beat in beats:
            for act in self.spec.get("beats", {}).get(beat, []) or []:
                self._action(beat, act)
        # observe payoff chime
        self.sounds.append((self.at("observe", "whoa"), sfx.chime()))

    def _action(self, beat, act):
        s, e, n = self.win(beat)
        a = self.at(beat, act.get("at"), 0.0)
        if "to" in act:
            b = self.at(beat, act["to"], 1.0)
        else:
            b = a + act.get("dur", 1.8)
        L = b - a
        fx = act.get("fx")
        if "show" in act:
            for k, pid in enumerate(act["show"]):
                t0 = a + k * act.get("stagger", 0.0)
                self.key(t0, t0 + 0.5, pid, "show", 1)
                self.sounds.append((t0, sfx.pop(480 + 40 * k)))
        if "hide" in act:
            for pid in act["hide"]:
                self.key(a, a + 0.4, pid, "show", 0)
        if fx is None and "set" in act:
            self._sets(a, b, act["set"])
        if fx == "move":
            self._sets(a, b, act["set"])
            self.sounds.append((a, sfx.whoosh(min(1.0, L)) * 0.6))
        elif fx == "pour":
            self._pour(act, a, b, L)
        elif fx == "drops":
            self._drops(act, a, b, L)
        elif fx == "sprinkle":
            self._sprinkle(act, a, b, L)
        elif fx == "drip":  # drops falling from a fixed point (e.g. a soaked strip's end) into a container
            t = a
            k = 0
            while t < b:
                pt = act["from"][k % len(act["from"])]
                self.fx.append((t, t + 0.5, "drop", {"from": pt, "into": act["into"], "color": act.get("colors", ["red"])[k % len(act.get("colors", ["red"]))]}))
                self.sounds.append((t + 0.5, sfx.plink(900) * 0.35))
                t += act.get("every", 0.7)
                k += 1
            self._sets(a, b, act.get("set"))
        elif fx == "rub":  # wiggle a part back and forth (rubbing, stroking)
            pid, axis = act["part"], act.get("axis", "x")
            base = self.state(a)[pid][axis]
            n = act.get("n", 6)
            step = L / (2 * n + 1)
            for i in range(2 * n):
                self.key(a + i * step, a + (i + 1) * step, pid, axis, base + act.get("amp", 50) * (1 if i % 2 == 0 else -1))
            self.key(a + 2 * n * step, b, pid, axis, base)
            for i in range(n):
                self.sounds.append((a + 2 * i * step, sfx.whoosh(step * 1.6) * 0.5))
            self._sets(a, b, act.get("set"))
        elif fx == "clock":
            self.clock_windows.append((a, b, act.get("minutes", 60)))
            t = a
            while t < b:
                self.sounds.append((t, sfx.tick()))
                t += 0.25
        if "sfx" in act:
            snd = {"chime": sfx.chime(), "pop": sfx.pop(), "whoosh": sfx.whoosh(), "plink": sfx.plink(),
                   "bubbles": sfx.bubbles(max(0.5, L)), "fizz": sfx.bubbles(max(0.5, L), density=40)}[act["sfx"]]
            self.sounds.append((a, snd))

    def _over(self, by, into, lift=150, t=None):
        """Where to hold a tool above a container — wherever the container is at that moment."""
        tgt = self.state(t)[into] if t is not None else self.parts[into]
        k, w, h, hw = P.container_geom(tgt)
        return tgt["x"], tgt["y"] - h - lift

    def _pour(self, act, a, b, L):
        by, into = act["by"], act["into"]
        tx, ty = self._over(by, into, act.get("lift", 40), a)
        side = act.get("side", 1)
        hx, hy = self.parts[by]["_home"]
        self.key(a, a + 0.25 * L, by, "x", tx + side * act.get("dx", 120))
        self.key(a, a + 0.25 * L, by, "y", ty)  # L >= 4 s keeps this travel unhurried
        self.key(a + 0.25 * L, a + 0.33 * L, by, "rot", -side * act.get("tilt", 45))
        self.key(a + 0.33 * L, a + 0.34 * L, by, "flow", 1)
        self.key(a + 0.33 * L, a + 0.34 * L, by, "_into", into)
        self._sets(a + 0.34 * L, a + 0.85 * L, act.get("set"))
        self.key(a + 0.85 * L, a + 0.86 * L, by, "flow", 0)
        self.key(a + 0.86 * L, a + 0.93 * L, by, "rot", 0)
        self.key(a + 0.93 * L, b, by, "x", hx)
        self.key(a + 0.93 * L, b, by, "y", hy)
        kind = self.parts[by].get("kind", "jug")
        snd = sfx.bubbles(0.5 * L) if kind != "box" else sfx.whoosh(0.5 * L) * 0.4
        self.sounds.append((a + 0.34 * L, snd))

    def _drops(self, act, a, b, L):
        by, into = act["by"], act["into"]
        tx, ty = self._over(by, into, act.get("lift", 30), a)
        hx, hy = self.parts[by]["_home"]
        self.key(a, a + 0.2 * L, by, "x", tx)
        self.key(a, a + 0.2 * L, by, "y", ty - 70)
        self.key(a, a + 0.01, by, "flip", 1)
        n = act.get("n", 3)
        col = self.parts[by].get("color", "red")
        for j in range(n):
            t0 = a + 0.25 * L + j * 0.4 * L / max(1, n)
            self.fx.append((t0, t0 + 0.45, "drop", {"from": by, "into": into, "color": col}))
            self.sounds.append((t0 + 0.45, sfx.plink(1150 - 110 * j)))
        if act.get("cloud", True) and self.parts[into].get("type") == "container":
            self.key(a + 0.3 * L, a + 0.31 * L, into, "cloudc", col)
            self.key(a + 0.3 * L, a + 0.31 * L, into, "cloud", 0.01)
            self.key(a + 0.31 * L, a + 0.9 * L, into, "cloud", 1.0)
        self._sets(a + 0.35 * L, a + 0.9 * L, act.get("set"))
        self.key(b - 0.1 * L, b, by, "flip", 0)
        self.key(b - 0.1 * L, b, by, "x", hx)
        self.key(b - 0.1 * L, b, by, "y", hy)

    def _sprinkle(self, act, a, b, L):
        by, into = act["by"], act["into"]
        tx, ty = self._over(by, into, act.get("lift", 60), a)
        hx, hy = self.parts[by]["_home"]
        self.key(a, a + 0.2 * L, by, "x", tx + act.get("dx", 0))
        self.key(a, a + 0.2 * L, by, "y", ty)
        self.key(a + 0.2 * L, a + 0.25 * L, by, "rot", act.get("tilt", 160))
        self.key(a + 0.25 * L, a + 0.26 * L, by, "shake", 1)
        self.fx.append((a + 0.25 * L, a + 0.75 * L, "specks", {"from": by, "into": into, "color": act.get("color", "white")}))
        t = a + 0.25 * L
        while t < a + 0.75 * L:
            self.sounds.append((t, sfx.tick() * 0.7))
            t += 0.12
        self._sets(a + 0.3 * L, a + 0.8 * L, act.get("set"))
        self.key(a + 0.75 * L, a + 0.76 * L, by, "shake", 0)
        self.key(a + 0.76 * L, a + 0.85 * L, by, "rot", 0)
        self.key(a + 0.85 * L, b, by, "x", hx)
        self.key(a + 0.85 * L, b, by, "y", hy)

    # ------------------------------------------------------------ state + draw
    def state(self, t):
        st = {pid: dict(p) for pid, p in self.parts.items()}
        for a, b, pid, param, v in sorted(self.keys, key=lambda k: k[0]):
            if t < a:
                continue
            u = ease((t - a) / (b - a))
            cur = st[pid].get(param, 0 if _num(v) else v)
            st[pid][param] = _blend(cur, v, u) if (_num(v) or isinstance(v, (list, tuple)) or v in P.PALETTE) else v
        for p in st.values():
            if p.get("type") == "pourer" and p.get("flow", 0) > 0 and p.get("_into") in st:
                tgt = st[p["_into"]]
                p["_target"] = (tgt["x"] + (tgt.get("sx_off", 0)), P.surface_y(tgt))
        return st

    def draw(self, t, bg, st=None, labels=False, clock=True):
        st = st or self.state(t)
        w, h = bg.size
        layers = [Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0)) for _ in range(3)]
        db, dl, df = (ImageDraw.Draw(L) for L in layers)
        ordered = [st[i] for i in self.order]
        for p in ordered:
            fb, fl, ff = P.DRAW[p["type"]]
            if fb:
                fb(db, p, t)
        for p in ordered:
            fb, fl, ff = P.DRAW[p["type"]]
            if fl:
                fl(dl, p, t)
        for p in ordered:  # things inside containers, then glass fronts, then everything else
            if p["type"] in P.INSIDE:
                P.DRAW[p["type"]][2](df, p, t)
        for p in ordered:
            if p["type"] == "container":
                P.container_front(df, p, t)
        for p in ordered:
            if p["type"] not in P.INSIDE and p["type"] != "container" and P.DRAW[p["type"]][2]:
                P.DRAW[p["type"]][2](df, p, t)
        for a, b, kind, d in self.fx:  # transient drops / specks, drawn from the tool's tip to the surface
            if not a <= t < b:
                continue
            f = (t - a) / (b - a)
            tgt = st[d["into"]]
            if isinstance(d["from"], (list, tuple)):
                tip = tuple(d["from"])
            else:
                src = st[d["from"]]
                tip = src.get("_tip", (src["x"], src["y"]))
            sy = P.surface_y(tgt) if tgt["type"] == "container" else tgt["y"]
            if kind == "drop":
                from lib.kids_science.draw import drop
                drop(df, lerp(tip[0], tgt["x"], f * 0.2), lerp(tip[1], sy, f * f), P.color(d["color"]))
            else:
                rng = np.random.default_rng(int(t * 40))
                for j in range(14):
                    q = rng.random()
                    px = lerp(tip[0], tgt["x"], q) + rng.uniform(-25, 25)
                    py = lerp(tip[1], sy, q)
                    col = P.color(d["color"])
                    df.ellipse([(px - 2.5) * SS, (py - 2.5) * SS, (px + 2.5) * SS, (py + 2.5) * SS], fill=col + (255,),
                               outline=(150, 150, 150, 255))
        if clock:
            for a, b, mins in self.clock_windows:
                if a - 0.3 <= t <= b + 1.0:
                    P.clock(df, {"x": 1185, "y": 250, "minutes": mins * ease((t - a) / (b - a)), "show": min(1, (t - a + 0.3) / 0.4)}, t)
        out = bg.convert("RGBA")
        for k, (lay, blur) in enumerate(zip(layers, (0.8, 0.6, 0.8))):
            lay = lay.filter(ImageFilter.GaussianBlur(blur * SS)).resize((w, h), Image.LANCZOS)
            if k == 1:  # watercolour liquids
                arr = np.asarray(lay, float)
                a = arr[..., 3:4] / 255
                soft = np.asarray(Image.fromarray(arr[..., 3].astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), float)[..., None] / 255
                edge = np.clip((a - soft) * 2.5, 0, 1)
                arr[..., :3] = np.clip(arr[..., :3] * _texture(w, h) * (1 - 0.25 * edge), 0, 255)
                lay = Image.fromarray(arr.astype(np.uint8), "RGBA")
            out.alpha_composite(lay)
        img = out.convert("RGB")
        tags = [p for p in ordered if p["type"] == "tag" and p.get("show", 1) > 0.05]
        if tags:
            dd = ImageDraw.Draw(img, "RGBA")
            f = _font(28)
            for p in tags:
                a = int(255 * min(1, p["show"]))
                tw = dd.textbbox((0, 0), p["text"], font=f)[2]
                x, y = p["x"], p["y"]
                if p.get("to"):
                    dd.line([(x, y + 18), tuple(p["to"])], fill=(120, 90, 70, a), width=3)
                    dd.ellipse([p["to"][0] - 5, p["to"][1] - 5, p["to"][0] + 5, p["to"][1] + 5], fill=(120, 90, 70, a))
                dd.rounded_rectangle((x - tw // 2 - 14, y, x + tw // 2 + 14, y + 40), radius=18, fill=(255, 250, 240, int(a * 0.92)))
                dd.text((x - tw // 2, y + 4), p["text"], font=f, fill=(120, 55, 25, a))
        if labels:
            dd = ImageDraw.Draw(img)
            f = _font(26)
            for p in ordered:
                if p.get("label") and p.get("show", 1) > 0.6:
                    lx, ly = p.get("label_at", (p["x"], 645))
                    tw = dd.textbbox((0, 0), p["label"], font=f)[2]
                    dd.text((lx - tw // 2, ly), p["label"], font=f, fill=(90, 60, 40))
        return img


# ---------------------------------------------------------------- the episode


def _wrap(d, text, font, maxw):
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
    if not panel and np.asarray(img.crop((300, 10, 980, 150)).convert("L"), float).mean() < 150:
        panel = True  # dark room: text needs its own light panel to stay readable
    d = ImageDraw.Draw(img, "RGBA")
    fh, fs = _font(52), _font(32)
    heads = _wrap(d, head, fh, W - 160)
    subs = _wrap(d, sub, fs, W - 220) if sub else []
    y0 = 26 + 64 * len(heads)
    if panel:
        d.rounded_rectangle((110, 18, W - 110, y0 + 14 + 42 * len(subs)), radius=24, fill=(255, 252, 245, 220))
    for i, line in enumerate(heads):
        tw = d.textbbox((0, 0), line, font=fh)[2]
        d.text(((W - tw) // 2, 26 + 64 * i), line, font=fh, fill=(120, 55, 25))
    for i, line in enumerate(subs):
        sw = d.textbbox((0, 0), line, font=fs)[2]
        d.text(((W - sw) // 2, y0 + 4 + i * 42), line, font=fs, fill=(70, 45, 30))
    return img


def picture(path, t, s, e, zoom=0.06):
    im = Image.open(path).convert("RGB").resize((W, H))
    z = 1 + zoom * ease((t - s) / max(0.1, e - s))
    cw, ch = W / z, H / z
    return im.crop(((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2)).resize((W, H), Image.LANCZOS)


def camera(img, z, fx=640, fy=430):
    if z <= 1.001:
        return img
    cw, ch = W / z, H / z
    x0 = min(max(0, fx - cw / 2), W - cw)
    y0 = min(max(0, fy - ch / 2), H - ch)
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)


_BADGE = {}


def badge(img, p, size=(240, 160), corner="br"):
    if "b" not in _BADGE:
        src = Image.open(BOYS / "hello.png").convert("RGB").resize((W, H))
        _BADGE["b"] = src.crop((390, 70, 930, 430))
    s = min(1.12, p / 0.6 * 1.12) if p < 0.6 else 1.12 - 0.12 * min(1, (p - 0.6) / 0.4)
    if s <= 0.02:
        return img
    bw, bh = int(size[0] * s), int(size[1] * s)
    c2 = _BADGE["b"].resize((bw, bh), Image.LANCZOS)
    m = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, bw - 1, bh - 1), radius=int(40 * s), fill=255)
    x, y = (W - 18 - bw, H - 18 - bh) if corner == "br" else (W - 18 - bw, 150)
    fr = Image.new("RGB", (bw + 12, bh + 12), (255, 255, 255))
    fm = Image.new("L", fr.size, 0)
    ImageDraw.Draw(fm).rounded_rectangle((0, 0, fr.width - 1, fr.height - 1), radius=int(44 * s), fill=255)
    img.paste(fr, (x - 6, y - 6), fm)
    img.paste(c2, (x, y), m)
    return img


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class Episode:
    def __init__(self, exp_id: str, out: Path):
        from lib.kids_science.quick import plan
        self.id = exp_id
        self.exp = ks.experiment(exp_id)
        self.spec = scenes()[exp_id]
        self.out = out
        _, _, self.lines = plan(self.exp)
        self.order = ["hook", "hello", "materials"] + [f"step{i}" for i in range(1, len(self.exp["steps"]) + 1)] + \
                     ["watch", "observe", "explain", "bye"]

    # ---------------------------------------------------------- narration
    def narrate(self):
        from faster_whisper import WhisperModel

        from tools.audio.kokoro_tts_mlx import KokoroTTSMLX
        from tools.story.story_narration import word_error
        adir = self.out / "audio"
        adir.mkdir(parents=True, exist_ok=True)
        voice = ks.series()["voices"]["narrator"]
        items = [(b, i, l["text"]) for b in self.order for i, l in enumerate(self.lines[b])]
        jobs = [{"text": t, "output_path": str(adir / f"{b}-{i}.wav"), "voice": voice["kokoro_voice"],
                 "lang_code": voice["lang_code"], "speed": VOICE_SPEED} for b, i, t in items]
        r = KokoroTTSMLX().execute({"jobs": jobs})
        if not r.success:
            raise RuntimeError(r.error)
        wm = WhisperModel("small", device="cpu", compute_type="int8")
        spans, words, t = {}, [], 0.6
        with wave.open(jobs[0]["output_path"]) as w0:
            params = w0.getparams()
        rate = params.framerate
        silence = lambda s: b"\x00\x00" * int(s * rate)  # noqa: E731
        with wave.open(str(adir / "narration_raw.wav"), "wb") as out:
            out.setparams(params)
            out.writeframes(silence(0.6))
            prev = None
            for (b, i, text), j in zip(items, jobs):
                if prev is not None:
                    hold = (STEP_HOLD if prev.startswith("step") else HOLD.get(prev, 0.5)) if b != prev else 0.0
                    out.writeframes(silence(GAP + hold))
                    t += GAP + hold
                segs = list(wm.transcribe(j["output_path"], language="en", word_timestamps=True, vad_filter=False)[0])
                heard = " ".join(s.text.strip() for s in segs)
                err = word_error(text, heard)
                if err > MAX_LINE_ERROR:
                    raise RuntimeError(f"{b}: voice doesn't match the script (heard {heard!r})")
                with wave.open(j["output_path"]) as w:
                    out.writeframes(w.readframes(w.getnframes()))
                    dur = w.getnframes() / w.getframerate()
                for s in segs:
                    for wd in s.words or []:
                        words.append((b, re.sub(r"[^a-z']", "", wd.word.lower()).replace("'", ""), round(t + wd.start, 3)))
                sp = spans.setdefault(b, [t, t])
                sp[1] = t + dur
                t += dur
                prev = b
            out.writeframes(silence(HOLD["bye"]))
        total = t + HOLD["bye"]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(adir / "narration_raw.wav"), "-af",
                        "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "1", str(adir / "narration.wav")], check=True)
        for k, b in enumerate(self.order):
            spans[b].append(spans[self.order[k + 1]][0] - 0.3 if k + 1 < len(self.order) else total)
        return spans, words, total

    # ---------------------------------------------------------- frames
    def setup(self, spans, words):
        self.spans, self.words = spans, words
        self.stage = Stage(self.id, spans, words)
        self.bg = background(W, H)
        self._final = None

    def final_state(self):
        if self._final is None:
            self._final = self.stage.state(1e9)
        return self._final

    def zoom_at(self, t):
        """Camera zoom: each beat's target (scenes.yaml `zoom`), eased in over 1.2 s, plus a slow drift."""
        zs = self.spec.get("zoom", {})
        b = self.beat_of(t)
        k = self.order.index(b)
        z1 = zs.get(b, 1.0)
        z0 = zs.get(self.order[k - 1], 1.0) if k else z1
        if self.order[k - 1] in ("hello", "materials") or b in ("hello", "materials"):
            z0 = z1  # a cut, not a move
        s, e, n = self.spans[b]
        return lerp(z0, z1, ease((t - s) / 1.2)) * (1 + 0.025 * ease((t - s) / max(0.1, n - s)))

    def beat_of(self, t):
        for b in self.order:
            if t < self.spans[b][2]:
                return b
        return "bye"

    def frame(self, t):
        b = self.beat_of(t)
        s, e, n = self.spans[b]
        st, ex, spec = self.stage, self.exp, self.spec
        if b == "hook":
            img = camera(st.draw(t, self.bg, st=self.final_state(), clock=False), self.zoom_at(t) * (1 + 0.03 * ease((t - s) / (n - s))),
                         *spec.get("focus", (640, 430)))
            return caption(img, ex["question"])
        if b == "hello":
            img = picture(BOYS / "hello.png", t, s - 0.3, n)
            d = ImageDraw.Draw(img, "RGBA")
            f = _font(34)
            for name, x in (("Arjun", 500), ("Dev", 820)):
                if t >= st.at("hello", name.lower(), 0.3):
                    tw = d.textbbox((0, 0), name, font=f)[2]
                    d.rounded_rectangle((x - tw // 2 - 18, 650, x + tw // 2 + 18, 700), radius=22, fill=(255, 250, 240, 230))
                    d.text((x - tw // 2, 656), name, font=f, fill=(120, 55, 25))
            return img
        if b == "materials":
            return caption(st.draw(t, self.bg, labels=True), "You'll need")
        if b.startswith("step") or b == "watch":
            img = camera(st.draw(t, self.bg), self.zoom_at(t), *spec.get("focus", (640, 430)))
            if b == "watch":
                return caption(img, "Now watch closely…")
            i = int(b[4:])
            return caption(img, f"Step {i}", ex["steps"][i - 1])
        if b == "observe":
            img = camera(st.draw(t, self.bg), self.zoom_at(t) * (1 + 0.04 * ease((t - s) / (n - s))), *spec.get("focus", (640, 430)))
            img = caption(img, "Whoa!", spec.get("observe_caption", ""))
            whoa = st.at("observe", "whoa")
            p = (t - whoa) / 0.5 if t < whoa + 3.5 else max(0, 1 - (t - whoa - 3.5) / 0.3)
            return badge(img, p)
        if b == "explain":
            res = st.draw(t, self.bg, st=self.final_state(), clock=False)
            img = self.bg.copy()
            img.paste(res.resize((int(W * 0.6), int(H * 0.6)), Image.LANCZOS), (10, 190))
            le = spec.get("explain", {})
            lv = ease((t - st.at("explain", le.get("from"), 0.2)) / max(0.5, st.at("explain", le.get("to"), 0.7) - st.at("explain", le.get("from"), 0.2)))
            p_in = ease((t - s) / 0.5)
            ax, ay = le.get("anchor", (640, 430))
            img = P.lens(img, 930, 400, 230 * (0.3 + 0.7 * p_in), lv, mode=le.get("lens", "bubbles"), t=t,
                         col=P.color(le.get("color", "red")), anchor=(10 + ax * 0.6, 190 + ay * 0.6))
            cap = le.get("captions") or [{"title": self.term(), "sub": ""}]
            cur = cap[0]
            for c in cap:
                if t >= st.at("explain", c.get("at"), 0.0):
                    cur = c
            return caption(img, cur["title"], cur.get("sub", ""), panel=True)
        # bye: real world → safety → the boys
        sents = _sentences(self.lines["bye"][0]["text"])
        bwords = [w for bb, w, tt in self.words if bb == "bye"]
        btimes = [tt for bb, w, tt in self.words if bb == "bye"]
        bounds, k = [], 0
        for sen in sents:
            bounds.append(btimes[min(k, len(btimes) - 1)] if btimes else s)
            k += len(sen.split())
        idx = max(i for i, x in enumerate(bounds) if t >= x - 0.3) if t >= bounds[0] - 0.3 else 0
        if idx == len(sents) - 1 or idx >= 2:
            return caption(picture(BOYS / "bye.png", t, bounds[-1] - 0.3, n), "Try it with a grown-up!", panel=True)
        img = camera(st.draw(t, self.bg, st=self.final_state(), clock=False), spec.get("zoom", {}).get("hook", 1.0),
                     *spec.get("focus", (640, 430)))
        if idx == 0:
            return caption(img, "Did you know?", sents[0], panel=True)
        return caption(img, "Stay safe!", sents[1], panel=True)

    def term(self):
        for f in self.exp["facts"]:
            m = re.search(r"called ([^.,]+)", f)
            if m:
                return m.group(1).strip().rstrip(".").capitalize() + "!"
        return "Here's why"

    # ---------------------------------------------------------- thumbnail
    def thumbnail(self, path: Path):
        """Bold, uncluttered: the finished result alone on a sunburst, a short hook, the boys' faces."""
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        cx, cy = W * 0.42, H * 0.55
        ang = np.arctan2(yy - cy, xx - cx)
        r = np.hypot(xx - cx, yy - cy) / np.hypot(W, H)
        rays = (np.sin(ang * 14) > 0).astype(float)[..., None]
        inner, outer = np.array([255, 244, 200.0]), np.array([255, 196, 110.0])
        base = inner * (1 - r[..., None] * 1.6).clip(0, 1) + outer * (r[..., None] * 1.6).clip(0, 1)
        base = base * (1 - 0.07 * rays)
        ty = int(H * 0.80)
        base[ty:] = np.array([236, 222, 200.0])
        base[ty:ty + 4] *= 0.9
        bg = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        st = copy.deepcopy(self.final_state())
        for pid, p in st.items():
            if p["type"] in ("pourer", "dropper", "shaker", "spoon", "tag", "swab", "hairdryer") or pid in self.spec.get("thumb_hide", []):
                p["show"] = 0
        img = self.stage.draw(1e9, bg, st=st, clock=False)
        fx, fy = self.spec.get("thumb_focus", self.spec.get("focus", (640, 430)))
        img = camera(img, self.spec.get("thumb_zoom", 1.3), fx + 120, fy - 20)
        d = ImageDraw.Draw(img, "RGBA")
        words = self.spec.get("thumb", self.exp["title"]).upper()
        f = _font(100, bold=True)
        lines = _wrap(d, words, f, W - 100)
        y = 24
        for line in lines:
            tw = d.textbbox((0, 0), line, font=f, stroke_width=12)[2]
            d.text(((W - tw) // 2, y), line, font=f, fill=(255, 220, 40), stroke_width=12, stroke_fill=(90, 30, 20))
            y += 112
        img = badge(img, 1.0, size=(360, 240))
        img.convert("RGB").save(path, optimize=True)

    # ---------------------------------------------------------- sound + render
    def sound(self, total):
        st = self.stage
        ev = list(st.sounds)
        ev.append((0.1, sfx.chime() * 0.5))
        ev += [(st.at("hello", w, 0.3), sfx.pop(600)) for w in ("arjun", "dev")]
        ev.append((self.spans["explain"][0], sfx.pop(420)))
        return sfx.mix_track(ev, total)

    def render(self, preview: list | None = None):
        self.out.mkdir(parents=True, exist_ok=True)
        tj = self.out / "audio" / "timing.json"
        if preview is not None and tj.exists():
            d = json.loads(tj.read_text())
            spans, words, total = d["spans"], [tuple(w) for w in d["words"]], d["total"]
        else:
            spans, words, total = self.narrate()
            tj.write_text(json.dumps({"spans": spans, "words": words, "total": total}))
        self.setup(spans, words)
        if preview is not None:
            times = preview or [spans[b][0] + f * (spans[b][2] - spans[b][0]) for b in self.order for f in (0.1, 0.5, 0.95)]
            cw, chh, cols = 400, 225, 6
            sheet = Image.new("RGB", (cw * cols, chh * ((len(times) + cols - 1) // cols)), "white")
            dd = ImageDraw.Draw(sheet)
            for k, t in enumerate(times):
                x, y = (k % cols) * cw, (k // cols) * chh
                sheet.paste(self.frame(t).resize((cw, chh)), (x, y))
                dd.rectangle((x, y, x + 50, y + 14), fill="black")
                dd.text((x + 2, y + 1), f"{t:.1f}s", fill="white")
            sheet.save(self.out / "preview.jpg", quality=85)
            self.thumbnail(self.out / "thumb_preview.png")
            return self.out / "preview.jpg"
        snd = self.sound(total)
        with wave.open(str(self.out / "audio" / "sfx.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sfx.RATE)
            w.writeframes((np.clip(snd, -1, 1) * 32767).astype(np.int16).tobytes())
        video = self.out / "audio" / "video.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(video)],
                              stdin=subprocess.PIPE)
        cupsy = {"hook", "watch", "observe"} | {b for b in self.order if b.startswith("step")}
        prev, last, fade = None, None, None
        for i in range(int(total * FPS)):
            t = i / FPS
            img = self.frame(t)
            sc = self.beat_of(t)
            sc = "stage" if sc in cupsy else sc
            if prev is not None and sc != prev:
                fade = (last, i)
            if fade and i - fade[1] < 9:
                img = Image.blend(fade[0], img, (i - fade[1] + 1) / 9)
            prev, last = sc, img
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        export = self.out / "export"
        export.mkdir(exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.exp["title"]).strip("-")
        final = export / f"{slug}.mp4"
        music = REPO / ks.series()["music"]["path"]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(self.out / "audio" / "narration.wav"),
                        "-i", str(self.out / "audio" / "sfx.wav"), "-stream_loop", "-1", "-i", str(music), "-filter_complex",
                        "[2:a]volume=0.5[s];[3:a]volume=0.10,afade=t=out:st=%.2f:d=2[m];"
                        "[1:a][s][m]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.8[a]" % (total - 2),
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}",
                        str(final)], check=True)
        thumb = export / f"{slug}_THUMBNAIL.png"
        self.thumbnail(thumb)
        self.publish_copy(final, thumb, total)
        return final

    def publish_copy(self, video: Path, thumb: Path, total: float):
        """Engagement-only YouTube copy, same shape the Backlot uploader reads."""
        ex = self.exp
        names = [c["name"] for c in ks.series()["cast"].values()]
        duo = " and ".join(names)
        title = f"{ex['question']} 🧪 | Quick Science with {' & '.join(names)}"[:100]
        topic = "#" + re.sub(r"[^a-z]", "", ex["id"].split("-")[0])
        description = "\n\n".join([
            title,
            f"{ex['question']} Watch the experiment step by step, then try it at home with a grown-up!",
            "You'll need: " + ", ".join(ex["materials"]) + ".",
            "What do you think will happen? Tell us in the comments!",
            f"Safety first: {ex['safety']} Always do experiments with a grown-up.",
            f"Subscribe for more quick science with {duo}.",
            " ".join(["#kidsscience", "#scienceforkids", "#STEM", "#scienceexperiment", topic]),
        ])
        tags = ["kids science", "science experiment for kids", "STEM for kids", "science for kids", "easy science experiment",
                duo, ex["title"], *ex["materials"][:3]]
        exp_dir = video.parent
        (exp_dir / "metadata.json").write_text(json.dumps({"title": title, "description": description, "tags": tags,
                                                           "chapters": []}, indent=2, ensure_ascii=False), encoding="utf-8")
        art = self.out / "artifacts"
        art.mkdir(exist_ok=True)
        (art / "publish_log.json").write_text(json.dumps({
            "version": "1.0",
            "entries": [{"platform": "local_export", "status": "exported", "export_path": str(video),
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
            "metadata": {"suggested_title": title, "suggested_description": description, "suggested_tags": tags,
                         "suggested_chapters": [], "suggested_category_id": "27", "suggested_made_for_kids": True,
                         "suggested_thumbnail": str(thumb.relative_to(self.out)), "contains_synthetic_media": True,
                         "hero_output": f"projects/{self.out.name}/{video.relative_to(self.out).as_posix()}",
                         "copy_policy": "engagement only: no tools, models or production tech named",
                         "publish_action": "exported locally only — nothing uploaded"},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.out / "project.json").write_text(json.dumps({"version": "1.0", "project_id": self.out.name,
                                                           "title": f"{ex['title']} (Quick Science, drawn)",
                                                           "series": SERIES,
                                                           "pipeline_type": "kids-science-story"}))
