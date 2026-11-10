"""Reusable code-drawn parts for every experiment.

A part is a type name plus a dict of numbers (position, scale, liquid levels,
colours, how far something has happened). The stage engine animates those
numbers; each draw function here turns them into pixels. Nothing is
generated, so a part can only ever show the state it was given.

Coordinates are final pixels (1280x720); `x, y` is a part's anchor (bottom
centre for things that stand on the table). Every part has `show` (0..1, pops
in with a little overshoot), `s` (scale) and `rot` (degrees).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from lib.kids_science.draw import SS, _path, _sc, ease, lerp, mix, rot

PALETTE = {
    "water": (170, 205, 235), "red": (214, 40, 52), "yellow": (246, 196, 30), "orange": (240, 120, 30),
    "blue": (40, 110, 220), "green": (70, 170, 80), "purple": (140, 70, 190), "honey": (214, 140, 30),
    "soap": (120, 200, 120), "oil": (246, 222, 120), "milk": (250, 248, 240), "vinegar": (236, 236, 222),
    "foam": (232, 80, 90), "white": (255, 255, 255), "brown": (150, 95, 45), "pink": (240, 130, 170),
    "fizzy": (225, 238, 245), "goo": (246, 244, 236), "lemon": (250, 225, 90), "celery": (150, 205, 110),
    "none": (0, 0, 0),
}
INK = (95, 75, 60)
GLASS = (120, 130, 150, 210)


def color(v):
    if isinstance(v, str):
        return PALETTE[v]
    return tuple(int(c) for c in v)


def oval(cx, cy, rx, ry, n=28):
    return [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def pop(v: float) -> float:
    """show 0..1 -> scale with a small overshoot (back-out)."""
    if v >= 1:
        return 1.0
    if v <= 0:
        return 0.0
    u = v - 1
    return 1 + 2.7 * u ** 3 + 1.7 * u ** 2


# ------------------------------------------------------------------ containers

def _hw_fn(kind: str, w: float, h: float):
    """Half-width of the inside of a container as a function of height above its base (0..h)."""
    if kind == "cup":
        return lambda z: lerp(w * 0.39, w / 2, z / h)
    if kind == "glass":
        return lambda z: lerp(w * 0.44, w / 2, z / h)
    if kind == "bottle":
        def f(z):
            r = z / h
            if r < 0.62:
                return w / 2
            if r < 0.8:
                return lerp(w / 2, w * 0.19, ease((r - 0.62) / 0.18))
            return w * 0.19
        return f
    if kind == "bowl":
        return lambda z: w / 2 * (0.62 + 0.38 * math.sqrt(max(0.0, z / h)))
    if kind == "bag":
        return lambda z: w / 2
    return lambda z: w / 2


SIZES = {"cup": (210, 290), "glass": (150, 330), "bottle": (130, 300), "bowl": (420, 150), "bag": (300, 360),
         "plate": (560, 46)}


def container_geom(p):
    k = p.get("kind", "cup")
    sc = p.get("s", 1) * pop(p.get("show", 1))
    w0, h0 = SIZES[k]
    w, h = w0 * sc * p.get("sw", 1), h0 * sc * p.get("sh", 1)
    return k, w, h, _hw_fn(k, w, h)


def surface_y(p, frac=None):
    """Height (pixel y) of the top liquid surface."""
    k, w, h, hw = container_geom(p)
    if k == "plate":
        return p["y"] - h * 1.3
    tot = sum(p.get(f"l{i}", 0) for i in range(4)) if frac is None else frac
    return p["y"] - 6 - (h - 12) * min(1.0, tot)


def _outline(p, hw, h):
    zs = np.linspace(0, h, 24)
    left = [(p["x"] - hw(z), p["y"] - z) for z in zs]
    right = [(p["x"] + hw(z), p["y"] - z) for z in zs[::-1]]
    return left + right


def container_back(d, p, t):
    k, w, h, hw = container_geom(p)
    if w < 2:
        return
    x, y = p["x"], p["y"]
    d.ellipse(_sc([(x - w * 0.55, y - 6), (x + w * 0.55, y + 12)]), fill=(120, 100, 90, 45))
    if k == "plate":
        d.ellipse(_sc([(x - w / 2, y - h * 2.6), (x + w / 2, y + 4)]), fill=(250, 250, 252, 255),
                  outline=(170, 175, 185, 255), width=3 * SS)
        d.ellipse(_sc([(x - w * 0.36, y - h * 2.6 + h * 0.5), (x + w * 0.36, y - h * 0.4)]),
                  outline=(215, 218, 226, 255), width=2 * SS)
        return
    if k == "bowl":
        d.polygon(_sc(_outline(p, hw, h)), fill=(205, 225, 240, 255) if p.get("opaque") else (236, 242, 250, 90))
        return
    d.polygon(_sc(_outline(p, hw, h)), fill=(236, 242, 250, 70))


def _layer_poly(p, hw, h, z0, z1):
    zs = np.linspace(z0, z1, 12)
    x, y = p["x"], p["y"]
    inset = 5 * p.get("s", 1)
    left = [(x - hw(z) + inset, y - 6 - z) for z in zs]
    right = [(x + hw(z) - inset, y - 6 - z) for z in zs[::-1]]
    return left + right


def container_liquid(d, p, t):
    k, w, h, hw = container_geom(p)
    if w < 24:  # popping in / shrinking away: too small to hold a visible liquid
        return
    x, y = p["x"], p["y"]
    if k == "plate":
        return plate_contents(d, p, t, w, h)
    inner = h - 12
    z = 0.0
    top_col = None
    for i in range(4):
        lv = p.get(f"l{i}", 0)
        if lv <= 0.003:
            continue
        col = mix(color(p.get(f"c{i}", "water")), (250, 250, 250), 0.55 * p.get("milky", 0))  # tint, never overlay:
        z1 = z + lv * inner                                                                    # PIL replaces pixels
        d.polygon(_sc(_layer_poly(p, hw, h, z, z1)), fill=col + (int(p.get(f"a{i}", 215)),))
        z, top_col = z1, col
    if top_col is None:
        return
    sy = y - 6 - z
    shw = hw(z) - 5
    if p.get("cloud", 0) > 0 and p.get("cloud", 0) < 1:
        cp = p["cloud"]
        depth = z
        rx, ry = shw * (0.25 + 0.7 * cp), depth * (0.15 + 0.4 * cp)
        cy = sy + depth * (0.1 + 0.4 * cp)
        d.ellipse(_sc([(x - rx, cy - ry), (x + rx, cy + ry)]), fill=mix(color(p.get("cloudc", "red")), top_col, cp) + (215,))
    surf_w = max(3, h * 0.05) if k != "bowl" else max(6, h * 0.18)
    d.ellipse(_sc([(x - shw, sy - surf_w), (x + shw, sy + surf_w)]), fill=mix(top_col, (255, 255, 255), 0.35) + (230,))
    if k == "bowl":
        bowl_surface(d, p, t, x, sy, shw, surf_w)
    # bubbles rising (fizz) and dancing blobs
    if p.get("fizz", 0) > 0.01:
        rng = np.random.default_rng(int(x))
        n = int(40 * p["fizz"])
        for j in range(n):
            bx = x + rng.uniform(-0.8, 0.8) * hw(0)
            sp = rng.uniform(0.3, 0.8)
            ph = (t * sp + rng.random()) % 1
            by = y - 8 - ph * z
            r = rng.uniform(2, 5) * p.get("s", 1)
            d.ellipse(_sc([(bx - r, by - r), (bx + r, by + r)]), outline=(255, 255, 255, 220), width=SS)
    if p.get("blobs", 0) > 0.01:
        rng = np.random.default_rng(3)
        z_water = p.get("l0", 0) * inner
        for j in range(6):
            sp = rng.uniform(0.12, 0.2)
            ph = (t * sp + j / 6) % 1
            up = math.sin(ph * math.pi)  # up then down again
            by = y - 6 - z_water - up * (z - z_water) * 0.95 * p["blobs"]
            bx = x + rng.uniform(-0.45, 0.45) * hw(0)
            r = rng.uniform(9, 15) * p.get("s", 1)
            col = color(p.get("blobc", "blue"))
            d.ellipse(_sc([(bx - r, by - r * 1.2), (bx + r, by + r)]), fill=col + (235,))
            for b in range(2):
                d.ellipse(_sc([(bx - 3 + b * 6, by - r * 1.2 - 5 - b * 3), (bx + 2 + b * 6, by - r * 1.2 - b * 3)]),
                          outline=(255, 255, 255, 230), width=SS)


def container_front(d, p, t):
    k, w, h, hw = container_geom(p)
    if w < 2:
        return
    x, y = p["x"], p["y"]
    if k == "plate":
        return
    d.polygon(_sc(_outline(p, hw, h)), outline=GLASS if not p.get("opaque") else (110, 140, 170, 255), width=3 * SS)
    tw = hw(h)
    if k == "bag":  # a flat bag: zip strip, no round rim
        d.line(_sc([(x - tw, y - h + 26), (x + tw, y - h + 26)]), fill=(120, 170, 220, 255), width=5 * SS)
        d.line(_sc([(x - tw, y - h + 36), (x + tw, y - h + 36)]), fill=(150, 190, 230, 255), width=2 * SS)
    else:
        d.ellipse(_sc([(x - tw, y - h - max(4, tw * 0.09)), (x + tw, y - h + max(4, tw * 0.09))]), outline=GLASS, width=3 * SS)
    if not p.get("opaque"):
        hx = x - hw(h * 0.8) * 0.6
        d.line(_sc([(hx, y - h * 0.9), (x - hw(h * 0.2) * 0.6, y - h * 0.12)]), fill=(255, 255, 255, 150), width=5 * SS)
    if p.get("foam", 0) > 0.01:
        foam(d, p, t, x, y, h, hw)


def foam(d, p, t, x, y, h, hw):
    f = p["foam"]
    col = color(p.get("foamc", "foam"))
    rng = np.random.default_rng(5)
    top = y - h
    rise = f * 110
    for j in range(40):
        a = rng.uniform(-1, 1)
        bx = x + a * hw(h) * (1 + f * 0.6)
        by = top - rng.uniform(0, rise) + 10
        r = rng.uniform(10, 22) * (0.4 + 0.6 * f)
        d.ellipse(_sc([(bx - r, by - r), (bx + r, by + r)]), fill=mix(col, (255, 255, 255), rng.uniform(0, 0.3)) + (245,))
    spill = max(0.0, f - 0.4) / 0.6  # foam drips running down the outside, then pooling at the base
    if spill > 0:
        for side, lens_ in ((-1, (1.0, 0.55)), (1, (0.75, 0.4))):
            for k, ln in enumerate(lens_):
                z_end = h - spill * ln * h
                zs = np.linspace(h - 4, z_end, 10)
                off = 10 + 16 * k
                pts = [(x + side * (hw(z) + off * (1 - (h - z) / h * 0.3) + 3 * math.sin(z / 18 + t * 2)), y - z) for z in zs]
                wdt = int((22 - 6 * k) * SS * (0.6 + 0.4 * spill))
                d.line(_sc(pts), fill=col + (245,), width=wdt, joint="curve")
                for q, (bx, by) in enumerate(pts[::2]):  # bubbly texture along the drip
                    rr = wdt / SS * (0.45 + 0.2 * ((q * 7 + k) % 3) / 2)
                    d.ellipse(_sc([(bx - rr, by - rr), (bx + rr, by + rr)]), fill=mix(col, (255, 255, 255), 0.25) + (250,),
                              outline=mix(col, (0, 0, 0), 0.12) + (255,))
                ex, ey = pts[-1]
                r = (13 - 3 * k) * (0.6 + 0.4 * spill)
                d.ellipse(_sc([(ex - r, ey - r), (ex + r, ey + r * 1.3)]), fill=col + (245,))
        if spill > 0.6:
            pw = hw(0) + 60 * (spill - 0.6) / 0.4 + 20
            d.ellipse(_sc([(x - pw, y - 10), (x + pw, y + 12)]), fill=col + (230,))


def bowl_surface(d, p, t, x, sy, shw, sh):
    """Things floating on a bowl's surface: pepper specks, soap touches, a cork with a needle."""
    if p.get("specks", 0) > 0:
        rng = np.random.default_rng(9)
        sc = p.get("scatter", 0)
        for j in range(int(160 * p["specks"])):
            a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
            r = r + (0.97 - r) * ease(sc)
            px, py = x + math.cos(a) * r * (shw - 6), sy + math.sin(a) * r * (sh - 2)
            d.ellipse(_sc([(px - 2, py - 2), (px + 2, py + 2)]), fill=(40, 35, 30, 255))
    if p.get("tissue", 0) > 0:
        sink = p.get("sink", 0)
        ty = sy + sink * 50
        a = int(255 * (1 - 0.7 * sink))
        d.polygon(_sc([(x - 70, ty - 10), (x + 60, ty - 14), (x + 72, ty + 8), (x - 62, ty + 12)]),
                  fill=(250, 248, 255, a), outline=(200, 200, 215, a), width=SS)
    if p.get("clip", 0) > 0:
        cy = sy - 2
        d.rounded_rectangle(_sc([(x - 34, cy - 8), (x + 34, cy + 8)]), radius=8 * SS, outline=(110, 115, 125, 255), width=3 * SS)
        d.rounded_rectangle(_sc([(x - 24, cy - 4), (x + 22, cy + 4)]), radius=4 * SS, outline=(110, 115, 125, 255), width=2 * SS)
        if p.get("dimple", 0) > 0:
            d.arc(_sc([(x - 50, cy - 6), (x + 50, cy + 16)]), 10, 170, fill=(255, 255, 255, 200), width=2 * SS)
    if p.get("cork", 0) > 0:
        ang = math.radians(p.get("needle", 0))
        dx, dy = math.cos(ang) * 120, math.sin(ang) * 22
        d.line(_sc([(x - dx, sy - 10 - dy), (x + dx, sy - 10 + dy)]), fill=(120, 125, 135, 255), width=4 * SS)
        d.ellipse(_sc([(x + dx - 6, sy - 10 + dy - 6), (x + dx + 6, sy - 10 + dy + 6)]), fill=(200, 60, 60, 255))
        d.ellipse(_sc([(x - 26, sy - 22), (x + 26, sy + 4)]), fill=(196, 150, 100, 255), outline=(150, 110, 70, 255), width=2 * SS)
    if p.get("goo", 0) > 0:
        rng = np.random.default_rng(2)
        for j in range(18):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.85)
            px, py = x + math.cos(a) * r * shw, sy + math.sin(a) * r * sh
            d.arc(_sc([(px - 20, py - 4), (px + 20, py + 4)]), 0, 180, fill=(225, 222, 210, 255), width=SS)
        if p.get("crack", 0) > 0.05:
            c = p["crack"]
            for k in range(6):
                a = k * math.pi / 3 + 0.3
                d.line(_sc([(x, sy), (x + math.cos(a) * 60 * c, sy + math.sin(a) * 14 * c)]), fill=(170, 165, 150, 255), width=2 * SS)


def plate_contents(d, p, t, w, h):
    x, y = p["x"], p["y"]
    cy = y - h * 1.3
    rx, ry = w * 0.4, h * 0.95
    if p.get("milk", 0) > 0:
        a = int(255 * p["milk"])
        d.ellipse(_sc([(x - rx, cy - ry), (x + rx, cy + ry)]), fill=(252, 250, 244, a))
    if p.get("dots", 0) > 0:
        burst = p.get("burst", 0)
        cols = [PALETTE["red"], PALETTE["blue"], PALETTE["yellow"]]
        for k, c in enumerate(cols):
            a0 = k * 2 * math.pi / 3 + 0.4
            if burst <= 0.01:
                px, py = x + math.cos(a0) * 34, cy + math.sin(a0) * 10
                r = 16 * p["dots"]
                d.ellipse(_sc([(px - r, py - r * 0.5), (px + r, py + r * 0.5)]), fill=c + (240,))
            else:
                for j in range(5):  # swirling arms pushed outward
                    a = a0 + j * 0.25 + burst * 1.6 + 0.3 * math.sin(t * 1.5 + j)
                    pts = []
                    for q in np.linspace(0.05, 0.25 + 0.7 * burst, 14):
                        aa = a + q * 2.2
                        pts.append((x + math.cos(aa) * q * rx, cy + math.sin(aa) * q * ry))
                    d.line(_sc(pts), fill=c + (190,), width=int((14 - j * 2) * SS), joint="curve")
    if p.get("candies", 0) > 0:
        cols = ["red", "orange", "yellow", "green", "blue", "purple"] * 3
        n = 16
        if p.get("wet", 0) > 0:  # water first: PIL replaces pixels, so later strokes must sit on top
            wr = 0.85 * p["wet"]
            d.ellipse(_sc([(x - rx * wr, cy - ry * wr), (x + rx * wr, cy + ry * wr)]), fill=(214, 232, 246, 255))
        dif = p.get("diffuse", 0)
        if dif > 0:
            for k in range(n):
                a = k * 2 * math.pi / n
                c = color(cols[k % 6])
                for j, wdt in enumerate((34, 22)):
                    reach = 0.92 - 0.8 * dif * (1 if j else 0.85)
                    px, py = x + math.cos(a) * rx * 0.92, cy + math.sin(a) * ry * 0.92
                    ex, ey = x + math.cos(a) * rx * reach, cy + math.sin(a) * ry * reach
                    d.line(_sc([(px, py), (ex, ey)]), fill=mix(c, (255, 255, 255), 0.45 - 0.25 * j) + (255,), width=int(wdt * SS))
        for k in range(n):
            if k >= int(n * p["candies"] + 0.5):
                break
            a = k * 2 * math.pi / n
            px, py = x + math.cos(a) * rx * 0.92, cy + math.sin(a) * ry * 0.92
            c = color(cols[k % 6])
            d.ellipse(_sc([(px - 14, py - 9), (px + 14, py + 9)]), fill=c + (255,), outline=mix(c, (0, 0, 0), 0.3) + (255,))


# ------------------------------------------------------------------ tools that act on containers

def pourer(d, p, t):
    """A jug / carton / bottle / jar that tilts and pours; stream drawn while `flow` > 0."""
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    col = color(p.get("color", "water"))
    kind = p.get("kind", "jug")
    x, y = p["x"], p["y"]
    cx, cy = x, y - 85 * sc
    tilt = p.get("rot", 0)
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], cx, cy, tilt))  # noqa: E731
    if kind == "jug":
        body = [(-60, -85), (60, -85), (70, 85), (-70, 85)]
        d.polygon(S(body), fill=(236, 242, 250, 110))
        d.polygon(S([(-64, -10), (64, -10), (70, 85), (-70, 85)]), fill=col + (200,))
        d.line(S([(60, -60), (105, -50), (105, 40), (66, 50)]), fill=GLASS, width=int(9 * sc * SS), joint="curve")
        d.polygon(S(body), outline=GLASS, width=3 * SS)
        d.polygon(S([(-60, -85), (-88, -102), (-58, -66)]), fill=(236, 242, 250, 160), outline=GLASS)
        tip = (-88, -102)
    elif kind == "jar":  # honey jar / soap bottle, opaque-ish with label colour
        d.polygon(S([(-50, -70), (50, -70), (50, 85), (-50, 85)]), fill=col + (255,), outline=mix(col, (0, 0, 0), 0.35) + (255,))
        d.polygon(S([(-22, -70), (22, -70), (22, -100), (-22, -100)]), fill=(245, 245, 245, 255), outline=(170, 170, 170, 255))
        d.polygon(S([(-34, -10), (34, -10), (34, 40), (-34, 40)]), fill=(255, 255, 255, 210))
        tip = (-22, -100)
    elif kind == "box":  # powder box (baking soda / cornstarch / salt)
        d.polygon(S([(-55, -90), (55, -90), (55, 85), (-55, 85)]), fill=(250, 250, 250, 255), outline=(150, 150, 160, 255), width=2 * SS)
        d.polygon(S([(-55, -30), (55, -30), (55, 30), (-55, 30)]), fill=col + (255,))
        tip = (-55, -90)
    else:  # bottle (vinegar, oil, fizzy water)
        d.polygon(S([(-40, -40), (40, -40), (40, 85), (-40, 85)]), fill=(236, 242, 250, 150), outline=GLASS, width=2 * SS)
        d.polygon(S([(-38, 0), (38, 0), (38, 83), (-38, 83)]), fill=col + (220,))
        d.polygon(S([(-15, -40), (15, -40), (15, -95), (-15, -95)]), fill=(236, 242, 250, 150), outline=GLASS, width=2 * SS)
        tip = (-15, -95)
    tx, ty = rot([(tip[0] * sc, tip[1] * sc)], cx, cy, tilt)[0]
    p["_tip"] = (tx, ty)
    if p.get("flow", 0) > 0.05 and "_target" in p:
        gx, gy = p["_target"]
        w = 10 * SS if kind != "box" else 0
        if kind == "box":
            rng = np.random.default_rng(int(t * 30))
            for j in range(18):
                f = rng.random()
                px, py = lerp(tx, gx, f) + rng.uniform(-6, 6), lerp(ty, gy, f)
                d.ellipse(_sc([(px - 3, py - 3), (px + 3, py + 3)]), fill=col + (255,))
        else:
            d.line(_sc([(tx, ty), (tx - 6, ty + 25), (gx, gy)]), fill=col + (235,), width=w, joint="curve")


def dropper(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    col = color(p.get("color", "red"))
    x, y = p["x"], p["y"]
    flip = p.get("flip", 0) > 0.5
    k = -1 if flip else 1
    cy = y - 55 * sc if not flip else y
    Y = lambda v: cy + k * v * sc  # noqa: E731
    box = lambda hw, a, b: _sc([(x - hw * sc, min(Y(a), Y(b))), (x + hw * sc, max(Y(a), Y(b)))])  # noqa: E731
    d.rounded_rectangle(box(30, -45, 55), radius=int(12 * sc * SS), fill=col + (255,), outline=mix(col, (0, 0, 0), 0.3) + (255,), width=2 * SS)
    d.rectangle(box(22, -5, 25), fill=(255, 255, 255, 200))
    d.rectangle(box(13, -65, -45), fill=(250, 250, 250, 255), outline=(170, 170, 170, 255), width=SS)
    tip = (x, Y(-88))
    d.polygon(_sc([(x - 9 * sc, Y(-65)), (x + 9 * sc, Y(-65)), tip]), fill=(250, 250, 250, 255), outline=(170, 170, 170, 255))
    p["_tip"] = tip


def shaker(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    cx, cy = x, y - 60 * sc
    tilt = p.get("rot", 0) + 18 * math.sin(t * 25) * p.get("shake", 0)
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], cx, cy, tilt))  # noqa: E731
    dark = p.get("kind", "salt") == "pepper"
    d.polygon(S([(-32, -40), (32, -40), (36, 60), (-36, 60)]), fill=(70, 60, 55, 255) if dark else (240, 242, 248, 255),
              outline=(120, 120, 130, 255), width=2 * SS)
    d.polygon(S([(-32, -40), (32, -40), (26, -62), (-26, -62)]), fill=(190, 190, 200, 255), outline=(120, 120, 130, 255))
    p["_tip"] = rot([(0, -62 * sc)], cx, cy, tilt)[0]


def spoon(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    tilt = p.get("rot", 0)
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], x, y, tilt))  # noqa: E731
    d.line(S([(20, -10), (150, -60)]), fill=(170, 175, 185, 255), width=int(10 * SS * sc))
    d.polygon(S(oval(-5, -8, 35, 18)), fill=(200, 204, 212, 255), outline=(150, 155, 165, 255))
    if p.get("heap", 0) > 0:
        c = color(p.get("color", "white"))
        d.polygon(S(oval(-5, -14, 26, 12)), fill=c + (255,))
    p["_tip"] = (x - 30 * sc, y)


# ------------------------------------------------------------------ objects

def strip(d, p, t):
    if p.get("shown", 0) <= 0:
        return
    pts = p["points"]
    path = _path(pts)
    n_show = max(2, int(len(path) * p["shown"]))
    q = [tuple(v * SS) for v in path[:n_show]]
    width = 15 * SS
    d.line(q, fill=(250, 248, 242, 255), width=width, joint="curve")
    d.line(q, fill=(205, 198, 188, 160), width=2 * SS)
    n_wet = int(len(path) * min(p.get("wick", 0), p["shown"]))
    if n_wet > 1:
        d.line([tuple(v * SS) for v in path[:n_wet]], fill=color(p.get("color", "red")) + (235,), width=width - 4 * SS, joint="curve")


def towel(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    f = p.get("fold", 0)
    w = lerp(160, 22, f) * sc
    dep = lerp(38, 10, f) * sc
    L = lerp(160, 230, f) * sc if p.get("long") else w
    pts = [(x - w / 2, y), (x + w / 2, y), (x + w / 2 - 20 * sc, y - dep), (x - w / 2 - 20 * sc, y - dep)]
    if f > 0.02:
        pts = [(x - L / 2, y), (x + L / 2, y), (x + L / 2 - 14 * sc, y - dep), (x - L / 2 - 14 * sc, y - dep)]
    col = color(p.get("color", "white"))
    d.polygon(_sc(pts), fill=col + (255,), outline=(190, 185, 176, 255), width=2 * SS)
    for i in range(1, 4):
        xx = lerp(pts[0][0], pts[1][0], i / 4)
        d.line(_sc([(xx, y - 2), (xx - 10 * sc, y - dep + 2)]), fill=(225, 220, 210, 255), width=SS)


def egg(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.ellipse(_sc([(x - 30 * sc, y - 80 * sc), (x + 30 * sc, y)]), fill=(246, 232, 210, 255), outline=(190, 160, 120, 255), width=2 * SS)
    d.ellipse(_sc([(x - 14 * sc, y - 66 * sc), (x - 4 * sc, y - 46 * sc)]), fill=(255, 250, 240, 220))


def balloon(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1)) * (0.25 + 0.75 * p.get("inflate", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # y = centre
    col = color(p.get("color", "red"))
    if p.get("horiz"):
        d.ellipse(_sc([(x - 95 * sc, y - 62 * sc), (x + 95 * sc, y + 62 * sc)]), fill=col + (255,), outline=mix(col, (0, 0, 0), 0.25) + (255,), width=2 * SS)
        d.ellipse(_sc([(x - 10 * sc, y - 45 * sc), (x + 40 * sc, y - 20 * sc)]), fill=mix(col, (255, 255, 255), 0.5) + (255,))
        d.polygon(_sc([(x - 104 * sc, y - 10 * sc), (x - 104 * sc, y + 10 * sc), (x - 92 * sc, y)]), fill=mix(col, (0, 0, 0), 0.2) + (255,))
    else:
        d.ellipse(_sc([(x - 70 * sc, y - 85 * sc), (x + 70 * sc, y + 85 * sc)]), fill=col + (255,), outline=mix(col, (0, 0, 0), 0.25) + (255,), width=2 * SS)
        d.ellipse(_sc([(x - 40 * sc, y - 60 * sc), (x - 15 * sc, y - 25 * sc)]), fill=mix(col, (255, 255, 255), 0.5) + (255,))
        d.polygon(_sc([(x - 9 * sc, y + 92 * sc), (x + 9 * sc, y + 92 * sc), (x, y + 82 * sc)]), fill=mix(col, (0, 0, 0), 0.2) + (255,))
    if p.get("charges", 0) > 0.05:
        n = int(10 * p["charges"])
        rng = np.random.default_rng(1)
        for j in range(n):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 0.75)
            cx, cy = x + math.cos(a) * 60 * r * sc, y + math.sin(a) * 75 * r * sc
            d.line(_sc([(cx - 9, cy), (cx + 9, cy)]), fill=(255, 255, 255, 255), width=4 * SS)
    if p.get("rocket", 0) > 0:  # air rushing out of the back
        for j in range(4):
            ph = (t * 3 + j / 4) % 1
            ax = x - 110 * sc - ph * 90
            d.line(_sc([(ax, y - 15 + j * 10), (ax - 30, y - 15 + j * 10)]), fill=mix((150, 190, 230), (250, 248, 242), ph) + (255,), width=3 * SS)


def wall(d, p, t):
    a = p.get("show", 1)
    if a < 0.02:
        return
    x0 = p["x"] + (1 - a) * 300  # slides in from the right
    d.rectangle(_sc([(x0, 0), (1280, 720)]), fill=(232, 214, 190, 255))
    for yy in range(0, 720, 60):
        d.line(_sc([(x0, yy), (1280, yy)]), fill=(222, 202, 176, 255), width=SS)
    d.line(_sc([(x0, 0), (x0, 720)]), fill=(200, 180, 150, 255), width=4 * SS)
    if p.get("plus", 0) > 0.05:
        for j in range(int(8 * p["plus"])):
            yy = p.get("py", 300) - 70 + j * 20
            d.line(_sc([(x0 + 14, yy), (x0 + 30, yy)]), fill=(200, 60, 60, 255), width=3 * SS)
            d.line(_sc([(x0 + 22, yy - 8), (x0 + 22, yy + 8)]), fill=(200, 60, 60, 255), width=3 * SS)


def sweater(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    col = color(p.get("color", "blue"))
    dark = mix(col, (0, 0, 0), 0.2) + (255,)
    P_ = lambda pts: _sc([(x + a * sc, y + b * sc) for a, b in pts])  # noqa: E731
    d.polygon(P_([(-150, -40), (-95, -150), (-40, -165), (40, -165), (95, -150), (150, -40), (115, -20), (85, -80),
                  (85, 0), (-85, 0), (-85, -80), (-115, -20)]), fill=col + (255,), outline=dark, width=2 * SS)
    d.chord(P_([(-40, -185), (40, -145)]), 0, 180, fill=dark)
    for yy in range(-140, -5, 14):  # knit rows
        for xx in range(-80, 80, 16):
            d.arc(P_([(xx, yy), (xx + 14, yy + 10)]), 20, 160, fill=mix(col, (255, 255, 255), 0.3) + (255,), width=SS)
    d.rectangle(P_([(-85, -12), (85, 0)]), fill=dark)


def head(d, p, t):
    """A simple friendly face whose hair can stand up toward something on its right."""
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # y = chin
    skin, hair = (226, 180, 145, 255), (45, 32, 25, 255)
    d.rectangle(_sc([(x - 30 * sc, y - 10 * sc), (x + 30 * sc, y + 40 * sc)]), fill=skin)  # neck
    d.ellipse(_sc([(x - 75 * sc, y - 175 * sc), (x + 75 * sc, y)]), fill=skin)
    stand = p.get("stand", 0)
    for j in range(22):  # hair strands from the top of the head, bending toward the balloon as `stand` grows
        a0 = math.pi * (1.0 + j / 21)
        bx, by = x + math.cos(a0) * 72 * sc, y - 95 * sc + math.sin(a0) * 82 * sc
        out = (math.cos(a0) * 18, math.sin(a0) * 18)
        tx = bx + (out[0] * (1 - stand) + stand * (55 + 10 * math.sin(j))) * sc
        ty = by + (out[1] * (1 - stand) - stand * (35 - j)) * sc
        d.line(_sc([(bx, by), (tx, ty)]), fill=hair, width=int(7 * SS * sc))
    d.chord(_sc([(x - 76 * sc, y - 180 * sc), (x + 76 * sc, y - 60 * sc)]), 180, 360, fill=hair)
    for ex in (-28, 28):
        d.ellipse(_sc([(x + (ex - 7) * sc, y - 95 * sc), (x + (ex + 7) * sc, y - 79 * sc)]), fill=(40, 30, 25, 255))
    d.arc(_sc([(x - 28 * sc, y - 70 * sc), (x + 28 * sc, y - 35 * sc)]), 20, 160, fill=(150, 70, 60, 255), width=int(4 * SS * sc))
    for ex in (-48, 48):
        d.ellipse(_sc([(x + (ex - 12) * sc, y - 66 * sc), (x + (ex + 12) * sc, y - 54 * sc)]), fill=(240, 150, 140, 255))


def ice(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.rounded_rectangle(_sc([(x - 40 * sc, y - 34 * sc), (x + 40 * sc, y + 22 * sc)]), radius=int(10 * SS * sc),
                        fill=(225, 242, 252, 240), outline=(150, 190, 220, 255), width=2 * SS)
    d.line(_sc([(x - 25 * sc, y - 22 * sc), (x - 5 * sc, y - 26 * sc)]), fill=(255, 255, 255, 255), width=3 * SS)


def string(d, p, t):
    if p.get("show", 1) < 0.05:
        return
    if "x2" in p:
        (x1, y1), (x2, y2) = (p["x"], p["y"]), (p["x2"], p["y2"])
        sag = p.get("sag", 0)
        pts = [(lerp(x1, x2, u), lerp(y1, y2, u) + sag * 4 * u * (1 - u)) for u in np.linspace(0, 1, 16)]
    else:
        pts = p["points"]
    d.line(_sc(pts), fill=(160, 120, 80, 255), width=3 * SS, joint="curve")


def salt_dots(d, p, t):
    if p.get("show", 1) < 0.05:
        return
    rng = np.random.default_rng(4)
    for j in range(int(40 * p.get("show", 1))):
        x = p["x"] + rng.uniform(-40, 40)
        y = p["y"] + rng.uniform(-10, 6)
        d.rectangle(_sc([(x - 1.5, y - 1.5), (x + 1.5, y + 1.5)]), fill=(255, 255, 255, 255), outline=(170, 170, 180, 255))


def raisins(d, p, t):
    """Raisins in a glass of fizzy water; `dance` 0..1 makes them rise with bubbles and sink."""
    rng = np.random.default_rng(8)
    x, y = p["x"], p["y"]
    top = p.get("top", y - 260)
    for j in range(int(p.get("n", 6) * p.get("show", 1))):
        ph = (t * rng.uniform(0.12, 0.2) + j / 6) % 1
        up = math.sin(ph * math.pi) ** 1.5 * p.get("dance", 0)
        rx = x + rng.uniform(-45, 45)
        ry = y - 22 - up * (y - 22 - top - 20)
        d.ellipse(_sc([(rx - 14, ry - 9), (rx + 14, ry + 9)]), fill=(90, 50, 60, 255))
        d.arc(_sc([(rx - 10, ry - 6), (rx + 8, ry + 4)]), 200, 340, fill=(130, 80, 90, 255), width=SS)
        if up > 0.1:
            for b in range(4):
                a = b * 1.6
                d.ellipse(_sc([(rx + math.cos(a) * 16 - 3, ry + math.sin(a) * 11 - 3), (rx + math.cos(a) * 16 + 3, ry + math.sin(a) * 11 + 3)]),
                          outline=(255, 255, 255, 255), width=SS)


def celery(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # y = base of stalk (inside the glass)
    tint, col = p.get("tint", 0), color(p.get("color", "blue"))
    d.polygon(_sc([(x - 16 * sc, y), (x + 16 * sc, y), (x + 10 * sc, y - 330 * sc), (x - 10 * sc, y - 330 * sc)]),
              fill=color("celery") + (255,), outline=(100, 150, 70, 255), width=2 * SS)
    for k in (-6, 0, 6):
        d.line(_sc([(x + k * sc, y - 4), (x + k * 0.6 * sc, y - 326 * sc)]), fill=(120, 175, 85, 255), width=SS)
    rng = np.random.default_rng(int(x))
    for j in range(7):
        lx = x + rng.uniform(-70, 70) * sc
        ly = y - (330 + rng.uniform(10, 90)) * sc
        leaf = mix(color("green"), col, 0.0)
        d.ellipse(_sc([(lx - 50 * sc, ly - 32 * sc), (lx + 50 * sc, ly + 32 * sc)]), fill=mix(color("green"), col, 0.9 * tint) + (255,))
        d.ellipse(_sc([(lx - (50 - 12 * tint) * sc, ly - (32 - 9 * tint) * sc), (lx + (50 - 12 * tint) * sc, ly + (32 - 9 * tint) * sc)]),
                  fill=leaf + (255,))  # coloured rim grows inward as the dye arrives
        d.line(_sc([(x, y - 330 * sc), (lx, ly)]), fill=(110, 165, 80, 255), width=3 * SS)
    if tint > 0.05:
        for k in (-8, 0, 8):
            d.ellipse(_sc([(x + k * sc - 3, y - 10 - 3), (x + k * sc + 3, y - 10 + 3)]), fill=col + (int(255 * tint),))


def flashlight(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    ang = p.get("rot", 0)
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], x, y, ang))  # noqa: E731
    if p.get("on", 0) > 0.05:
        L = p.get("beam", 400)
        spread = p.get("spread", 60)
        d.polygon(S([(40, -18), (40 + L, -spread), (40 + L, spread), (40, 18)]), fill=(255, 245, 190, int(110 * p["on"])))
    d.polygon(S([(-110, -16), (10, -16), (10, 16), (-110, 16)]), fill=(70, 80, 100, 255))
    d.polygon(S([(10, -16), (40, -26), (40, 26), (10, 16)]), fill=(90, 100, 125, 255))
    d.line(S([(40, -24), (40, 24)]), fill=(255, 250, 200, 255), width=5 * SS)


def sheet(d, p, t):
    """A sheet of white paper standing up (screen) or lying flat with a hidden message."""
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    if p.get("kind") == "screen":
        d.rectangle(_sc([(x - 110 * sc, y - 290 * sc), (x + 110 * sc, y)]), fill=(255, 255, 255, 255), outline=(190, 185, 176, 255), width=2 * SS)
        if p.get("rainbow", 0) > 0:
            cols = ["red", "orange", "yellow", "green", "blue", "purple"]
            for k, c in enumerate(cols):  # colours mixed into the white, never transparent (PIL replaces pixels)
                r = (88 - k * 9) * sc
                d.arc(_sc([(x - r, y - 150 * sc - r), (x + r, y - 150 * sc + r)]), 200, 340,
                      fill=mix((255, 255, 255), color(c), p["rainbow"]) + (255,), width=int(9 * SS * sc))
        return
    w, h = 300 * sc, 200 * sc
    d.polygon(_sc([(x - w / 2, y), (x + w / 2, y), (x + w / 2 - 40 * sc, y - h * 0.5), (x - w / 2 - 40 * sc, y - h * 0.5)]),
              fill=(255, 255, 252, 255), outline=(190, 185, 176, 255), width=2 * SS)
    msg_p = p.get("write", 0)
    if msg_p > 0:
        wet, brown = p.get("wet", 1), p.get("reveal", 0)
        ink = mix((235, 222, 150), (120, 70, 30), brown)
        col = mix((255, 255, 252), ink, max(wet * 0.6, brown))  # mixed into the paper: dry ink is truly invisible
        a = 255
        letters = [[(-90, 0), (-90, -60)], [(-90, -30), (-50, -30)], [(-50, 0), (-50, -60)], [(-15, 0), (-15, -60)],
                   [(30, -60), (30, -15)], [(30, 0), (30, -4)]]
        for L in letters[: max(0, int(len(letters) * msg_p + 0.99))]:
            d.line(_sc([(x + a_ * sc - 18 * sc * (-(b_) / 60), y - 25 * sc + b_ * sc * 0.55) for a_, b_ in L]),
                   fill=col + (a,), width=int(8 * SS * sc))


def lemon(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.chord(_sc([(x - 60 * sc, y - 60 * sc), (x + 60 * sc, y + 60 * sc)]), 180, 360, fill=color("lemon") + (255,), outline=(220, 180, 40, 255), width=3 * SS)
    d.chord(_sc([(x - 50 * sc, y - 50 * sc), (x + 50 * sc, y + 50 * sc)]), 180, 360, fill=(252, 240, 170, 255))
    for k in range(6):
        a = math.pi + k * math.pi / 6 + math.pi / 12
        d.line(_sc([(x, y), (x + math.cos(a) * 46 * sc, y + math.sin(a) * 46 * sc)]), fill=(240, 215, 110, 255), width=2 * SS)


def swab(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # y = tip
    tilt = p.get("rot", 0)
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], x, y, tilt))  # noqa: E731
    d.line(S([(0, 0), (0, -170)]), fill=(240, 240, 245, 255), width=int(6 * SS * sc))
    tipc = color(p.get("color", "white")) if p.get("dipped", 0) > 0.5 else (255, 255, 255)
    d.polygon(S(oval(0, 0, 9, 12)), fill=tipc + (255,), outline=(200, 200, 210, 255))
    d.polygon(S(oval(0, -170, 9, 12)), fill=(255, 255, 255, 255), outline=(200, 200, 210, 255))


def hairdryer(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.rounded_rectangle(_sc([(x - 70 * sc, y - 40 * sc), (x + 60 * sc, y + 10 * sc)]), radius=int(20 * SS * sc), fill=(230, 120, 160, 255))
    d.rounded_rectangle(_sc([(x + 10 * sc, y), (x + 50 * sc, y + 110 * sc)]), radius=int(12 * SS * sc), fill=(210, 100, 140, 255))
    d.ellipse(_sc([(x - 84 * sc, y - 36 * sc), (x - 60 * sc, y + 6 * sc)]), fill=(120, 60, 90, 255))
    if p.get("on", 0) > 0.05:
        for j in range(4):
            ph = (t * 2.5 + j / 4) % 1
            ax = x - 90 * sc - ph * 120
            d.arc(_sc([(ax - 20, y - 40 + j * 10 + ph * 40), (ax + 20, y - 20 + j * 10 + ph * 40)]), 90, 270,
                  fill=(250, 170, 90, int(255 * (1 - ph) * p["on"])), width=3 * SS)


def chair(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    c, k = (170, 120, 80, 255), (-1 if p.get("flip") else 1)
    d.rectangle(_sc([(x - 60 * sc, y - 150 * sc), (x + 60 * sc, y - 136 * sc)]), fill=c)
    for lx in (-55, 50):
        d.rectangle(_sc([(x + lx * sc, y - 140 * sc), (x + (lx + 8) * sc, y)]), fill=c)
    bx = x + k * 55 * sc
    d.rectangle(_sc([(min(bx, bx + k * 10 * sc), y - 300 * sc), (max(bx, bx + k * 10 * sc), y - 140 * sc)]), fill=c)
    d.rectangle(_sc([(min(bx, bx - k * 60 * sc), y - 300 * sc), (max(bx, bx - k * 60 * sc) + 0, y - 286 * sc)]), fill=c)


def straw(d, p, t):
    if p.get("show", 1) < 0.05:
        return
    x, y = p["x"], p["y"]
    d.rounded_rectangle(_sc([(x - 40, y - 8), (x + 40, y + 8)]), radius=6 * SS, fill=(250, 250, 255, 255), outline=(220, 90, 90, 255), width=2 * SS)
    for k in range(-30, 40, 16):
        d.line(_sc([(x + k, y - 7), (x + k + 6, y + 7)]), fill=(225, 80, 80, 255), width=3 * SS)


def pencil(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # y = the point
    S = lambda pts: _sc(rot([(a * sc, b * sc) for a, b in pts], x, y, p.get("rot", 0)))  # noqa: E731
    d.polygon(S([(-9, -30), (9, -30), (9, -230), (-9, -230)]), fill=(250, 200, 50, 255), outline=(200, 150, 30, 255))
    d.polygon(S([(-9, -230), (9, -230), (9, -255), (-9, -255)]), fill=(240, 140, 150, 255))
    d.polygon(S([(-9, -30), (9, -30), (0, 0)]), fill=(240, 215, 170, 255))
    d.polygon(S([(-3, -9), (3, -9), (0, 0)]), fill=(60, 60, 60, 255))


def clip(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.rounded_rectangle(_sc([(x - 34 * sc, y - 8 * sc), (x + 34 * sc, y + 8 * sc)]), radius=int(8 * SS * sc), outline=(110, 115, 125, 255), width=3 * SS)
    d.rounded_rectangle(_sc([(x - 24 * sc, y - 4 * sc), (x + 22 * sc, y + 4 * sc)]), radius=int(4 * SS * sc), outline=(110, 115, 125, 255), width=2 * SS)


def magnet(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.rectangle(_sc([(x - 60 * sc, y - 20 * sc), (x, y + 20 * sc)]), fill=(214, 50, 50, 255))
    d.rectangle(_sc([(x, y - 20 * sc), (x + 60 * sc, y + 20 * sc)]), fill=(60, 90, 200, 255))


def needle(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.line(_sc([(x - 120 * sc, y), (x + 120 * sc, y)]), fill=(140, 145, 155, 255), width=4 * SS)
    d.ellipse(_sc([(x + 114 * sc, y - 4), (x + 126 * sc, y + 4)]), fill=(200, 60, 60, 255))


def bean(d, p, t):
    """A bean against a damp towel, with a root growing down and a shoot growing up."""
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    root, shoot = p.get("root", 0), p.get("shoot", 0)
    if root > 0.01:
        pts = [(x + 6 * math.sin(k / 3), y + 14 + k / 10 * 120 * root) for k in range(11)]
        d.line(_sc(pts), fill=(170, 130, 80, 255), width=10 * SS, joint="curve")
        d.line(_sc(pts), fill=(236, 214, 170, 255), width=6 * SS, joint="curve")
        for k in range(3, 10, 3):
            if k / 10 < root:
                bx, by = pts[k]
                d.line(_sc([(bx, by), (bx + 16, by + 12)]), fill=(170, 130, 80, 255), width=3 * SS)
    if shoot > 0.01:
        pts = [(x + 4 * math.sin(k / 2), y - 16 - k / 10 * 110 * shoot) for k in range(11)]
        d.line(_sc(pts), fill=(110, 180, 80, 255), width=6 * SS, joint="curve")
        if shoot > 0.6:
            tx, ty = pts[-1]
            for s_ in (-1, 1):
                d.ellipse(_sc([(tx + s_ * 22 - 18, ty - 10), (tx + s_ * 22 + 18, ty + 8)]), fill=(120, 195, 90, 255))
    d.ellipse(_sc([(x - 26 * sc, y - 17 * sc), (x + 26 * sc, y + 17 * sc)]), fill=(150, 60, 50, 255), outline=(110, 40, 35, 255), width=2 * SS)
    d.arc(_sc([(x - 8 * sc, y - 6 * sc), (x + 8 * sc, y + 10 * sc)]), 180, 360, fill=(240, 220, 210, 255), width=2 * SS)


def tray(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y, w = p["x"], p["y"], p.get("w", 700) * sc
    d.rounded_rectangle(_sc([(x - w / 2, y - 22), (x + w / 2, y + 18)]), radius=16 * SS, fill=(176, 206, 196, 255), outline=(120, 160, 148, 255), width=3 * SS)
    d.rounded_rectangle(_sc([(x - w / 2 + 12, y - 14), (x + w / 2 - 12, y + 8)]), radius=10 * SS, fill=(196, 222, 212, 255))


def tablet(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.ellipse(_sc([(x - 22 * sc, y - 8 * sc), (x + 22 * sc, y + 8 * sc)]), fill=(250, 250, 250, 255), outline=(180, 180, 190, 255), width=2 * SS)


def finger(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]  # fingertip
    d.rounded_rectangle(_sc([(x - 22 * sc, y - 220 * sc), (x + 22 * sc, y)]), radius=int(22 * SS * sc), fill=(226, 180, 145, 255),
                        outline=(190, 140, 110, 255), width=2 * SS)
    d.rounded_rectangle(_sc([(x - 12 * sc, y - 40 * sc), (x + 12 * sc, y - 8 * sc)]), radius=int(8 * SS * sc), fill=(244, 214, 200, 255))


def drip(d, p, t):
    """Goo dripping in a thick stream from a scoop above a bowl."""
    if p.get("show", 0) < 0.05:
        return
    x, y, L = p["x"], p["y"], p.get("len", 0) * 120
    d.ellipse(_sc([(x - 40, y - 24), (x + 40, y + 10)]), fill=color("goo") + (255,), outline=(200, 196, 182, 255), width=2 * SS)
    if L > 2:
        wob = 4 * math.sin(t * 3)
        d.line(_sc([(x, y + 6), (x + wob, y + L * 0.5), (x, y + L)]), fill=color("goo") + (255,), width=12 * SS, joint="curve")


def clock(d, p, t):
    from lib.kids_science.draw import clock as _clock
    if p.get("show", 1) > 0.05:
        _clock(d, p["x"], p["y"], 62 * pop(p.get("show", 1)), p.get("minutes", 0))


def eye(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    d.ellipse(_sc([(x - 46 * sc, y - 24 * sc), (x + 46 * sc, y + 24 * sc)]), fill=(255, 255, 255, 255), outline=INK + (255,), width=3 * SS)
    d.ellipse(_sc([(x - 16 * sc, y - 16 * sc), (x + 16 * sc, y + 16 * sc)]), fill=(80, 120, 170, 255))
    d.ellipse(_sc([(x - 7 * sc, y - 7 * sc), (x + 7 * sc, y + 7 * sc)]), fill=(20, 20, 30, 255))


def glow(d, p, t):
    """A soft coloured glow (e.g. the blue side-glow or orange beam seen through milky water)."""
    a = p.get("show", 0)
    if a < 0.02:
        return
    x, y, r = p["x"], p["y"], p.get("r", 120)
    col = color(p.get("color", "blue"))
    for k in range(6):
        rr = r * (1 - k / 7)
        d.ellipse(_sc([(x - rr, y - rr), (x + rr, y + rr)]), fill=col + (int((18 + 16 * k) * a),))


def dark(d, p, t):
    """Dims the room (drawn first, behind everything)."""
    a = p.get("amount", 0)
    if a > 0.01:
        d.rectangle(_sc([(0, 0), (1280, 720)]), fill=(25, 28, 48, int(215 * a)))


def beam(d, p, t):
    """A cone of light from (x, y) to (x2, y2), drawn behind things so glass and water sit in it."""
    on = p.get("on", 0) * p.get("show", 1)
    if on < 0.02:
        return
    x1, y1, x2, y2 = p["x"], p["y"], p["x2"], p["y2"]
    s0, s1 = p.get("spread0", 18), p.get("spread", 70)
    ang = math.atan2(y2 - y1, x2 - x1)
    nx, ny = -math.sin(ang), math.cos(ang)
    col = color(p.get("color", [255, 246, 200]))
    d.polygon(_sc([(x1 + nx * s0, y1 + ny * s0), (x2 + nx * s1, y2 + ny * s1), (x2 - nx * s1, y2 - ny * s1), (x1 - nx * s0, y1 - ny * s0)]),
              fill=col + (int(170 * on),))


def cork(d, p, t):
    sc = p.get("s", 1) * pop(p.get("show", 1))
    if sc < 0.02:
        return
    x, y = p["x"], p["y"]
    if p.get("needle", None) is not None and p.get("pierced", 0) > 0.5:
        ang = math.radians(p["needle"])
        dx, dy = math.cos(ang) * 120 * sc, math.sin(ang) * 22 * sc
        d.line(_sc([(x - dx, y - 10 - dy), (x + dx, y - 10 + dy)]), fill=(120, 125, 135, 255), width=4 * SS)
        d.ellipse(_sc([(x + dx - 6, y - 10 + dy - 6), (x + dx + 6, y - 10 + dy + 6)]), fill=(200, 60, 60, 255))
    d.ellipse(_sc([(x - 26 * sc, y - 22 * sc), (x + 26 * sc, y + 4 * sc)]), fill=(196, 150, 100, 255), outline=(150, 110, 70, 255), width=2 * SS)


def label(d, p, t):
    pass  # drawn by the stage (text needs the font)


DRAW = {
    "container": (container_back, container_liquid, container_front),
    "pourer": (None, None, pourer), "dropper": (None, None, dropper), "shaker": (None, None, shaker),
    "spoon": (None, None, spoon), "strip": (None, None, strip), "towel": (None, None, towel), "egg": (None, None, egg),
    "balloon": (None, None, balloon), "wall": (wall, None, None), "sweater": (None, None, sweater), "head": (None, None, head),
    "ice": (None, None, ice), "string": (None, None, string), "salt": (None, None, salt_dots), "raisins": (None, None, raisins),
    "celery": (None, None, celery), "flashlight": (None, None, flashlight), "sheet": (None, None, sheet),
    "lemon": (None, None, lemon), "swab": (None, None, swab), "hairdryer": (None, None, hairdryer), "chair": (None, None, chair),
    "straw": (None, None, straw), "magnet": (None, None, magnet), "pencil": (None, None, pencil), "clip": (None, None, clip), "needle": (None, None, needle), "bean": (None, None, bean),
    "tray": (tray, None, None), "tablet": (None, None, tablet), "finger": (None, None, finger), "drip": (None, None, drip),
    "clock": (None, None, clock), "eye": (None, None, eye), "glow": (glow, None, None),
    "tag": (None, None, None), "dark": (dark, None, None), "beam": (beam, None, None), "cork": (None, None, cork),  # text label with a pointer; drawn by the stage (needs the font)
}
# Parts that sit *inside* a container are drawn between its liquid and its front glass.
INSIDE = {"egg", "raisins", "celery", "ice", "bean", "tablet", "strip"}


# ------------------------------------------------------------------ close-up lens for explanations

def lens(img: Image.Image, cx, cy, r, level, mode="fibers", t=0.0, col=(230, 95, 105), anchor=None) -> Image.Image:
    """A magnifying glass that shows what is happening at a tiny scale; `level` 0..1 animates it."""
    w, h = img.size
    lay = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if anchor:
        d.line(_sc([anchor, (cx - r * 0.7, cy + r * 0.7)]), fill=(150, 130, 110, 160), width=3 * SS)
        d.ellipse(_sc([(anchor[0] - 10, anchor[1] - 10), (anchor[0] + 10, anchor[1] + 10)]), outline=(150, 130, 110, 220), width=3 * SS)
    inner = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    di.ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=(253, 251, 245, 255))
    rng = np.random.default_rng(5)
    if mode == "fibers":
        fibres = [cx - r + 18 + i * 30 + rng.uniform(-6, 6) for i in range(int(2 * r / 30))]
        surf = cy + r - level * (1.55 * r)
        xs = np.linspace(cx - r, cx + r, 160)
        ys = [surf - 38 * max(math.exp(-((x - f) / 9) ** 2) for f in fibres) for x in xs]
        di.polygon(_sc([(cx - r, cy + r + 5)] + list(zip(xs, ys)) + [(cx + r, cy + r + 5)]), fill=col + (200,))
        for i, f in enumerate(fibres):
            pts = [(f + 6 * math.sin(y / 23 + i), y) for y in np.linspace(cy - r, cy + r, 30)]
            di.line(_sc(pts), fill=(205, 196, 180, 255), width=9 * SS, joint="curve")
            di.line(_sc(pts), fill=(248, 244, 234, 255), width=5 * SS, joint="curve")
    elif mode == "tubes":
        for i in range(-2, 3):
            x = cx + i * r * 0.32
            di.rectangle(_sc([(x - 18, cy - r), (x + 18, cy + r)]), fill=(205, 235, 190, 255), outline=(110, 170, 90, 255), width=2 * SS)
            top = cy + r - level * 2 * r
            di.rectangle(_sc([(x - 14, top), (x + 14, cy + r)]), fill=col + (220,))
            for k in range(3):
                yy = top + 20 + ((t * 60 + k * 50 + i * 17) % max(20, (cy + r - top)))
                if yy < cy + r:
                    di.polygon(_sc([(x - 7, yy + 8), (x + 7, yy + 8), (x, yy - 4)]), fill=(255, 255, 255, 220))
    elif mode == "skin":  # surface tension: water particles holding hands at the surface
        sy = cy - r * 0.15
        di.rectangle(_sc([(cx - r, sy), (cx + r, cy + r)]), fill=(185, 215, 240, 255))
        xs = list(np.arange(cx - r, cx + r, 34))
        for k, x in enumerate(xs):
            dip = 0
            if mode == "skin" and level > 0:
                dip = 26 * level * math.exp(-((x - cx) / (r * 0.35)) ** 2)
            di.ellipse(_sc([(x - 13, sy + dip - 13), (x + 13, sy + dip + 13)]), fill=(90, 150, 220, 255))
            if k + 1 < len(xs):
                x2 = xs[k + 1]
                dip2 = 26 * level * math.exp(-((x2 - cx) / (r * 0.35)) ** 2) if level > 0 else 0
                di.line(_sc([(x + 13, sy + dip), (x2 - 13, sy + dip2)]), fill=(60, 110, 190, 255), width=4 * SS)
        for k in range(10):
            x, y = cx + rng.uniform(-r, r) * 0.8, sy + 50 + rng.uniform(0, r * 0.7)
            di.ellipse(_sc([(x - 11, y - 11), (x + 11, y + 11)]), fill=(120, 170, 230, 255))
        if level > 0:
            di.rounded_rectangle(_sc([(cx - 70, sy - 30), (cx + 70, sy + 26 * level - 8)]), radius=12 * SS, fill=(130, 135, 145, 255))
    elif mode == "particles":  # dense vs. less dense: tightly packed vs. spread out
        for half, n in ((-1, 30), (1, 9)):
            for k in range(n):
                x = cx + half * r * 0.45 + rng.uniform(-r * 0.32, r * 0.32)
                y = cy + rng.uniform(-r * 0.6, r * 0.6)
                di.ellipse(_sc([(x - 10, y - 10), (x + 10, y + 10)]), fill=(col if half < 0 else (150, 200, 240)) + (255,))
        di.line(_sc([(cx, cy - r), (cx, cy + r)]), fill=(200, 190, 180, 255), width=3 * SS)
    elif mode == "bubbles":
        for k in range(16):
            x = cx + rng.uniform(-0.8, 0.8) * r
            y = cy + r - ((t * 0.3 + rng.random()) % 1) * 2 * r
            rr = rng.uniform(8, 20)
            di.ellipse(_sc([(x - rr, y - rr), (x + rr, y + rr)]), outline=(120, 170, 220, 255), width=3 * SS)
            di.text((x * SS - 6 * SS, y * SS - 7 * SS), "", fill=(0, 0, 0))
    elif mode == "charges":
        di.rectangle(_sc([(cx + r * 0.25, cy - r), (cx + r, cy + r)]), fill=(232, 214, 190, 255))
        di.ellipse(_sc([(cx - r * 0.9, cy - r * 0.6), (cx + r * 0.2, cy + r * 0.6)]), fill=(214, 40, 52, 255))
        for k in range(8):
            x, y = cx - r * 0.35 + rng.uniform(-0.3, 0.3) * r, cy + rng.uniform(-0.4, 0.4) * r
            di.line(_sc([(x - 10, y), (x + 10, y)]), fill=(255, 255, 255, 255), width=5 * SS)
        for k in range(6):
            y = cy - r * 0.5 + k * r * 0.2
            x = cx + r * 0.32
            di.line(_sc([(x - 9, y), (x + 9, y)]), fill=(200, 60, 60, 255), width=4 * SS)
            di.line(_sc([(x, y - 9), (x, y + 9)]), fill=(200, 60, 60, 255), width=4 * SS)
        for k in range(3):
            y = cy - r * 0.3 + k * r * 0.3
            di.line(_sc([(cx - 5, y), (cx + r * 0.2 * level, y)]), fill=(120, 110, 100, 255), width=3 * SS)
    elif mode == "light":
        di.rectangle(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=(40, 45, 60, 255))
        di.polygon(_sc([(cx - r * 0.3, cy - r * 0.5), (cx + r * 0.3, cy - r * 0.5), (cx + r * 0.3, cy + r * 0.5), (cx - r * 0.3, cy + r * 0.5)]),
                   fill=(170, 205, 235, 120))
        di.line(_sc([(cx - r, cy), (cx - r * 0.3, cy)]), fill=(255, 255, 240, 255), width=8 * SS)
        cols = ["red", "orange", "yellow", "green", "blue", "purple"]
        for k, c in enumerate(cols):
            dy = (k - 2.5) * 10 * level
            di.line(_sc([(cx - r * 0.3, cy), (cx + r * 0.3, cy + dy * 0.6), (cx + r, cy + dy * 2.4)]), fill=color(c) + (255,), width=4 * SS)
    elif mode == "grains":
        di.ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=(205, 228, 246, 255))
        jam = level
        for k in range(36):
            gx = cx + ((k % 6) - 2.5) * (44 - 14 * jam) + rng.uniform(-8, 8) * (1 - jam)
            gy = cy + ((k // 6) - 2.5) * (44 - 14 * jam) + rng.uniform(-8, 8) * (1 - jam)
            di.ellipse(_sc([(gx - 15, gy - 15), (gx + 15, gy + 15)]), fill=(250, 248, 240, 255), outline=(200, 195, 180, 255), width=2 * SS)
    elif mode == "dissolve":
        for k in range(40):
            spread = 0.15 + 0.8 * level
            x = cx + rng.normal(0, 1) * r * spread * 0.45
            y = cy + rng.normal(0, 1) * r * spread * 0.45
            di.ellipse(_sc([(x - 6, y - 6), (x + 6, y + 6)]), fill=col + (230,))
    elif mode == "seed":
        di.ellipse(_sc([(cx - r * 0.7, cy - r * 0.45), (cx + r * 0.7, cy + r * 0.45)]), fill=(200, 120, 90, 255), outline=(150, 70, 50, 255), width=4 * SS)
        di.ellipse(_sc([(cx - r * 0.6, cy - r * 0.35), (cx + r * 0.1, cy + r * 0.35)]), fill=(245, 225, 180, 255))
        di.ellipse(_sc([(cx + r * 0.1, cy - r * 0.15), (cx + r * 0.45, cy + r * 0.15)]), fill=(160, 200, 110, 255))
    elif mode == "ice":  # salt melts the top a little, then it freezes again around the string
        top = cy - r * 0.15
        di.rectangle(_sc([(cx - r, top), (cx + r, cy + r)]), fill=(222, 240, 252, 255))
        for k in range(-4, 5):
            di.line(_sc([(cx + k * 40, top), (cx + k * 40 + 30, cy + r)]), fill=(200, 225, 245, 255), width=2 * SS)
        melt = min(1.0, level * 2)
        refreeze = max(0.0, level * 2 - 1)
        if melt > 0:
            di.rectangle(_sc([(cx - r, top - 6), (cx + r, top + 26 * melt)]), fill=mix((222, 240, 252), (150, 200, 240), melt * (1 - refreeze)) + (255,))
        di.line(_sc([(cx - r, top - 4 + 10 * melt), (cx + r, top - 4 + 10 * melt)]), fill=(160, 120, 80, 255), width=6 * SS)
        for k in range(14):
            x = cx - r * 0.8 + k * r * 0.12
            di.rectangle(_sc([(x - 4, top - 14), (x + 4, top - 6)]), fill=(255, 255, 255, 255), outline=(170, 170, 180, 255))
        if refreeze > 0:
            for k in range(-5, 6):
                x = cx + k * 28
                di.line(_sc([(x, top + 24), (x + 10, top - 10 * refreeze)]), fill=(255, 255, 255, 255), width=3 * SS)
    elif mode == "push":
        bx = cx + r * 0.1 + level * r * 0.25
        di.ellipse(_sc([(bx - 70, cy - 50), (bx + 70, cy + 50)]), fill=(214, 40, 52, 255))
        for k in range(5):
            yy = cy - 30 + k * 15
            di.line(_sc([(bx - 80, yy), (bx - 80 - 60 - 40 * level, yy)]), fill=(120, 170, 230, 255), width=4 * SS)
        di.line(_sc([(bx + 80, cy), (bx + 150, cy)]), fill=(60, 140, 60, 255), width=8 * SS)
        di.polygon(_sc([(bx + 150, cy - 16), (bx + 150, cy + 16), (bx + 180, cy)]), fill=(60, 140, 60, 255))
    elif mode == "magnet":
        di.ellipse(_sc([(cx - r * 0.55, cy - r * 0.55), (cx + r * 0.55, cy + r * 0.55)]), fill=(90, 150, 210, 255))
        di.text(_sc([(cx - 10, cy - r * 0.85)])[0], "N", fill=(200, 60, 60, 255))
        for k in range(5):
            a = k * 0.6 - 1.2
            di.arc(_sc([(cx - r * (0.6 + 0.07 * k), cy - r * 0.9), (cx + r * (0.6 + 0.07 * k), cy + r * 0.9)]), 270 - 60, 270 + 60,
                   fill=(200, 60, 60, int(255 * level)), width=2 * SS)
    mask = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(mask).ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=255)
    lay.paste(inner, (0, 0), mask)
    d = ImageDraw.Draw(lay)
    d.ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), outline=(95, 75, 60, 255), width=14 * SS)
    hx, hy = cx + r * 0.72, cy + r * 0.72
    d.line(_sc([(hx, hy), (hx + 90, hy + 90)]), fill=(95, 75, 60, 255), width=26 * SS)
    d.arc(_sc([(cx - r + 30, cy - r + 30), (cx + r - 30, cy + r - 30)]), 200, 250, fill=(255, 255, 255, 200), width=6 * SS)
    lay = lay.filter(ImageFilter.GaussianBlur(0.7 * SS)).resize((w, h), Image.LANCZOS)
    out = img.convert("RGBA")
    out.alpha_composite(lay)
    return out.convert("RGB")
