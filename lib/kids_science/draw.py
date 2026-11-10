"""Code-drawn experiment parts: exact, animatable, never hallucinated.

Image models cannot keep a physics experiment straight (cups became mugs,
strips appeared from nowhere). Here the experiment is a small scene of parts
whose state is numbers — a cup's liquid level and colour, a paper strip's
wicked length — so every frame is computed from the experiment's facts and a
picture can never disagree with the narration.

Rendering is supersampled (2x) PIL with soft "painted" edges and a paper
texture, to sit next to the series' watercolour characters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 2  # supersampling factor


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(x, y, t)) for x, y in zip(c1, c2))


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


WATER = (170, 205, 235)
RED = (214, 40, 52)
YELLOW = (246, 196, 30)
ORANGE = (240, 120, 30)


@dataclass
class Cup:
    cx: float           # centre x (px, final resolution)
    top: float          # rim y
    bottom: float       # base y
    w_top: float
    w_bottom: float
    level: float = 0.0  # 0..1 of the cup's height
    color: tuple = WATER
    drops: list = field(default_factory=list)  # [(x, y, color)] falling drops to draw
    ripples: list = field(default_factory=list)  # ripple phases 0..1 on the surface
    cloud: tuple | None = None  # (colour, 0..1) dye spreading through the water

    def half_width(self, y: float) -> float:
        t = (y - self.top) / (self.bottom - self.top)
        return lerp(self.w_top, self.w_bottom, t) / 2

    def outline(self, s=SS):
        ht, hb = self.w_top / 2, self.w_bottom / 2
        return [((self.cx - ht) * s, self.top * s), ((self.cx + ht) * s, self.top * s),
                ((self.cx + hb) * s, self.bottom * s), ((self.cx - hb) * s, self.bottom * s)]


@dataclass
class Strip:
    """A paper towel strip along a path; `wick` (0..1) is how far colour has climbed from its start."""
    points: list          # path points (final px), from the coloured cup's bottom to the target cup
    shown: float = 1.0    # 0..1 of the path drawn (strip being placed)
    wick: float = 0.0
    color: tuple = RED


def _path(points, n=120):
    """Resample a polyline through Catmull-Rom-ish smoothing into n points."""
    pts = np.array(points, float)
    out = []
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, len(pts) - 1)]
        for t in np.linspace(0, 1, n // (len(pts) - 1), endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.array(out)


def background(w: int, h: int, top=(232, 226, 245), bottom=(250, 246, 240), table=(244, 238, 228), table_y=0.80):
    """Soft painted studio backdrop with a tabletop and paper grain."""
    rng = np.random.default_rng(3)
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.array(top, float) * (1 - y) + np.array(bottom, float) * y
    img = np.repeat(img, w, axis=1)
    ty = int(h * table_y)
    img[ty:] = np.array(table, float)
    img[ty:ty + 3] *= 0.93
    blots = Image.new("L", (w // 8, h // 8))
    blots.putdata(list((rng.random(w // 8 * (h // 8)) * 255).astype(int)))
    blots = np.asarray(blots.resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(18)), float)[..., None] / 255
    img = img * (0.96 + 0.06 * blots)
    grain = rng.normal(0, 3.0, (h, w, 1))
    return Image.fromarray(np.clip(img + grain, 0, 255).astype(np.uint8))


_TEX: dict = {}


def _texture(w: int, h: int) -> np.ndarray:
    """Watercolour pigment variation (HxWx1, ~0.88..1.08), fixed per size so it doesn't flicker."""
    if (w, h) not in _TEX:
        rng = np.random.default_rng(7)
        a = np.zeros((h, w))
        for scale, amp in ((6, 0.6), (20, 0.3), (60, 0.1)):
            small = Image.fromarray((rng.random((max(1, h // scale), max(1, w // scale))) * 255).astype(np.uint8))
            a += amp * np.asarray(small.resize((w, h), Image.BICUBIC), float) / 255
        _TEX[(w, h)] = (0.88 + 0.2 * (a - a.min()) / (np.ptp(a) + 1e-6))[..., None]
    return _TEX[(w, h)]


def rot(points, cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in points]


def _sc(pts):
    return [(x * SS, y * SS) for x, y in pts]


def render(bg: Image.Image, cups: list, strips: list, front=(), back=()) -> Image.Image:
    """Compose one frame. `front`/`back` are callables f(draw) drawing extra parts
    (in final-pixel coordinates via `_sc`) in front of / behind the cups."""
    w, h = bg.size
    W, H = w * SS, h * SS
    back_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    liq_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    front_l = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    db, dl, d = ImageDraw.Draw(back_l), ImageDraw.Draw(liq_l), ImageDraw.Draw(front_l)

    for f in back:
        f(db)
    for c in cups:  # shadow, then the back of the glass
        db.ellipse(((c.cx - c.w_bottom * 0.75) * SS, (c.bottom - 6) * SS, (c.cx + c.w_bottom * 0.75) * SS,
                    (c.bottom + 14) * SS), fill=(120, 100, 90, 45))
        db.polygon(c.outline(), fill=(236, 242, 250, 70))

    for c in cups:  # liquid, ripples, spreading dye
        if c.level > 0.002:
            ly = lerp(c.bottom - 6, c.top + 10, c.level)
            hw = c.half_width(ly) - 5
            hb = c.w_bottom / 2 - 5
            dl.polygon(_sc([(c.cx - hw, ly), (c.cx + hw, ly), (c.cx + hb, c.bottom - 6), (c.cx - hb, c.bottom - 6)]),
                       fill=c.color + (215,))
            if c.cloud:
                col, p = c.cloud
                depth = (c.bottom - 6) - ly
                rx, ry = hw * (0.25 + 0.7 * p), depth * (0.15 + 0.4 * p)
                cy = ly + depth * (0.1 + 0.4 * p)
                dl.ellipse(_sc([(c.cx - rx, cy - ry), (c.cx + rx, cy + ry)]), fill=col + (int(200 * (1 - 0.6 * p)),))
            dl.ellipse(_sc([(c.cx - hw, ly - 7), (c.cx + hw, ly + 7)]), fill=mix(c.color, (255, 255, 255), 0.35) + (230,))
            for ph in c.ripples:
                if 0 <= ph < 1:
                    rx = hw * (0.15 + 0.8 * ph)
                    dl.ellipse(_sc([(c.cx - rx, ly - rx * 0.07 - 1), (c.cx + rx, ly + rx * 0.07 + 1)]),
                               outline=(255, 255, 255, int(220 * (1 - ph))), width=2 * SS)

    for s in strips:  # paper strips: over the liquid, behind the front glass
        path = _path(s.points)
        n_show = max(2, int(len(path) * s.shown))
        if s.shown <= 0:
            continue
        pts = [tuple(q * SS) for q in path[:n_show]]
        width = 15 * SS
        d.line(pts, fill=(250, 248, 242, 255), width=width, joint="curve")
        d.line(pts, fill=(205, 198, 188, 160), width=2 * SS)
        n_wet = int(len(path) * min(s.wick, s.shown))
        if n_wet > 1:
            d.line([tuple(q * SS) for q in path[:n_wet]], fill=s.color + (235,), width=width - 4 * SS, joint="curve")

    for c in cups:  # front glass, glints, falling drops
        d.polygon(c.outline(), outline=(120, 130, 150, 200), width=3 * SS)
        d.ellipse(_sc([(c.cx - c.w_top / 2, c.top - 9), (c.cx + c.w_top / 2, c.top + 9)]),
                  outline=(120, 130, 150, 200), width=3 * SS)
        hx = c.cx - c.w_top * 0.28
        d.line(_sc([(hx, c.top + 18), (c.cx - c.w_bottom * 0.3, c.bottom - 18)]), fill=(255, 255, 255, 160), width=6 * SS)
        d.line(_sc([(hx + 14, c.top + 30), (hx + 10, c.top + 70)]), fill=(255, 255, 255, 120), width=3 * SS)
        for (x, y, col) in c.drops:
            drop(d, x, y, col)
    for f in front:
        f(d)

    out = bg.convert("RGBA")
    for lay, blur in ((back_l, 0.8), (liq_l, 0.6), (front_l, 0.8)):
        lay = lay.filter(ImageFilter.GaussianBlur(blur * SS)).resize((w, h), Image.LANCZOS)
        if blur == 0.6:  # watercolour: pigment variation + darker pooled edges
            arr = np.asarray(lay, float)
            a = arr[..., 3:4] / 255
            soft = np.asarray(Image.fromarray(arr[..., 3].astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), float)[..., None] / 255
            edge = np.clip((a - soft) * 2.5, 0, 1)
            arr[..., :3] = np.clip(arr[..., :3] * _texture(w, h) * (1 - 0.25 * edge), 0, 255)
            lay = Image.fromarray(arr.astype(np.uint8), "RGBA")
        out.alpha_composite(lay)
    return out.convert("RGB")


def drop(d, x, y, col, r=7):
    d.ellipse(_sc([(x - r, y - r * 1.35), (x + r, y + r)]), fill=col + (240,))
    d.ellipse(_sc([(x - r / 2.5, y - r), (x, y - r / 3)]), fill=(255, 255, 255, 200))


GLASS = (120, 130, 150, 210)


def jug(d, cx, cy, s=1.0, tilt=0.0, level=0.65):
    """A clear water jug centred at (cx, cy); tilt in degrees (negative pours to the left).
    Returns the spout tip."""
    body = [(-60, -85), (60, -85), (70, 85), (-70, 85)]
    sc = lambda pts: rot([(x * s, y * s) for x, y in pts], cx, cy, tilt)  # noqa: E731
    d.polygon(_sc(sc(body)), fill=(236, 242, 250, 110))
    wy = 85 - 170 * level
    d.polygon(_sc(sc([(-60 - 10 * (wy + 85) / 170, wy), (60 + 10 * (wy + 85) / 170, wy), (70, 85), (-70, 85)])),
              fill=WATER + (200,))
    d.line(_sc(sc([(60, -60), (105, -50), (105, 40), (66, 50)])), fill=GLASS, width=int(9 * s * SS), joint="curve")
    d.polygon(_sc(sc(body)), outline=GLASS, width=3 * SS)
    d.polygon(_sc(sc([(-60, -85), (-88, -102), (-58, -66)])), fill=(236, 242, 250, 160), outline=GLASS)
    d.line(_sc(sc([(-40, -70), (-48, 60)])), fill=(255, 255, 255, 150), width=5 * SS)
    return sc([(-88, -102)])[0]


def bottle(d, cx, cy, color, s=1.0, flip=False):
    """A small squeeze bottle of food colouring; flip=True points the nozzle down. Returns the nozzle tip."""
    k = -1 if flip else 1
    Y = lambda y: cy + k * y * s  # noqa: E731
    band = lambda y0, y1: _sc([(0, min(Y(y0), Y(y1))), (0, max(Y(y0), Y(y1)))])  # noqa: E731

    def box(x, y0, y1):
        (_, t), (_, b) = band(y0, y1)
        return [((cx - x * s) * SS, t), ((cx + x * s) * SS, b)]

    d.rounded_rectangle(box(30, -45, 55), radius=int(12 * s * SS), fill=color + (255,),
                        outline=mix(color, (0, 0, 0), 0.3) + (255,), width=2 * SS)
    d.rectangle(box(22, -5, 25), fill=(255, 255, 255, 200))
    d.rectangle(box(13, -65, -45), fill=(250, 250, 250, 255), outline=(170, 170, 170, 255), width=SS)
    tip = (cx, Y(-88))
    d.polygon(_sc([(cx - 9 * s, Y(-65)), (cx + 9 * s, Y(-65)), tip]), fill=(250, 250, 250, 255), outline=(170, 170, 170, 255))
    return tip


def towel_flat(d, cx, base, s=1.0):
    pts = [(cx - 75 * s, base), (cx + 85 * s, base), (cx + 65 * s, base - 38 * s), (cx - 95 * s, base - 38 * s)]
    d.polygon(_sc(pts), fill=(252, 251, 247, 255), outline=(190, 185, 176, 255), width=2 * SS)
    for i in range(1, 4):
        x = cx - 75 * s + i * 40 * s
        d.line(_sc([(x, base - 3), (x - 20 * s, base - 35 * s)]), fill=(225, 220, 210, 255), width=SS)


def towel_fold(d, cx, cy, width, height, fold_p=0.0):
    """A paper towel seen face-on, `width` wide; while fold_p in (0,1) its right half is flipping over."""
    x0, x1 = cx - width / 2, cx + width / 2
    d.rectangle(_sc([(x0, cy - height / 2), (x1, cy + height / 2)]), fill=(252, 251, 247, 255),
                outline=(190, 185, 176, 255), width=2 * SS)
    if 0 < fold_p < 1:
        mid = cx
        fx = mid + (x1 - mid) * math.cos(math.pi * fold_p)
        d.polygon(_sc([(mid, cy - height / 2), (fx, cy - height / 2 - 10 * math.sin(math.pi * fold_p)),
                       (fx, cy + height / 2 + 10 * math.sin(math.pi * fold_p)), (mid, cy + height / 2)]),
                  fill=(232, 228, 218, 255), outline=(190, 185, 176, 255), width=2 * SS)
    for i in range(1, max(2, int(width // 30))):
        x = x0 + i * width / max(2, int(width // 30))
        d.line(_sc([(x, cy - height / 2 + 4), (x, cy + height / 2 - 4)]), fill=(228, 223, 213, 255), width=SS)


def clock(d, cx, cy, r, minutes):
    d.ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=(255, 253, 248, 255), outline=(110, 80, 60, 255), width=4 * SS)
    for i in range(12):
        a = math.radians(i * 30)
        d.line(_sc([(cx + math.sin(a) * r * 0.8, cy - math.cos(a) * r * 0.8),
                    (cx + math.sin(a) * r * 0.92, cy - math.cos(a) * r * 0.92)]), fill=(110, 80, 60, 255), width=2 * SS)
    am, ah = math.radians(minutes * 6), math.radians(90 + minutes * 0.5)
    d.line(_sc([(cx, cy), (cx + math.sin(ah) * r * 0.5, cy - math.cos(ah) * r * 0.5)]), fill=(80, 55, 40, 255), width=5 * SS)
    d.line(_sc([(cx, cy), (cx + math.sin(am) * r * 0.78, cy - math.cos(am) * r * 0.78)]), fill=(200, 70, 50, 255), width=3 * SS)
    d.ellipse(_sc([(cx - 5, cy - 5), (cx + 5, cy + 5)]), fill=(80, 55, 40, 255))


def tray(d, x0, x1, y):
    d.rounded_rectangle(_sc([(x0, y - 22), (x1, y + 18)]), radius=16 * SS, fill=(176, 206, 196, 255),
                        outline=(120, 160, 148, 255), width=3 * SS)
    d.rounded_rectangle(_sc([(x0 + 12, y - 14), (x1 - 12, y + 8)]), radius=10 * SS, fill=(196, 222, 212, 255))


def plant(d, cx, base, water):
    """A potted plant whose stem is drawn as a see-through tube; `water` (0..1) is how far water has risen."""
    d.polygon(_sc([(cx - 90, base - 150), (cx + 90, base - 150), (cx + 65, base), (cx - 65, base)]),
              fill=(206, 120, 80, 255), outline=(160, 90, 60, 255), width=3 * SS)
    d.rectangle(_sc([(cx - 98, base - 170), (cx + 98, base - 145)]), fill=(214, 130, 90, 255), outline=(160, 90, 60, 255),
                width=3 * SS)
    d.ellipse(_sc([(cx - 85, base - 182), (cx + 85, base - 160)]), fill=(110, 80, 60, 255))
    top = base - 520
    stem = [(cx, base - 170 - (base - 170 - top) * i / 40) for i in range(41)]
    stem = [(x + 18 * math.sin(i / 40 * math.pi * 1.5), y) for i, (x, y) in enumerate(stem)]
    d.line(_sc(stem), fill=(90, 160, 80, 255), width=26 * SS, joint="curve")
    d.line(_sc(stem), fill=(200, 235, 190, 255), width=14 * SS, joint="curve")
    n = int(len(stem) * water)
    if n > 1:
        d.line(_sc(stem[:n]), fill=(90, 150, 230, 255), width=10 * SS, joint="curve")
    for k, side in ((14, -1), (24, 1), (34, -1), (40, 1)):
        x, y = stem[k]
        wet = water * 40 >= k
        col = (110, 190, 100, 255) if wet else (150, 190, 120, 255)
        leaf = rot([(0, -18), (70, -30), (120, 0), (70, 26), (0, 14)], x, y, -25 if side > 0 else 205)
        d.polygon(_sc(leaf), fill=col, outline=(70, 130, 60, 255))


def lens(img: Image.Image, cx, cy, r, level, water_col=(230, 95, 105), anchor=None) -> Image.Image:
    """A magnifying glass showing paper fibres; water climbs between them as `level` goes 0→1."""
    w, h = img.size
    lay = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if anchor:
        d.line(_sc([anchor, (cx - r * 0.7, cy + r * 0.7)]), fill=(150, 130, 110, 160), width=3 * SS)
        d.ellipse(_sc([(anchor[0] - 10, anchor[1] - 10), (anchor[0] + 10, anchor[1] + 10)]),
                  outline=(150, 130, 110, 220), width=3 * SS)
    inner = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    di.ellipse(_sc([(cx - r, cy - r), (cx + r, cy + r)]), fill=(253, 251, 245, 255))
    rng = np.random.default_rng(5)
    fibres = [cx - r + 18 + i * 30 + rng.uniform(-6, 6) for i in range(int(2 * r / 30))]
    surf = cy + r - level * (1.55 * r)
    xs = np.linspace(cx - r, cx + r, 160)
    ys = [surf - 38 * max(math.exp(-((x - f) / 9) ** 2) for f in fibres) for x in xs]
    di.polygon(_sc([(cx - r, cy + r + 5)] + list(zip(xs, ys)) + [(cx + r, cy + r + 5)]), fill=water_col + (200,))
    for i, f in enumerate(fibres):
        pts = [(f + 6 * math.sin(y / 23 + i), y) for y in np.linspace(cy - r, cy + r, 30)]
        di.line(_sc(pts), fill=(205, 196, 180, 255), width=9 * SS, joint="curve")
        di.line(_sc(pts), fill=(248, 244, 234, 255), width=5 * SS, joint="curve")
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
