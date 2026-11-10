"""Synthesised sound effects for code-drawn experiments — free, deterministic, license-clean.

Small, friendly sounds a kids' science video uses: drop plinks, pouring
bubbles, a paper whoosh, clock ticks, item pops and a payoff chime. Each
returns a mono float32 array at RATE; `mix_track` places them on a timeline.
"""

from __future__ import annotations

import numpy as np

RATE = 48000
_rng = np.random.default_rng(11)


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def plink(pitch: float = 1100.0) -> np.ndarray:
    t = _t(0.35)
    f = pitch * np.exp(-t * 6) + pitch * 0.45
    phase = 2 * np.pi * np.cumsum(f) / RATE
    return (np.sin(phase) * np.exp(-t * 14) * 0.5).astype(np.float32)


def pop(pitch: float = 520.0) -> np.ndarray:
    t = _t(0.16)
    f = pitch + 700 * t / 0.16
    phase = 2 * np.pi * np.cumsum(f) / RATE
    return (np.sin(phase) * np.exp(-t * 22) * 0.45).astype(np.float32)


def bubbles(seconds: float, density: float = 18.0) -> np.ndarray:
    out = np.zeros(int(seconds * RATE), np.float32)
    for _ in range(int(seconds * density)):
        start = int(_rng.uniform(0, seconds - 0.08) * RATE)
        b = plink(_rng.uniform(380, 900)) * _rng.uniform(0.15, 0.35)
        out[start:start + len(b)] += b[: len(out) - start]
    noise = np.convolve(_rng.normal(0, 1, len(out)), np.ones(40) / 40, "same") * 0.05
    env = np.minimum(1, np.minimum(np.arange(len(out)), len(out) - np.arange(len(out))) / (0.15 * RATE))
    return ((out + noise) * env).astype(np.float32)


def whoosh(seconds: float = 0.6) -> np.ndarray:
    n = int(seconds * RATE)
    noise = _rng.normal(0, 1, n)
    smooth = np.convolve(noise, np.ones(24) / 24, "same")
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    return (smooth * env * 0.5).astype(np.float32)


def tick() -> np.ndarray:
    t = _t(0.04)
    return (np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 120) * 0.35).astype(np.float32)


def chime() -> np.ndarray:
    out = np.zeros(int(1.6 * RATE), np.float32)
    for k, freq in enumerate((1046.5, 1318.5, 1568.0, 2093.0)):  # C6 E6 G6 C7
        t = _t(1.2)
        note = (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)) * np.exp(-t * 3.5) * 0.18
        s = int(k * 0.09 * RATE)
        out[s:s + len(note)] += note.astype(np.float32)
    return out


def mix_track(events: list[tuple[float, np.ndarray]], seconds: float) -> np.ndarray:
    """Place (time, sound) events on a mono track, soft-limited."""
    out = np.zeros(int(seconds * RATE) + RATE, np.float32)
    for at, snd in events:
        s = int(max(0.0, at) * RATE)
        e = min(len(out), s + len(snd))
        out[s:e] += snd[: e - s]
    return np.tanh(out * 1.2) * 0.8
