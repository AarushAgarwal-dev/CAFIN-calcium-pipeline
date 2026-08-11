#!/usr/bin/env python3
"""Create a small deterministic two-channel CAFIN demo recording."""
from pathlib import Path
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "demo_data"


def make_demo(force=False):
    mem_dir, ca_dir = OUT / "membrane", OUT / "ca2"
    expected = mem_dir / "demo_mem_0000.tif"
    if expected.exists() and not force:
        return OUT
    mem_dir.mkdir(parents=True, exist_ok=True)
    ca_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    h = w = 256
    yy, xx = np.mgrid[:h, :w]
    centers = [(45 + 42 * c + (r % 2) * 20, 45 + 42 * r)
               for r in range(5) for c in range(5)]
    for t in range(12):
        dx = int(round(3 * np.sin(t / 3)))
        dy = int(round(2 * np.cos(t / 4)))
        mem = np.full((h, w), 180.0, np.float32)
        ca = np.full((h, w), 110.0 + 0.8 * t, np.float32)
        for i, (cx, cy) in enumerate(centers):
            rr = np.sqrt((xx - (cx + dx)) ** 2 + (yy - (cy + dy)) ** 2)
            mem += 1200 * np.exp(-((rr - 16) ** 2) / 3.2)
            pulse = 900 * np.exp(-0.5 * ((t - (2 + i % 8)) / 1.1) ** 2)
            ca += (170 + pulse) * (rr <= 14)
        mem += rng.normal(0, 18, mem.shape)
        ca += rng.normal(0, 12, ca.shape)
        tifffile.imwrite(mem_dir / f"demo_mem_{t:04d}.tif",
                         np.clip(mem, 0, 65535).astype(np.uint16))
        tifffile.imwrite(ca_dir / f"demo_ca_{t:04d}.tif",
                         np.clip(ca, 0, 65535).astype(np.uint16))
    (OUT / "README.txt").write_text(
        "Synthetic CAFIN demo only. It is not biological evidence and must not be used in a paper.\n",
        encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(f"Demo data ready: {make_demo()}")
