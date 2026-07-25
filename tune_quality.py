#!/usr/bin/env python3
"""Sweep OUTPUT settings against a fixed composition and save every result.

The point is a fair comparison: build the scene ONCE, then vary one knob at a
time with the seed pinned, so differences between images are the setting and
nothing else. Run against an already-running app.

    python tune_quality.py --out tuning
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7860/control"
TIMEOUT = 1800
SEED = 12345  # pinned: the sweep must not also be sampling noise


def call(path: str, payload: dict | None = None, raw: bool = False):
    url = f"{BASE}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data is not None else "GET"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = resp.read()
    return body if raw else json.loads(body)


def save(out_dir: Path, name: str) -> Path:
    blob = call("/render/output.png", raw=True)
    path = out_dir / f"{name}.png"
    path.write_bytes(blob)
    return path


def refine(out_dir: Path, name: str, **knobs) -> float:
    body = {"seed": SEED, **knobs}
    start = time.time()
    call("/refine", body)
    elapsed = time.time() - start
    save(out_dir, name)
    print(f"  {name:<34} {elapsed:5.1f}s  {knobs}", flush=True)
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tuning")
    ap.add_argument("--backdrop", default="an old stone bridge over a river in autumn woodland")
    ap.add_argument("--sticker", default="a red vintage bicycle")
    ap.add_argument("--skip-build", action="store_true", help="Reuse the current composition")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = call("/state")
    if not state["models_ready"]:
        print("Models are not loaded yet.")
        return 1

    if not args.skip_build:
        print("Building the reference composition (once)…", flush=True)
        call("/reset", {})
        call("/background", {"prompt": args.backdrop, "seed": SEED})
        sticker = call("/sticker", {"prompt": args.sticker, "seed": SEED})
        call("/place", {"id": sticker["id"], "x": 380, "y": 660, "scale": 0.40})
        print(f"  sticker {sticker['id']} placed", flush=True)

    print("\nDENOISE sweep (steps=6, cfg=1.0) — how far the refine may drift:", flush=True)
    for d in (0.20, 0.30, 0.40, 0.50, 0.65):
        refine(out_dir, f"denoise_{d:.2f}", denoise=d, steps=6, cfg=1.0)

    print("\nSTEPS sweep (denoise=0.35, cfg=1.0) — Hyper is distilled for few:", flush=True)
    for s in (4, 6, 8, 12):
        refine(out_dir, f"steps_{s:02d}", denoise=0.35, steps=s, cfg=1.0)

    print("\nCFG sweep (denoise=0.35, steps=8):", flush=True)
    for c in (1.0, 1.5, 2.0, 3.0):
        refine(out_dir, f"cfg_{c:.1f}", denoise=0.35, steps=8, cfg=c)

    print(f"\nWrote {len(list(out_dir.glob('*.png')))} images to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"Could not reach the app: {exc.reason}")
        raise SystemExit(1) from None
