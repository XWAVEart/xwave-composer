#!/usr/bin/env python3
"""Drive a running xwave-composer from the command line.

The app must already be running (Start XWAVE Composer.bat). This talks to the
control API on the same port the UI uses, so anything done here shows up in
the browser and vice versa.

    python xwave.py load
    python xwave.py background "an empty misty lake at dawn"
    python xwave.py sticker "a small wooden rowboat"
    python xwave.py place --x 300 --y 700 --scale 0.4
    python xwave.py refine
    python xwave.py improve --notes "the boat is too small"
    python xwave.py save output.png

Or the whole thing in one go:

    python xwave.py compose "a misty lake at dawn" -s "a rowboat" -s "a heron"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:7860/control"
# Generation, refine and especially SeedVR2 export are all slow; a short
# timeout would abort work that is progressing perfectly well.
TIMEOUT_S = 1800


class ApiError(RuntimeError):
    pass


def call(base: str, path: str, payload: dict | None = None, raw: bool = False):
    url = f"{base}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
        raise ApiError(f"{exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ApiError(
            f"Could not reach {url} — is the app running? ({exc.reason})"
        ) from None
    return body if raw else json.loads(body)


def post(base: str, path: str, payload: dict | None = None):
    return call(base, path, payload if payload is not None else {})


# ---------------------------------------------------------------- presenting
def show_state(state: dict) -> None:
    canvas = state.get("canvas", {})
    print(f"  status   : {state.get('status', '')}")
    print(f"  canvas   : {canvas.get('width')}x{canvas.get('height')}")
    bg = state.get("background", {})
    print(f"  backdrop : {'yes' if bg.get('has_image') else 'none'}  {bg.get('prompt', '')[:60]}")
    layers = state.get("layers", [])
    if layers:
        print(f"  stickers : {len(layers)} (bottom to top)")
        for layer in layers:
            mark = "*" if layer["id"] == state.get("selected_id") else " "
            print(
                f"   {mark} {layer['id']}  ({layer['x']:.0f},{layer['y']:.0f})"
                f"  x{layer['scale_x']:.2f}  {layer['prompt'][:44]}"
            )
    else:
        print("  stickers : none")
    vram = state.get("vram")
    if vram:
        print(f"  vram     : {vram['used_gb']:.1f} / {vram['total_gb']:.1f} GB")
    flags = []
    if state.get("can_undo"):
        flags.append("undo")
    if state.get("can_redo"):
        flags.append("redo")
    if state.get("has_output"):
        flags.append("output")
    if not state.get("models_ready"):
        flags.append("MODELS NOT LOADED")
    if flags:
        print(f"  flags    : {', '.join(flags)}")


def report(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    if "state" in result:
        show_state(result["state"])
    else:
        print(json.dumps(result, indent=2))


# ------------------------------------------------------------------ commands
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xwave", description="Drive a running xwave-composer."
    )
    parser.add_argument("--base", default=DEFAULT_BASE, help="Control API base URL")
    parser.add_argument("--json", action="store_true", help="Print raw JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state", help="Show what is on the canvas")
    sub.add_parser("load", help="Load the models (slow the first time)")
    sub.add_parser("undo")
    sub.add_parser("redo")
    sub.add_parser("reset", help="Clear the workspace")

    p = sub.add_parser("background", help="Generate the backdrop")
    p.add_argument("prompt")
    p.add_argument("--seed", type=int, default=-1)

    p = sub.add_parser("sticker", help="Generate a new cut-out sticker")
    p.add_argument("prompt")
    p.add_argument("--isolation", default="", help="Override the isolation prompt")
    p.add_argument("--no-cutout", action="store_true", help="Place the full image")
    p.add_argument("--seed", type=int, default=-1)

    p = sub.add_parser("place", help="Move, scale or rotate a sticker")
    p.add_argument("--id", default=None, help="Defaults to the selected layer")
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.add_argument("--scale", type=float)
    p.add_argument("--rotation", type=float)
    p.add_argument("--opacity", type=float)

    p = sub.add_parser("select")
    p.add_argument("id")

    p = sub.add_parser("delete")
    p.add_argument("--id", default=None)

    p = sub.add_parser("refine", help="Re-render the OUTPUT")
    p.add_argument("--denoise", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("improve", help="Critique the image and propose a better prompt")
    p.add_argument("--notes", default="", help="What you think is wrong")
    p.add_argument("--edit", action="store_true", help="Refine the existing image instead")
    p.add_argument("--apply", action="store_true", help="Apply the improved prompt")
    p.add_argument(
        "--target",
        choices=("layer", "output"),
        default=None,
        help="What to look at. Default: the selected layer, else the OUTPUT.",
    )

    sub.add_parser("apply", help="Apply the last improved prompt")
    sub.add_parser("history", help="Show improve iterations")

    p = sub.add_parser("export", help="2x upscale and save")
    p.add_argument("--steps", type=int)

    p = sub.add_parser("save", help="Download work.png / output.png / a layer")
    p.add_argument("dest")
    p.add_argument("--what", default="output", help="output | work | <layer id>")

    p = sub.add_parser("compose", help="Backdrop plus stickers in one command")
    p.add_argument("background")
    p.add_argument("-s", "--sticker", action="append", default=[], help="Repeatable")
    p.add_argument("--no-refine", action="store_true")

    args = parser.parse_args(argv)
    base = args.base.rstrip("/")

    try:
        if args.cmd == "state":
            report({"state": call(base, "/state")}, args.json)

        elif args.cmd == "load":
            print("Loading models — this takes many minutes on a cold start…")
            report(post(base, "/load"), args.json)

        elif args.cmd in ("undo", "redo", "reset"):
            report(post(base, f"/{args.cmd}"), args.json)

        elif args.cmd == "background":
            report(
                post(base, "/background", {"prompt": args.prompt, "seed": args.seed}), args.json
            )

        elif args.cmd == "sticker":
            result = post(
                base,
                "/sticker",
                {
                    "prompt": args.prompt,
                    "isolation_prompt": args.isolation,
                    "cutout": not args.no_cutout,
                    "seed": args.seed,
                },
            )
            if not args.json:
                print(f"sticker id: {result.get('id')}")
            report(result, args.json)

        elif args.cmd == "place":
            body = {
                k: v
                for k, v in {
                    "id": args.id,
                    "x": args.x,
                    "y": args.y,
                    "scale": args.scale,
                    "rotation": args.rotation,
                    "opacity": args.opacity,
                }.items()
                if v is not None
            }
            report(post(base, "/place", body), args.json)

        elif args.cmd == "select":
            report(post(base, "/select", {"id": args.id}), args.json)

        elif args.cmd == "delete":
            report(post(base, "/delete", {"id": args.id}), args.json)

        elif args.cmd == "refine":
            body = {
                k: v
                for k, v in {
                    "denoise": args.denoise,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "seed": args.seed,
                }.items()
                if v is not None
            }
            report(post(base, "/refine", body), args.json)

        elif args.cmd == "improve":
            result = post(
                base,
                "/improve",
                {
                    "notes": args.notes,
                    "edit_mode": args.edit,
                    "apply": args.apply,
                    "target": args.target,
                },
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                # Say what was judged: the default is the selected layer, which
                # is not always what the notes were about.
                if result.get("looked_at"):
                    print(f"looked at: {result['looked_at']}\n")
                if result.get("critique"):
                    print("CRITIQUE\n" + result["critique"] + "\n")
                if result.get("improved_prompt"):
                    print("IMPROVED PROMPT\n" + result["improved_prompt"] + "\n")
                if result.get("error"):
                    print(f"note: {result['error']}")
                if result.get("applied"):
                    print(result["applied"])
                elif result.get("ok"):
                    print("(run `xwave.py apply` to use it)")

        elif args.cmd == "apply":
            result = post(base, "/improve/apply")
            print(result.get("message", ""))

        elif args.cmd == "history":
            data = call(base, "/improve/history")
            if args.json:
                print(json.dumps(data, indent=2))
            else:
                print(f"target: {data.get('target') or '(none)'}")
                for it in data.get("iterations", []):
                    print(f"\n--- pass {it['n']} ---")
                    if it.get("user_notes"):
                        print(f"you said: {it['user_notes']}")
                    print(it["critique"])

        elif args.cmd == "export":
            print("Exporting with SeedVR2 — models unload and reload, so this is slow…")
            result = post(base, "/export", {"refine_steps": args.steps})
            print(f"saved: {result.get('path')}")

        elif args.cmd == "save":
            what = args.what
            path = {
                "output": "/render/output.png",
                "work": "/render/work.png",
            }.get(what, f"/render/layer/{what}.png")
            blob = call(base, path, raw=True)
            with open(args.dest, "wb") as handle:
                handle.write(blob)
            print(f"wrote {args.dest} ({len(blob):,} bytes)")

        elif args.cmd == "compose":
            print(f"backdrop: {args.background}")
            post(base, "/background", {"prompt": args.background, "seed": -1})
            for prompt in args.sticker:
                print(f"sticker : {prompt}")
                result = post(
                    base,
                    "/sticker",
                    {"prompt": prompt, "isolation_prompt": "", "cutout": True, "seed": -1},
                )
                print(f"          -> {result.get('id')}")
            if not args.no_refine:
                print("refining…")
                result = post(base, "/refine", {})
                report(result, args.json)

    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
