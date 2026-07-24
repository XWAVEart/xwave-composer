"""CLI entrypoint for xwave-composer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-layer AI image composition (Flux 2 klein + SDXL Hyper)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    parser.add_argument("--host", type=str, default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument(
        "--profile",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=None,
        help="Blackwell compute profile (overrides optimization.profile)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load Flux + SDXL Hyper at startup (uses VRAM immediately)",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    # Ensure src is importable when run as script
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from xwave_composer.config import AppConfig
    from xwave_composer.ui.gradio_app import build_app

    config = AppConfig.load(args.config)
    if args.host:
        config.raw.setdefault("server", {})["host"] = args.host
    if args.port is not None:
        config.raw.setdefault("server", {})["port"] = args.port
    if args.share:
        config.raw.setdefault("server", {})["share"] = True
    if args.profile:
        config.raw.setdefault("optimization", {})["profile"] = args.profile

    config.ensure_dirs()
    from xwave_composer.optimization import configure_blackwell_runtime, configured_profile

    caps = configure_blackwell_runtime(config)
    logging.getLogger(__name__).info(
        "Starting xwave-composer | device=%s | profile=%s | %s",
        config.device,
        configured_profile(config),
        caps.summary(),
    )

    demo = build_app(config)

    if args.preload:
        import threading

        session = getattr(demo, "_xwave_session", None)
        if session is not None:
            def _preload() -> None:
                logging.getLogger(__name__).info("Preloading models…")
                logging.getLogger(__name__).info(session.preload_core())

            threading.Thread(target=_preload, daemon=True).start()

    from xwave_composer.ui.gradio_app import _launch_kwargs

    demo.queue(default_concurrency_limit=1).launch(**_launch_kwargs(demo, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
