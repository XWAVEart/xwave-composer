"""CLI entrypoint for xwave-composer."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # Windows consoles commonly default to a legacy code page (cp1252) that
    # cannot encode the status glyphs in our log messages, e.g. the check and
    # cross in ComputeCapabilities.summary(). Without this the first startup
    # log raises UnicodeEncodeError and the capability line is lost.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached or non-reconfigurable stream
                pass
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

    # Persist the TorchInductor cache inside the project.
    #
    # optimization.compile is on by default, and regionally compiling the Flux
    # transformer costs about 8 minutes on a first run. Inductor caches the
    # result, but its default cache lives under the system temp directory,
    # which is cleared often enough that the cost is paid again and again.
    # Pointing it at the project turns later startups into a cache hit.
    #
    # An explicit TORCHINDUCTOR_CACHE_DIR always wins.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / ".cache" / "inductor"))

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

    log = logging.getLogger(__name__)
    launch_kwargs = _launch_kwargs(demo, config)
    # Launch without blocking so the control API can be attached to the FastAPI
    # app Gradio builds, then block explicitly. One process, one port, one copy
    # of the models shared by the UI and any script driving it.
    demo.queue(default_concurrency_limit=1).launch(prevent_thread_lock=True, **launch_kwargs)

    session = getattr(demo, "_xwave_session", None)
    fastapi_app = getattr(demo, "app", None)
    if session is not None and fastapi_app is not None:
        try:
            from xwave_composer.api.control import register_control_api

            register_control_api(fastapi_app, session)
            log.info(
                "Control API ready: http://127.0.0.1:%s/control/state",
                launch_kwargs.get("server_port", 7860),
            )
        except Exception:  # noqa: BLE001 - the UI must still come up
            log.exception("Control API failed to mount; the UI is unaffected")
    else:
        log.warning("Control API not mounted (no session or app on the Blocks object)")

    demo.block_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
