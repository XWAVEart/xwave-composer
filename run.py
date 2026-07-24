#!/usr/bin/env python3
"""Launch xwave-composer from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from xwave_composer.app import main

if __name__ == "__main__":
    raise SystemExit(main())
