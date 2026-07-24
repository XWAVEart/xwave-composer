"""Style presets loaded from the XWAVE-COMPOSER-STYLES.csv spreadsheet.

Each preset carries a prefix/suffix prompt pair, a negative prompt, and
sampler values (CFG, denoise, eta). The user prompt is injected between
prefix and suffix. Values remain user-editable after a preset loads.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from xwave_composer.config import AppConfig

if TYPE_CHECKING:
    from xwave_composer.canvas.layers import WorkDocument

logger = logging.getLogger(__name__)

# Old embedding tokens like <3D>, <DEETS> from a previous setup — no matching
# textual inversions are loaded here, so they would be tokenized as junk text.
_TOKEN_RE = re.compile(r"<[^<>]{1,24}>")
_SPACE_COMMA_RE = re.compile(r"\s*,\s*(?:,\s*)+")


def _clean(text: str) -> str:
    text = _TOKEN_RE.sub("", text or "")
    text = _SPACE_COMMA_RE.sub(", ", text)
    return text.strip(" ,")


@dataclass(frozen=True)
class StylePreset:
    name: str
    cfg: float
    denoise: float
    eta: float
    prefix: str
    suffix: str
    negative: str


class StyleManager:
    """Load style presets from the CSV and resolve the active preset."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.path = self._find_csv(config)
        self.presets: dict[str, StylePreset] = {}
        self.active_name: str | None = None
        self.load()

    @staticmethod
    def _find_csv(config: AppConfig) -> Path:
        configured = config.get("style", "presets_file", default=None)
        if configured:
            path = config.path("style", "presets_file", default=str(configured))
        else:
            path = config.root / "XWAVE-COMPOSER-STYLES.csv"
        if path.exists():
            return path
        # Linux paths are case-sensitive; accept .csv / .CSV interchangeably.
        for candidate in sorted(config.root.glob("XWAVE-COMPOSER-STYLES.*")):
            if candidate.suffix.lower() == ".csv":
                return candidate
        return path

    def load(self) -> None:
        self.presets = {}
        if not self.path.exists():
            logger.warning("Style presets CSV not found: %s", self.path)
            return
        with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        # Skip the title row ("XWAVE COMPOSER STYLES") and locate the header.
        header_i = next(
            (i for i, r in enumerate(rows) if r and r[0].strip().lower() == "style name"),
            None,
        )
        if header_i is None:
            logger.warning("Style CSV header row not found in %s", self.path)
            return
        for row in rows[header_i + 1 :]:
            if len(row) < 7 or not row[0].strip():
                continue
            name = row[0].strip()
            try:
                preset = StylePreset(
                    name=name,
                    cfg=float(row[1] or 1.0),
                    denoise=float(row[2] or 0.35),
                    eta=float(row[3] or 0.0),
                    prefix=row[4] or "",
                    suffix=_clean(row[5]),
                    negative=_clean(row[6]),
                )
            except ValueError as exc:
                logger.warning("Skipping style row %r: %s", name, exc)
                continue
            self.presets[name] = preset
        logger.info("Loaded %d style presets from %s", len(self.presets), self.path)

    def names(self) -> list[str]:
        return sorted(self.presets.keys(), key=str.lower)

    def get(self, name: str | None) -> StylePreset | None:
        if not name:
            return None
        return self.presets.get(name)

    def set_active(self, name: str | None) -> StylePreset | None:
        preset = self.get(name)
        self.active_name = preset.name if preset else None
        return preset

    @property
    def active(self) -> StylePreset | None:
        return self.get(self.active_name)


def build_output_prompt(
    doc: "WorkDocument",
    preset: StylePreset | None,
    order: str = "pcs",
    manual_prefix: str | None = None,
    manual_suffix: str | None = None,
) -> str:
    """Build the OUTPUT prompt from style prefix/suffix + layer prompts.

    order="pcs" (default): prefix is fused into the user prompts with a space
    (no comma), then ", " + suffix — matching the CSV style sheets.
    order="psc": prefix, suffix, then user prompts — comma-separated.
    """
    parts: list[str] = []
    if doc.background_prompt.strip():
        parts.append(doc.background_prompt.strip())
    for obj in doc.objects:
        if obj.prompt_enabled and obj.prompt.strip():
            parts.append(obj.prompt.strip())
    content = ", ".join(parts)

    if manual_prefix is not None or manual_suffix is not None:
        prefix = _clean(manual_prefix or "")
        suffix = _clean(manual_suffix or "")
    elif preset is not None:
        prefix = _clean(preset.prefix)
        suffix = _clean(preset.suffix)
    else:
        return content

    if order == "psc":
        pieces = [p for p in (prefix, suffix, content) if p]
        return ", ".join(pieces)

    # pcs: PREFIX USER SUFFIX — fuse prefix into the user text (no comma).
    if prefix and content:
        lead = f"{prefix} {content}"
    else:
        lead = prefix or content
    if suffix:
        return f"{lead}, {suffix}" if lead else suffix
    return lead
