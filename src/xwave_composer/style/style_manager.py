"""Style presets loaded from the XWAVE-COMPOSER-STYLES.csv spreadsheet.

Each preset carries a family tag, a prefix/suffix prompt pair, a negative
prompt, and sampler values (CFG, denoise, eta). The user prompt is injected
between prefix and suffix. Values remain user-editable after a preset loads.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from xwave_composer.config import AppConfig

if TYPE_CHECKING:
    from xwave_composer.canvas.layers import WorkDocument

logger = logging.getLogger(__name__)

# Canonical family labels (order used in UI). New families can be added later.
STYLE_FAMILIES: tuple[str, ...] = ("Art", "Render", "Photo", "Sculpture", "Other")
_FAMILY_ALIAS = {f.lower(): f for f in STYLE_FAMILIES}

# Old embedding tokens like <3D>, <DEETS> from a previous setup — no matching
# textual inversions are loaded here, so they would be tokenized as junk text.
_TOKEN_RE = re.compile(r"<[^<>]{1,24}>")
_SPACE_COMMA_RE = re.compile(r"\s*,\s*(?:,\s*)+")


def _clean(text: str) -> str:
    text = _TOKEN_RE.sub("", text or "")
    text = _SPACE_COMMA_RE.sub(", ", text)
    return text.strip(" ,")


def normalize_family(value: str | None) -> str:
    """Map a CSV family cell to a canonical label; unknown/empty → Other."""
    raw = (value or "").strip()
    if not raw:
        return "Other"
    known = _FAMILY_ALIAS.get(raw.lower())
    if known:
        return known
    # Allow future families without code edits — Title Case the token.
    return raw[:1].upper() + raw[1:] if raw else "Other"


@dataclass(frozen=True)
class StylePreset:
    name: str
    family: str
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
        header = [c.strip().lower() for c in rows[header_i]]

        def col(*aliases: str, default: int | None = None) -> int | None:
            for a in aliases:
                if a in header:
                    return header.index(a)
            return default

        i_name = col("style name", default=0)
        i_family = col("family")
        # Support both old (no Family) and new layouts.
        if i_family is None:
            i_cfg, i_denoise, i_eta = col("cfg", default=1), col("denoise", default=2), col(
                "eta", default=3
            )
            i_prefix, i_suffix, i_neg = col("prefix", default=4), col("suffix", default=5), col(
                "negative", default=6
            )
            min_cols = 7
        else:
            i_cfg, i_denoise, i_eta = col("cfg", default=2), col("denoise", default=3), col(
                "eta", default=4
            )
            i_prefix, i_suffix, i_neg = col("prefix", default=5), col("suffix", default=6), col(
                "negative", default=7
            )
            min_cols = 8

        assert i_name is not None
        assert i_cfg is not None and i_denoise is not None and i_eta is not None
        assert i_prefix is not None and i_suffix is not None and i_neg is not None

        for row in rows[header_i + 1 :]:
            if len(row) < min_cols or not row[i_name].strip():
                continue
            name = row[i_name].strip()
            family = normalize_family(row[i_family] if i_family is not None else None)
            try:
                preset = StylePreset(
                    name=name,
                    family=family,
                    cfg=float(row[i_cfg] or 1.0),
                    denoise=float(row[i_denoise] or 0.35),
                    eta=float(row[i_eta] or 0.0),
                    prefix=row[i_prefix] or "",
                    suffix=_clean(row[i_suffix]),
                    negative=_clean(row[i_neg]),
                )
            except ValueError as exc:
                logger.warning("Skipping style row %r: %s", name, exc)
                continue
            self.presets[name] = preset
        logger.info("Loaded %d style presets from %s", len(self.presets), self.path)

    def names(self, families: Iterable[str] | None = None) -> list[str]:
        """Sorted style names, optionally filtered to one or more families."""
        if families is None:
            pool = self.presets.values()
        else:
            allowed = {normalize_family(f) for f in families if f}
            if not allowed:
                return []
            pool = (p for p in self.presets.values() if p.family in allowed)
        return sorted((p.name for p in pool), key=str.lower)

    def families_present(self) -> list[str]:
        """Families that have at least one preset, canonical order then extras."""
        present = {p.family for p in self.presets.values()}
        ordered = [f for f in STYLE_FAMILIES if f in present]
        extras = sorted(present - set(STYLE_FAMILIES), key=str.lower)
        return ordered + extras

    def names_by_family(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {f: [] for f in self.families_present()}
        for name in self.names():
            fam = self.presets[name].family
            out.setdefault(fam, []).append(name)
        return out

    def grouped_choices(self, include_none: str | None = None) -> list:
        """Gradio Dropdown choices: family optgroups, optional none sentinel first."""
        groups: list = []
        if include_none is not None:
            groups.append(include_none)
        for fam, names in self.names_by_family().items():
            if names:
                groups.append((fam, names))
        return groups

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


# UI labels → internal order keys (Compose + Infinite Canvas).
CONCAT_ORDERS: dict[str, str] = {
    "Prefix · Prompts · Suffix": "pcs",
    "Prefix · Suffix · Prompts": "psc",
    "Prompts · Prefix · Suffix": "cps",
    "Suffix · Prefix · Prompts": "spc",
}


def assemble_style_prompt(
    content: str,
    prefix: str,
    suffix: str,
    order: str = "pcs",
) -> str:
    """Combine user content with style prefix/suffix using a concat order.

    order="pcs" (default): prefix is fused into the user text with a space
    (no comma), then ", " + suffix — matching the CSV style sheets.
    order="psc": prefix, suffix, then content — comma-separated.
    order="cps": content, prefix, suffix — comma-separated.
    order="spc": suffix, prefix, then content — comma-separated.
    """
    content = (content or "").strip()
    prefix = _clean(prefix)
    suffix = _clean(suffix)
    if order == "psc":
        pieces = [p for p in (prefix, suffix, content) if p]
        return ", ".join(pieces)
    if order == "cps":
        pieces = [p for p in (content, prefix, suffix) if p]
        return ", ".join(pieces)
    if order == "spc":
        pieces = [p for p in (suffix, prefix, content) if p]
        return ", ".join(pieces)

    # pcs: PREFIX USER SUFFIX — fuse prefix into the user text (no comma).
    if prefix and content:
        lead = f"{prefix} {content}"
    else:
        lead = prefix or content
    if suffix:
        return f"{lead}, {suffix}" if lead else suffix
    return lead


def build_output_prompt(
    doc: "WorkDocument",
    preset: StylePreset | None,
    order: str = "pcs",
    manual_prefix: str | None = None,
    manual_suffix: str | None = None,
) -> str:
    """Build the OUTPUT prompt from style prefix/suffix + layer prompts."""
    parts: list[str] = []
    if doc.background_prompt.strip():
        parts.append(doc.background_prompt.strip())
    for obj in doc.objects:
        if obj.prompt_enabled and obj.prompt.strip():
            parts.append(obj.prompt.strip())
    content = ", ".join(parts)

    if manual_prefix is not None or manual_suffix is not None:
        prefix = manual_prefix or ""
        suffix = manual_suffix or ""
    elif preset is not None:
        prefix = preset.prefix
        suffix = preset.suffix
    else:
        return content

    return assemble_style_prompt(content, prefix, suffix, order=order)
