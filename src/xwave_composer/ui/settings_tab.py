"""Settings tab — interface preferences (color scheme, density, motion).

Purely client-side: the panel is static HTML and settings.js applies and
persists choices (localStorage) by setting data-* attributes on <html>.
No Gradio events, no server state.
"""

from __future__ import annotations

import gradio as gr

# (accent key, display name, chip color) — chip shows the dark-mode accent hue.
ACCENTS = (
    ("teal", "Teal", "#5eead4"),
    ("sky", "Sky", "#7dd3fc"),
    ("amber", "Amber", "#fbbf24"),
    ("lime", "Lime", "#a3e635"),
    ("rose", "Rose", "#fb7185"),
    ("graphite", "Graphite", "#a8b0bc"),
)


def _seg(attr: str, options: tuple[tuple[str, str], ...], label: str) -> str:
    buttons = "".join(
        f'<button type="button" data-set-{attr}="{value}" '
        f'title="{label}: {text}" aria-pressed="false">{text}</button>'
        for value, text in options
    )
    return f'<div class="xwave-set-seg" role="group" aria-label="{label}">{buttons}</div>'


def render_settings_html() -> str:
    swatches = "".join(
        f'<button type="button" class="xwave-set-swatch" data-set-accent="{key}" '
        f'title="Accent: {name}" aria-pressed="false">'
        f'<span class="xwave-set-chip" style="--chip:{chip}"></span>{name}</button>'
        for key, name, chip in ACCENTS
    )
    return f"""
<div class="xwave-set-wrap" id="xwave-settings-root">
  <div class="xwave-set-head">
    <h1 class="xwave-set-title">Settings</h1>
    <p class="xwave-set-sub">Interface preferences. Applied instantly, stored in this browser.</p>
  </div>

  <section class="xwave-set-section">
    <h2 class="xwave-set-section-title">Appearance</h2>
    <p class="xwave-set-label">Mode</p>
    {_seg("mode", (("dark", "Dark"), ("light", "Light")), "Color mode")}
    <p class="xwave-set-label">Accent</p>
    <div class="xwave-set-grid">{swatches}</div>
  </section>

  <section class="xwave-set-section">
    <h2 class="xwave-set-section-title">Density</h2>
    {_seg("density", (("compact", "Compact"), ("comfortable", "Comfortable")), "Layout density")}
  </section>

  <section class="xwave-set-section">
    <h2 class="xwave-set-section-title">Motion</h2>
    {_seg("motion", (("full", "Full"), ("reduced", "Reduced")), "Interface motion")}
  </section>

  <div class="xwave-set-foot">
    <p class="xwave-set-note">Defaults: Dark mode, Teal accent, Compact density, Full motion.
    Canvas areas stay dark in Light mode so image judging stays consistent.</p>
    <button type="button" class="xwave-set-reset" id="xwave-set-reset"
      title="Reset all interface preferences to defaults">Reset to defaults</button>
  </div>
</div>
"""


def build_settings_tab() -> None:
    """Build the Settings tab inside the current Tab context."""
    gr.HTML(value=render_settings_html(), padding=False)
