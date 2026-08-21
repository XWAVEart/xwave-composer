"""Gradio UI for the image library tab."""

from __future__ import annotations

from typing import Any

import gradio as gr

from xwave_composer.library.render import pack_library_views, selected_item_outputs
from xwave_composer.library.store import ImageLibrary, PAGE_SIZE
from xwave_composer.pipeline.large_canvas import (
    IMPORT_CENTER,
    IMPORT_FIT,
    IMPORT_STAMP,
)

IC_IMPORT_PLACEMENTS = ("Fit canvas", "Paste centered", "Paste in stamp")
_PLACE_KEYS = {
    "Fit canvas": IMPORT_FIT,
    "Paste centered": IMPORT_CENTER,
    "Paste in stamp": IMPORT_STAMP,
}


def placement_key(label: str | None) -> str:
    return _PLACE_KEYS.get(str(label or "").strip(), IMPORT_FIT)


def shift_page(library: ImageLibrary, offset: int, delta_pages: int) -> int:
    page = library.page(int(offset or 0), PAGE_SIZE)
    nxt = page.offset + int(delta_pages) * page.limit
    nxt = max(0, nxt)
    if page.total:
        nxt = min(nxt, ((page.total - 1) // page.limit) * page.limit)
    return int(library.page(nxt, PAGE_SIZE).offset)


def build_library_tab(library: ImageLibrary) -> dict[str, Any]:
    """Build Library tab widgets inside the current Tab context."""
    views = pack_library_views(library)
    preview, name, meta, download = selected_item_outputs(library, None)

    browse_offset = gr.State(0)
    compose_offset = gr.State(0)
    ic_offset = gr.State(0)
    selected_id = gr.State("")

    with gr.Row(elem_classes=["xwave-row"], equal_height=False):
        with gr.Column(scale=3, min_width=380, elem_classes=["xwave-col"]):
            lib_html = gr.HTML(value=views[0], elem_classes=["xwave-lib-browse"])
            with gr.Row(elem_classes=["xwave-lib-pager"]):
                prev_btn = gr.Button("Prev", size="sm", scale=0, min_width=72)
                pager = gr.Textbox(
                    value=views[1],
                    show_label=False,
                    interactive=False,
                    container=False,
                    scale=1,
                )
                next_btn = gr.Button("Next", size="sm", scale=0, min_width=72)
        with gr.Column(scale=2, min_width=280, elem_classes=["xwave-col"]):
            preview_img = gr.Image(
                value=preview,
                label="Preview",
                type="filepath",
                format="jpeg",
                interactive=False,
                height=280,
                buttons=["fullscreen"],
            )
            meta_box = gr.Textbox(
                value=meta,
                label="Info",
                interactive=False,
                lines=2,
            )
            name_box = gr.Textbox(
                value=name,
                label="Name",
                placeholder="Optional name",
                lines=1,
            )
            with gr.Row():
                rename_btn = gr.Button("Save name", size="sm")
                download_file = gr.File(
                    value=download,
                    label="Download",
                    interactive=False,
                )
            with gr.Row():
                delete_btn = gr.Button("Delete", size="sm")
            gr.Markdown('<p class="xwave-section-head">Send to</p>')
            to_compose_btn = gr.Button("Import to Compose layer", size="sm")
            to_edit_btn = gr.Button("Import to Edit", size="sm")
            ic_place = gr.Radio(
                choices=list(IC_IMPORT_PLACEMENTS),
                value="Fit canvas",
                label="Infinite Canvas placement",
            )
            to_ic_btn = gr.Button("Import to Infinite Canvas", size="sm")

    return {
        "browse_offset": browse_offset,
        "compose_offset": compose_offset,
        "ic_offset": ic_offset,
        "selected_id": selected_id,
        "html": lib_html,
        "pager": pager,
        "preview": preview_img,
        "meta": meta_box,
        "name": name_box,
        "download": download_file,
        "prev": prev_btn,
        "next": next_btn,
        "rename": rename_btn,
        "delete": delete_btn,
        "to_compose": to_compose_btn,
        "to_edit": to_edit_btn,
        "to_ic": to_ic_btn,
        "ic_place": ic_place,
        "compose_pager_seed": views[3],
        "ic_pager_seed": views[5],
        "compose_html_seed": views[2],
        "ic_html_seed": views[4],
    }
