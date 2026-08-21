"""HTML thumbnail grid for the image library."""

from __future__ import annotations

import html as html_lib

from xwave_composer.library.store import ImageLibrary, LibraryItem, LibraryPage, PAGE_SIZE


def _gradio_file_url(path) -> str:
    return f"/gradio_api/file={path.resolve()}"


def pager_label(page: LibraryPage) -> str:
    if page.total == 0:
        return "No images yet"
    start = page.offset + 1
    end = min(page.offset + len(page.items), page.total)
    return f"{start}–{end} of {page.total} · page {page.page_index + 1}/{page.page_count}"


def render_library_grid(
    page: LibraryPage,
    *,
    library_root,
    selected_id: str | None,
    context: str,
    compact: bool = False,
    empty: str = "Library is empty — save an image from Compose or Infinite Canvas.",
) -> str:
    """Server-rendered thumbnail grid. Clicks go through library.js."""
    ctx = html_lib.escape(context)
    extra = " xwave-lib-grid-compact" if compact else ""
    if not page.items:
        return (
            f'<div class="xwave-lib-grid{extra}" data-lib-context="{ctx}">'
            f'<p class="xwave-lib-empty">{html_lib.escape(empty)}</p></div>'
        )
    cards: list[str] = []
    for item in page.items:
        sel = " is-selected" if selected_id and item.id == selected_id else ""
        thumb = library_root / item.thumb
        url = _gradio_file_url(thumb) if thumb.is_file() else ""
        bg = (
            f' style="background-image:url(&quot;{html_lib.escape(url)}&quot;)"'
            if url
            else ""
        )
        lid = html_lib.escape(item.id)
        title = html_lib.escape(item.label())
        cards.append(
            f'<button type="button" class="xwave-lib-item{sel}" data-lib-id="{lid}" '
            f'title="{title}" aria-label="{title}">'
            f'<span class="xwave-lib-thumb"{bg}></span>'
            f'<span class="xwave-lib-caption">{title}</span>'
            f"</button>"
        )
    return (
        f'<div class="xwave-lib-grid{extra}" data-lib-context="{ctx}">'
        f"{''.join(cards)}</div>"
    )


def item_meta_text(item: LibraryItem | None) -> str:
    if item is None:
        return ""
    src = {
        "compose": "Compose",
        "infinite_canvas": "Infinite Canvas",
        "edit": "Edit",
    }.get(item.source, item.source or "Library")
    created = item.created.replace("T", " ").replace("+00:00", " UTC")
    name = item.name.strip() or "untitled"
    return f"{name} · {src} · {item.width}×{item.height} · {created}"


def pack_library_views(
    library: ImageLibrary,
    *,
    browse_offset: int = 0,
    compose_offset: int = 0,
    ic_offset: int = 0,
    edit_offset: int = 0,
    selected_id: str | None = None,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Browse, Compose, Infinite Canvas, and Edit picker grids + pagers."""
    browse = library.page(int(browse_offset or 0), PAGE_SIZE)
    compose = library.page(int(compose_offset or 0), PAGE_SIZE)
    ic = library.page(int(ic_offset or 0), PAGE_SIZE)
    edit = library.page(int(edit_offset or 0), PAGE_SIZE)
    return (
        render_library_grid(
            browse,
            library_root=library.root,
            selected_id=selected_id,
            context="browse",
        ),
        pager_label(browse),
        render_library_grid(
            compose,
            library_root=library.root,
            selected_id=selected_id,
            context="compose",
            compact=True,
            empty="Save an image to use it as a layer.",
        ),
        pager_label(compose),
        render_library_grid(
            ic,
            library_root=library.root,
            selected_id=selected_id,
            context="infinite",
            compact=True,
            empty="Save an image to paste it on the canvas.",
        ),
        pager_label(ic),
        render_library_grid(
            edit,
            library_root=library.root,
            selected_id=selected_id,
            context="edit",
            compact=True,
            empty="Save an image to open it in Edit.",
        ),
        pager_label(edit),
    )


def selected_item_outputs(library: ImageLibrary, item_id: str | None):
    """Preview path, name, meta, download path for the Library tab."""
    item = library.get(str(item_id or ""))
    if item is None:
        return None, "", "", None
    path = str(library.full_path(item))
    if not library.full_path(item).is_file():
        return None, item.name, item_meta_text(item), None
    return path, item.name, item_meta_text(item), path
