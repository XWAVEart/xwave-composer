"""Disk-backed image library that persists between app sessions."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image

logger = logging.getLogger(__name__)

PAGE_SIZE = 24
THUMB_MAX_SIDE = 256
FULL_JPEG_QUALITY = 95
THUMB_JPEG_QUALITY = 70
INDEX_NAME = "index.json"
LibrarySource = Literal["compose", "infinite_canvas", "edit"]

_SOURCES = frozenset({"compose", "infinite_canvas", "edit"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clamp_name(name: str | None) -> str:
    text = " ".join(str(name or "").strip().split())
    return text[:80]


@dataclass(frozen=True)
class LibraryItem:
    id: str
    created: str
    source: str
    width: int
    height: int
    name: str
    full: str
    thumb: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created": self.created,
            "source": self.source,
            "width": int(self.width),
            "height": int(self.height),
            "name": self.name,
            "full": self.full,
            "thumb": self.thumb,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LibraryItem":
        return cls(
            id=str(raw.get("id") or ""),
            created=str(raw.get("created") or ""),
            source=str(raw.get("source") or "compose"),
            width=int(raw.get("width") or 0),
            height=int(raw.get("height") or 0),
            name=str(raw.get("name") or ""),
            full=str(raw.get("full") or ""),
            thumb=str(raw.get("thumb") or ""),
        )

    def label(self) -> str:
        if self.name.strip():
            return self.name.strip()
        src = {
            "compose": "Compose",
            "infinite_canvas": "Infinite Canvas",
            "edit": "Edit",
        }.get(self.source, self.source or "Library")
        return f"{src} · {self.width}×{self.height}"


@dataclass(frozen=True)
class LibraryPage:
    items: list[LibraryItem]
    offset: int
    limit: int
    total: int

    @property
    def page_index(self) -> int:
        if self.limit <= 0:
            return 0
        return self.offset // self.limit

    @property
    def page_count(self) -> int:
        if self.limit <= 0:
            return 1
        return max(1, (self.total + self.limit - 1) // self.limit)


class ImageLibrary:
    """Catalog of saved RGB stills under ``library/`` (full + thumbs + index)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.full_dir = self.root / "full"
        self.thumbs_dir = self.root / "thumbs"
        self.index_path = self.root / INDEX_NAME
        self._lock = threading.RLock()
        self._items: list[LibraryItem] = []
        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.index_path.exists():
            self._items = []
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read library index %s", self.index_path)
            self._items = []
            return
        entries = raw.get("items", raw) if isinstance(raw, dict) else raw
        items: list[LibraryItem] = []
        if isinstance(entries, list):
            for row in entries:
                if not isinstance(row, dict):
                    continue
                item = LibraryItem.from_dict(row)
                if item.id:
                    items.append(item)
        self._items = items

    def _dump(self) -> None:
        payload = {"items": [item.to_dict() for item in self._items]}
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, item_id: str) -> LibraryItem | None:
        key = str(item_id or "").strip()
        if not key:
            return None
        with self._lock:
            for item in self._items:
                if item.id == key:
                    return item
        return None

    def full_path(self, item: LibraryItem) -> Path:
        return (self.root / item.full).resolve()

    def thumb_path(self, item: LibraryItem) -> Path:
        return (self.root / item.thumb).resolve()

    def open_full(self, item_id: str) -> Image.Image | None:
        item = self.get(item_id)
        if item is None:
            return None
        path = self.full_path(item)
        if not path.is_file():
            return None
        with Image.open(path) as im:
            return im.convert("RGB")

    def page(self, offset: int = 0, limit: int = PAGE_SIZE) -> LibraryPage:
        limit = max(1, int(limit or PAGE_SIZE))
        with self._lock:
            total = len(self._items)
            if total == 0:
                return LibraryPage(items=[], offset=0, limit=limit, total=0)
            off = max(0, min(int(offset or 0), total - 1))
            off = (off // limit) * limit
            return LibraryPage(
                items=list(self._items[off : off + limit]),
                offset=off,
                limit=limit,
                total=total,
            )

    def add(
        self,
        image: Image.Image,
        source: str,
        name: str | None = None,
    ) -> LibraryItem:
        if image is None:
            raise ValueError("No image to save.")
        src = str(source or "compose").strip().lower()
        if src not in _SOURCES:
            src = "compose"
        rgb = image.convert("RGB")
        item_id = uuid.uuid4().hex[:12]
        full_rel = f"full/{item_id}.jpg"
        thumb_rel = f"thumbs/{item_id}.jpg"
        full_path = self.root / full_rel
        thumb_path = self.root / thumb_rel
        self._ensure_dirs()
        rgb.save(
            full_path,
            format="JPEG",
            quality=FULL_JPEG_QUALITY,
            subsampling=0,
            optimize=True,
        )
        thumb = _make_thumb(rgb)
        thumb.save(
            thumb_path,
            format="JPEG",
            quality=THUMB_JPEG_QUALITY,
            optimize=True,
        )
        item = LibraryItem(
            id=item_id,
            created=_utc_now(),
            source=src,
            width=int(rgb.width),
            height=int(rgb.height),
            name=_clamp_name(name),
            full=full_rel,
            thumb=thumb_rel,
        )
        with self._lock:
            self._items.insert(0, item)
            self._dump()
        return item

    def delete(self, item_id: str) -> bool:
        key = str(item_id or "").strip()
        if not key:
            return False
        with self._lock:
            found = next((item for item in self._items if item.id == key), None)
            if found is None:
                return False
            self._items = [item for item in self._items if item.id != key]
            self._dump()
        for path in (self.full_path(found), self.thumb_path(found)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not delete library file %s", path, exc_info=True)
        return True

    def set_name(self, item_id: str, name: str | None) -> LibraryItem | None:
        key = str(item_id or "").strip()
        if not key:
            return None
        cleaned = _clamp_name(name)
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id != key:
                    continue
                updated = LibraryItem(
                    id=item.id,
                    created=item.created,
                    source=item.source,
                    width=item.width,
                    height=item.height,
                    name=cleaned,
                    full=item.full,
                    thumb=item.thumb,
                )
                self._items[i] = updated
                self._dump()
                return updated
        return None


def _make_thumb(rgb: Image.Image) -> Image.Image:
    w, h = rgb.size
    scale = min(1.0, THUMB_MAX_SIDE / max(w, h, 1))
    if scale >= 1.0:
        return rgb.copy()
    return rgb.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.BILINEAR,
    )
