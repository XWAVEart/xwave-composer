"""A plain HTTP control surface over the composer session.

The Gradio UI and this API drive the same ComposerSession, which serialises
its own mutations, so a script and a person can work on one composition at
the same time. Routes are attached to the FastAPI app Gradio already serves,
so there is one process, one port, and one set of loaded models.

Every mutating route returns the same state envelope, so a caller never has
to follow up with a second request to find out what happened.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from xwave_composer.device import vram_stats
from xwave_composer.pipeline.session import ComposerSession

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- payloads
class GenerateBody(BaseModel):
    prompt: str
    seed: int = -1


class StickerBody(BaseModel):
    prompt: str
    isolation_prompt: str = ""
    cutout: bool = True
    seed: int = -1


class PlaceBody(BaseModel):
    id: str | None = None
    x: float | None = None
    y: float | None = None
    scale: float | None = Field(default=None, description="Sets both axes at once.")
    scale_x: float | None = None
    scale_y: float | None = None
    rotation: float | None = None
    opacity: float | None = None


class LayerRef(BaseModel):
    id: str | None = None


class RefineBody(BaseModel):
    denoise: float | None = None
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None


class ImproveBody(BaseModel):
    notes: str = ""
    edit_mode: bool = False
    apply: bool = Field(default=False, description="Write the improved prompt straight back.")


class ExportBody(BaseModel):
    refine_steps: int | None = None


# ------------------------------------------------------------------- shaping
def _layer_view(obj: Any) -> dict[str, Any]:
    t = obj.transform
    return {
        "id": obj.id,
        "name": obj.name,
        "prompt": obj.prompt,
        "prompt_enabled": obj.prompt_enabled,
        "has_image": obj.image is not None,
        "x": t.x,
        "y": t.y,
        "scale_x": t.scale_x,
        "scale_y": t.scale_y,
        "rotation": t.rotation,
        "opacity": t.opacity,
        "visible": t.visible,
    }


def _state(session: ComposerSession) -> dict[str, Any]:
    doc = session.doc
    vram = vram_stats()
    improve = session.last_improve
    return {
        "status": session.status,
        "canvas": {"width": doc.width, "height": doc.height},
        "background": {
            "prompt": doc.background_prompt,
            "has_image": doc.background is not None,
        },
        # Bottom-to-top draw order, which is also the order the compositor uses.
        "layers": [_layer_view(o) for o in doc.objects],
        "selected_id": doc.selected_id,
        "has_output": session.last_output is not None,
        "can_undo": session.can_undo,
        "can_redo": session.can_redo,
        "models_ready": session.core_ready,
        "vram": {"used_gb": round(vram[0], 2), "total_gb": round(vram[1], 2)} if vram else None,
        "output_settings": {
            "denoise": session.output_settings.denoise,
            "steps": session.output_settings.steps,
            "cfg": session.output_settings.cfg,
            "seed": session.output_settings.seed,
            "prompt_locked": session.output_settings.prompt_locked,
        },
        "improve": (
            {
                "ok": improve.ok,
                "critique": improve.critique,
                "improved_prompt": improve.improved_prompt,
                "error": improve.error,
                "target": session.last_improve_target,
                "iterations": len(session.improve_iterations()),
            }
            if improve is not None
            else None
        ),
    }


def _png(image: Image.Image) -> Response:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _require_models(session: ComposerSession) -> None:
    if not session.core_ready:
        raise HTTPException(
            status_code=409,
            detail="Models are not loaded. POST /control/load first (it takes a while).",
        )


# -------------------------------------------------------------------- routes
def register_control_api(app: Any, session: ComposerSession) -> APIRouter:
    """Attach the control routes to an existing FastAPI application."""
    router = APIRouter(prefix="/control", tags=["control"])

    @router.get("/state")
    def get_state() -> dict[str, Any]:
        return _state(session)

    @router.post("/load")
    def load_models() -> dict[str, Any]:
        message = session.preload_core()
        return {"message": message, "state": _state(session)}

    @router.post("/background")
    def make_background(body: GenerateBody) -> dict[str, Any]:
        _require_models(session)
        session.doc.selected_id = "__bg__"
        session.generate_background(body.prompt, seed=body.seed)
        return {"state": _state(session)}

    @router.post("/sticker")
    def make_sticker(body: StickerBody) -> dict[str, Any]:
        _require_models(session)
        # A sticker is a new layer every time; reusing the selection would
        # silently overwrite whatever the caller had selected.
        layer = session.add_empty_object()
        session.doc.selected_id = layer.id
        session.generate_selected(
            body.prompt,
            isolation_prompt=body.isolation_prompt,
            seed=body.seed,
            isolate=body.cutout,
        )
        return {"id": layer.id, "state": _state(session)}

    @router.post("/regenerate")
    def regenerate(body: StickerBody) -> dict[str, Any]:
        """Re-run generation into the selected layer, keeping its id and pose."""
        _require_models(session)
        session.push_history()
        session.generate_selected(
            body.prompt,
            isolation_prompt=body.isolation_prompt,
            seed=body.seed,
            isolate=body.cutout,
        )
        return {"state": _state(session)}

    @router.post("/place")
    def place(body: PlaceBody) -> dict[str, Any]:
        layer_id = body.id or session.doc.selected_id
        if not layer_id or layer_id == "__bg__":
            raise HTTPException(status_code=400, detail="No object layer selected.")
        if session.doc.find_by_id(layer_id) is None:
            raise HTTPException(status_code=404, detail=f"No layer {layer_id}.")
        session.push_history()
        scale_x = body.scale_x if body.scale_x is not None else body.scale
        scale_y = body.scale_y if body.scale_y is not None else body.scale
        session.update_transform_by_id(
            layer_id,
            x=body.x,
            y=body.y,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=body.rotation,
            opacity=body.opacity,
        )
        return {"state": _state(session)}

    @router.post("/select")
    def select(body: LayerRef) -> dict[str, Any]:
        if body.id == "__bg__":
            session.doc.selected_id = "__bg__"
        else:
            session.select_layer_id(body.id)
        return {"state": _state(session)}

    @router.post("/delete")
    def delete(body: LayerRef) -> dict[str, Any]:
        session.delete_layer(body.id)
        return {"state": _state(session)}

    @router.post("/undo")
    def undo() -> dict[str, Any]:
        session.undo()
        return {"state": _state(session)}

    @router.post("/redo")
    def redo() -> dict[str, Any]:
        session.redo()
        return {"state": _state(session)}

    @router.post("/refine")
    def refine(body: RefineBody) -> dict[str, Any]:
        _require_models(session)
        s = session.output_settings
        if body.denoise is not None:
            s.denoise = float(body.denoise)
        if body.steps is not None:
            s.steps = int(body.steps)
        if body.cfg is not None:
            s.cfg = float(body.cfg)
        if body.seed is not None:
            s.seed = int(body.seed)
        work = session.refresh_work()
        session.run_output(work)
        return {"state": _state(session)}

    @router.post("/improve")
    def improve(body: ImproveBody) -> dict[str, Any]:
        _require_models(session)
        result = session.improve(user_notes=body.notes, edit_mode=body.edit_mode)
        applied = session.apply_improved() if (body.apply and result.ok) else None
        return {
            "ok": result.ok,
            "critique": result.critique,
            "improved_prompt": result.improved_prompt,
            "error": result.error,
            "applied": applied,
            "state": _state(session),
        }

    @router.post("/improve/apply")
    def improve_apply() -> dict[str, Any]:
        return {"message": session.apply_improved(), "state": _state(session)}

    @router.get("/improve/history")
    def improve_history() -> dict[str, Any]:
        return {
            "target": session.last_improve_target,
            "iterations": [
                {
                    "n": it.n,
                    "critique": it.critique,
                    "improved_prompt": it.improved_prompt,
                    "user_notes": it.user_notes,
                }
                for it in session.improve_iterations()
            ],
        }

    @router.post("/export")
    def export(body: ExportBody) -> dict[str, Any]:
        _require_models(session)
        if session.last_output is None:
            raise HTTPException(status_code=400, detail="Refine an OUTPUT before exporting.")
        # export_final returns (image, path); SeedVR2 runs in a short-lived
        # subprocess after Flux/SDXL unload, so this call is slow by design.
        _, path = session.export_final(refine_steps=body.refine_steps)
        return {"path": str(path), "state": _state(session)}

    @router.post("/reset")
    def reset() -> dict[str, Any]:
        session.reset_workspace()
        return {"state": _state(session)}

    @router.get("/render/work.png")
    def render_work() -> Response:
        image = session.last_work or session.refresh_work()
        if image is None:
            raise HTTPException(status_code=404, detail="Nothing composed yet.")
        return _png(image)

    @router.get("/render/output.png")
    def render_output() -> Response:
        if session.last_output is None:
            raise HTTPException(status_code=404, detail="No OUTPUT yet.")
        return _png(session.last_output)

    @router.get("/render/layer/{layer_id}.png")
    def render_layer(layer_id: str) -> Response:
        obj = session.doc.find_by_id(layer_id)
        if obj is None or obj.image is None:
            raise HTTPException(status_code=404, detail=f"No image on layer {layer_id}.")
        return _png(obj.image)

    app.include_router(router)
    logger.info("Control API mounted at /control")
    return router
