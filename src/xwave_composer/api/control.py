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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
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
    preview: bool = Field(
        default=False,
        description="Fast low-resolution pass for interactive feedback. Export and "
        "refine always re-render at full quality first, so a preview is never saved.",
    )


class ImproveBody(BaseModel):
    notes: str = ""
    edit_mode: bool = False
    target: str | None = Field(
        default=None, description='"layer", "output", or null to choose automatically.'
    )
    apply: bool = Field(default=False, description="Write the improved prompt straight back.")
    regenerate: bool = Field(
        default=False,
        description="Apply the improved prompt AND regenerate the target in the same "
        "call, so one press goes from critique to a new image. Implies apply.",
    )


class EnhanceBody(BaseModel):
    prompt: str
    kind: str = Field(default="sticker", description='"sticker" or "backdrop".')


class BackdropRef(BaseModel):
    id: str


class ReorderBody(BaseModel):
    ids: list[str] = Field(description="Layer ids bottom-to-top draw order.")


class StyleBody(BaseModel):
    name: str | None = Field(default=None, description="Preset name, or null to clear.")


class TimelineGoto(BaseModel):
    index: int


class ApplyBody(BaseModel):
    text: str | None = None


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
        "rev": obj.rev,
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
    timeline = session.timeline_view()
    return {
        "status": session.status,
        "canvas": {"width": doc.width, "height": doc.height},
        "background": {
            "prompt": doc.background_prompt,
            "has_image": doc.background is not None,
            "rev": session.background_rev,
        },
        "backdrops": [
            {
                "id": b["id"],
                "prompt": b["prompt"],
                "active": b["id"] == session.active_backdrop_id,
            }
            for b in session.backdrops
        ],
        "timeline": {
            "pos": timeline["pos"],
            "count": len(timeline["entries"]),
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

    # The studio page is plain HTML on top of these routes. It is read from
    # disk per request rather than cached, so the front end can be edited and
    # reloaded without restarting the app -- which otherwise means waiting for
    # the models to load again.
    studio_file = Path(__file__).with_name("studio.html")

    @router.get("/studio", response_class=HTMLResponse)
    def studio() -> HTMLResponse:
        try:
            return HTMLResponse(studio_file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"studio.html missing: {exc}") from None

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
        # generate_selected pushes its own history; pushing here too would put
        # two identical snapshots on the stack and make one Ctrl+Z do nothing.
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
        session.run_output(work, preview=body.preview)
        return {"state": _state(session)}

    @router.post("/improve")
    def improve(body: ImproveBody) -> dict[str, Any]:
        _require_models(session)
        # Report what was looked at: with no explicit target the selected layer
        # wins, which surprises anyone whose notes were about the whole picture.
        looked_at = session.improve_target_label(body.edit_mode, body.target)
        result = session.improve(
            user_notes=body.notes, edit_mode=body.edit_mode, target=body.target
        )
        applied = None
        if result.ok and body.regenerate:
            # In-situ: critique -> prompt -> regenerated image in one call.
            # Coalesced so the whole thing is ONE undo entry: separately, the
            # first Ctrl+Z would restore the old picture but keep the new
            # wording, which is a state the user never asked for.
            with session.coalesced_history():
                session.apply_improved()
                target_key = session.last_improve_target
                if target_key.startswith("layer:"):
                    layer_id = target_key.split(":", 1)[1]
                    obj = session.doc.find_by_id(layer_id)
                    if obj is not None:
                        session.select_layer_id(layer_id)
                        session.generate_selected(
                            obj.prompt,
                            isolation_prompt=obj.isolation_prompt,
                            # Honour how this layer was made. Defaulting to
                            # True would turn a full-image layer into a cutout.
                            isolate=obj.cutout,
                        )
                session.run_output(session.refresh_work())
            # apply_improved's message ends "press Generate", which is stale
            # once the regeneration has already happened in this same call.
            applied = "Improved and remade."
        elif result.ok and body.apply:
            # apply_improved pushes history, so the wording change is undoable
            # whether the target was a sticker prompt or the OUTPUT prompt.
            applied = session.apply_improved()
        return {
            "ok": result.ok,
            "looked_at": looked_at,
            "critique": result.critique,
            "improved_prompt": result.improved_prompt,
            "error": result.error,
            "applied": applied,
            "state": _state(session),
        }

    @router.post("/improve/apply")
    def improve_apply(body: ApplyBody | None = None) -> dict[str, Any]:
        # text lets the caller apply an edited version of the proposal, so the
        # model suggests and the person decides on the final wording.
        text = body.text if body is not None else None
        return {"message": session.apply_improved(text), "state": _state(session)}

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

    @router.post("/enhance")
    def enhance(body: EnhanceBody) -> dict[str, Any]:
        # Needs only the LLM, not the diffusion stack, so no _require_models.
        enhanced = session.enhance_prompt(body.prompt, kind=body.kind)
        # ok distinguishes "the model looked and left it alone" from "the model
        # never ran". Both return the original text, and blaming the user's
        # wording for an LLM crash is worse than saying nothing.
        return {
            "enhanced": enhanced,
            "changed": enhanced.strip() != body.prompt.strip(),
            "ok": not session.last_enhance_failed,
            "status": session.status,
        }

    @router.post("/backdrop")
    def switch_backdrop(body: BackdropRef) -> dict[str, Any]:
        # The library evicts past BACKDROP_LIMIT, so a client can legitimately
        # hold a stale id. Say so rather than returning 200 for a no-op.
        if not any(b["id"] == body.id for b in session.backdrops):
            raise HTTPException(status_code=404, detail=f"No backdrop {body.id}.")
        session.set_background(body.id)
        return {"state": _state(session)}

    @router.post("/backdrop/remove")
    def remove_backdrop(body: BackdropRef) -> dict[str, Any]:
        session.remove_backdrop(body.id)
        return {"state": _state(session)}

    @router.post("/reorder")
    def reorder(body: ReorderBody) -> dict[str, Any]:
        session.reorder_layers([str(i) for i in body.ids])
        return {"state": _state(session)}

    @router.get("/styles")
    def styles() -> dict[str, Any]:
        mgr = session.styles
        return {
            "styles": mgr.names() if mgr else [],
            "active": mgr.active_name if mgr else None,
        }

    @router.post("/style")
    def set_style(body: StyleBody) -> dict[str, Any]:
        session.apply_style_preset(body.name)
        return {"state": _state(session)}

    @router.get("/timeline")
    def timeline() -> dict[str, Any]:
        return session.timeline_view()

    @router.post("/timeline/goto")
    def timeline_goto(body: TimelineGoto) -> dict[str, Any]:
        session.timeline_goto(body.index)
        return {"state": _state(session)}

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

    @router.get("/render/background.png")
    def render_background() -> Response:
        """The backdrop alone. work.png is the full composite, so a client that
        draws its own layers needs this or it paints every sticker twice."""
        if session.doc.background is None:
            raise HTTPException(status_code=404, detail="No background yet.")
        return _png(session.doc.background)

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

    @router.get("/render/backdrop/{backdrop_id}.png")
    def render_backdrop(backdrop_id: str) -> Response:
        entry = next((b for b in session.backdrops if b["id"] == backdrop_id), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No backdrop {backdrop_id}.")
        return _png(entry["image"])

    app.include_router(router)

    # Also expose it at the short path. Gradio owns "/", so the studio lives
    # beside it rather than replacing it -- the expert UI stays one click away.
    @app.get("/studio", response_class=HTMLResponse)
    def studio_root() -> HTMLResponse:
        return studio()

    logger.info("Control API mounted at /control; studio UI at /studio")
    return router
