"""Contract tests for the /control HTTP surface, no models loaded.

Everything here runs against a CPU session with no diffusion stack, which is
exactly the state a client sees between server start and Load models — so the
guard behaviour (409s, 400s, 404s) is the thing under test, plus the routes
that must work without models: state, place, select, undo, timeline, reorder.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from xwave_composer.api.control import register_control_api
from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


def _make(tmp_path):
    config = AppConfig(
        raw={
            "device": "cpu",
            "paths": {
                "models_cache": "models",
                "layers_dir": "layers",
                "workspace_dir": "workspace",
            },
            "style": {"lora_dir": "loras", "embedding_dir": "embeddings"},
            "export": {"output_dir": "exports"},
        },
        root=tmp_path,
    )
    session = ComposerSession(config)
    app = FastAPI()
    register_control_api(app, session)
    return TestClient(app), session


def _add_layer(session, name="A"):
    layer = ObjectLayer(
        name=name, prompt=name.lower(),
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
    )
    session.doc.add_object(layer)
    return layer


def test_state_shape(tmp_path):
    client, _ = _make(tmp_path)
    s = client.get("/control/state").json()
    assert s["models_ready"] is False
    assert s["layers"] == []
    assert s["backdrops"] == []
    assert s["timeline"] == {"pos": 0, "count": 1}
    assert "rev" in s["background"]


def test_generation_routes_refuse_without_models(tmp_path):
    client, _ = _make(tmp_path)
    for path, body in [
        ("/control/sticker", {"prompt": "x"}),
        ("/control/background", {"prompt": "x"}),
        ("/control/refine", {}),
    ]:
        r = client.post(path, json=body)
        assert r.status_code == 409, path


def test_place_requires_a_layer(tmp_path):
    client, _ = _make(tmp_path)
    assert client.post("/control/place", json={"x": 1}).status_code == 400
    assert client.post("/control/place", json={"id": "nope", "x": 1}).status_code == 404


def test_place_moves_and_is_undoable(tmp_path):
    client, session = _make(tmp_path)
    layer = _add_layer(session)

    r = client.post("/control/place", json={"id": layer.id, "x": 300, "y": 200}).json()
    moved = r["state"]["layers"][0]
    assert (moved["x"], moved["y"]) == (300, 200)

    r = client.post("/control/undo").json()
    back = r["state"]["layers"][0]
    assert (back["x"], back["y"]) == (512, 512)


def test_reorder_route(tmp_path):
    client, session = _make(tmp_path)
    a = _add_layer(session, "A")
    b = _add_layer(session, "B")

    r = client.post("/control/reorder", json={"ids": [b.id, a.id]}).json()

    assert [l["id"] for l in r["state"]["layers"]] == [b.id, a.id]


def test_timeline_routes(tmp_path):
    client, session = _make(tmp_path)
    layer = _add_layer(session)
    layer.transform.x = 100.0
    session.record("at 100")
    layer.transform.x = 400.0
    session.record("at 400")

    tl = client.get("/control/timeline").json()
    assert tl["entries"] == ["start", "at 100", "at 400"]

    r = client.post("/control/timeline/goto", json={"index": 1}).json()
    assert r["state"]["layers"][0]["x"] == 100
    assert r["state"]["timeline"]["pos"] == 1


def test_layer_render_includes_rev_in_state(tmp_path):
    client, session = _make(tmp_path)
    layer = _add_layer(session)
    s = client.get("/control/state").json()
    assert s["layers"][0]["rev"] == layer.rev

    png = client.get(f"/control/render/layer/{layer.id}.png")
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"


def test_backdrop_library_switch(tmp_path):
    client, session = _make(tmp_path)
    blue = Image.new("RGB", (32, 32), "blue")
    red = Image.new("RGB", (32, 32), "red")
    session.backdrops = [
        {"id": "aaa", "prompt": "blue", "image": blue},
        {"id": "bbb", "prompt": "red", "image": red},
    ]
    session.active_backdrop_id = "bbb"
    session.doc.background = red

    r = client.post("/control/backdrop", json={"id": "aaa"}).json()

    assert session.doc.background is blue
    active = [b for b in r["state"]["backdrops"] if b["active"]]
    assert [b["id"] for b in active] == ["aaa"]
    # And the library render route serves the non-active entry too.
    assert client.get("/control/render/backdrop/bbb.png").status_code == 200
    assert client.get("/control/render/backdrop/zzz.png").status_code == 404


def test_styles_routes(tmp_path):
    client, _ = _make(tmp_path)
    listing = client.get("/control/styles").json()
    assert "styles" in listing and "active" in listing
    # No CSV in the temp workspace, so the list is empty — but applying a
    # missing name must not error, just clear.
    r = client.post("/control/style", json={"name": None})
    assert r.status_code == 200
