from PIL import Image

from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import HISTORY_LIMIT, ComposerSession


def _session(tmp_path) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "sdxl_hyper": {"default_denoise": 0.3, "default_steps": 6},
            "paths": {
                "models_cache": "models",
                "layers_dir": "layers",
                "workspace_dir": "workspace",
            },
            "style": {
                "lora_dir": "loras",
                "embedding_dir": "embeddings",
            },
            "export": {"output_dir": "exports"},
        },
        root=tmp_path,
    )
    session = ComposerSession(config)
    session.doc.background = Image.new("RGB", (64, 64), "white")
    return session


def _with_layer(session, name: str = "A") -> ObjectLayer:
    layer = ObjectLayer(
        name=name,
        prompt=name.lower(),
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
    )
    session.doc.add_object(layer)
    return layer


def test_undo_restores_previous_pose(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)
    layer.transform.x = 100.0

    session.push_history()
    session.update_transform_by_id(layer.id, x=400.0)
    assert session.doc.find_by_id(layer.id).transform.x == 400.0

    session.undo()

    assert session.doc.find_by_id(layer.id).transform.x == 100.0


def test_redo_reapplies_the_undone_pose(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)
    layer.transform.x = 100.0

    session.push_history()
    session.update_transform_by_id(layer.id, x=400.0)
    session.undo()
    session.redo()

    assert session.doc.find_by_id(layer.id).transform.x == 400.0


def test_undo_restores_a_deleted_layer(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)

    session.delete_layer(layer.id)
    assert session.doc.objects == []

    session.undo()

    assert [o.id for o in session.doc.objects] == [layer.id]
    # The image came back with it, not just the id.
    assert session.doc.find_by_id(layer.id).image is not None


def test_undo_restores_draw_order(tmp_path):
    session = _session(tmp_path)
    first = _with_layer(session, "A")
    second = _with_layer(session, "B")
    original = [first.id, second.id]

    session.reorder_layers([second.id, first.id])
    assert [o.id for o in session.doc.objects] == [second.id, first.id]

    session.undo()

    assert [o.id for o in session.doc.objects] == original


def test_new_edit_clears_the_redo_branch(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)

    session.push_history()
    session.update_transform_by_id(layer.id, x=400.0)
    session.undo()
    assert session.can_redo

    session.push_history()  # a fresh edit diverges from the undone branch

    assert not session.can_redo


def test_undo_on_empty_history_is_a_no_op(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)
    layer.transform.x = 42.0

    assert not session.can_undo
    session.undo()

    assert session.doc.find_by_id(layer.id).transform.x == 42.0
    assert session.status == "Nothing to undo."


def test_history_is_bounded(tmp_path):
    session = _session(tmp_path)
    layer = _with_layer(session)

    for i in range(HISTORY_LIMIT + 25):
        session.push_history()
        session.update_transform_by_id(layer.id, x=float(i))

    assert len(session._undo_stack) == HISTORY_LIMIT


def test_snapshot_does_not_alias_the_live_transform(tmp_path):
    """Transforms mutate in place, so a snapshot must copy rather than point."""
    session = _session(tmp_path)
    layer = _with_layer(session)
    layer.transform.x = 10.0

    session.push_history()
    layer.transform.x = 999.0  # mutate the same object the snapshot came from

    session.undo()

    assert session.doc.find_by_id(layer.id).transform.x == 10.0
