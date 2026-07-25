"""Timeline: an append-only log of composition states, scrubbable by index.

It never truncates on branch — editing after scrubbing back appends the new
state — so the log stays a complete chronology of everything the composition
has been. Snapshots share image references with the layers, so entries are
cheap and restoring one brings back the exact pixels.
"""

from PIL import Image

from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import TIMELINE_LIMIT, ComposerSession


def _session(tmp_path) -> ComposerSession:
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
    return ComposerSession(config)


def _layer(session, name="A") -> ObjectLayer:
    layer = ObjectLayer(
        name=name, prompt=name.lower(),
        image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
    )
    session.doc.add_object(layer)
    return layer


def test_seeded_with_a_start_entry(tmp_path):
    session = _session(tmp_path)
    view = session.timeline_view()
    assert view["entries"] == ["start"]
    assert view["pos"] == 0


def test_record_appends_and_advances(tmp_path):
    session = _session(tmp_path)
    _layer(session)
    session.record("added a layer")
    view = session.timeline_view()
    assert view["entries"] == ["start", "added a layer"]
    assert view["pos"] == 1


def test_goto_restores_the_recorded_pose(tmp_path):
    session = _session(tmp_path)
    layer = _layer(session)
    layer.transform.x = 100.0
    session.record("at 100")
    layer.transform.x = 400.0
    session.record("at 400")

    session.timeline_goto(1)

    assert session.doc.find_by_id(layer.id).transform.x == 100.0
    assert session.timeline_view()["pos"] == 1


def test_goto_start_restores_the_blank_canvas(tmp_path):
    session = _session(tmp_path)
    _layer(session)
    session.record("added")

    session.timeline_goto(0)

    assert session.doc.objects == []


def test_editing_after_scrub_appends_rather_than_truncates(tmp_path):
    """The log is a chronology, not a branch tree: nothing is thrown away."""
    session = _session(tmp_path)
    layer = _layer(session)
    layer.transform.x = 1.0
    session.record("one")
    layer.transform.x = 2.0
    session.record("two")

    session.timeline_goto(1)  # back to "one"
    layer = session.doc.find_by_id(layer.id)
    layer.transform.x = 3.0
    session.record("three")

    view = session.timeline_view()
    assert view["entries"] == ["start", "one", "two", "three"]
    assert view["pos"] == 3
    # "two" is still reachable even though "three" branched from "one".
    session.timeline_goto(2)
    assert session.doc.find_by_id(layer.id).transform.x == 2.0


def test_goto_is_undoable(tmp_path):
    session = _session(tmp_path)
    layer = _layer(session)
    layer.transform.x = 100.0
    session.record("at 100")
    layer.transform.x = 400.0

    session.timeline_goto(1)  # jumps the pose back to 100
    assert session.doc.find_by_id(layer.id).transform.x == 100.0

    session.undo()  # takes the scrub itself back

    assert session.doc.find_by_id(layer.id).transform.x == 400.0


def test_goto_restores_images_not_just_poses(tmp_path):
    """generate mutates obj.image in place; the snapshot must keep the old ref."""
    session = _session(tmp_path)
    layer = _layer(session)
    old_image = layer.image
    session.record("first image")

    layer.image = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
    layer.rev += 1
    session.record("second image")

    session.timeline_goto(1)

    assert session.doc.find_by_id(layer.id).image is old_image


def test_goto_restores_the_backdrop(tmp_path):
    session = _session(tmp_path)
    first = Image.new("RGB", (32, 32), "blue")
    session.doc.background = first
    session.doc.background_prompt = "blue"
    session.record("blue backdrop")

    session.doc.background = Image.new("RGB", (32, 32), "red")
    session.doc.background_prompt = "red"
    session.record("red backdrop")

    session.timeline_goto(1)

    assert session.doc.background is first
    assert session.doc.background_prompt == "blue"


def test_restore_bumps_revs_for_cache_busting(tmp_path):
    session = _session(tmp_path)
    layer = _layer(session)
    session.record("added")
    rev_before = layer.rev
    bg_rev_before = session.background_rev

    session.timeline_goto(0)
    session.timeline_goto(1)

    assert session.doc.find_by_id(layer.id).rev > rev_before
    assert session.background_rev > bg_rev_before


def test_timeline_is_bounded(tmp_path):
    session = _session(tmp_path)
    layer = _layer(session)
    for i in range(TIMELINE_LIMIT + 30):
        layer.transform.x = float(i)
        session.record(f"step {i}")
    assert len(session.timeline_view()["entries"]) == TIMELINE_LIMIT


def test_reset_clears_the_timeline(tmp_path):
    session = _session(tmp_path)
    _layer(session)
    session.record("added")
    session.reset_workspace()
    view = session.timeline_view()
    assert view["entries"] == ["start"]
    assert view["pos"] == 0
    assert session.backdrops == []
