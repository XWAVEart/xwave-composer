"""Snapshots are full state, so every mutation MUST push history first.

The failure this pins down: an edit that skips push_history leaves the newest
undo entry describing a much older composition. Because _restore replaces
layers, images AND the backdrop, one Ctrl+Z then reverts everything since --
a backdrop plus a sticker collapses to a blank canvas in one press.
"""

from PIL import Image

from xwave_composer.canvas.layers import ObjectLayer
from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeFlux:
    """Returns flat images; enough to exercise the state transitions."""

    ready = True

    def generate_background(self, *, prompt, width, height, seed=None):
        return Image.new("RGB", (width, height), "navy")

    def generate_object(self, *, object_prompt, isolation_prompt, width, height, seed=None):
        return Image.new("RGB", (width, height), "orange")

    def generate(self, *, prompt, width, height, seed=None):
        return Image.new("RGB", (width, height), "orange")


class FakeIsolator:
    def isolate(self, image, click_xy=None, prefer=None):
        rgba = image.convert("RGBA")
        return rgba, "fake"


def _session(tmp_path) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "canvas": {"width": 64, "height": 64},
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
    session.flux = FakeFlux()
    session.isolator = FakeIsolator()
    return session


def test_undo_after_sticker_removes_only_the_sticker(tmp_path):
    """The regression: this used to blank the whole composition."""
    session = _session(tmp_path)
    session.generate_background("a beach")
    assert session.doc.background is not None

    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a crab")
    assert len(session.doc.objects) == 1

    session.undo()

    # The backdrop must survive; only the sticker's content is rolled back.
    assert session.doc.background is not None, "undo wiped the backdrop"
    assert session.doc.background_prompt == "a beach"


def test_undo_restores_the_previous_image_after_regeneration(tmp_path):
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("first")
    first_image = session.doc.find_by_id(layer.id).image

    session.generate_selected("second")
    assert session.doc.find_by_id(layer.id).image is not first_image

    session.undo()

    restored = session.doc.find_by_id(layer.id)
    assert restored.image is first_image
    assert restored.prompt == "first"


def test_generate_selected_records_the_cutout_choice(tmp_path):
    """improve --fix regenerates; forcing isolation would change the layer."""
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id

    session.generate_selected("a panel", isolate=False)
    assert session.doc.find_by_id(layer.id).cutout is False

    session.generate_selected("a crab", isolate=True)
    assert session.doc.find_by_id(layer.id).cutout is True


def test_cutout_flag_survives_undo(tmp_path):
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a panel", isolate=False)

    session.generate_selected("a crab", isolate=True)
    session.undo()

    assert session.doc.find_by_id(layer.id).cutout is False


def test_backdrop_generation_is_undoable_without_losing_stickers(tmp_path):
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a crab")

    session.doc.selected_id = "__bg__"
    session.generate_background("a beach")
    session.undo()

    assert [o.id for o in session.doc.objects] == [layer.id]
    assert session.doc.find_by_id(layer.id).image is not None


def test_active_backdrop_follows_undo(tmp_path):
    session = _session(tmp_path)
    session.generate_background("first")
    first_id = session.active_backdrop_id
    session.generate_background("second")
    assert session.active_backdrop_id != first_id

    session.undo()

    # The shelf highlight must match what the canvas shows.
    assert session.active_backdrop_id == first_id


def test_apply_improved_is_undoable(tmp_path):
    from xwave_composer.models.llm_rewriter import ImproveResult

    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a plain crab")

    session.last_improve = ImproveResult(ok=True, improved_prompt="a magnificent crab")
    session.last_improve_target = f"layer:{layer.id}"
    session.apply_improved()
    assert session.doc.find_by_id(layer.id).prompt == "a magnificent crab"

    session.undo()

    assert session.doc.find_by_id(layer.id).prompt == "a plain crab"


def test_scene_improve_locks_a_prompt_that_undo_releases(tmp_path):
    from xwave_composer.models.llm_rewriter import ImproveResult

    session = _session(tmp_path)
    session.generate_background("a beach")
    session.last_improve = ImproveResult(ok=True, improved_prompt="a golden beach at dusk")
    session.last_improve_target = "output"

    session.apply_improved()
    assert session.output_settings.prompt_locked is True
    assert session.output_settings.custom_prompt == "a golden beach at dusk"

    session.undo()

    assert session.output_settings.prompt_locked is False
    assert session.output_settings.custom_prompt == ""


def test_import_background_joins_the_library_and_bumps_rev(tmp_path):
    session = _session(tmp_path)
    session.doc.selected_id = "__bg__"
    rev_before = session.background_rev

    session.import_into_selected(Image.new("RGB", (64, 64), "teal"), cutout="none")

    assert session.background_rev > rev_before, "clients would keep a stale backdrop"
    assert len(session.backdrops) == 1
    assert session.backdrops[0]["id"] == session.active_backdrop_id
    assert "imported backdrop" in session.timeline_view()["entries"]


def test_reset_clears_improve_anchors(tmp_path):
    from xwave_composer.models.llm_rewriter import ImproveResult

    session = _session(tmp_path)
    session.improve_history["output"] = {"concept": "old concept", "iterations": []}
    session.last_improve = ImproveResult(ok=True, improved_prompt="x")
    session.last_improve_target = "output"

    session.reset_workspace()

    assert session.improve_history == {}
    assert session.last_improve is None
    assert session.last_improve_target == ""


def test_undo_leaves_the_timeline_cursor_describing_the_canvas(tmp_path):
    """The cursor must never point at a step the canvas is not showing.

    Undo can land between recorded steps — here the layer exists but its image
    has been rolled back to None — so the state is appended rather than
    mispointed at an entry that does not match.
    """
    session = _session(tmp_path)
    session.generate_background("a beach")
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a crab")

    session.undo()

    view = session.timeline_view()
    pos = view["pos"]
    assert 0 <= pos < len(view["entries"])
    # Scrubbing to where the cursor already points must be a no-op: that is
    # what "the cursor describes the canvas" means.
    before = (session.doc.background, session.doc.find_by_id(layer.id).image)
    session.timeline_goto(pos)
    after = (session.doc.background, session.doc.find_by_id(layer.id).image)
    assert before == after


def test_undo_that_lands_on_a_recorded_step_reuses_it(tmp_path):
    """No pointless entries when the restored state IS a recorded step."""
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("first")
    count_before = len(session.timeline_view()["entries"])

    session.generate_selected("second")
    session.undo()

    view = session.timeline_view()
    # "second" added one entry; the undo returned to the "first" state, which
    # is already on the log, so nothing new should be appended.
    assert len(view["entries"]) == count_before + 1
    assert view["entries"][view["pos"]].endswith("first")


def test_timeline_view_does_not_block_on_the_session_lock(tmp_path):
    """Status polling must stay responsive during a long generation."""
    import threading

    session = _session(tmp_path)
    holding = threading.Event()
    release = threading.Event()

    def hold_lock():
        with session._lock:
            holding.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold_lock, daemon=True)
    worker.start()
    assert holding.wait(timeout=5)
    try:
        # Would hang until the generation finished if this took the lock.
        done = threading.Event()

        def poll():
            session.timeline_view()
            done.set()

        threading.Thread(target=poll, daemon=True).start()
        assert done.wait(timeout=2), "timeline_view blocked on the session lock"
    finally:
        release.set()
        worker.join(timeout=5)


def test_coalesced_history_is_one_undo_entry(tmp_path):
    """Improve-and-regenerate must revert prompt and pixels in one press."""
    session = _session(tmp_path)
    layer = session.add_empty_object()
    session.doc.selected_id = layer.id
    session.generate_selected("a plain crab")
    original_prompt = session.doc.find_by_id(layer.id).prompt
    original_image = session.doc.find_by_id(layer.id).image
    depth_before = len(session._undo_stack)

    with session.coalesced_history():
        obj = session.doc.find_by_id(layer.id)
        obj.prompt = "a magnificent crab"
        session.generate_selected("a magnificent crab")

    assert len(session._undo_stack) == depth_before + 1, "should be one entry, not two"

    session.undo()

    restored = session.doc.find_by_id(layer.id)
    assert restored.prompt == original_prompt
    assert restored.image is original_image


def test_coalesced_history_depth_resets_on_error(tmp_path):
    session = _session(tmp_path)
    try:
        with session.coalesced_history():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # A leaked depth would silently disable undo for the rest of the session.
    assert session._history_depth == 0
    depth = len(session._undo_stack)
    session.push_history()
    assert len(session._undo_stack) == depth + 1
