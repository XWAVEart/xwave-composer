"""core_ready guards the OUTPUT path against running while SDXL is unloaded.

Every call site reads it bare -- `if not session.core_ready` -- so it has to
stay a property. Turned into a plain method, each of those guards would test a
bound method object instead, which is always truthy, and every check would
silently stop firing while still looking correct. These tests pin that down.
"""

from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeSDXL:
    def __init__(self, ready: bool) -> None:
        self._ready = ready

    @property
    def ready(self) -> bool:
        return self._ready


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


def test_core_ready_is_a_property_not_a_method(tmp_path):
    session = _session(tmp_path)
    session.sdxl = FakeSDXL(ready=True)
    # Reading the attribute must give a bool, not a callable.
    assert session.core_ready is True
    assert not callable(session.core_ready)


def test_falsy_when_sdxl_unloaded(tmp_path):
    session = _session(tmp_path)
    session.sdxl = FakeSDXL(ready=False)
    assert session.core_ready is False
    # The exact expression used by the UI guards.
    assert (not session.core_ready) is True


def test_falsy_when_sdxl_absent(tmp_path):
    session = _session(tmp_path)
    session.sdxl = None
    assert session.core_ready is False


def test_run_output_keeps_accepted_output_when_sdxl_unloaded(tmp_path):
    """The guard's actual purpose: never overwrite a good OUTPUT with WORK."""
    session = _session(tmp_path)
    session.sdxl = FakeSDXL(ready=False)
    accepted = Image.new("RGB", (32, 32), "blue")
    session.last_output = accepted
    session.last_work = Image.new("RGB", (32, 32), "red")

    result = session.run_output(session.last_work)

    assert result is accepted
    assert session.last_output is accepted
    # And it reports the real reason rather than falling into the except branch.
    assert "not ready" in session.status.lower()
