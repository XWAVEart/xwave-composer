"""A three-minute button press must say which wait it is.

Cold, the vision model is ~7.5 GB off disk and enhance blocks for minutes;
warm it is seconds. Reporting the same "Enhancing…" for both makes the first
press indistinguishable from a crash -- which is exactly how it read in use.
"""

from xwave_composer.config import AppConfig
from xwave_composer.pipeline.session import ComposerSession


class FakeLLM:
    def __init__(self, ready: bool) -> None:
        self._ready = ready
        self.unloaded = False

    @property
    def ready(self) -> bool:
        return self._ready

    def enhance(self, text, kind="sticker"):
        self._ready = True
        return f"{text}, richly detailed"

    def unload(self):
        self.unloaded = True
        self._ready = False


class ExplodingLLM(FakeLLM):
    def enhance(self, text, kind="sticker"):
        raise RuntimeError("CUDA alloc failed")


def _session(tmp_path, keep_loaded: bool) -> ComposerSession:
    config = AppConfig(
        raw={
            "device": "cpu",
            "llm": {"keep_loaded": keep_loaded},
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


def test_cold_press_explains_the_long_wait(tmp_path):
    session = _session(tmp_path, keep_loaded=True)
    session.llm = FakeLLM(ready=False)

    # Capture what the UI would show while the load is in flight.
    seen = []
    original = FakeLLM.enhance

    def spy(self, text, kind="sticker"):
        seen.append(session.status)
        return original(self, text, kind)

    FakeLLM.enhance = spy
    try:
        session.enhance_prompt("a cactus")
    finally:
        FakeLLM.enhance = original

    assert seen, "enhance was never called"
    assert "minutes" in seen[0].lower(), seen[0]


def test_warm_press_does_not_warn_about_minutes(tmp_path):
    session = _session(tmp_path, keep_loaded=True)
    session.llm = FakeLLM(ready=True)

    seen = []
    original = FakeLLM.enhance

    def spy(self, text, kind="sticker"):
        seen.append(session.status)
        return original(self, text, kind)

    FakeLLM.enhance = spy
    try:
        session.enhance_prompt("a cactus")
    finally:
        FakeLLM.enhance = original

    assert "minutes" not in seen[0].lower(), seen[0]


def test_enhance_returns_the_expanded_text(tmp_path):
    session = _session(tmp_path, keep_loaded=True)
    session.llm = FakeLLM(ready=True)
    assert session.enhance_prompt("a cactus") == "a cactus, richly detailed"
    assert session.last_enhance_failed is False


def test_failure_returns_the_original_and_flags_it(tmp_path):
    """The client must be able to tell a crash from 'already good'."""
    session = _session(tmp_path, keep_loaded=True)
    session.llm = ExplodingLLM(ready=True)

    result = session.enhance_prompt("a cactus")

    assert result == "a cactus"
    assert session.last_enhance_failed is True


def test_failure_still_unloads_when_not_keeping_loaded(tmp_path):
    """A half-initialised model must not squat on VRAM Flux and SDXL need."""
    session = _session(tmp_path, keep_loaded=False)
    session.llm = ExplodingLLM(ready=True)

    session.enhance_prompt("a cactus")

    assert session.llm.unloaded is True


def test_blank_prompt_is_a_no_op(tmp_path):
    session = _session(tmp_path, keep_loaded=True)
    session.llm = ExplodingLLM(ready=True)  # would raise if it ran
    assert session.enhance_prompt("   ") == ""
