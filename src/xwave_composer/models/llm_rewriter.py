"""Optional vision-LLM prompt rewrite for the OUTPUT stage.

Qwen2.5-VL-3B-Instruct receives the concatenated prompt (style prefix +
content + suffix) plus the WORK canvas image, and rewrites the prompt while
transferring the image's composition. Off by default; the user enables it.

VRAM note: Flux + SDXL usually fill the GPU, so we load the VL model only for
the rewrite call (prefer GPU, fall back to CPU on OOM) and unload afterward
unless keep_loaded is set.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from xwave_composer.config import AppConfig
from xwave_composer.device import empty_cache, gpu_summary

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert SDXL prompt engineer.\n"
    "You will receive a user's text prompt and a reference image.\n"
    "Your job:\n"
    "1. Analyze only the **composition** of the reference image (framing, camera angle, "
    "subject placement, depth, balance, negative space).\n"
    "2. Rewrite the user's prompt into a strong, detailed SDXL prompt.\n"
    "3. Apply the composition from the reference image to the new prompt.\n"
    "Rules:\n"
    "- Keep the main subject and idea from the user's prompt.\n"
    "- Do NOT describe the content of the reference image.\n"
    "- Do NOT take any style from the reference image: ignore its colors, lighting, "
    "rendering, medium, texture, and mood entirely.\n"
    "- All style, medium, and aesthetic wording must come only from the user's text "
    "prompt; preserve their meaning while integrating them naturally.\n"
    "- Only transfer composition, camera, and framing from the image.\n"
    "- The result MUST be a substantial rewrite, not a copy of the input. Expand and "
    "reorganize it with concrete camera, framing, placement, depth, and spatial details "
    "observed in the reference image.\n"
    "- Keep the final prompt concise: one comma-separated line of 40–60 words so it fits "
    "SDXL's CLIP text window.\n"
    "- Make the prompt natural and effective for SDXL.\n"
    "- Output only the final rewritten prompt. Nothing else."
)


CRITIQUE_SYSTEM = (
    "You are an expert visual art critic and prompt engineer.\n"
    "You look at a generated image, judge it against the ORIGINAL CONCEPT, and return "
    "two things: a short critique of what did not work, and an improved prompt that "
    "fixes it without drifting from the concept.\n"
    "Process:\n"
    "1. The ORIGINAL CONCEPT is the source of truth. The prompt may have drifted; judge "
    "against the concept, not the prompt.\n"
    "2. Look for missing elements, wrong style, bad lighting or colour, weak composition, "
    "anatomy or perspective errors, and anything that contradicts the concept.\n"
    "3. Name the 3 to 5 biggest problems, briefly and concretely.\n"
    "4. Rewrite the prompt to stay true to the concept, be far more specific where the "
    "image failed, strengthen wording for what came out weak, and keep what worked.\n"
    "Return ONLY a JSON object, no markdown fence and no preamble:\n"
    '{"critique": "...", "improved_prompt": "..."}'
)

CRITIQUE_EDIT_SYSTEM = (
    "You are a senior art director reviewing an IMAGE EDIT.\n"
    "You get two images: the ORIGINAL first, then the RESULT after the edit.\n"
    "Judge whether the edit achieved the intent.\n"
    "Process:\n"
    "1. Did it preserve what should have been preserved?\n"
    "2. Did it change what was meant to change?\n"
    "3. Is the result a recognisable evolution of the original, not a replacement?\n"
    "4. Look for over-editing, under-editing, misread instructions, lost original "
    "elements, and badly blended additions.\n"
    "5. Name the 3 to 5 biggest problems with the EDIT specifically, not with general "
    "image quality.\n"
    "6. Rewrite the edit instructions to be more specific about what to change, more "
    "explicit about what to preserve, and to warn against what went wrong this time.\n"
    "Return ONLY a JSON object, no markdown fence and no preamble:\n"
    '{"critique": "...", "improved_prompt": "..."}'
)


@dataclass
class ImproveIteration:
    """One pass of the improve loop, kept as context for the next pass."""

    n: int
    critique: str
    improved_prompt: str
    user_notes: str = ""


@dataclass
class ImproveResult:
    ok: bool
    critique: str = ""
    improved_prompt: str = ""
    error: str = ""
    raw: str = field(default="", repr=False)


def _parse_critique_json(text: str) -> tuple[str, str] | None:
    """Pull (critique, improved_prompt) out of a model response.

    A 3B model does not reliably honour "JSON only", so accept a fenced block
    or an object embedded in prose before giving up.
    """
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    attempts = [candidate]
    brace = re.search(r"\{.*\}", candidate, re.DOTALL)
    if brace:
        attempts.append(brace.group(0))
    for attempt in attempts:
        try:
            data = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        critique = data.get("critique") or data.get("Critique")
        improved = (
            data.get("improved_prompt")
            or data.get("improvedPrompt")
            or data.get("improved")
        )
        if isinstance(critique, list):
            critique = "\n".join(str(item) for item in critique)
        if critique and improved:
            return str(critique).strip(), str(improved).strip()
    return None


class PromptRewriter:
    """Local vision LLM for composition-aware prompt rewriting."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.model: Any = None
        self.processor: Any = None
        self.model_id_loaded: str | None = None
        self._runtime_device: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self, force: bool = False, prefer_device: str | None = None) -> str:
        if self.model is not None and not force:
            return f"LLM already loaded: {self.model_id_loaded}"

        model_id = str(self.config.get("llm", "model_id", default="Qwen/Qwen2.5-VL-3B-Instruct"))
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        target = prefer_device or self.device
        logger.info("Loading vision prompt LLM: %s on %s", model_id, target)
        empty_cache()
        processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if target.startswith("cuda"):
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["device_map"] = target
        else:
            load_kwargs["torch_dtype"] = torch.float32

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            if not target.startswith("cuda"):
                model = model.to(target)
        except torch.OutOfMemoryError:
            self.unload()
            if target.startswith("cuda"):
                logger.warning("LLM GPU OOM — retrying on CPU")
                return self.load(force=True, prefer_device="cpu")
            raise

        model.eval()
        self.processor = processor
        self.model = model
        self.model_id_loaded = model_id
        self._runtime_device = target
        return f"Prompt LLM loaded: {model_id} ({target}) | {gpu_summary()}"

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def rewrite(
        self,
        concatenated_prompt: str,
        reference_image: Image.Image | None = None,
        *,
        content_prompt: str = "",
        style_name: str = "",
        style_prefix: str = "",
        style_suffix: str = "",
    ) -> str:
        """Rewrite the prompt, transferring composition from the WORK image."""
        text = concatenated_prompt.strip()
        if not text:
            return text

        self.ensure_loaded()
        assert self.model is not None and self.processor is not None

        max_new = int(self.config.get("llm", "max_new_tokens", default=160))
        temperature = float(self.config.get("llm", "temperature", default=0.0))
        do_sample = bool(self.config.get("llm", "do_sample", default=False))
        min_new = int(self.config.get("llm", "min_new_tokens", default=12))

        images = None
        image_part: list[dict[str, Any]] = []
        if reference_image is not None:
            # Keep the vision token budget small; composition survives downscale.
            ref = reference_image.convert("RGB")
            ref.thumbnail((512, 512), Image.Resampling.BILINEAR)
            image_part.append({"type": "image"})
            images = [ref]

        context = (
            f"CONTENT / SUBJECT:\n{content_prompt.strip() or '(derive from source prompt)'}\n\n"
            f"STYLE PRESET NAME:\n{style_name.strip() or '(none)'}\n\n"
            f"STYLE PREFIX:\n{style_prefix.strip() or '(none)'}\n\n"
            f"STYLE SUFFIX:\n{style_suffix.strip() or '(none)'}"
        )
        instructions = (
            "Write a new, composition-aware SDXL prompt using the WORK image. "
            "Synthesize the separately labeled subject and style fields; do not merely join "
            "or repeat them. Include explicit framing, camera angle, subject placement, "
            "foreground/background depth, and negative-space details inferred from the image. "
            "Return one newly worded, concise comma-separated line of 40 to 60 words."
        )

        best_changed = ""
        style_terms = [
            token.casefold()
            for token in re.findall(r"[A-Za-z]{5,}", f"{style_name} {style_prefix}")
            if token.casefold()
            not in {"style", "render", "image", "manual", "detailed", "beautiful"}
        ]
        for attempt in range(3):
            correction = ""
            if attempt:
                correction = (
                    "\n\nYour previous answer copied or failed to rewrite the source. "
                    "Try again with clearly different wording and at least three concrete "
                    "composition details. The final text must not equal the source prompt "
                    "and must stay between 40 and 60 words. Explicitly integrate recognizable "
                    "language from the supplied style preset."
                )
            user_content = image_part + [
                {"type": "text", "text": f"{instructions}{correction}\n\n{context}"}
            ]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat], images=images, return_tensors="pt", padding=True
            )
            device = next(self.model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            prompt_length = int(inputs["input_ids"].shape[-1])

            try:
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        min_new_tokens=min(min_new, max_new),
                        do_sample=do_sample,
                        **(
                            {
                                "temperature": max(temperature, 1e-5),
                                "top_p": 0.9,
                            }
                            if do_sample
                            else {}
                        ),
                    )
            except torch.OutOfMemoryError:
                # Generation OOM on GPU — retry whole rewrite on CPU once.
                if str(self._runtime_device or "").startswith("cuda"):
                    logger.warning("LLM generate OOM — reloading on CPU")
                    self.unload()
                    self.load(force=True, prefer_device="cpu")
                    return self.rewrite(
                        concatenated_prompt,
                        reference_image=reference_image,
                        content_prompt=content_prompt,
                        style_name=style_name,
                        style_prefix=style_prefix,
                        style_suffix=style_suffix,
                    )
                raise

            generated = out[0, prompt_length:]
            result = self.processor.decode(generated, skip_special_tokens=True).strip()
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1].strip()
            for prefix in ("Assistant:", "assistant:", "Rewritten prompt:", "Prompt:"):
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix) :].strip()
            word_count = len(result.split())
            copied_source = bool(
                result and text.casefold() in result.casefold()
            )
            style_ok = not style_terms or any(
                term in result.casefold() for term in style_terms
            )
            changed = bool(
                result
                and result.casefold() != text.casefold()
                and not copied_source
            )
            if changed and style_ok:
                best_changed = result
            if changed and style_ok and 35 <= word_count <= 65 and len(result) <= 420:
                logger.info(
                    "LLM rewrite changed prompt (%d → %d chars)", len(text), len(result)
                )
                return result
            logger.warning(
                "LLM rewrite attempt %d rejected (changed=%s, style=%s, words=%d, chars=%d)",
                attempt + 1,
                changed,
                style_ok,
                word_count,
                len(result),
            )

        if best_changed:
            # Last-resort CLIP-window guard. Prefer cutting at a complete
            # comma-delimited phrase instead of allowing SDXL to truncate in
            # the middle of the final composition detail.
            clipped = best_changed[:420]
            if len(best_changed) > 420 and "," in clipped:
                clipped = clipped.rsplit(",", 1)[0]
            clipped = clipped.strip(" ,")
            logger.info(
                "LLM rewrite accepted after length guard (%d → %d chars)",
                len(text),
                len(clipped),
            )
            return clipped
        if result and style_prefix:
            # A model can occasionally obey the composition request but omit
            # style language on every retry. Keep its visual analysis and
            # deterministically restore the preset prefix rather than falling
            # back to the unchanged concatenation.
            restored = f"{style_prefix.strip()} {result.strip()}".strip()
            return restored[:420].strip(" ,")
        return text

    def critique(
        self,
        *,
        original_concept: str,
        current_prompt: str,
        result_image: Image.Image,
        base_image: Image.Image | None = None,
        user_notes: str = "",
        history: list[ImproveIteration] | None = None,
        edit_mode: bool = False,
    ) -> ImproveResult:
        """Judge a generated image and return a critique plus a better prompt.

        In edit mode the model sees the image before the edit followed by the
        result, and rewrites the edit instructions rather than the whole prompt,
        so an iteration refines the picture instead of regenerating a new one.

        The loop shape (visible critique, optional user notes, accumulated
        iterations, and a mode switch between whole-prompt and edit) follows the
        operator's art-generator workflow in listeningrooms-vibesmithing.
        """
        if edit_mode and base_image is None:
            return ImproveResult(ok=False, error="Edit mode needs the pre-edit image.")

        try:
            self.ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - surface load failures to the caller
            return ImproveResult(ok=False, error=f"LLM load failed: {exc}")
        assert self.model is not None and self.processor is not None

        def _small(img: Image.Image) -> Image.Image:
            # Two images double the vision token cost; keep the budget sane.
            out = img.convert("RGB")
            out.thumbnail((512, 512), Image.Resampling.BILINEAR)
            return out

        images: list[Image.Image] = []
        parts: list[dict[str, Any]] = []
        if edit_mode and base_image is not None:
            images.append(_small(base_image))
            parts.append({"type": "image"})
        images.append(_small(result_image))
        parts.append({"type": "image"})

        sections = [
            f"ORIGINAL CONCEPT:\n{original_concept.strip() or '(none given)'}",
            f"CURRENT {'EDIT INSTRUCTIONS' if edit_mode else 'PROMPT'}:\n"
            f"{current_prompt.strip() or '(none)'}",
        ]
        if user_notes.strip():
            # User notes outrank the model's own read of the image.
            sections.append(
                "WHAT THE USER SAYS IS WRONG (address these specifically):\n"
                f"{user_notes.strip()}"
            )
        for it in history or []:
            sections.append(
                f"EARLIER PASS {it.n}:\ncritique: {it.critique[:300]}\n"
                f"produced: {it.improved_prompt[:200]}"
            )
        if edit_mode:
            sections.append(
                "The first image is the ORIGINAL, the second is the RESULT of the edit. "
                "Critique the edit and return improved edit instructions."
            )
        else:
            sections.append(
                "Critique the attached image against the concept and return an "
                "improved prompt."
            )

        messages = [
            {"role": "system", "content": CRITIQUE_EDIT_SYSTEM if edit_mode else CRITIQUE_SYSTEM},
            {"role": "user", "content": parts + [{"type": "text", "text": "\n\n".join(sections)}]},
        ]

        max_new = int(self.config.get("llm", "critique_max_new_tokens", default=420))
        raw = ""
        for attempt in range(2):
            chat = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[chat], images=images, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            prompt_length = int(inputs["input_ids"].shape[-1])
            try:
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                    )
            except torch.OutOfMemoryError:
                if str(self._runtime_device or "").startswith("cuda"):
                    logger.warning("Critique OOM on GPU — reloading the LLM on CPU")
                    self.unload()
                    self.load(force=True, prefer_device="cpu")
                    return self.critique(
                        original_concept=original_concept,
                        current_prompt=current_prompt,
                        result_image=result_image,
                        base_image=base_image,
                        user_notes=user_notes,
                        history=history,
                        edit_mode=edit_mode,
                    )
                return ImproveResult(ok=False, error="Out of memory during critique.")

            raw = self.processor.decode(out[0, prompt_length:], skip_special_tokens=True).strip()
            parsed = _parse_critique_json(raw)
            if parsed is not None:
                critique, improved = parsed
                return ImproveResult(
                    ok=True, critique=critique, improved_prompt=improved, raw=raw
                )
            if attempt == 0:
                # Nudge once, then accept a plain-text fallback rather than failing.
                messages.append({"role": "assistant", "content": raw[:400]})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "That was not valid JSON. Reply with ONLY this, filled in:\n"
                                    '{"critique": "...", "improved_prompt": "..."}'
                                ),
                            }
                        ],
                    }
                )

        # Both attempts missed the format. The text is usually still useful, so
        # hand it back as the critique instead of throwing the work away.
        return ImproveResult(
            ok=False,
            critique=raw,
            error="Model did not return usable JSON; showing its raw critique.",
            raw=raw,
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.model_id_loaded = None
        self._runtime_device = None
        empty_cache()
