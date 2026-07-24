"""Optional vision-LLM prompt rewrite for the OUTPUT stage.

Qwen2.5-VL-3B-Instruct receives the concatenated prompt (style prefix +
content + suffix) plus the WORK canvas image, and rewrites the prompt while
transferring the image's composition. Off by default; the user enables it.

VRAM note: Flux + SDXL usually fill the GPU, so we load the VL model only for
the rewrite call (prefer GPU, fall back to CPU on OOM) and unload afterward
unless keep_loaded is set.
"""

from __future__ import annotations

import logging
import re
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

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.model_id_loaded = None
        self._runtime_device = None
        empty_cache()
