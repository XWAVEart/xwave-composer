/**
 * Native title tooltips for xwave controls (esp. emoji-only buttons).
 * Re-applies after Gradio re-renders via MutationObserver.
 */
(function () {
  "use strict";

  /** @type {Array<[string, string]>} */
  const BY_SELECTOR = [
    [".xwave-gen-btn", "Generate this layer"],
    [".xwave-roll-seed", "Roll a random seed"],
    [".xwave-iso-white", "Isolation backdrop: white"],
    [".xwave-iso-black", "Isolation backdrop: black"],
    [".xwave-canvas-apply", "Apply selected canvas size"],
    [".xwave-export-btn", "Export accepted OUTPUT at 2× with SeedVR2"],
    [".xwave-lib-save-btn", "Save the current result to the library (no upscale)"],
    [".xwave-edit-preview", "Hover an object until it glows, then click to lift it onto a layer"],
    [".xwave-edit-canvas", "Hover an object until it glows, then click to lift it onto a layer"],
    ["#xwave-fx-apply", "Apply the selected glitch effect to the current layer"],
    [".xwave-add-btn", "Add a new object layer"],
    [".xwave-cutout-chk", "Cut out the subject after generate"],
    [".xwave-del", "Delete this layer"],
    [".xwave-flipbook-lock", "Flipbook: lock seed across all styles"],
    [".xwave-flipbook-dice", "Flipbook: new random seed per style"],
    [".xwave-flipbook-run", "Render OUTPUT under many styles and stitch a randomized MP4"],
    [".xwave-quality-fast", "Fast OUTPUT: lower denoise/steps for snappier live refine"],
    [".xwave-quality-hq", "Quality OUTPUT: higher denoise/steps so objects blend into the scene"],
    [".xwave-work-grid-type", "WORK alignment grid (overlay only — not in OUTPUT)"],
    [".xwave-work-grid-color", "WORK grid line color"],
  ];

  const BY_TEXT = {
    "⬜": "Isolation backdrop: white",
    "⬛": "Isolation backdrop: black",
    "▶️": "Generate this layer",
    "🎲": "Roll a random seed",
    "✂️": "Cut out the subject after generate",
    "Load models": "Load Flux, SDXL, and SAM2 into VRAM",
    "Free VRAM": "Unload Flux and optional Compose models (keeps SDXL). Switching to Infinite Canvas or Edit does this automatically.",
    BF16: "Full precision (bf16) — highest quality, most VRAM",
    MXFP8: "MXFP8 quantized — balanced quality and VRAM",
    NVFP4: "NVFP4 quantized — lowest VRAM, fastest on Blackwell",
    Fast: "Fast OUTPUT: lower denoise/steps for snappier live refine",
    Quality: "Quality OUTPUT: higher denoise/steps so objects blend into the scene",
    "+ Layer": "Add a new object layer",
    "Roll all": "Re-roll every generated layer (muted too). Imports stay put. Duplicates keep pose; pixels follow the original. Clears pending SAM, then re-rolls OUTPUT.",
    Apply: "Apply selected canvas size",
    Mute: "Mute this layer’s prompt in the final concat",
    Unmute: "Include this layer’s prompt in the final concat",
    Hide: "Hide this layer on WORK and OUTPUT (prompt still included)",
    Show: "Show this layer again on WORK and OUTPUT",
    Dup: "Duplicate the selected layer",
    "Re-cut": "Re-run cutout (rembg / SAM2) on this layer",
    Delete: "Delete the selected layer",
    "Import into layer": "Import the image into a new object layer",
    Load: "Load the selected base model",
    "Load LoRA": "Load a LoRA adapter onto the output model",
    "Load TI": "Load a textual inversion embedding",
    "Update OUTPUT now": "Re-run SDXL refine on the current composition",
    "Refine OUTPUT": "Refine OUTPUT with current strength/steps",
    "Export accepted OUTPUT 2× with SeedVR2":
      "Upscale the accepted OUTPUT 2× with SeedVR2",
    "Run style flipbook":
      "Render OUTPUT under many styles and stitch a randomized MP4",
    "Roll seed": "Roll a random seed",
    "🔒": "Flipbook: lock seed across styles",
  };

  function normalize(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  /** @param {HTMLElement} el */
  function tipForReset(el) {
    if (normalize(el.textContent) !== "Reset") return "";
    if (el.closest(".xwave-layers-head")) {
      return "Reset the whole workspace";
    }
    if (el.closest(".xwave-action-row")) {
      return "Reset this layer’s transform";
    }
    return "Reset";
  }

  /** @param {Element} el */
  function applyTip(el) {
    if (!(el instanceof HTMLElement)) return;
    const text = normalize(el.textContent);
    // Allow Mute/Unmute and Hide/Show to refresh when the label flips.
    const flipLabels = new Set(["Mute", "Unmute", "Hide", "Show"]);
    if (el.getAttribute("data-xwave-tip") === "1" && el.title && !flipLabels.has(text)) {
      return;
    }

    let tip = tipForReset(el);
    if (!tip) {
      for (const [sel, value] of BY_SELECTOR) {
        if (el.matches(sel)) {
          tip = value;
          break;
        }
      }
    }
    if (!tip) {
      tip = BY_TEXT[text] || "";
    }
    if (!tip) return;
    el.title = tip;
    if (!el.getAttribute("aria-label")) {
      el.setAttribute("aria-label", tip);
    }
    el.setAttribute("data-xwave-tip", "1");
  }

  function scan(root) {
    const scope =
      root && root.nodeType === 1 && root.querySelectorAll
        ? root
        : document.querySelector("gradio-app") || document;
    const selfMatch =
      scope.matches &&
      (scope.matches("button") ||
        scope.matches(".xwave-cutout-chk") ||
        scope.matches(".xwave-del"));
    if (selfMatch) applyTip(scope);
    const nodes = scope.querySelectorAll(
      "button, .xwave-cutout-chk, .xwave-del"
    );
    nodes.forEach(applyTip);
    // Gradio image toolbar icons (download / fullscreen)
    scope
      .querySelectorAll(
        'button[aria-label="Download"], button[aria-label="Fullscreen"]'
      )
      .forEach((btn) => {
        if (btn instanceof HTMLElement && !btn.title) {
          btn.title = btn.getAttribute("aria-label") || "";
        }
      });
  }

  function boot() {
    scan(document);
    const app = document.querySelector("gradio-app") || document.body;
    const obs = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType === 1) scan(node);
        }
        // Mute ↔ Unmute label changes
        if (m.type === "characterData" || m.type === "childList") {
          const target = m.target;
          const btn =
            target && target.nodeType === 1 && target.closest
              ? target.closest("button")
              : target && target.parentElement
                ? target.parentElement.closest("button")
                : null;
          if (btn instanceof HTMLElement) {
            btn.removeAttribute("data-xwave-tip");
            applyTip(btn);
          }
        }
      }
    });
    obs.observe(app, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
