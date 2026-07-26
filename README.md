# xwave-composer

Multi-layer AI image composition for local use on an **NVIDIA RTX 5090**.

| Stage | Model | Role |
|--------|--------|------|
| Background + objects | **FLUX.2 [klein] 4B** (Apache 2.0) | Layer generation |
| Isolation | **rembg** (auto) / **SAM2** (click re-cut) | Transparent object layers |
| OUTPUT canvas | **SDXL base of your choice + Hyper LoRA** img2img | Near-real-time refine |
| Prompt rewrite (optional) | **Qwen2.5-VL-3B-Instruct** | Composition-aware rewrite (prompt + WORK image) |
| Export 2× | **SeedVR2** (or Real-ESRGAN / LANCZOS) | Final upscale |

Hybrid design: compose freely on the WORK canvas, refine with a fast SDXL Hyper pass that uses the WORK image as init.

## Requirements

- Linux, Python 3.10+
- NVIDIA GPU (designed for RTX 5090, ~32 GB VRAM)
- CUDA-capable PyTorch

## Setup

```bash
cd /home/will/xwave-composer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install the CUDA 13 build used by the RTX 5090 setup
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
# MSLK supplies the accelerated NVFP4 kernels and must match CUDA/PyTorch.
pip install --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -e .
# Install the export-only SeedVR2 CLI runtime. Weights download on first export.
python scripts/setup_seedvr2.py
```

Edit `config.yaml` for model IDs, paths, and defaults.

### Hugging Face access

Some models (Flux family, SDXL base) may require a Hugging Face token and license acceptance:

```bash
huggingface-cli login
```

Place optional style assets under:

- `models/loras/` — style LoRAs for OUTPUT  
- `models/embeddings/` — textual inversions for OUTPUT  

## Run

```bash
source .venv/bin/activate
python run.py
# or: xwave-composer
# or: python run.py --host 0.0.0.0 --port 7860
# optional: python run.py --preload --profile mxfp8
```

Open on this machine: `http://127.0.0.1:7860`  
Open on the LAN: `http://<this-host-ip>:7860`

## User guide

This guide uses ASD-STE100 Simplified Technical English.
UI labels are technical names. Keep the exact label when you operate the control.
Put the pointer on a control to see a short tooltip.

### 1. Start the application

1. Open a terminal.
2. Go to the project directory.
3. Activate the virtual environment.
4. Start the application:

```bash
source .venv/bin/activate
python run.py
# optional: python run.py --preload --profile mxfp8
```

5. Open `http://127.0.0.1:7860` in a browser.
6. For access on the local network, use `http://<host-ip>:7860`.

### 2. Load models and set compute

1. Press **Load models**.
2. Wait until the status shows that the models are ready.
3. The load includes Flux, SDXL Hyper, and the isolation models (rembg / SAM2).
4. To set the compute profile, press **BF16**, **MXFP8**, or **NVFP4** in the top bar.
5. Read the VRAM meter on the same row.
6. Press **Free VRAM** when you must release GPU memory after export or rewrite.
7. Set the profile also in the **Output model** panel.
8. Read the **Applied compute** field for the active path.

### 3. Set the canvas size

1. In the **Layers** column, open **Canvas size**.
2. Select an `aspect · pixels` value.
3. The list order is portrait, then **1:1**, then landscape.
4. Press **Apply**.

### 4. Work with layers

The **Background** layer exists when you start.
Object layers are cutout layers that you add.

1. Press **+ Layer** to add an object layer.
2. Select a layer card in the **Layers** list.
3. Selection is only from the list. A click on the WORK canvas does not select a layer.
4. Each card shows the layer prompt.
5. Drag a card to change the layer order.
6. Press **×** on a card to delete that layer.
7. Press **Reset** in the **Layers** header to clear the full workspace.

### 5. Generate a layer

1. Select a layer.
2. Type the layer prompt in the **Layer** panel.
3. Set the generate seed next to **▶️**. Use `-1` for a random seed.
4. For an object layer, set cutout-after-generate with **✂️** when the control is shown.
5. Press **⬜** or **⬛** to set the isolation backdrop when those controls are shown.
6. Press **▶️**, or press Enter in the prompt field.
7. Flux generates the layer image.

### 6. Cut out and re-cut an object

1. Select an object layer.
2. Select **rembg**, **SAM2**, or **none** under the layer actions.
3. Press **Re-cut** to cut the object again with the selected mode.
4. To re-cut with SAM2 by point, click the subject in the **Raw** preview.
5. Mode **none** keeps the full image with no cutout.

### 7. Import an image into a layer

1. Select the target layer.
2. Open **Import image**.
3. Drop or upload an image.
4. Set the cutout mode if you need a cutout.
5. Press **Import into layer**.

### 8. Transform layers

#### Background

1. Select the **Background** layer.
2. Set **Scale**, **Rot°**, **X**, and **Y**.
3. Set **Flip H** or **Flip V** if you need a mirror.

#### Object

1. Select an object layer.
2. Set **Scale** and **Rot°**.
3. Set **Flip H** or **Flip V** if you need a mirror.
4. Set **Opacity** (0 to 1).
5. Set **Feather** to soften the cutout edge inward (0 to 128).
6. Set **Blend**. The blend modes are:
   Normal, Multiply, Screen, Overlay, Soft Light, Hard Light,
   Add, Subtract, Difference, Darken, Lighten.

#### Layer actions

1. Press **Mute** to remove the layer prompt from the OUTPUT prompt build.
2. Press **Unmute** to include the prompt again.
3. Press **Dup** to copy the selected object layer.
4. Press **Reset** in the **Layer** panel to reset the transform of the selected layer.
5. Press **Delete** to remove the selected layer.

### 9. Use the WORK canvas

The WORK canvas shows the composed layers.

1. Drag a selected object to move it.
2. Drag a corner handle to stretch it.
3. Drag the orange handle to rotate it.
4. Change scale, rotation, and flip also from the **Layer** panel.
5. The WORK image updates when the composition changes.

### 10. Control live OUTPUT refine

The OUTPUT canvas shows an SDXL Hyper img2img refine of the WORK image.

1. Set **CFG**, **Denoise**, **Steps**, and **Eta** in the top-right bar.
2. The application can refresh OUTPUT after WORK or parameter changes.
3. Press **Update OUTPUT now** to run a refine immediately.
4. Live refine uses the current OUTPUT settings.
5. Raise **Steps** when you need higher quality.

### 11. Set style for OUTPUT

Style changes the OUTPUT stage only. Object layer prompts stay free of style text.

1. In the **Style** panel, select a family, or select **All families**.
2. Select a style preset.
3. The family values are **Art**, **Render**, **Photo**, **Sculpture**, and **Other**.
4. The preset sets prefix, suffix, negative prompt, CFG, Denoise, and Eta.
5. Edit those values after the preset loads if you must.
6. Set the OUTPUT seed. Press **🎲** for a new random seed.
7. Open **Prompt options** for more controls:
   1. Set **Prompt order** (prefix, prompts, and suffix order).
   2. Select **Manual style** to type your own prefix and suffix.
   3. Edit the **Negative prompt**.
8. Read **Built prompt** to see the text that SDXL receives.
9. Select **Edit built prompt (lock auto-build)** to edit that text by hand.
10. Clear the lock to rebuild the prompt from layers and style.

Style presets are in `XWAVE-COMPOSER-STYLES.csv`
(name, family, CFG, denoise, eta, prefix, suffix, negative).
The loader removes old embedding tokens such as `<3D>`.

### 12. Use LLM rewrite (optional)

LLM rewrite is off by default.

1. Select **LLM rewrite**.
2. The application loads **Qwen2.5-VL-3B-Instruct** when needed.
3. The model receives the built prompt and the WORK image.
4. The model rewrites the prompt for SDXL.
5. The rewrite keeps composition cues from the WORK image.
6. Read the LLM prompt field.
7. Select the LLM prompt lock if you must edit that text by hand.

### 13. Change the OUTPUT model and adapters

1. In **Output model**, select an SDXL base:
   SDXL Base, DreamShaper XL, Juggernaut XL v9, epiCRealism XL, or RealVisXL V5.
2. Or type a Hugging Face repo id or a CivitAI `.safetensors` link.
3. Press **Load**.
4. The Hyper LoRA loads again on the new base.
5. Open **Style adapters (LoRA / TI)** to load adapters:
   1. Type a LoRA path or Hugging Face id.
   2. Set **LoRA scale**.
   3. Press **Load LoRA**.
   4. Type a textual inversion path or id.
   5. Press **Load TI**.
6. Put local adapter files in `models/loras/` or `models/embeddings/`.

### 14. Refine and accept OUTPUT

1. Set **Refine strength** and **Refine steps** in the Final output area.
2. These values are separate from the live OUTPUT knobs.
3. Press **Refine OUTPUT**.
4. Review the result on the OUTPUT canvas.
5. Accept the image before you export.

### 15. Export 2× with SeedVR2

1. Open **SeedVR2 export settings** when you must change defaults.
2. Select a **Preset**:
   Quality — 7B FP16,
   Balanced — 7B FP8,
   Low VRAM — 3B FP8 + BlockSwap,
   or Fast — 3B FP8 + compile.
3. Or select a **Model** from the list.
4. Set **Color fidelity** (`lab`, `wavelet`, `wavelet_adaptive`, or `none`).
5. Set **Artifact reduction** and **Detail softness** when you must tune the export.
6. Set **Seed** for a fixed export seed.
7. Open **Advanced — VRAM / speed** to set BlockSwap, Swap I/O components, DiT offload, VAE offload, and Compile DiT.
8. Press **Export accepted OUTPUT 2× with SeedVR2**.
9. Flux and SDXL unload for the export.
10. SeedVR2 runs once, then stops.
11. The application writes a high-quality JPEG to `exports/`.
12. Read the export path and the export preview.
13. If SeedVR2 fails, the status shows the error.
14. Set `export.allow_fallback: true` in `config.yaml` only when a LANCZOS fallback is required.

### 16. Export a style flipbook

1. Refine OUTPUT first. Frame 0 is always the current OUTPUT image.
2. Open **Export Style Flipbook**.
3. Select mode **Styles** or **Seeds**.

#### Styles mode

1. Select one or more **Families**.
2. Set **Styles** count. Use `0` for all styles in the selected families.
3. Set **FPS** to 24, 30, 48, or 60.
4. Set **Frames per image** (hold length).
5. Press **🔒** to use the same seed for each style.
6. Press **🎲** to use a new seed for each style.
7. Press **Run style flipbook**.
8. The application re-refines WORK with each selected style.
9. CFG, Denoise, and Eta stay as set.
10. Only prefix, suffix, and negative prompt change per style.
11. The MP4 starts with the current OUTPUT. The other stills are shuffled.
12. Read the flipbook path and the video preview.

#### Seeds mode

1. Keep the current OUTPUT style and prompt settings.
2. Set **Frames** to the total still count. The minimum is 2.
3. Frame 0 is the current OUTPUT. The other frames use new seeds.
4. Set **FPS** and **Frames per image**.
5. Press **🔒** to use `seed+1`, `seed+2`, and so on.
6. Press **🎲** to use a random seed for each new frame.
7. Press **Run seed flipbook**.
8. The stills stay in generation order.

### 17. Recommended full sequence

1. Load models.
2. Set canvas size.
3. Generate the background.
4. Add object layers. Generate and cut out each object.
5. Arrange layers on the WORK canvas.
6. Select a style family and a style preset.
7. Adjust CFG, Denoise, Steps, and Eta.
8. Press **Update OUTPUT now** or wait for a live refresh.
9. Press **Refine OUTPUT** when the result is good.
10. Export 2× with SeedVR2, or run a flipbook.

## Project layout

```
xwave-composer/
  config.yaml
  XWAVE-COMPOSER-STYLES.csv  # style presets (family + prefix/suffix/negative + CFG/denoise/eta)
  run.py
  requirements.txt
  pyproject.toml
  src/xwave_composer/
    app.py                 # CLI entry (--preload loads models at startup)
    config.py
    canvas/                # layers + compositor
    models/                # Flux, SDXL Hyper, SAM2/rembg, LLM, upscaler
    pipeline/session.py    # session orchestration
    style/                 # CSV preset loader + prompt build
    ui/gradio_app.py       # Gradio UI
    ui/assets/             # app.css, work_canvas.js, tooltips.js
  workspace/               # runtime layer images
  exports/                 # 2× exports
  models/                  # local cache / LoRAs / embeddings
```

## Configuration notes

| Key | Meaning |
|-----|---------|
| `flux.model_id` | `black-forest-labs/FLUX.2-klein-4B` (Apache 2.0; 9B is non-commercial) |
| `optimization.profile` | `bf16`, `mxfp8` (default), or `nvfp4`; also selectable in the UI |
| `optimization.compile` | Regionally compile repeated diffusion blocks after quantization |
| `sdxl_hyper.base_model_id` | Startup SDXL base; swappable at runtime from the Output model menu |
| `isolation.preferred` | `sam2` or `rembg` (default backend; the Layer panel rembg / SAM2 / none selector overrides for Re-cut / import) |
| `llm.model_id` | Qwen2.5-VL-3B-Instruct (vision rewrite) |
| `style.presets_file` | CSV of style presets (default project root) |
| `export.upscaler` | `seedvr2` default; export-only subprocess (7B/3B, FP16/FP8) |
| `export.seedvr2_model` | DiT weights filename; UI presets cover Quality / Balanced / Low VRAM / Fast |
| `export.seedvr2_blocks_to_swap` | BlockSwap count (0=off); forces DiT offload to CPU when > 0 |
| `export.seedvr2_compile_dit` | `torch.compile` for DiT (slower first export) |
| `server.host` | `0.0.0.0` for LAN access |

SeedVR2 export failures are shown explicitly by default instead of silently substituting a LANCZOS resize. Set `export.allow_fallback: true` only if that fallback is desired.

## Performance (RTX 5090)

- The **Performance** selector can reload the diffusion pipelines in three modes:
  - **BF16** — no quantization; highest fidelity and broadest adapter compatibility.
  - **MXFP8 Balanced** — default; selective MXFP8 Flux plus conservative FP8 SDXL linear weights.
  - **NVFP4 Maximum** — selective NVFP4 Flux for maximum VRAM savings; SDXL remains on its safer FP8 path.
- Quantization only touches large eligible linear layers. Embeddings, normalization/output layers, SAM2, and rembg retain their quality-oriented precision.
- Repeated Flux/SDXL blocks use regional `torch.compile`. The first generation at a new canvas shape compiles kernels and is slower; later generations reuse them.
- If TorchAO, MSLK, a model source, or an adapter is incompatible, the loader discards the partial pipeline and reloads a clean BF16 pipeline. The Applied compute field reports the actual path.
- Keep Flux and SDXL Hyper loaded for interactive work.  
- Load the LLM only when rewrite is enabled.  
- Load the upscaler only on export.  
- Use the **Free VRAM** button after export/rewrite.  
- Live OUTPUT uses low steps (default 6) and moderate denoise (default 0.3). Raise steps for final quality.

## License

Application code: Apache-2.0.  
Third-party models keep their own licenses (see requirements doc for commercial notes).
