# xwave-composer

Multi-layer AI image composition for local use on a high-VRAM **NVIDIA** GPU.
Developed and tuned for an **RTX 5090** (~32 GB VRAM).

| Stage | Model | Role |
|--------|--------|------|
| Background + objects | **FLUX.2 [klein] 4B** (Apache 2.0) | Layer generation |
| Isolation | **rembg** (auto) / **SAM2** (click re-cut) | Transparent object layers |
| OUTPUT canvas | **SDXL base of your choice + Hyper LoRA** img2img | Near-real-time refine |
| Prompt rewrite (optional) | **Qwen2.5-VL-3B-Instruct** | Composition-aware rewrite (prompt + WORK image) |
| Export 2× | **SeedVR2** (or Real-ESRGAN / LANCZOS) | Final upscale |

Hybrid design: compose freely on the WORK canvas, refine with a fast SDXL Hyper pass that uses the WORK image as init.

The UI has two **exclusive** tabs:

| Tab | Stack | Use case |
|-----|--------|----------|
| **Compose** | Flux + isolator + SDXL OUTPUT | Multi-layer scenes with live OUTPUT refine |
| **Infinite Canvas** | SDXL Hyper only | Large flat canvases built from overlapping region stamps |

Only one tab owns the GPU stack at a time. Switching tabs unloads the other mode’s models to free VRAM.

## Requirements

- Linux
- Python 3.10 or newer
- An NVIDIA GPU with enough VRAM for Flux + SDXL Hyper (about 24 GB or more recommended; developed on an RTX 5090 with ~32 GB)
- A CUDA-capable PyTorch build that matches your driver

## Setup

1. Clone or copy the repository, then enter the project directory.
2. Create and activate a virtual environment:

```bash
cd /path/to/xwave-composer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

3. Install PyTorch for your CUDA version. Use the index that matches your system
   (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).
   Example for a CUDA 13 wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

4. Install the project and its dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

5. Optional — for **NVFP4** compute, install MSLK. The MSLK build must match
   your CUDA and PyTorch versions. Example for a CUDA 13 nightly wheel:

```bash
pip install --pre mslk --index-url https://download.pytorch.org/whl/nightly/cu130
```

Skip this step if you use only **BF16** or **MXFP8**.

6. Optional — install the SeedVR2 CLI used for 2× export.
   Weights download on the first export:

```bash
python scripts/setup_seedvr2.py
```

7. Edit `config.yaml` for model IDs, paths, and defaults.

### Hugging Face access

Some models (Flux family, SDXL bases) may need a Hugging Face token and license acceptance:

```bash
huggingface-cli login
```

### Optional style assets

Place local adapters here if you use them:

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

Open locally: `http://127.0.0.1:7860`  
Open on the LAN: `http://<host-ip>:7860`

## User guide

UI labels match the controls in the app. Put the pointer on a control to see a short tooltip.

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
8. Press **Roll all** next to **Reset** / **+ Layer** to re-generate every Flux layer (muted included). Imported layers stay unchanged. Pending SAM is cleared. Duplicates keep their position, rotation, and flips while receiving the origin’s new pixels. OUTPUT seed is re-rolled and refined afterward.

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
3. Press **Hide** to remove the layer from WORK and OUTPUT canvases without muting its prompt. Press **Show** to bring it back.
4. Press **Dup** to copy the selected object layer.
5. Press **Reset** in the **Layer** panel to reset the transform of the selected layer.
6. Press **Delete** to remove the selected layer.

### 9. Use the WORK canvas

The WORK canvas shows the composed layers.

1. Drag a selected object to move it.
2. Drag a corner handle to stretch it.
3. Drag the orange handle to rotate it.
4. Change scale, rotation, and flip also from the **Layer** panel.
5. The WORK image updates when the composition changes.
6. Optional **Grid** overlays help align objects: Off, Center, Thirds, Golden, Quadrants, Diagonals, Safe margins.
7. Set grid **Color** to Black, White, Cyan, or Magenta.
8. Grids are a WORK view only. They do not change the composed image or the OUTPUT.

### 10. Control live OUTPUT refine

The OUTPUT canvas shows an SDXL Hyper img2img refine of the WORK image.

1. Press **Fast** or **Quality** in the top-right bar to load a denoise/steps pack.
2. **Quality** (default) uses higher denoise and steps so objects blend into the scene.
3. **Fast** uses lower denoise and steps for a snappier live refine.
4. You can still edit **CFG**, **Denoise**, **Steps**, and **Eta** after you select a pack.
5. On drag release, OUTPUT runs one full refine after a short debounce.
6. Press **Update OUTPUT now** to run a refine immediately.

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

### 18. Infinite Canvas Mode

Infinite Canvas is a separate tab for building **large flat images** from overlapping SDXL Hyper region stamps. It does not use the Compose WORK/OUTPUT layer stack, Flux, rembg/SAM2, or the LLM rewriter. It **does** share the same style preset CSV, SDXL base presets, and Hyper img2img pipeline as Compose.

Typical uses: outpainting beyond a fixed frame, tiled murals, panoramic scenes, and iterative “paint forward” workflows where each stamp extends or refines what came before.

#### Switching between Compose and Infinite Canvas

1. Use the **Compose** and **Infinite Canvas** tabs at the top of the app.
2. Switching to **Infinite Canvas** unloads Flux, the isolator, and the LLM. SDXL is kept when already loaded; otherwise press **Load SDXL** in that tab.
3. Switching back to **Compose** re-enables the Compose stack. Press **Load models** if Flux was unloaded.
4. Compose live OUTPUT and Infinite Canvas generation share one SDXL mutex — only one can run inference at a time.
5. Pending Compose OUTPUT refreshes are deferred while you stay on Infinite Canvas; they resume when you return to Compose.

#### Load SDXL

1. Open the **Infinite Canvas** tab.
2. In **Model**, choose an SDXL **Base** (same presets as Compose OUTPUT) or type a custom Hugging Face repo / CivitAI link.
3. Set **Performance** to **BF16**, **MXFP8 Balanced**, or **NVFP4 Maximum**, then press **Load SDXL**.
4. Read **Loaded** for the applied compute path (quantization + compile status).
5. Press **Free VRAM** to unload Flux and other Compose-only models without dropping SDXL.

#### Create, expand, and reset the canvas

1. Open **Canvas / expand**.
2. Set **W** and **H**, or pick a **Preset** (`2048×2048`, `3072×3072`, `4096×4096`, `2048×3072`, `3072×2048`) to fill those fields.
3. Press **Create**. Sizes snap to multiples of **64** (SDXL latent alignment).
4. To grow the document, set expansion pixels **L** / **R** / **T** / **B** and press **Expand**. Existing paint shifts to stay in place; the region box moves with the expansion.
5. Press **Reset** to clear all pixels and occupancy while keeping the current canvas size and region placement.
6. **Create** and **Reset** also reset the lazy global blueprint and world-anchored noise field.

Canvas presets and region sizes always stay on the 64 px grid.

#### Navigate the viewport

The main view is an interactive canvas (not a static image).

1. **Drag** on empty space to pan.
2. **Shift-drag** also pans.
3. Scroll the **mouse wheel** to zoom.
4. **Double-click** to fit the full canvas in view.
5. After **Create**, **Reset**, or **Expand**, the view recenters once to fit.

#### Place and size the generation region

The teal box is the **region stamp** — the area that will be written on the next **Generate region** pass. It may extend **outside** the canvas edge for outpainting; at least one 64 px cell always stays recoverable on-canvas.

1. **Drag** the teal box to move it. On release it snaps smoothly to the 64 px grid.
2. Open **Region numbers** for exact **X**, **Y**, **W**, **H**, or press **Apply** after editing.
3. Set **Aspect** (`1:1`, `4:3`, `3:2`, `16:9`, `9:16`, `3:4`, `2:3`) to resize the stamp while keeping proportions.
4. Press **Size +** / **Size −** to grow or shrink by 64 px on the short side.
5. Default stamp size is **1024** on the short side unless you change aspect or size.

#### Generate a region

1. Type a **Prompt** for this region.
2. Optionally pick a **Family** and **Style** (Infinite Canvas–local; same CSV as Compose).
3. Open **Prompt options** for **Order**, **Manual prefix/suffix**, and **Built prompt** preview.
4. Set **CFG**, **Steps**, **Eta**, and **Seed** (`-1` = random).
5. Press **Generate region**.

What happens under the hood:

- **SDF strength map + Differential Diffusion** on the shared Hyper SDXL stack.
- **Blank** areas get a full rewrite; **existing paint** fades in via the strength map so seams stay soft.
- A **dilated read window** (region + feather + overlap + context pad) feeds the UNet; large windows are tiled at native **1024²** views.
- **Persistent latents** store committed regions so re-encodes do not drift at boundaries.
- **World-anchored noise** keeps overlapping stamps consistent at edges.
- On a largely empty canvas, a low-res **blueprint** pass primes global composition before the first stamp (lazy; reused until you expand or reset).
- Status shows stage timing when available, e.g. `timing prep=…s model=…s commit=…s total=…s`.

Default generation knobs (adjustable under **Blending**): **Steps** 8, **CFG** 1.0, **Context pad** 128, **Read overlap** 256, **Feather** 96, **Falloff** 0.35, **Overpaint effect** 0.85.

#### Blending controls

Open **Blending** when seams or overpaint strength need tuning.

| Control | Role |
|---------|------|
| **Context pad** | Extra context beyond the stamp edge (64 px steps) |
| **Read overlap** | Wider read window for smoother transitions into existing paint |
| **Feather** | Spatial fade width at the stamp boundary in the strength map |
| **Falloff** | Shape of the SDF falloff inside the feather zone |
| **Overpaint effect** | Denoise strength when the stamp sits entirely on existing paint; edges still blend via feather/falloff |

Higher overlap and feather widen the read window and increase VRAM/time per stamp.

#### Undo

1. Press **Undo** to revert the last **Generate region** or **Run fused refine**.
2. One level of undo is kept (snapshot before each write).

#### Fused refine (full-canvas pass)

After placing several regions, run a global polish pass:

1. Open **Final refine**.
2. Set **Prompt source**:
   - **Selected style** — style prefix/suffix/negative only
   - **Custom prompt** — your text exactly, no style injection
   - **Custom prompt + selected style** — both
3. Set **Denoise**, **Tile** (512–1536), and **Overlap**.
4. Press **Run fused refine**.

This runs a **MultiDiffusion-style** tiled ε-fusion refine across the full canvas using the same SDXL Hyper stack. It snapshots undo state first, like region generation.

#### Export

1. Open **Export**.
2. Choose **PNG**, **JPEG**, or **WebP**.
3. Optionally enable **Upscale 2× (SeedVR2)** — uses the same export subprocess as Compose (Flux/SDXL unload during upscale).
4. Press **Export canvas**.
5. Files are written to `exports/` as `large_canvas_<WxH>_<hash>.<ext>` and offered for download.

Preview JPEGs for the live viewport are cached under `workspace/large_canvas_cache/`.

#### Infinite Canvas — recommended sequence

1. Open **Infinite Canvas** and **Load SDXL** with your preferred base and performance profile.
2. **Create** a canvas (or **Expand** an existing one).
3. Drag the teal region to the area you want, set aspect/size, and write a prompt (+ optional style).
4. Press **Generate region**; repeat for adjacent areas, outpainting past edges as needed.
5. Tune **Blending** if seams show; use **Undo** to step back one write.
6. **Run fused refine** for a final global polish.
7. **Export** PNG/JPEG/WebP (optionally 2× SeedVR2).

#### Performance notes (Infinite Canvas)

- First stamp on a cold canvas may include **blueprint** generation and **kernel compile** — later stamps at the same shape are faster.
- A 1024² stamp typically expands to a ~1664² read window (feather + overlap + context), which can require **multiple 1024² UNet views** per step.
- **CFG 1** (default) is recommended; CFG above 1 roughly doubles UNet work.
- **NVFP4 Maximum** applies real NVFP4 to SDXL; regional compile is skipped for that path. Batched UNet views run one at a time under NVFP4.
- Read the **Status** line for per-stage timing when tuning speed vs quality.

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
    pipeline/session.py    # Compose session orchestration
    pipeline/large_canvas.py  # Infinite Canvas session state
    pipeline/region_fill.py   # stamp fill + fused refine
    pipeline/infinite_canvas/ # strength / priming / fuse / photometric
    style/                 # CSV preset loader + prompt build
    ui/gradio_app.py       # Gradio UI (Compose + Infinite Canvas tabs)
    ui/large_canvas_tab.py # Infinite Canvas tab builder
    ui/assets/             # app.css, work_canvas.js, large_canvas.js, tooltips.js
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
| `sdxl_hyper.default_quality_mode` | `quality` or `fast` — top-bar pack for denoise/steps |
| `sdxl_hyper.quality_modes.*` | Per-mode `denoise`, `steps`, `preview_steps` (easy to retune) |
| `sdxl_hyper.preview_scale` | Optional low-res preview scale (unused by the live UI; kept for API/tests) |
| `sdxl_hyper.settle_debounce_s` | Unused by the live UI (single full refine after debounce) |
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
  - **NVFP4 Maximum** — selective NVFP4 on Flux and SDXL linear layers for maximum VRAM savings; SDXL skips regional compile under NVFP4 and uses single-view UNet batches.
- Quantization only touches large eligible linear layers. Embeddings, normalization/output layers, SAM2, and rembg retain their quality-oriented precision.
- Repeated Flux/SDXL blocks use regional `torch.compile`. The first generation at a new canvas shape compiles kernels and is slower; later generations reuse them.
- If TorchAO, MSLK, a model source, or an adapter is incompatible, the loader discards the partial pipeline and reloads a clean BF16 pipeline. The Applied compute field reports the actual path.
- Keep Flux and SDXL Hyper loaded for interactive work.  
- Load the LLM only when rewrite is enabled.  
- Load the upscaler only on export.  
- Use the **Free VRAM** button after export/rewrite.  
- Live OUTPUT uses low steps (default 6) and moderate denoise (default 0.3). Raise steps for final quality.

## License

Application code: [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — noncommercial use only. Contact XWAVEart for commercial licensing.

Third-party models keep their own licenses (see requirements doc for commercial notes).
