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

### Windows

Verified working on Windows 11 with an RTX 5090. No source changes are needed; the
differences are all in the environment.

```powershell
py -3.12 -m venv .venv                       # 3.13 is ahead of onnxruntime-gpu / rembg
.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pip install triton-windows   # see below
$env:HF_HOME = "D:\hf-cache"                 # the models want ~25 GB
.venv\Scripts\python run.py --profile bf16
```

- **Triton.** Windows PyTorch ships without Triton, so `optimization.compile` silently
  falls back to eager — the failure is caught and logged, not raised. `triton-windows`
  restores real compilation.
- **NVFP4 is Linux-only in practice.** The `mslk` wheel for `win_amd64` is
  `py3-none-any`-style: 78 Python files and no compiled artifact, whose `__init__` loads a
  hardcoded `mslk.so` that the wheel never ships. Use `bf16` or `mxfp8`.
- **VRAM.** With Flux, SDXL and SAM2 all resident, `bf16` peaks around 31.5 GB of 32 GB.
  It fits, but `mxfp8` is the sane working profile.
- **First load is slow.** Compiling Flux and SDXL takes roughly 18 minutes on a cold start.
  Later starts reuse the cache but only save about a minute of that, so if you restart
  often, consider `optimization.compile: false`.

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

## Two surfaces

| | |
|---|---|
| **`/studio`** | The product. Two panes, a sticker shelf, one prompt bar, Improve. Nothing else. |
| **`/`** | The full workspace: every sampler knob, compute profile, style preset, LoRA slot and export setting. |

The studio is plain HTML on the control API below, so it does not inherit the
Gradio layout and can be edited and reloaded without restarting the app. Expert
is one click away from it, and both drive the same session — so a change made in
one shows up in the other, and in the CLI.

## Simple and advanced mode

The app opens in **Simple** mode: a prompt, the two canvases, the layer tray, Improve,
and export. The sampler knobs, quantization profiles, style presets, output-model picker,
LoRA slots and SeedVR2 settings are hidden behind the **Advanced** toggle beside the title.
The choice is remembered. Toggling is instant and client-side, so it never re-renders the
canvas or interrupts what you are typing.

Everything below describes the full set of controls, i.e. Advanced mode.

## Keyboard and canvas

| | |
|---|---|
| click an object | selects it — transparent pixels fall through to whatever is behind |
| drag / corner handles / orange handle | move / stretch / rotate (`Shift` snaps rotation to 15°) |
| mouse wheel over the canvas | scale the selected sticker |
| arrows, `Shift`+arrows | nudge 1px / 10px (a burst is one undo step) |
| `[` / `]` | push the selected sticker back / bring it forward |
| `Ctrl`+`Z`, `Ctrl`+`Shift`+`Z` | undo / redo, including undeleting a layer with its cutout |
| `Delete` | remove the selected layer |
| `Ctrl`+`E` in the prompt field | enhance the prompt (studio) |
| double-click a sticker chip | remake it: its prompt loads into the bar, Make regenerates in place |

Pointer events are used throughout, so pen and touch work the same as a mouse.

## Workflow

1. **Load models** — Flux + SDXL Hyper + SAM2/rembg  
2. **Layers** — the Background layer exists from the start; **+ Layer** adds object layers.
   Select a layer, type its prompt in the Layer panel, press **⟡ Generate** (or Enter).
   Layer cards show the prompt directly — no separate titles.  
3. **Re-cut** — with an object selected, click the subject in the raw preview (SAM2) or use Re-cut (rembg)  
4. **WORK canvas** — drag to move, corner handles to stretch, orange handle to rotate; drag cards to reorder, × to delete  
5. **OUTPUT** — CFG / Denoise / Steps / Eta knobs sit directly below the OUTPUT canvas; style preset dropdown, negative prompt, optional LLM rewrite in the Style panel  
6. **Output model** — pick an SDXL base (SDXL Base, DreamShaper XL, Juggernaut XL v9, epiCRealism XL, RealVisXL V5) or paste any HF repo id / CivitAI `.safetensors` link, then press Load; the Hyper LoRA is re-applied on top  
7. **Final output** — use **Refine OUTPUT** to review an SDXL pass at the chosen strength/steps; once accepted, export that exact image 2× with SeedVR2. Flux/SDXL unload, SeedVR2 runs once, then exits and saves a high-quality JPEG to `exports/`

## Style presets

Styles live in `XWAVE-COMPOSER-STYLES.csv` (name, CFG, denoise, eta, prefix, suffix, negative).
Loading a preset builds the OUTPUT prompt as `prefix + [background + object prompts] + suffix`,
fills the negative prompt, and sets the CFG / denoise / eta knobs — all still editable afterwards.
Old embedding tokens like `<3D>` in the sheet are stripped automatically.
Object layers stay style-free; style only affects the OUTPUT stage.

## LLM rewrite (optional)

Off by default — the OUTPUT prompt is the plain concatenation. Ticking **LLM rewrite**
loads **Qwen2.5-VL-3B-Instruct** and sends it the concatenated prompt (style prefix/suffix
included) plus the current WORK canvas image. The model rewrites the prompt for SDXL and
transfers only the image's composition (framing, camera angle, subject placement).

## Studio niceties

- **Enhance (✦ / Ctrl+E)** — expands a naive prompt into a detailed one, in the same
  field, using the local vision LLM. Sticker mode keeps it to one isolated object;
  backdrop mode writes a full scene. Your original wording is one native Ctrl+Z away.
- **Backdrop library** — every backdrop you generate stays on the shelf as an asset.
  Click one to swap it in behind the composition (undoable); × removes it from the
  shelf without touching the composition.
- **Timeline** — a slider above the shelf scrubs through every state the composition
  has been in, one entry per step ("backdrop: …", "sticker: …", "moved…"). It is an
  append-only chronology: editing after scrubbing back appends rather than erasing,
  so nothing you made is ever lost. Scrubbing is itself undoable.
- **Look picker** — the style presets from the CSV, applied to the result with one
  dropdown.

## Improve loop

Press **✧ Improve** and the vision model looks at what you just made, says what is wrong
with it, and writes a better prompt. Optionally tell it what *you* think is wrong first —
your notes outrank its own reading of the image. The proposed prompt lands in an editable
box, so you can adjust the wording before pressing **Use this**.

In the studio the default target is **the scene** — the composed OUTPUT — because notes
are usually about the whole picture; switch to **Selected sticker** to iterate on one
element. The primary button, **✦ Improve it**, does the whole loop in one press: look,
critique, rewrite the prompt, apply it, and regenerate the image, with one Ctrl+Z
reverting prompt and pixels together. "Just look" keeps the two-step manual path.

Two modes:

- **Concept** (default) — judges the image against the original concept and rewrites the
  whole prompt. The concept is captured on the first pass and stays the yardstick, so
  repeated passes converge instead of drifting.
- **Edit mode** — compares the WORK composite against the refined OUTPUT and rewrites the
  *edit instructions* instead, so iterations refine the picture in front of you rather than
  starting a new one. Tick it once the composition is roughly right.

Earlier passes are fed back as context. The critic reuses the same Qwen2.5-VL weights as
the LLM rewrite, so it costs no extra VRAM, and unloads afterwards unless `llm.keep_loaded`.

## Command line

The app exposes a control API on the same port as the UI, so a script or an agent can
drive the same session a person is looking at. `xwave.py` needs nothing beyond the
standard library.

```bash
python xwave.py load                                   # models into VRAM
python xwave.py background "an empty misty lake at dawn"
python xwave.py sticker    "a small wooden rowboat"    # prints the new layer id
python xwave.py place --x 300 --y 700 --scale 0.4
python xwave.py refine
python xwave.py enhance "a rowboat"                    # naive idea -> rich prompt
python xwave.py improve --notes "the boat is too small" --fix   # critique + regenerate
python xwave.py timeline                               # every state so far
python xwave.py goto 3                                 # scrub back to step 3
python xwave.py save out.png
python xwave.py export                                 # 2x via SeedVR2

# or the whole composition in one command
python xwave.py compose "a misty lake at dawn" -s "a rowboat" -s "a heron"
```

`xwave.py state` prints what is on the canvas. Add `--json` to any command for the raw
envelope. The routes themselves are under `/control` — `GET /control/state`,
`POST /control/sticker`, `GET /control/render/output.png` and so on — and every mutating
route returns the full state, so a caller never needs a follow-up request.

## Project layout

```
xwave-composer/
  config.yaml
  XWAVE-COMPOSER-STYLES.csv  # style presets (prefix/suffix/negative + CFG/denoise/eta)
  run.py
  requirements.txt
  pyproject.toml
  xwave.py                 # command-line client for a running instance
  src/xwave_composer/
    app.py                 # CLI entry (--preload loads models at startup)
    config.py
    api/control.py         # HTTP control surface mounted on the Gradio app
    canvas/                # layers + compositor
    models/                # Flux, SDXL Hyper, SAM2/rembg, LLM, upscaler
    pipeline/session.py    # session orchestration, history, improve loop
    style/                 # CSV preset loader + prompt build
    ui/gradio_app.py       # Gradio UI + WORK canvas JS
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
| `isolation.preferred` | `sam2` or `rembg` (auto-isolation always uses rembg; SAM2 is for click re-cut) |
| `llm.model_id` | Qwen2.5-VL-3B-Instruct (vision rewrite) |
| `style.presets_file` | CSV of style presets (default project root) |
| `export.upscaler` | `seedvr2` default; export-only 7B FP16 runtime |
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
- Use the **Free optional VRAM** button after export/rewrite.  
- Live OUTPUT uses low steps (default 6) and moderate denoise (default 0.3). Raise steps for final quality.

## License

Application code: Apache-2.0.  
Third-party models keep their own licenses (see requirements doc for commercial notes).
