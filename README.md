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

## Project layout

```
xwave-composer/
  config.yaml
  XWAVE-COMPOSER-STYLES.csv  # style presets (prefix/suffix/negative + CFG/denoise/eta)
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
