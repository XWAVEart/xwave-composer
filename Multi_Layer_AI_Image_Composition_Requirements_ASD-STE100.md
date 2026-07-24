# Multi-Layer AI Image Composition Application
## Requirements Specification in ASD-STE100 Simplified Technical English

**Document Status:** Updated Draft for Implementation  
**Target Platform:** Local network Gradio web application  
**Primary Hardware:** NVIDIA RTX 5090 (Blackwell architecture)  
**Date:** 2026-07-22  
**Update Note:** Hybrid architecture (Flux 2 klein + SDXL Hyper) and commercial licensing considerations added.

---

## 1. Purpose

This document gives the requirements for an application.  
The application lets AI artists compose images.  
The artists generate one background layer and multiple object layers.  
The artists place and transform the object layers on a WORK canvas.  
The application shows a refined version of the composition on an OUTPUT canvas.  
The OUTPUT canvas updates almost in real time when the user moves objects.  
The application exports a high-quality upscaled version of the OUTPUT image.

---

## 2. System Overview

The application runs on a local computer with an RTX 5090 GPU.  
The application serves a Gradio web interface over the local network.  
The system has two types of layers.

**Background layer**  
The background layer is one full-size image.  
The background layer has no transparency.

**Object layer**  
Each object layer contains one isolated object.  
The system generates the object on a plain background.  
The system then removes the background to create transparency.

The user places object layers on the WORK canvas.  
The user can move, resize, stretch, and rotate each object layer.  
A separate OUTPUT canvas shows a refined version of the full composition.  
The OUTPUT canvas uses a fast model.  
The OUTPUT canvas updates when the user changes the position or size of objects.

---

## 3. Hardware Requirements

The application must run on a computer with an NVIDIA RTX 5090 GPU.  
The application must use the GPU for all image generation and processing.  
The application must support local network access so that other devices on the same network can open the Gradio interface.

---

## 4. Software Requirements

The application must use Gradio as the main web interface.  
The application must allow custom frontend components inside Gradio.  
The custom components must support interactive multi-layer canvas operations.  
If a better framework than Gradio exists for complex canvas interaction, the implementer may evaluate it.  
The preferred solution remains Gradio with custom components.

The application must store user style phrases in a local JSON file.

---

## 5. Functional Requirements

### 5.1 Layer Generation

#### 5.1.1 Background Layer

The system must generate one background layer.  
The background layer uses **Flux 2 klein** (Apache 2.0 licensed version).  
The background layer has no transparency.  
The user supplies the prompt for the background layer.

#### 5.1.2 Object Layers

The system must generate object layers.  
Each object layer uses **Flux 2 klein**.  
The system must generate each object on a plain background.  
The isolation prompt must be a separate editable text input on the Gradio interface.  
This lets the user test different isolation prompts.

After generation, the system must remove the background of the object.  
The preferred method is Segment Anything Model 2 (SAM2).  
SAM2 must allow the user to click on the desired object in the generated image.  
If SAM2 is not available or fails, the system must fall back to a standard machine-learning background removal tool (for example rembg).

Object layers must contain only the basic object.  
Object layers must not contain baked-in artistic styles.

The user must be able to add, remove, and reorder object layers.  
The user must be able to select which object layer is currently controlled.

### 5.2 WORK Canvas

The WORK canvas is the interactive composition area.  
The default resolution of the WORK canvas is 1024 by 1024 pixels.  
The user must be able to choose a different canvas size and aspect ratio.

The user places object layers on top of the background layer on the WORK canvas.  
The user can perform these operations on any selected object layer:
- Move the object to any position
- Resize the object
- Stretch the object
- Rotate the object

The WORK canvas must update immediately when the user changes an object layer.

### 5.3 OUTPUT Canvas (Hybrid Architecture)

The OUTPUT canvas shows a refined version of the current WORK canvas composition.  

**Architecture decision:**  
The composed image from the WORK canvas is used as the init image for an image-to-image process.  
The fast model for this process is **SDXL Hyper** (or Hyper-SDXL / equivalent low-step distilled SDXL model).

This hybrid approach (Flux 2 klein for layer generation + SDXL Hyper for live refinement) prioritizes commercial licensing clarity while keeping near-real-time performance.

The OUTPUT process applies light denoising and style injection.  
Because the spatial composition already exists in the init image, the prompt adherence requirements on the SDXL model are reduced.

**User-controllable parameters for OUTPUT:**
- Denoise strength (user knob)
- Number of steps (user knob)

These controls must be available in the interface so the user can balance speed versus quality during live interaction and during final output.

**Prompt construction for OUTPUT**  
The system builds the prompt in this order:  
[style injection phrases] + [background prompt] + [object 1 prompt] + [object 2 prompt] + ...

The system must support two modes for prompt construction:  
1. Simple concatenation of the parts listed above.  
2. Optional rewrite by a small local language model.

Both modes must be available to the user.

**Recommended small LLM for prompt rewriting:**  
Use a strong instruction-following model in the 3B–8B range that runs efficiently alongside the diffusion models.  
Preferred options (in order):
- Qwen2.5-7B-Instruct or Qwen3-7B/8B (strong instruction following and format obedience)
- Llama 3.2 3B or Llama 3.3 8B
- Phi-4-mini (3.8B) when VRAM is constrained

The LLM must rewrite the concatenated prompt into a shorter, higher-signal version that fits SDXL’s limited context while preserving the most important style and content information.

**Style injection without heavy token cost:**  
The system must support loading of LoRAs and textual inversions (embeddings) that apply style.  
These style adapters must be usable on the OUTPUT stage so that style guidance does not consume large numbers of tokens in the text prompt.

The OUTPUT canvas must update as the user moves or changes object layers on the WORK canvas.

### 5.4 Style System

The system must provide a menu of stylization phrases.  
These phrases inject style into the OUTPUT prompt only.  
Object layers remain free of style.

In addition to text phrases, the system must support style LoRAs and textual inversions as described above.

The user must be able to add new stylization phrases.  
The user must be able to save the phrases for later use.  
The system stores the phrases in a local JSON file.

### 5.5 Export Function

The system must provide a final export button.  
When the user activates the export button, the system takes the current OUTPUT image.  
The system upscales the image by a factor of 2.  
The preferred upscaler is SeedVR2 or an equivalent high-quality diffusion-based upscaler.  
The upscale process does not need to be real-time.  
The goal of the upscale is to increase quality while keeping the composition almost unchanged.

The user must also be able to control the number of steps used in any final higher-quality refinement pass before or during export.

The system then saves the upscaled image to disk.

---

## 6. User Interface Requirements

The Gradio interface must contain these main areas:

- Background prompt input and generate button
- Object prompt input, isolation prompt input, and generate button
- SAM2 (or fallback) interactive selection area for object isolation
- Layer list that shows all object layers and allows selection, reordering, and deletion
- WORK canvas with interactive controls for move, resize, stretch, and rotate
- Style phrase menu with add and save functions
- Controls for loading style LoRAs and textual inversions
- Denoise strength slider for OUTPUT
- Steps control for OUTPUT and final export
- OUTPUT canvas that updates automatically
- Toggle between simple prompt concatenation and LLM prompt rewrite
- Export button

The interface must work over the local network.  
Any device on the same network must be able to open the Gradio page.

---

## 7. Performance Requirements

Background and object generation must be reasonably fast.  
The OUTPUT reprocessing must be as fast as possible.  
The OUTPUT update must feel almost real-time when the user drags or transforms objects.

All heavy computation must use the RTX 5090 GPU.

The small LLM used for prompt rewriting must run with low enough latency that it does not break the interactive feel of the OUTPUT canvas.

---

## 8. Licensing Considerations

The chosen models prioritize commercial usability:

- Flux 2 klein → Apache 2.0
- SDXL Hyper / Hyper-SDXL → generally commercial-friendly (OpenRAIL++ family)
- SeedVR2 → Apache 2.0
- Recommended LLMs (Qwen family) → typically Apache 2.0

This combination reduces legal risk if the application is later used commercially.

---

## 9. Future Options

The specification keeps the design open for these later improvements:

- Full automatic use of SAM2 without a separate isolation prompt
- Improved regional prompting or attention control for better multi-object handling
- Additional upscale factors beyond 2x
- Support for more advanced canvas interactions
- Caching of rewritten prompts to further reduce latency

---

## 10. Implementation Notes for the Builder LLM

Use technical names such as Flux 2 klein, SDXL Hyper, Hyper-SDXL, SAM2, SeedVR2, Gradio, and RTX 5090 as required.  
Keep all user-facing text and internal documentation in clear language.  
Prefer active voice and short sentences in any generated code comments or user help text.  

The primary goal is a practical tool that lets an artist compose with generated layers and receive a fast refined preview while maintaining a path to commercial use.

---

**End of Requirements Specification**
