<h1 align=center>iqView</h1>

<p align=center>iqView is a powerful fork of <b>qView</b> that integrates local, GPU-accelerated AI tools for rapid image editing without sacrificing minimalism.</p>

<h3 align=center>
    Retouch | Creative Fill | Isolate | Local AI
</h3>

<p align=center>
    <img alt="Screenshot" src="https://interversehq.com/qview/assets/img/screenshot3.png">
</p>

## ✨ What is iqView?
While original qView is a fantastic minimalist viewer, **iqView** expands it into a lightweight creative toolkit powered by local AI on your CUDA-capable GPU (developed on an NVIDIA RTX 3090).

## 🛠 AI Features
- **Object Removal (R)** — *LaMa inpainting*: mask distracting objects, text, or photobombers and they vanish in a fraction of a second. Brush and lasso masking tools included.
- **Creative Fill (G)** — *FLUX.2 klein*: mask a region, type a prompt, and generate photo-real replacement content in seconds.
- **Isolate (S)** — *SAM 3*: one-click subject cutout — removes the background and gives you a transparent PNG.
- **Zero Configuration**: the Python environment and AI models are set up and downloaded automatically on first use.
- **Idle Prefetch (optional)**: enable "Preload Retouch model when idle" in Settings to warm up LaMa a few seconds after you stop on an image, so `R` never has a cold-start delay. Off by default; only activates once the AI environment is already set up.
- **Privacy First**: all AI processing happens locally on your machine — your photos never leave your computer. (Gated models like FLUX and SAM 3 require a free Hugging Face token to download weights.)

## 🎮 Shortcuts
| Key | Action | Engine |
| --- | --- | --- |
| **R** | Toggle Retouch Mode (Cycle: Brush -> Lasso -> Off) / Apply if masked | LaMa |
| **G** | Creative Fill: mask + prompt | FLUX.2 klein |
| **S** | Isolate: remove the background | SAM 3 |
| **Enter** | Apply Retouch / Confirm | - |
| **Esc** | Cancel current AI job / Exit Mode | - |
| **Middle Click** | Apply Retouch (Quick Action) | - |
| **Right Click** | Cancel Retouch (Quick Action) | - |
| **[ / ]** | Adjust Brush Size | - |
| **1**–**5** | Star rating (press the same number again to clear) | - |
| **0** | Clear rating | - |
| **X** | Reject | - |
| **Ctrl+Z** | Undo edit (up to 5 steps) | - |
| **Ctrl+Shift+Z** | Redo edit | - |
| **Ctrl+S** | Save Image (offers to overwrite the original after an AI edit) | - |
| **Ctrl+Shift+S** | Save Image As... | - |
| **Ctrl+F** | Flip Image | - |

> **Note:** AI results are written to your temp folder until you save them. Use **Ctrl+S** to keep an edit permanently.

## 🌟 Culling
Rate while you browse: **1**–**5** for stars, **X** to reject, **0** to clear (pressing a
number the image already has clears it too). Ratings are written to XMP sidecar files next to
each image, using the same `xmp:Rating` convention Lightroom, Bridge, digiKam and Photo
Mechanic read — so culling here and finishing elsewhere works, and nothing is locked inside
iqView.

**Edit → Rate → Export Keepers...** copies every image at or above a star threshold into a
folder of your choosing, sidecars included.

## Installation
No Python installation required — iqView bootstraps its own portable Python environment (via [uv](https://github.com/astral-sh/uv)) the first time you use an AI feature. An NVIDIA GPU with CUDA is strongly recommended (CPU/MPS fallback works but is slower).
C++ code builds with **Qt 6 / CMake** on Windows, Linux, and macOS.

The image viewer itself needs no setup — the AI environment is only touched the first time you actually use Retouch, Creative Fill, or Isolate. Opening and browsing images works immediately, with no Hugging Face account required.

## 🖥 Platform Support

| Platform | Viewer | AI Features |
| --- | --- | --- |
| **Windows + NVIDIA** | ✅ | ✅ Verified — this is the primary dev environment (RTX 3090) |
| **Windows (no NVIDIA GPU)** | ✅ | ⚠️ Works via CPU fallback, but slow |
| **macOS** | ✅ (builds via CI) | ❓ Untested on real hardware, but should work — torch ships with MPS acceleration built in on Apple Silicon, and the setup flow no longer needs a pre-installed Python |
| **Linux + NVIDIA** | ✅ (builds via CI) | ❓ Untested on real hardware — CUDA path exists in code but hasn't been run on real Linux hardware |
| **Linux (no NVIDIA GPU)** | ✅ (builds via CI) | ⚠️ Works via CPU fallback, but slow |

If something goes wrong with the AI features, **Help → Export Debug Report...** writes a single
text file with your platform details, environment paths and the tail of every AI log. Attach it
to an issue — it deliberately excludes your Hugging Face token and any image data.

**Looking for testers on macOS and Linux.** The C++ viewer compiles and passes CI on all three platforms, but the Python AI pipeline (LaMa, FLUX.2, SAM 3) has only ever been run on Windows with an NVIDIA GPU. If you try iqView on Mac or Linux — even just confirming the viewer itself opens and browses images smoothly — please [open an issue](https://github.com/cyberhirsch/iqView/issues) and let us know what happened.

## 📜 Model Licenses
AI features use third-party models with their own license terms, downloaded on demand (never bundled):
- **LaMa** — [Apache 2.0](https://github.com/advimman/lama)
- **FLUX.2-klein** — [Apache 2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), cleared for commercial use
- **SAM 3** — [Meta's SAM License](https://huggingface.co/facebook/sam3) (gated — requires accepting Meta's terms with your own Hugging Face token before downloading)

---
*Based on the original [qView](https://github.com/jurplel/qView) by Jurplel.*
