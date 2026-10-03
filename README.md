# DLSS 5 Neural Rendering for Blender

Runs Blender's 3D viewport and still renders through NVIDIA's DLSS 5 neural
renderer (NGX feature 18).

**Status: 0.9.0, experimental.** Working and measured, but read
[Licensing of the runtime](#licensing-of-the-runtime) before you rely on it.

> Unofficial community project. Not affiliated with or endorsed by NVIDIA or the
> Blender Foundation. This repository contains **only the add-on source**; it does not
> include or redistribute any NVIDIA or third-party binaries.

---

## What it does

Two independent paths, both driving the same native worker:

**Still render** — takes the Render Result, pushes it through DLSS, and hands
back an image datablock plus an optional PNG. The preview is pre-compensated
for the scene's view transform, so it looks right without you having to switch
the whole scene to Standard.

**Live viewport** — takes the frame the viewport has already drawn, pushes it
through DLSS and draws the result back, rate-limited so Blender stays usable.

There are two possible sources for that frame, and which one you can use is the
single most important thing to understand about this add-on:

| Source | What it gives | What it costs |
|---|---|---|
| **Screen capture** (default) | Exactly what you are looking at, including accumulated Cycles samples | Always full resolution, so upscaling saves nothing; overlays are processed too |
| **Off-screen render** (experimental) | A clean frame with no overlays, at a resolution we choose — the only way to get real reconstruction and an actual speed win | Cannot render a Cycles Rendered viewport, and returned an entirely black frame in a real production file |

Off-screen is the architecturally better answer and it is implemented, but it
is not trustworthy yet, so Automatic means screen capture. If you switch to
off-screen, the panel tells you when the captured frame came back empty rather
than silently painting the viewport black.

## Requirements

| | |
|---|---|
| OS | Windows |
| GPU | NVIDIA RTX (Ampere / Ada / Blackwell) |
| Driver | 590 or newer |
| Blender | 4.2+ (developed and measured on 5.2 LTS) |
| Python deps | none — numpy ships with Blender |

A DLSS 5 runtime folder containing all five of:

```
nvngx.dll              native D3D12 worker (an executable despite the name)
dxgi.dll               ReShade carrier
renodx-dlss5.addon64   RenoDX DLSS 5 add-on
nvngx_dlss.dll         DLSS super resolution runtime
nvngx_dlssnr.dll       DLSS neural rendering runtime
```

Take all five from the *same* package. Mixing builds fails: the worker checks
the neural runtime's SHA-256, and NVIDIA's own Streamline SDK ships a
`nvngx_dlssnr.dll` that is a different build from the one the worker expects.

## Install

1. Copy `dlss5_blender.py` into
   `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
   (or Edit → Preferences → Add-ons → the ⌄ menu → Install from Disk).
2. Enable **DLSS 5 Neural Rendering** in Preferences → Add-ons.
3. Set **Runtime Folder** in the add-on preferences. The panel tells you which
   files are missing if the folder is incomplete.
4. Render Properties → DLSS 5 → Diagnostics → **Test Runtime**. This pushes a
   synthetic frame through the worker and reports the negotiated sizes.

If your antivirus quarantines `nvngx.dll`, add the runtime folder to its
exclusion list — it is an executable with a `.dll` extension, which reliably
trips heuristics.

## Using it

**Still render.** F12, then Render Properties → DLSS 5 → *Enhance Render
Result*. The report line gives the size change and how much the frame actually
moved (`delta`).

**Live viewport.** Expand *Live Viewport* and tick the header checkbox. The
panel then shows, live:

```
578x394 -> 1156x788
scene 1.8 ms + dlss 9.6 ms = 11.4 ms (88 fps)
40 frames, 10 history resets
delta 1.76 avg, 98 max, 87% pixels
```

`delta` is the honest answer to "is this doing anything": mean absolute
difference between the frame that went in and the frame that came back. **0.00
means the worker handed the frame straight back.**

## Reading the delta

Measured on this build, same settings, different inputs:

| Input | delta (mean) |
|---|---|
| Noisy path-traced frame (synthetic) | ~11.0 |
| Default Blender scene (flat grey cube) | ~1.8 |
| A real production head scene, live viewport | ~0.06 |

That last row is the honest one, and it is small. In a real file at a real
viewport size the frame-to-frame change is slight; the effect is clearer on a
single still render than in the live view. Judge it with Compare > Split before
deciding the live path is worth its cost.

The model reconstructs detail. Give it a clean, flat, low-detail image and
there is little for it to do, and the result will look unchanged — that is the
model behaving correctly, not a broken install. Give it a detailed or noisy
frame and the difference is large.

Use **Compare → Split** to judge it: left half untouched, right half processed,
white seam down the middle. Use **Compare → Tint** if you suspect the frame is
not reaching the screen at all — that is a different failure from DLSS doing
nothing.

## Settings, and what they actually do

Measured against the v4.0 runtime, not copied from upstream docs.

| Control | Effect |
|---|---|
| **Mode** | In the viewport, the resolution the scene is rendered at. DLAA = 1:1, Performance = half. On a still render it only enlarges the finished frame. |
| **Model Preset** | The largest single lever. Default/J/K stay soft; L and M rebuild noticeably more skin and hair texture. |
| **Style** | The only control that measurably moves tone. On a grey wedge: Default range 220, Cinematic 215 with blacks lifted to 22, Natural 194 — Natural is the *flattest*, despite the name. |
| **Intensity** | Below 1.0 fades back toward the source. Above 1.0 does nothing on this runtime. |
| **Local Tone** | Local tone mapping. Little effect on flat areas by construction. |
| **Local Structure** | Fine detail reconstruction. Raise it when the result looks too smooth. |
| **Skin Structure** | Only active with Automatic Mask on. |
| **NR Preset** | Nothing. #1, #2 and #3 return byte-identical frames. Hidden from the panel for that reason. |

Changing any of these restarts the worker (~4 s), because the worker takes
every control in a one-time header at session start. Compare and the still
render options do not restart it.

## Known limits

- **It is not a denoiser.** DLSS reconstructs; it does not remove
  path-tracing noise. The panel warns when Cycles viewport denoising is off,
  because that setting matters far more than anything here.
- **Blender still draws the viewport underneath.** A Python add-on cannot stop
  the region from being drawn, so the off-screen render is paid on top of
  Blender's own. Performance mode reduces *our* cost, not Blender's.
- **Camera motion resets temporal history.** Motion is read exactly from the
  region's perspective matrix rather than guessed with optical flow. Accurate,
  but it means quality builds up only while the view is still.
- **Timings depend on what else the GPU is doing.** The DLSS round trip
  measured 9.5 ms at 1154x786 in an idle scene and 47-63 ms in a production
  file with Cycles sampling at 1024 in the background. The update rate limit
  exists because of this.
- **The worker is a separate process.** Every frame crosses a pipe: about
  25 MB round trip at 1080p. Measured throughput up to ~1 GB/s, which is what
  keeps this viable at all.
- **Windows and NVIDIA only**, and the worker needs a real D3D12 device even
  though Blender is running OpenGL.

## Troubleshooting

| Symptom | Where to look |
|---|---|
| Panel says the runtime is incomplete | It names the missing files. All five, same package. |
| Live Viewport turns itself off | Panel shows the failure; full traceback in `dlss5_error.txt` in the runtime folder. |
| Nothing appears at all | Compare → Tint. No magenta means the draw is not landing, which is not a DLSS problem. |
| No visible change | Check `delta`. If it is non-zero the pipeline works and the input simply has little for the model to rebuild. |
| Worker will not start | Antivirus. `nvngx.dll` is an executable with a library extension. |

Diagnostics → **Show Status** prints a summary to the system console and reads
the worker's own log. Note that the worker logs per *feature creation*, not per
frame, so its evaluation count is proof that DLSS ran — never a frame count.
The panel's frame counter is the real one.

## Licensing of the runtime

The add-on source in this repository is MIT. The runtime it drives is not.

- `nvngx_dlss.dll` and `nvngx_dlssnr.dll` are **NVIDIA proprietary**, under
  NVIDIA's own licence.
- `nvngx.dll` (the worker) is a third-party build. The upstream project states
  its source is not included.
- `renodx-dlss5.addon64` is a community build distributed outside the main
  RenoDX repository.
- `dxgi.dll` is ReShade (BSD-3-Clause), which is the only unambiguous one.

The upstream project's own `BINARIES.md` tells users to obtain these
themselves and verify the terms. That is a strong hint that redistributing
them is not settled.

**Practical consequence:** this repository ships only source that *drives* a
runtime you obtain and install yourself. Do not open issues or PRs that add
those binaries; they will be closed. Bundling the runtime, or selling a product
that requires it, needs a lawyer rather than a README.

Related: NVIDIA's own DLSS Ray Reconstruction integration lands in Blender 5.3
(PR #153077, milestone 5.3, ships 10 November 2026) for Cycles viewport
denoising. That is native, supported, and free. It solves a narrower problem
than this add-on: viewport denoising, not neural rendering of arbitrary frames.

## Testing

A stubbed unit-test suite (`bpy` and `gpu` mocked: wire format sizes, header
field order, output size resolution, resize, delta measurement, sRGB round
trip) existed during development but is **not included yet**. Restoring it is a
good first contribution.

The GPU paths cannot be tested headlessly: Blender does not initialise the
`gpu` module in `--background`. They were verified by running a real GUI
session under script control and writing results to JSON — that is how the
off-screen pipeline numbers above were obtained.
