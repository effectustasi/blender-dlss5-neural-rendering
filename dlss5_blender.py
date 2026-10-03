bl_info = {
    "name": "DLSS 5 Neural Rendering",
    "author": "effectustasi",
    "version": (0, 9, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Render > DLSS 5 Neural Rendering",
    "description": (
        "Run the viewport and still renders through NVIDIA's DLSS 5 neural "
        "renderer (NGX feature 18). Requires a separately obtained runtime"
    ),
    "warning": "Windows and NVIDIA RTX only; needs a DLSS 5 runtime you supply",
    "category": "Render",
}

# ---------------------------------------------------------------------------
# How this works
#
# The neural work happens in a native D3D12 worker process that hosts ReShade
# plus the RenoDX DLSS 5 add-on and evaluates NGX feature 18. This file speaks
# its version-4 stdin/stdout protocol directly, so nothing outside Blender is
# needed: numpy ships with Blender, and the torch/OpenCV/PyAV dependencies of
# the upstream ComfyUI nodes are only required for video, which this does not
# do.
#
# Two paths feed that worker:
#
#   Still render  Render Result -> save_render (applies the view transform)
#                 -> worker -> image datablock, with the preview
#                 pre-compensated for the scene's view transform.
#
#   Live viewport GPUOffScreen.draw_view3d at render resolution -> worker at
#                 the region resolution -> drawn back over the region. Because
#                 we choose the off-screen size, Performance mode really does
#                 render the scene at half and reconstruct, rather than being a
#                 post effect on a frame that was already drawn at full size.
#
# Measured on an RTX 5070 Ti, driver 591.44, Blender 5.2 LTS, OpenGL backend:
#   off-screen scene render   2.2-2.5 ms at 1154x786, 1.8 ms at 578x394
#   DLSS round trip           9.5 ms at 1154x786, 8.3 ms at 960x540,
#                             19.9 ms at 1920x1080
#   worker start-up           ~4 s, once per session or settings change
#
# LICENSING: the runtime this drives (nvngx.dll worker, ReShade carrier, RenoDX
# add-on, NVIDIA NGX DLLs) is not distributed with this add-on and carries
# terms that have to be checked by whoever installs it. Read README.md before
# redistributing or selling anything built on top of this.
# ---------------------------------------------------------------------------

import math
import os
import struct
import subprocess
import threading
import traceback
import time
from collections import deque

import bpy
import gpu
import numpy as np
from gpu_extras.presets import draw_texture_2d
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)


# --------------------------------------------------------------------------
# Worker protocol (version 4)
# --------------------------------------------------------------------------

VIDEO_MAGIC = 0x34563544
SETUP_MAGIC = 0x34505553
FRAME_MAGIC = 0x314D5246
OUT_MAGIC = 0x3154554F

VIDEO_HEADER = struct.Struct("<14I4f")
SETUP_RESPONSE = struct.Struct("<12I")
FRAME_HEADER = struct.Struct("<4Iq")
RESULT_HEADER = struct.Struct("<5Iq")

WORKER_NAME = "nvngx.dll"  # an executable despite the name
REQUIRED_RUNTIME_FILES = (
    WORKER_NAME,
    "dxgi.dll",
    "renodx-dlss5.addon64",
    "nvngx_dlss.dll",
    "nvngx_dlssnr.dll",
)

# label -> (scale factor, worker perf/quality id)
UPSCALING_MODES = {
    "DLAA": (1.0, 5),
    "QUALITY": (1.5, 2),
    "BALANCED": (1.724, 1),
    "PERFORMANCE": (2.0, 0),
    "ULTRA_PERFORMANCE": (3.0, 3),
}

DLSS_MODEL_PRESETS = {"Default": 0, "J": 10, "K": 11, "L": 12, "M": 13}
NR_PRESETS = {"Default": 0, "Preset #1": 1, "Preset #2": 2, "Preset #3": 3}
NR_STYLES = {"Default": 0, "Natural": 1, "Cinematic": 2}

MAX_LONG_EDGE = 7680
MAX_SHORT_EDGE = 4320


def _even(value):
    """DLSS only accepts even dimensions."""
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


def resolve_output_size(width, height, factor):
    out_w = _even(int(width) * factor)
    out_h = _even(int(height) * factor)
    if max(out_w, out_h) > MAX_LONG_EDGE or min(out_w, out_h) > MAX_SHORT_EDGE:
        raise ValueError(
            "A {}x{} output exceeds the supported {}x{} boundary. "
            "Pick a lower upscaling mode or render smaller.".format(
                out_w, out_h, MAX_LONG_EDGE, MAX_SHORT_EDGE
            )
        )
    return out_w, out_h


def _read_exact(stream, size):
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError("worker stopped after {} of {} bytes".format(offset, size))
        offset += count
    return bytes(buffer)


def _read_into(stream, target):
    view = memoryview(target).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError("worker stopped after {} of {} bytes".format(offset, len(view)))
        offset += count


class DlssRuntimeError(RuntimeError):
    pass


def validate_runtime(root):
    """Raise unless every binary the worker needs is present in ``root``."""
    if not root:
        raise DlssRuntimeError(
            "No DLSS 5 runtime folder is set. Point the add-on preferences at the "
            "folder that contains nvngx.dll, dxgi.dll and renodx-dlss5.addon64."
        )
    root = bpy.path.abspath(root)
    if not os.path.isdir(root):
        raise DlssRuntimeError("The runtime folder does not exist:\n  {}".format(root))
    missing = [n for n in REQUIRED_RUNTIME_FILES if not os.path.isfile(os.path.join(root, n))]
    if missing:
        raise DlssRuntimeError(
            "The DLSS 5 runtime in\n  {}\nis incomplete. Missing:\n  {}".format(
                root, "\n  ".join(missing)
            )
        )
    return root


class DlssSession:
    """One frame stream against the native feature-18 worker.

    The worker must run with the runtime folder as its working directory or the
    ReShade carrier will not load.
    """

    def __init__(self, root, settings, input_width, input_height, frame_count=1):
        self.root = root
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.logs = deque(maxlen=200)
        self._closed = False

        settings = {k: v for k, v in settings.items() if k != "always_reset"}
        factor, perf_quality = UPSCALING_MODES[settings["upscaling"]]
        self.output_width, self.output_height = resolve_output_size(
            self.input_width, self.input_height, factor
        )

        worker = os.path.join(root, WORKER_NAME)
        try:
            self._worker = subprocess.Popen(
                [worker, "--video"],
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise DlssRuntimeError(
                "The native DLSS worker at\n  {}\ncould not be started: {}.\n\n"
                "It is an executable despite the .dll name, so antivirus software "
                "often quarantines it. Add the runtime folder to the exclusion "
                "list.".format(worker, exc)
            ) from exc

        self._log_thread = threading.Thread(target=self._drain, daemon=True)
        self._log_thread.start()

        try:
            self.setup = self._handshake(settings, perf_quality, frame_count)
        except BaseException:
            self.abort()
            raise

    def _drain(self):
        for line in iter(self._worker.stderr.readline, b""):
            self.logs.append(line.decode("utf-8", "replace").rstrip())

    def _handshake(self, settings, perf_quality, frame_count):
        header = VIDEO_HEADER.pack(
            VIDEO_MAGIC,
            self.input_width,
            self.input_height,
            self.output_width,
            self.output_height,
            0,                              # warmup frames
            max(1, int(frame_count)),
            perf_quality,
            DLSS_MODEL_PRESETS[settings["model_preset"]],
            0,                              # profile
            NR_PRESETS[settings["nr_preset"]],
            NR_STYLES[settings["nr_style"]],
            int(bool(settings["auto_mask"])),
            0,                              # ui correction
            float(settings["intensity"]),
            float(settings["local_tone"]),
            float(settings["local_structure"]),
            float(settings["skin_structure"]),
        )
        try:
            self._worker.stdin.write(header)
            self._worker.stdin.flush()
            payload = _read_exact(self._worker.stdout, SETUP_RESPONSE.size)
        except (EOFError, BrokenPipeError, OSError) as exc:
            try:
                code = self._worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                code = None
            details = "\n".join(list(self.logs)[-40:]) or "The worker produced no output."
            raise DlssRuntimeError(
                "The worker failed during DLSS setup or does not speak the "
                "version-4 protocol (exit {}):\n{}".format(code, details)
            ) from exc

        fields = SETUP_RESPONSE.unpack(payload)
        if fields[0] != SETUP_MAGIC:
            raise DlssRuntimeError(
                "The worker did not answer with a version-4 setup response. "
                "The installed runtime is incompatible with this add-on."
            )
        (ok, result, render_w, render_h, out_w, out_h,
         _min_w, _min_h, _max_w, _max_h, applied_preset) = fields[1:]

        if not ok:
            details = "\n".join(list(self.logs)[-40:])
            raise DlssRuntimeError(
                "DLSS is unavailable for {}x{} (NGX 0x{:08X}). Pick a lower "
                "upscaling mode or update the NVIDIA driver.{}".format(
                    self.output_width, self.output_height, result,
                    "\n" + details if details else "",
                )
            )
        if (out_w, out_h) != (self.output_width, self.output_height):
            raise DlssRuntimeError(
                "The worker negotiated {}x{} instead of the requested {}x{}.".format(
                    out_w, out_h, self.output_width, self.output_height
                )
            )
        requested = DLSS_MODEL_PRESETS[settings["model_preset"]]
        if applied_preset != requested:
            raise DlssRuntimeError(
                "The worker applied a different DLSS model preset than requested. "
                "This runtime does not support that model - set Model Preset to Default."
            )
        if render_w < 64 or render_h < 64:
            raise DlssRuntimeError(
                "DLSS returned an unusable render size {}x{}; both edges must be "
                "at least 64 pixels.".format(render_w, render_h)
            )

        self.render_width = render_w
        self.render_height = render_h
        return fields

    def submit(self, rgba, index=0, reset=True, pts=0):
        """Send one render-resolution RGBA8 frame, return the enhanced frame."""
        if self._closed:
            raise DlssRuntimeError("This DLSS session is already closed.")

        # A single still has no previous frame, so the temporal guide is zero and
        # history is reset. Motion is FP16 (dx, dy) at render resolution.
        motion = np.zeros((self.render_height, self.render_width, 2), dtype=np.float16)

        stdin, stdout = self._worker.stdin, self._worker.stdout
        try:
            stdin.write(FRAME_HEADER.pack(FRAME_MAGIC, index, int(reset), 0, pts))
            stdin.write(memoryview(np.ascontiguousarray(rgba, dtype=np.uint8)).cast("B"))
            stdin.write(memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B"))
            stdin.flush()
            header = RESULT_HEADER.unpack(_read_exact(stdout, RESULT_HEADER.size))
        except (EOFError, BrokenPipeError, OSError, struct.error) as exc:
            details = "\n".join(list(self.logs)[-40:])
            raise DlssRuntimeError(
                "The worker died while processing the frame.{}".format(
                    "\n" + details if details else ""
                )
            ) from exc

        magic, out_index, ok, byte_count, ngx_result, out_pts = header
        expected = self.output_width * self.output_height * 4
        if magic != OUT_MAGIC:
            raise DlssRuntimeError("The worker sent a malformed frame result header.")
        if not ok or out_index != index or byte_count != expected:
            raise DlssRuntimeError(
                "The worker returned an invalid response for frame {} "
                "(index {}, {} of {} bytes).".format(index, out_index, byte_count, expected)
            )
        if ngx_result != 1:
            raise DlssRuntimeError(
                "Feature-18 evaluation failed: 0x{:08X}".format(ngx_result)
            )

        output = np.empty((self.output_height, self.output_width, 4), dtype=np.uint8)
        _read_into(stdout, output)
        return output

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._worker.stdin.close()
        except OSError:
            pass
        try:
            self._worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._worker.kill()
        for pipe in (self._worker.stdout, self._worker.stderr):
            try:
                pipe.close()
            except OSError:
                pass

    def abort(self):
        self._closed = True
        try:
            self._worker.kill()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------

def _resize_rgba8(src, width, height):
    """Bilinear resize, used only when DLSS negotiates a render size that
    differs from the rendered frame (rounding on odd resolutions)."""
    sh, sw = src.shape[:2]
    if (sw, sh) == (width, height):
        return np.ascontiguousarray(src)

    ys = (np.arange(height, dtype=np.float32) + 0.5) * sh / height - 0.5
    xs = (np.arange(width, dtype=np.float32) + 0.5) * sw / width - 0.5
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]

    y0c, y1c = np.clip(y0, 0, sh - 1), np.clip(y0 + 1, 0, sh - 1)
    x0c, x1c = np.clip(x0, 0, sw - 1), np.clip(x0 + 1, 0, sw - 1)

    rows0 = src[y0c].astype(np.float32)
    rows1 = src[y1c].astype(np.float32)
    top = rows0[:, x0c] + (rows0[:, x1c] - rows0[:, x0c]) * wx
    bottom = rows1[:, x0c] + (rows1[:, x1c] - rows1[:, x0c]) * wx
    out = top + (bottom - top) * wy
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def image_to_rgba8(image):
    """Read a Blender image as top-down RGBA8 without a colour transform.

    ``Non-Color`` keeps Blender from linearising the buffer on the way out, so
    the bytes handed to DLSS are exactly the display-referred pixels the neural
    model expects.
    """
    previous = image.colorspace_settings.name
    try:
        image.colorspace_settings.name = "Non-Color"
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
    finally:
        image.colorspace_settings.name = previous

    rgba = buffer.reshape(height, width, 4)
    rgba = np.flipud(rgba)  # Blender stores images bottom-up
    return np.ascontiguousarray(
        np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
    )


def measure_view_transform(scene, samples=1024, max_linear=64.0):
    """Sample the scene's active view transform: scene-linear -> display code.

    Blender's image editor always runs the view transform over a datablock's
    buffer, and there is no per-image override. Rather than making the user
    switch the whole scene to Standard - which changes how their render is
    graded - measure whatever transform is active (including look, exposure and
    gamma) so the preview can be pre-compensated for it.

    The ramp runs well past 1.0 because AgX only reaches ~197/255 at
    scene-linear 1.0; the brighter display codes live further up the curve.
    """
    ramp = np.concatenate(
        ([0.0], np.geomspace(1e-4, max_linear, samples - 1))
    ).astype(np.float32)

    probe = bpy.data.images.new(
        "dlss5_probe", samples, 1, alpha=True, float_buffer=True
    )
    path = os.path.join(bpy.app.tempdir, "dlss5_probe.png")
    settings = scene.render.image_settings
    saved = (settings.file_format, settings.color_mode, settings.color_depth)
    try:
        pixels = np.zeros((1, samples, 4), dtype=np.float32)
        pixels[..., :3] = ramp[None, :, None]
        pixels[..., 3] = 1.0
        probe.pixels.foreach_set(pixels.ravel())

        settings.file_format = "PNG"
        settings.color_mode = "RGBA"
        settings.color_depth = "8"
        probe.save_render(filepath=path, scene=scene)
    finally:
        (settings.file_format, settings.color_mode, settings.color_depth) = saved
        bpy.data.images.remove(probe)

    loaded = bpy.data.images.load(path, check_existing=False)
    try:
        loaded.colorspace_settings.name = "Non-Color"
        buffer = np.empty(len(loaded.pixels), dtype=np.float32)
        loaded.pixels.foreach_get(buffer)
    finally:
        bpy.data.images.remove(loaded)

    display = buffer.reshape(1, samples, 4)[0, :, 0] * 255.0
    # np.interp needs a strictly increasing table; the curve flattens at both
    # ends, so nudge ties apart and keep it monotonic.
    display = np.maximum.accumulate(display)
    display += np.arange(display.size, dtype=np.float32) * 1e-6
    return ramp, display


def invert_view_transform(rgba, ramp, display):
    """Map display-referred bytes to the scene-linear values that reproduce them."""
    target = rgba[..., :3].astype(np.float32)
    linear = np.interp(target.ravel(), display, ramp).astype(np.float32)
    out = np.empty(rgba.shape, dtype=np.float32)
    out[..., :3] = linear.reshape(target.shape)
    out[..., 3] = rgba[..., 3].astype(np.float32) / 255.0
    return out


def srgb_to_linear(values):
    """Undo the sRGB transfer function on a 0..1 float array."""
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    )


def rgba8_to_image(rgba, name, compensate_for=None):
    """Create (or refresh) a Blender image datablock from top-down RGBA8.

    DLSS returns display-referred sRGB bytes, but Blender's ``pixels`` buffer is
    always scene-linear and the image editor applies the scene's view transform
    on top of it. Storing the bytes verbatim would therefore run the view
    transform over an already-graded frame - under the default AgX that lifts
    midtones (118 -> ~165) while pulling highlights down (237 -> ~194), which
    reads as "too bright, no contrast".

    So linearise on the way in and tag the datablock sRGB: the image then
    displays correctly under a Standard view transform, and saving re-applies
    the sRGB curve to give back exactly the bytes DLSS produced.
    """
    height, width = rgba.shape[:2]
    image = bpy.data.images.get(name)
    if image is None or tuple(image.size) != (width, height):
        if image is not None:
            bpy.data.images.remove(image)
        image = bpy.data.images.new(name, width, height, alpha=True)

    if compensate_for is None:
        image.colorspace_settings.name = "sRGB"
        buffer = np.flipud(rgba).astype(np.float32) / 255.0
        buffer[..., :3] = srgb_to_linear(buffer[..., :3])  # alpha stays linear
    else:
        # Pre-compensate: store the scene-linear values that the active view
        # transform maps back onto the exact bytes DLSS produced.
        ramp, display = compensate_for
        image.colorspace_settings.name = "Non-Color"
        buffer = invert_view_transform(np.flipud(rgba), ramp, display)

    image.pixels.foreach_set(np.ascontiguousarray(buffer).ravel())
    image.update()
    return image


def save_rgba8_png(rgba, path):
    """Write the untouched DLSS bytes to a PNG, independent of the preview."""
    height, width = rgba.shape[:2]
    image = bpy.data.images.new("dlss5_save", width, height, alpha=True)
    try:
        image.colorspace_settings.name = "sRGB"
        buffer = np.flipud(rgba).astype(np.float32) / 255.0
        buffer[..., :3] = srgb_to_linear(buffer[..., :3])
        image.pixels.foreach_set(np.ascontiguousarray(buffer).ravel())
        image.filepath_raw = path
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)
    return path


def save_render_to_temp(scene):
    """Write the Render Result to an 8-bit RGBA PNG and load it back.

    Going through ``save_render`` applies the scene's view transform, which is
    what makes the frame display-referred - the domain the neural model works in.
    """
    render_result = bpy.data.images.get("Render Result")
    if render_result is None:
        raise DlssRuntimeError("There is no Render Result. Render a frame first (F12).")

    path = os.path.join(bpy.app.tempdir, "dlss5_input.png")
    settings = scene.render.image_settings
    saved = (settings.file_format, settings.color_mode, settings.color_depth)
    try:
        settings.file_format = "PNG"
        settings.color_mode = "RGBA"
        settings.color_depth = "8"
        render_result.save_render(filepath=path, scene=scene)
    finally:
        (settings.file_format, settings.color_mode, settings.color_depth) = saved

    loaded = bpy.data.images.load(path, check_existing=False)
    try:
        return image_to_rgba8(loaded)
    finally:
        bpy.data.images.remove(loaded)


# --------------------------------------------------------------------------
# Live viewport
# --------------------------------------------------------------------------

# Measured on an RTX 5070 Ti: 8.3 ms/frame at 960x540, 11.4 at 720p, 19.9 at
# 1080p once the session is warm. That is fast enough to sit in a draw handler,
# which is why this exists at all - the round trip is what gates it.

def _wrap(text, width):
    """Break a message into label-sized chunks; Blender labels do not wrap."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:6]


def _buffer_to_array(buffer, size):
    """Read a gpu.types.Buffer as uint8 without copying element by element."""
    try:
        return np.asarray(buffer, dtype=np.uint8).reshape(size)
    except (TypeError, ValueError):
        # Older builds may not expose the buffer protocol; this is slow but
        # keeps the feature working rather than failing outright.
        return np.frombuffer(bytes(bytearray(buffer)), dtype=np.uint8)[:size]


# Which upload path worked last time, so the fallbacks are tried once and then
# the winner is reused every frame.
def _make_texture(gpu, rgba):
    """Upload a top-down RGBA8 frame as a GPUTexture.

    GPUTexture takes FLOAT buffers only - handing it a UBYTE buffer raises
    "Only Buffer of format `FLOAT` is currently supported" - so the frame is
    normalised to 0..1 and uploaded as RGBA16F. At draw time that samples
    identically to an RGBA8 UNORM texture, so the image itself is unchanged.

    The buffer is filled through the buffer protocol: building a Python list of
    eight million floats would cost far more than the DLSS pass that made them.
    """
    height, width = rgba.shape[:2]
    flat = np.ascontiguousarray(np.flipud(rgba), dtype=np.uint8).ravel()
    values = flat.astype(np.float32) / 255.0

    buffer = gpu.types.Buffer("FLOAT", values.size)
    try:
        np.asarray(buffer)[:] = values
    except (TypeError, ValueError):
        buffer = gpu.types.Buffer("FLOAT", values.size, values.tolist())
    return gpu.types.GPUTexture((width, height), format="RGBA16F", data=buffer)


class ViewportStream:
    """A long-lived DLSS session fed by an off-screen render of the viewport.

    Reading the on-screen framebuffer also works, but it hands DLSS a frame
    that already contains the grid, gizmos and text, and it is always at the
    region's full size - so DLSS could only ever be a post effect on top of a
    render that already happened.

    Rendering the view into our own off-screen buffer instead costs about
    2.7 ms at 1048x694 and 2.1 ms at half that (measured, RTX 5070 Ti, OpenGL
    backend) and buys three things: a clean frame with no overlays in it, a
    render resolution we choose, and therefore a real DLSS reconstruction
    rather than a 1:1 pass.
    """

    def __init__(self, root, settings, render_width, render_height,
                 source="OFFSCREEN"):
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.source = source
        self.always_reset = bool(settings.get('always_reset', True))
        # The worker takes every control in its one-time video header, so a
        # settings change cannot be pushed into a live session - it needs a new
        # one. Keep a copy so the draw handler can notice.
        self.settings = dict(settings)
        # The worker wants a frame budget up front; this is an open-ended
        # stream, so ask for far more than a session will use and tear it down
        # with abort() rather than a clean end-of-stream.
        self.session = DlssSession(
            root, settings, self.render_width, self.render_height,
            frame_count=1_000_000,
        )
        # Only the off-screen source needs a buffer; screen capture reads the
        # region straight out of the framebuffer that is already on screen.
        self.offscreen = (
            gpu.types.GPUOffScreen(
                self.render_width, self.render_height, format="RGBA8"
            )
            if self.source == "OFFSCREEN" else None
        )

        self.index = 0
        self.resets = 0
        self.last_view = None
        self.recent_ms = deque(maxlen=30)
        self.recent_render_ms = deque(maxlen=30)
        # How far the returned frame actually is from the one that went in.
        # 0.00 means the worker handed the frame straight back.
        self.delta_mean = None
        self.delta_max = None
        self.delta_pct = None
        # Last processed frame, kept so a throttled redraw has something to
        # show without paying for another round trip.
        self.cached_texture = None
        self.cached_view = None
        self.cached_at = 0.0
        self.skipped = 0
        # Set when a capture came back black, which is a source problem rather
        # than a DLSS one.
        self.frame_empty = False
        # Screen capture reads the framebuffer we ourselves drew into. When the
        # renderer has stopped redrawing, the next capture is our own output -
        # processing that again converges on a fixed point and the effect
        # disappears. Keep the last output to recognise it.
        self.last_output = None
        self.last_input = None
        self.feedback_score = None
        self.feedback_hits = 0

    @property
    def output_size(self):
        return self.session.output_width, self.session.output_height

    @property
    def average_ms(self):
        return sum(self.recent_ms) / len(self.recent_ms) if self.recent_ms else 0.0

    @property
    def average_render_ms(self):
        values = self.recent_render_ms
        return sum(values) / len(values) if values else 0.0

    def matches(self, render_width, render_height, settings, source):
        return (
            (self.render_width, self.render_height)
            == (int(render_width), int(render_height))
            and self.settings == settings
            and self.source == source
        )

    def capture(self, context, space, region):
        """Get the frame to process, from whichever source this stream uses."""
        started = time.perf_counter()
        if self.source == "OFFSCREEN":
            frame = self._render_offscreen(context, space, region)
        else:
            frame = self._read_screen(region)
        self.recent_render_ms.append((time.perf_counter() - started) * 1000.0)
        return frame

    def _render_offscreen(self, context, space, region):
        """Render the 3D view into our own buffer, overlays excluded.

        do_color_management applies the scene's view transform, so what comes
        back is display-referred - the domain the neural model works in.

        This cannot reproduce a Cycles Rendered viewport: measured against a
        converged on-screen render it differs by 27 levels on average and
        returns a byte-identical image on every call, because it rasterises a
        preview rather than reusing the accumulated path-traced result. Use the
        screen source for Rendered shading.
        """
        region_3d = space.region_3d
        self.offscreen.draw_view3d(
            context.scene,
            context.view_layer,
            space,
            region,
            region_3d.view_matrix,
            region_3d.window_matrix,
            do_color_management=True,
            draw_background=True,
        )
        buffer = self.offscreen.texture_color.read()
        frame = np.asarray(buffer, dtype=np.uint8).reshape(
            self.render_height, self.render_width, 4
        )
        return np.ascontiguousarray(np.flipud(frame))   # GPU hands back bottom-up

    def _read_screen(self, region):
        """Read the region out of the window framebuffer.

        This must run from a timer, not from a draw handler: inside a
        POST_PIXEL callback the region framebuffer reads back black. From a
        timer the active framebuffer is the whole window, so the region's own
        rectangle is where its pixels live.
        """
        framebuffer = gpu.state.active_framebuffer_get()
        width = region.width & ~1
        height = region.height & ~1
        buffer = framebuffer.read_color(
            region.x, region.y, width, height, 4, 0, "UBYTE"
        )
        frame = np.flipud(np.asarray(buffer, dtype=np.uint8).reshape(height, width, 4))
        if (width, height) != (self.render_width, self.render_height):
            frame = _resize_rgba8(frame, self.render_width, self.render_height)
        return np.ascontiguousarray(frame)

    def process(self, rgba, view_key):
        """Run one frame; reset temporal history whenever the view moved.

        Camera motion is read straight from the region's perspective matrix
        rather than estimated with optical flow: it is exact, costs nothing,
        and a wrong guess here would show up as smearing.
        """
        # We have no real motion vectors - the buffer we send is zero - so DLSS
        # cannot know where a pixel came from. Letting it accumulate across
        # frames therefore drags content from the previous viewpoint into the
        # new one, which is what turns the image into invented detail that has
        # nothing to do with the scene. Unless accumulation is explicitly asked
        # for, every frame starts clean.
        reset = True if self.always_reset else (view_key != self.last_view)
        self.last_view = view_key
        if reset:
            self.resets += 1


        started = time.perf_counter()
        output = self.session.submit(
            rgba, index=self.index, reset=reset, pts=self.index
        )
        self.recent_ms.append((time.perf_counter() - started) * 1000.0)
        self.index += 1
        return output

    def close(self):
        self.cached_texture = None
        self.cached_view = None
        self.last_output = None
        try:
            self.session.abort()
        except Exception:
            pass
        try:
            if self.offscreen is not None:
                self.offscreen.free()
        except Exception:
            pass


def measure_delta(output, source):
    """How much the worker actually changed the frame.

    Returns (mean absolute difference, max difference, percent of pixels
    touched) across RGB. A mean of 0.00 means the worker handed the frame
    straight back; squinting at a screenshot cannot settle that, this can.
    Sampled every 4th pixel, because it runs per frame.
    """
    out_h, out_w = output.shape[:2]
    reference = _resize_rgba8(source, out_w, out_h)
    a = output[::4, ::4, :3].astype(np.int16)
    b = reference[::4, ::4, :3].astype(np.int16)
    difference = np.abs(a - b)
    return (
        float(difference.mean()),
        int(difference.max()),
        float((difference.any(axis=2)).mean() * 100.0),
    )


def _apply_debug_view(output, source, mode):
    """Comparison overlays drawn straight into the returned frame.

    At DLAA with default controls DLSS is a subtle pass - easy to mistake for
    "nothing happened". SPLIT puts the untouched frame beside the processed one
    so the difference is visible live; TINT proves the drawn texture reaches
    the screen at all, which no amount of logging settles.
    """
    if mode == "OFF":
        return output

    out_h, out_w = output.shape[:2]
    if mode == "TINT":
        tinted = output.copy()
        tinted[..., 1] = (tinted[..., 1].astype(np.uint16) // 3).astype(np.uint8)
        return tinted

    reference = _resize_rgba8(source, out_w, out_h)
    split = output.copy()
    middle = out_w // 2
    split[:, :middle] = reference[:, :middle]
    seam = max(1, out_w // 400)
    split[:, middle:middle + seam, :3] = 255
    return split


_stream = None
_draw_handle = None
_in_draw = False
_last_error = None
_suppress_paint = False
_suppressed_done = False
_last_capture_key = None
_last_capture_at = 0.0
_last_seen_key = None
_view_changed_at = 0.0
_needs_capture = True

# Execution trace. "Nothing happened and nothing raised" is the hardest kind of
# bug to find by reasoning, so count every path instead.
_trace = {}


def _mark(name):
    _trace[name] = _trace.get(name, 0) + 1


def _stop_stream():
    global _stream
    if _stream is not None:
        _stream.close()
        _stream = None


# Engines whose viewport an off-screen pass can reproduce. EEVEE and Workbench
# rasterise a complete frame every time, so re-rendering the view into our own
# buffer gives the same picture. Cycles does not: measured against a converged
# viewport its off-screen pass came back black in a production file, and
# byte-identical on every call in a simple one.
RASTER_ENGINES = {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"}


def resolve_frame_source(settings, space, scene=None):
    """Where the frame to process should come from.

    Off-screen is the better source wherever it works: it is independent of
    what is on screen, so nothing can feed back, no paint has to be withheld to
    get a clean capture, and the render resolution is ours to choose - which is
    what makes real reconstruction possible instead of a 1:1 pass.

    It only works under a rasterising engine. Under Cycles the same call
    returns black, so there the frame has to be read off the screen.
    """
    if settings.frame_source != "AUTO":
        return settings.frame_source
    engine = getattr(getattr(scene, "render", None), "engine", None)
    shading = getattr(getattr(space, "shading", None), "type", "SOLID")
    if engine in RASTER_ENGINES or shading in {"SOLID", "MATERIAL"}:
        return "OFFSCREEN"
    return "SCREEN"


def _view_key(region_data):
    """A cheap, exact "did the view move" token."""
    if region_data is None:
        return None
    return tuple(round(v, 5) for row in region_data.perspective_matrix for v in row)


def draw_cached(gpu, texture, region):
    """Draw an already-built texture with the same state handling as a fresh one."""
    previous_blend = gpu.state.blend_get()
    previous_depth = gpu.state.depth_test_get()
    previous_mask = gpu.state.depth_mask_get()
    gpu.state.blend_set("NONE")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    try:
        draw_texture_2d(texture, (0, 0), region.width, region.height)
    finally:
        gpu.state.depth_mask_set(previous_mask)
        gpu.state.depth_test_set(previous_depth)
        gpu.state.blend_set(previous_blend)


def _viewport_draw():
    """Paint the last processed frame. All the work happens in the timer."""
    global _in_draw

    _mark("called")
    context = bpy.context
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "dlss5", None) if scene else None
    if settings is None or not settings.live_viewport:
        _mark("bail_disabled")
        return
    if _suppress_paint:
        # One redraw with our result withheld, so the timer can capture the
        # viewport rather than a picture of our own output. Acknowledge it, so
        # the timer knows the screen really is showing the raw frame - reading
        # before this point captures our own paint and the effect compounds,
        # which looks like the image slowly deforming.
        global _suppressed_done
        _suppressed_done = True
        _mark("suppressed")
        return
    if _stream is None or _stream.cached_texture is None:
        _mark("bail_no_texture")
        return

    region = context.region
    if region is None:
        _mark("bail_no_region")
        return

    # The cached frame belongs to the view it was captured from. Painting it
    # over a view that has since moved shows the user a stale image on top of
    # a live one, which reads as the result being dragged along from before the
    # move. While the view differs, show Blender's own render instead.
    space = context.space_data
    if _stream.cached_view is not None:
        current = _view_key(getattr(space, "region_3d", None))
        if current != _stream.cached_view:
            # This callback sees the region as it is being drawn, so it knows
            # the current view before a timer reading of the same data does.
            # Flag the work rather than leaving the timer to notice.
            global _needs_capture, _last_seen_key, _view_changed_at
            if current != _last_seen_key:
                _last_seen_key = current
                _view_changed_at = time.perf_counter()
            _needs_capture = True
            _mark("bail_stale_view")
            return

    _in_draw = True
    try:
        draw_cached(gpu, _stream.cached_texture, region)
        _mark("drew")
    except Exception as exc:
        _report_failure(context, settings, exc)
    finally:
        _in_draw = False


def _viewport_region():
    """The largest 3D viewport's area and window region, or (None, None)."""
    window = getattr(bpy.context, "window", None)
    screen = getattr(window, "screen", None) if window else None
    if screen is None:
        return None, None
    areas = [a for a in screen.areas if a.type == "VIEW_3D"]
    if not areas:
        return None, None
    area = max(areas, key=lambda a: a.width * a.height)
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    return area, region


def _report_failure(context, settings, exc):
    global _last_error
    _last_error = "{}: {}".format(type(exc).__name__, exc)
    details = traceback.format_exc()
    print("DLSS 5 live viewport disabled:\n" + details)
    try:
        prefs = context.preferences.addons[__name__].preferences
        with open(os.path.join(bpy.path.abspath(prefs.runtime_dir),
                               "dlss5_error.txt"), "w", encoding="utf-8") as handle:
            handle.write(details)
    except Exception:
        pass
    _stop_stream()
    try:
        settings.live_viewport = False
    except Exception:
        pass


def _process_once(context, settings, area, region, space, source, interval):
    """Capture one frame, run it through DLSS and build the texture to paint."""
    global _stream, _last_capture_key, _last_capture_at, _needs_capture

    try:
        out_w, out_h = _even(region.width), _even(region.height)
        # Only the off-screen source can render at a size of our choosing, so
        # only there does the upscaling ratio mean anything. A screen capture
        # is whatever Blender already drew, at full size.
        factor = UPSCALING_MODES[settings.upscaling][0] if source == "OFFSCREEN" else 1.0
        render_w = max(64, _even(out_w / factor))
        render_h = max(64, _even(out_h / factor))

        stream_settings = dict(settings.as_dict())
        if factor == 1.0:
            stream_settings["upscaling"] = "DLAA"

        if _stream is not None and not _stream.matches(
            render_w, render_h, stream_settings, source
        ):
            _stop_stream()
        if _stream is None:
            prefs = context.preferences.addons[__name__].preferences
            root = validate_runtime(prefs.runtime_dir)
            _mark("creating_stream")
            _stream = ViewportStream(
                root, stream_settings, render_w, render_h, source=source
            )

        frame = _stream.capture(context, space, region)
        _stream.frame_empty = bool(frame[..., :3].max() < 4)
        if _stream.frame_empty:
            _mark("bail_frame_empty")
            area.tag_redraw()
            return interval

        view_key = _view_key(getattr(space, "region_3d", None))
        output = _stream.process(frame, view_key)
        (_stream.delta_mean, _stream.delta_max,
         _stream.delta_pct) = measure_delta(output, frame)
        _stream.last_output = output
        _stream.last_input = frame

        _last_capture_key = view_key
        _needs_capture = False
        _last_capture_at = time.perf_counter()
        shown = _apply_debug_view(output, frame, settings.debug_view)
        _stream.cached_texture = _make_texture(gpu, shown)
        _stream.cached_view = view_key
        _stream.cached_at = time.perf_counter()
        _mark("processed")
        area.tag_redraw()
    except Exception as exc:
        _report_failure(context, settings, exc)
        return None

    return interval


def _viewport_timer():
    """Capture, process and build the texture. Runs outside any draw callback."""
    global _stream, _suppress_paint, _suppressed_done
    global _last_capture_key, _last_capture_at
    global _last_seen_key, _view_changed_at, _needs_capture

    context = bpy.context
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "dlss5", None) if scene else None
    if settings is None or not settings.live_viewport:
        _suppress_paint = False
        return None                      # unregister; the toggle restarts it

    interval = 1.0 / max(1, settings.update_rate)
    area, region = _viewport_region()
    if area is None or region is None or region.width < 64 or region.height < 64:
        return interval

    space = area.spaces.active
    source = resolve_frame_source(settings, space, scene)

    if source == "OFFSCREEN":
        # Nothing on screen is involved, so capture immediately: no withheld
        # paint, no flash, and no waiting for the renderer to settle.
        _mark("timer_direct")
        return _process_once(context, settings, area, region, space, source, interval)

    if not _suppress_paint:
        # Screen capture only. Reading the framebuffer while our own result is
        # painted over it would feed the model its own output, so one redraw
        # has to go by unpainted first.
        view_key = _view_key(getattr(space, "region_3d", None))
        now = time.perf_counter()

        if view_key != _last_seen_key:
            _last_seen_key = view_key
            _view_changed_at = now
            _mark("timer_moving")
            return interval

        if (now - _view_changed_at) < settings.idle_refresh:
            _mark("timer_settling")
            return interval

        if _stream is not None and _stream.cached_texture is not None                 and not _needs_capture and view_key == _last_capture_key:
            _mark("timer_idle")
            return interval

        _mark("timer_suppress")
        _suppress_paint = True
        _suppressed_done = False
        area.tag_redraw()
        return 0.01

    if not _suppressed_done:
        _mark("timer_waiting")
        return 0.01

    # Phase 2: the viewport was redrawn without our paint, so read it now.
    _suppress_paint = False
    return _process_once(context, settings, area, region, space, source, interval)


def _update_live_viewport(self, context):
    global _draw_handle, _last_error, _suppress_paint
    global _needs_capture
    _needs_capture = True
    if self.live_viewport:
        _last_error = None      # a fresh attempt gets a clean slate
        _trace.clear()
    _suppress_paint = False

    if self.live_viewport and _draw_handle is None:
        _mark("handler_registered")
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _viewport_draw, (), "WINDOW", "POST_PIXEL"
        )
        if not bpy.app.timers.is_registered(_viewport_timer):
            bpy.app.timers.register(_viewport_timer, first_interval=0.1)
    elif not self.live_viewport and _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
        if bpy.app.timers.is_registered(_viewport_timer):
            bpy.app.timers.unregister(_viewport_timer)
        _stop_stream()

    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

class DLSS5Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    runtime_dir: StringProperty(
        name="Runtime Folder",
        subtype="DIR_PATH",
        description=(
            "Folder holding nvngx.dll (the worker), dxgi.dll, "
            "renodx-dlss5.addon64, nvngx_dlss.dll and nvngx_dlssnr.dll"
        ),
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "runtime_dir")
        column = layout.column(align=True)
        try:
            root = validate_runtime(self.runtime_dir)
        except DlssRuntimeError as exc:
            for index, line in enumerate(str(exc).splitlines()):
                column.label(text=line, icon="ERROR" if index == 0 else "NONE")
        else:
            column.label(text="Runtime looks complete: {}".format(root), icon="CHECKMARK")


class DLSS5Settings(bpy.types.PropertyGroup):
    # Every description here is what the control actually does on the v4.0
    # runtime, measured rather than copied from upstream marketing.
    upscaling: EnumProperty(
        name="Mode",
        items=[
            ("DLAA", "DLAA - native (1x)",
             "Render the viewport at its full size and let DLSS refine it. "
             "Best quality, no speed saved"),
            ("QUALITY", "Quality (1.5x)",
             "Render at 1/1.5 of the viewport size and reconstruct. "
             "Near-native quality, noticeably cheaper"),
            ("BALANCED", "Balanced (1.724x)",
             "Render at 1/1.724 of the viewport size and reconstruct"),
            ("PERFORMANCE", "Performance (2x)",
             "Render at half size and reconstruct. Roughly quarter the scene "
             "cost; softer, and the usual choice for heavy scenes"),
            ("ULTRA_PERFORMANCE", "Ultra Performance (3x)",
             "Render at a third and reconstruct. Cheapest and softest"),
        ],
        default="DLAA",
        description=(
            "How much of the frame DLSS reconstructs. In the live viewport "
            "this sets the resolution the scene is actually rendered at, so it "
            "is a real quality/speed trade. On a still render it only enlarges "
            "the finished frame"
        ),
    )
    model_preset: EnumProperty(
        name="Model Preset",
        items=[
            ("Default", "Default", "Whatever the runtime picks"),
            ("J", "J", "Older model, softer result"),
            ("K", "K", "Older model, softer result"),
            ("L", "L", "Newer model, more reconstructed texture"),
            ("M", "M", "Newest model, the most skin and hair detail"),
        ],
        default="M",
        description=(
            "Which reconstruction model runs. This has the largest effect on "
            "the result: Default, J and K stay soft, while L and M rebuild "
            "noticeably more skin and hair texture. Fall back to Default if "
            "the runtime refuses the model"
        ),
    )
    # Measured against the v4.0 runtime: #1, #2 and #3 return byte-identical
    # frames, so the control is kept for the protocol but left out of the panel.
    nr_preset: EnumProperty(
        name="NR Preset",
        items=[(k, k, k) for k in NR_PRESETS],
        default="Default",
        description="No effect on current runtime builds - verified identical output",
    )
    nr_style: EnumProperty(
        name="Style",
        items=[
            ("Default", "Default",
             "Widest tonal range. Measured on a grey wedge: black 17, "
             "white 237, range 220"),
            ("Natural", "Natural",
             "Flattest of the three - compresses the range to 194 and lowers "
             "contrast. Not the one to pick if the image already looks soft"),
            ("Cinematic", "Cinematic",
             "Lifts the blacks to 22 and keeps most of the range at 215. "
             "Slightly milkier shadows, punchier midtones"),
        ],
        default="Default",
        description=(
            "The only control that measurably changes tone and contrast. The "
            "numbers in each entry are from a grey step wedge pushed through "
            "the runtime, so they describe this build, not the marketing"
        ),
    )
    intensity: FloatProperty(
        name="Intensity", default=1.0, min=0.0, max=2.0,
        description=(
            "Blends the neural pass back toward the source. Below 1.0 it fades "
            "out progressively; above 1.0 does nothing on this runtime"
        ),
    )
    local_tone: FloatProperty(
        name="Local Tone", default=1.0, min=0.0, max=2.0,
        description=(
            "Local tone mapping strength. Works on local contrast, so it does "
            "little on flat areas and more where the image already has "
            "structure"
        ),
    )
    local_structure: FloatProperty(
        name="Local Structure", default=1.5, min=0.0, max=2.0,
        description=(
            "How much fine detail and material response is rebuilt. Raise it "
            "when the result looks too smooth"
        ),
    )
    skin_structure: FloatProperty(
        name="Skin Structure", default=2.0, min=-1.0, max=2.0,
        description=(
            "Skin and pore reconstruction, applied only inside the regions the "
            "model detects as skin. Requires Automatic Mask; -1.0 leaves the "
            "decision to the model"
        ),
    )
    auto_mask: BoolProperty(
        name="Automatic Mask", default=True,
        description=(
            "Let the model find skin regions. Skin Structure has no effect "
            "while this is off"
        ),
    )

    live_viewport: BoolProperty(
        name="Live Viewport",
        default=False,
        update=_update_live_viewport,
        description=(
            "Render the 3D view into an off-screen buffer, push it through "
            "DLSS and draw the result back, on every redraw. The off-screen "
            "render costs about 2.7 ms at 1048x694 and the DLSS round trip "
            "8-20 ms depending on size, and both are paid on top of Blender's "
            "own viewport draw"
        ),
    )
    always_reset: BoolProperty(
        name="Reset Every Frame",
        default=True,
        description=(
            "Start each frame with no temporal history. This add-on cannot "
            "supply real motion vectors, so letting DLSS accumulate makes it "
            "warp content from the previous viewpoint into the current one - "
            "detail that belongs to nothing in the scene. Turn this off only "
            "for a locked-off view, where accumulation is valid and does "
            "reduce noise further"
        ),
    )
    idle_refresh: FloatProperty(
        name="Settle Delay",
        default=4.0, min=0.05, max=60.0, subtype="TIME",
        description=(
            "How long the view must sit still before the frame is captured. "
            "Nothing is processed while you navigate, and nothing is processed "
            "while the renderer is still resolving - a half-converged frame is "
            "what makes the result look invented. Raise it for a heavy scene "
            "that takes longer to settle, lower it for a quicker response"
        ),
    )
    update_rate: IntProperty(
        name="Update Rate",
        default=15, min=1, max=60, subtype="UNSIGNED",
        description=(
            "How many times a second the viewport may be processed. Cycles "
            "redraws continuously while it samples, and a round trip on every "
            "one of those redraws locks the interface up; between updates the "
            "last result is reused. Lower this if the viewport feels heavy"
        ),
    )
    frame_source: EnumProperty(
        name="Frame Source",
        items=[
            ("AUTO", "Automatic",
             "Screen capture - the source that works in every scene tested"),
            ("OFFSCREEN", "Off-screen Render (experimental)",
             "Render the view again at a resolution of our choosing. Clean - "
             "no overlays - and the only source that allows real "
             "reconstruction, so it is the one worth having. But it cannot "
             "reproduce a Cycles Rendered viewport, and in some production "
             "files it returns an empty frame. Check the panel says the frame "
             "is not empty before trusting it"),
            ("SCREEN", "Screen Capture",
             "Read the region from the framebuffer already on screen. Shows "
             "exactly what you see, including accumulated Cycles samples and "
             "overlays, but is always full resolution so upscaling saves "
             "nothing"),
        ],
        default="AUTO",
        description=(
            "Where the frame handed to DLSS comes from. Rendered shading has "
            "to use screen capture: an off-screen render rasterises a preview "
            "instead of the path-traced result"
        ),
    )
    debug_view: EnumProperty(
        name="Compare",
        items=[
            ("OFF", "Off", "Show the processed frame"),
            ("SPLIT", "Split",
             "Left half untouched, right half through DLSS, with a white seam "
             "down the middle. The honest way to judge whether it is helping"),
            ("TINT", "Tint (test)",
             "Force a magenta tint. If the viewport does not turn magenta the "
             "frame is not reaching the screen, which is a different problem "
             "from DLSS doing nothing"),
        ],
        default="OFF",
        description=(
            "Comparison overlays. Changing this does not restart the worker"
        ),
    )
    match_view_transform: BoolProperty(
        name="Match View Transform",
        default=True,
        description=(
            "Still renders only. DLSS returns a display-referred frame, and "
            "Blender's image editor would run the view transform over it a "
            "second time - under AgX that lifts midtones and flattens "
            "contrast. This measures the active transform and pre-compensates "
            "for it, so the preview is exact without changing how the render "
            "itself is graded"
        ),
    )
    auto_run: BoolProperty(
        name="Run After Render",
        default=False,
        description="Enhance automatically when a still render finishes",
    )
    save_beside_output: BoolProperty(
        name="Save Next To Output",
        default=False,
        description=(
            "Write the untouched DLSS bytes to dlss5_result.png beside the "
            "scene's output path. The saved file is never view-transform "
            "compensated, only the on-screen preview is"
        ),
    )

    def as_dict(self):
        # always_reset is consumed by ViewportStream, not sent to the worker,
        # but it belongs in this dict so changing it restarts the session.
        return {
            "upscaling": self.upscaling,
            "model_preset": self.model_preset,
            "nr_preset": self.nr_preset,
            "nr_style": self.nr_style,
            "intensity": self.intensity,
            "local_tone": self.local_tone,
            "local_structure": self.local_structure,
            "skin_structure": self.skin_structure,
            "auto_mask": self.auto_mask,
            "always_reset": self.always_reset,
        }


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

def _runtime_root(context):
    prefs = context.preferences.addons[__name__].preferences
    return validate_runtime(prefs.runtime_dir)


def _show_in_image_editor(context, image):
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            area.spaces.active.image = image
            return True
    return False


class DLSS5_OT_render_upscaled(bpy.types.Operator):
    bl_idname = "dlss5.render_upscaled"
    bl_label = "Render Upscaled"
    bl_description = (
        "Render at a fraction of the output resolution and let DLSS "
        "reconstruct the rest. This is the one place the add-on saves real "
        "time: at Performance the renderer does a quarter of the pixels"
    )
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        settings = scene.dlss5

        factor = UPSCALING_MODES[settings.upscaling][0]
        if factor <= 1.0:
            self.report(
                {"ERROR"},
                "Mode is DLAA (1x), so there is nothing to save. Pick Quality, "
                "Balanced, Performance or Ultra Performance.",
            )
            return {"CANCELLED"}

        full_percentage = scene.render.resolution_percentage
        full_w = int(scene.render.resolution_x * full_percentage / 100)
        full_h = int(scene.render.resolution_y * full_percentage / 100)

        # Render at output/factor, so the reconstruction lands on the size the
        # scene was actually set up to produce.
        reduced = max(1, int(round(full_percentage / factor)))

        try:
            root = _runtime_root(context)
        except DlssRuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        started = time.perf_counter()
        try:
            scene.render.resolution_percentage = reduced
            bpy.ops.render.render(write_still=False)
            render_seconds = time.perf_counter() - started
            rgba = save_render_to_temp(scene)
        except (DlssRuntimeError, RuntimeError, ValueError) as exc:
            scene.render.resolution_percentage = full_percentage
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            scene.render.resolution_percentage = full_percentage

        height, width = rgba.shape[:2]
        try:
            with DlssSession(root, settings.as_dict(), width, height) as session:
                frame = _resize_rgba8(rgba, session.render_width, session.render_height)
                output = session.submit(frame)
        except (DlssRuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        compensate = (measure_view_transform(scene)
                      if settings.match_view_transform else None)
        image = rgba8_to_image(output, "DLSS5 Result", compensate_for=compensate)
        _show_in_image_editor(context, image)

        if settings.save_beside_output:
            base = bpy.path.abspath(scene.render.filepath) or bpy.app.tempdir
            directory = base if os.path.isdir(base) else os.path.dirname(base)
            os.makedirs(directory, exist_ok=True)
            save_rgba8_png(output, os.path.join(directory, "dlss5_result.png"))

        self.report(
            {"INFO"},
            "Rendered {}x{} in {:.1f}s -> reconstructed {}x{} (target was {}x{})".format(
                width, height, render_seconds,
                output.shape[1], output.shape[0], full_w, full_h,
            ),
        )
        return {"FINISHED"}


class DLSS5_OT_enhance_render(bpy.types.Operator):
    bl_idname = "dlss5.enhance_render"
    bl_label = "Enhance Render Result"
    bl_description = "Send the current Render Result through DLSS 5 neural rendering"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene
        settings = scene.dlss5

        try:
            root = _runtime_root(context)
            rgba = save_render_to_temp(scene)
            height, width = rgba.shape[:2]

            with DlssSession(root, settings.as_dict(), width, height) as session:
                frame = _resize_rgba8(rgba, session.render_width, session.render_height)
                output = session.submit(frame)

            compensate = None
            if settings.match_view_transform:
                compensate = measure_view_transform(scene)
            image = rgba8_to_image(output, "DLSS5 Result", compensate_for=compensate)

            if settings.save_beside_output:
                base = bpy.path.abspath(scene.render.filepath) or bpy.app.tempdir
                directory = base if os.path.isdir(base) else os.path.dirname(base)
                os.makedirs(directory, exist_ok=True)
                target = save_rgba8_png(
                    output, os.path.join(directory, "dlss5_result.png")
                )
                self.report({"INFO"}, "Saved {}".format(target))

        except (DlssRuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _show_in_image_editor(context, image)
        # Same measurement as the live viewport, so the two paths can be
        # compared honestly instead of by eye against a noisy preview.
        delta_mean, delta_max, delta_pct = measure_delta(output, rgba)
        self.report(
            {"INFO"},
            "DLSS 5: {}x{} -> {}x{} | delta {:.2f} avg, {} max, {:.0f}% pixels".format(
                width, height, output.shape[1], output.shape[0],
                delta_mean, delta_max, delta_pct,
            ),
        )
        return {"FINISHED"}


def read_feature_report(root):
    """Summarise what the worker's own log says about the last session.

    ReShade rewrites this file each time the worker starts, so it always
    describes the most recent run - which is exactly the question "did it
    actually do anything just now".
    """
    path = os.path.join(root, "ReShade.log")
    if not os.path.isfile(path):
        return {"log": None}

    created = evaluated = 0
    sizes = []
    ngx_init = False
    errors = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "feature 18 created" in line:
                created += 1
            if "feature 18 evaluation succeeded" in line:
                evaluated += 1
            if "NVSDK_NGX_D3D12_Init" in line and "Success" in line:
                ngx_init = True
            if "NR input " in line:
                start = line.find("NR input ") + len("NR input ")
                sizes.append(line[start:].split()[0])
            # vtable::Hook misses are expected: the add-on falls back to the
            # signed snippet path, which is the one that succeeds.
            if "| ERROR" in line and "vtable::Hook" not in line:
                errors.append(line.strip()[-140:])

    return {
        "log": path,
        "ngx_init": ngx_init,
        "created": created,
        "evaluated": evaluated,
        "sizes": sorted(set(sizes)),
        "errors": errors[-3:],
    }


class DLSS5_OT_diagnostics(bpy.types.Operator):
    bl_idname = "dlss5.diagnostics"
    bl_label = "Show Status"
    bl_description = "Report what the worker actually did, from its own log"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            root = _runtime_root(context)
        except DlssRuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        report = read_feature_report(root)
        lines = ["DLSS 5 status", "  runtime: {}".format(root)]

        if _stream is not None:
            lines.append(
                "  live viewport: {}x{}, {} frames, {} history resets, {:.1f} ms/frame".format(
                    _stream.width, _stream.height, _stream.index,
                    _stream.resets, _stream.average_ms,
                )
            )
        else:
            lines.append("  live viewport: not running")

        if report["log"] is None:
            lines.append("  worker log: none yet - nothing has run")
        else:
            # Successful evaluations are the signal that matters: the add-on
            # can reach feature 18 through the signed snippet without logging
            # an explicit NGX init, so a missing init line is not a failure.
            # The worker logs a feature creation, not every frame, so this is
            # proof that DLSS ran - never a frame count. The live counter above
            # is the one that says whether frames are still flowing.
            lines.append("  feature 18: {} created, {} logged evaluations "
                         "(logged per feature, not per frame)".format(
                             report["created"], report["evaluated"]))
            lines.append("  NGX D3D12 init line: {}".format(
                "logged" if report["ngx_init"] else "not logged (harmless)"))
            if report["sizes"]:
                lines.append("  processed sizes: {}".format(", ".join(report["sizes"])))
            for error in report["errors"]:
                lines.append("  error: {}".format(error))

        for line in lines:
            print(line)

        if report.get("evaluated"):
            self.report({"INFO"}, "feature 18 OK - {} evaluations at {}".format(
                report["evaluated"], ", ".join(report["sizes"]) or "?"))
        else:
            self.report({"WARNING"},
                        "No successful feature-18 evaluation in the worker log")
        return {"FINISHED"}


class DLSS5_OT_enable_viewport_denoise(bpy.types.Operator):
    bl_idname = "dlss5.enable_viewport_denoise"
    bl_label = "Enable Viewport Denoising"
    bl_description = (
        "Turn on Cycles viewport denoising, so DLSS is handed a usable frame "
        "instead of raw path-tracing noise"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == "CYCLES"

    def execute(self, context):
        cycles = getattr(context.scene, "cycles", None)
        if cycles is None:
            self.report({"ERROR"}, "Cycles settings are unavailable")
            return {"CANCELLED"}
        cycles.use_preview_denoising = True
        self.report({"INFO"}, "Viewport denoising enabled")
        return {"FINISHED"}


class DLSS5_OT_use_standard_view(bpy.types.Operator):
    bl_idname = "dlss5.use_standard_view"
    bl_label = "Set View Transform to Standard"
    bl_description = (
        "Switch Color Management to Standard so the enhanced frame is shown "
        "exactly as DLSS produced it"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.view_settings.view_transform = "Standard"
        self.report({"INFO"}, "View Transform set to Standard")
        return {"FINISHED"}


class DLSS5_OT_test_runtime(bpy.types.Operator):
    bl_idname = "dlss5.test_runtime"
    bl_label = "Test Runtime"
    bl_description = "Push one synthetic frame through the worker to verify the install"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            root = _runtime_root(context)
            frame = np.zeros((256, 256, 4), dtype=np.uint8)
            frame[..., 0] = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
            frame[..., 1] = np.linspace(0, 255, 256, dtype=np.uint8)[:, None]
            frame[..., 3] = 255

            with DlssSession(root, context.scene.dlss5.as_dict(), 256, 256) as session:
                fitted = _resize_rgba8(frame, session.render_width, session.render_height)
                output = session.submit(fitted)
        except (DlssRuntimeError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            "Runtime OK - render {}x{} -> output {}x{}, feature 18 evaluated".format(
                session.render_width, session.render_height,
                output.shape[1], output.shape[0],
            ),
        )
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def scene_render_sizes(scene, factor):
    """(reduced_w, reduced_h, full_w, full_h) for the upscaled render path."""
    percentage = scene.render.resolution_percentage
    full_w = int(scene.render.resolution_x * percentage / 100)
    full_h = int(scene.render.resolution_y * percentage / 100)
    reduced = max(1, int(round(percentage / factor)))
    return (int(scene.render.resolution_x * reduced / 100),
            int(scene.render.resolution_y * reduced / 100),
            full_w, full_h)


def viewport_denoise_state(scene):
    """Whether Cycles is denoising the viewport, and at what sample count.

    DLSS reconstructs; it does not remove path-tracing noise. Handing it a
    heavily noisy preview wastes most of what the model can do, and the fix
    lives in Blender's own settings rather than in this add-on - so say so
    instead of silently producing a poor result.
    """
    if scene.render.engine != "CYCLES":
        return None
    cycles = getattr(scene, "cycles", None)
    if cycles is None:
        return None
    return {
        "denoise": bool(getattr(cycles, "use_preview_denoising", False)),
        "samples": int(getattr(cycles, "preview_samples", 0)),
    }


class DLSS5_PT_panel(bpy.types.Panel):
    bl_label = "DLSS 5 Neural Rendering"
    bl_idname = "DLSS5_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.label(icon="SHADERFX")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.dlss5

        try:
            validate_runtime(
                context.preferences.addons[__name__].preferences.runtime_dir
            )
        except DlssRuntimeError as exc:
            box = layout.box()
            box.label(text="Runtime not ready", icon="ERROR")
            for line in str(exc).splitlines()[:4]:
                box.label(text=line)
            box.label(text="Set the runtime folder in the add-on preferences")
            return

        column = layout.column(align=True)
        column.prop(settings, "upscaling")
        column.prop(settings, "model_preset")
        column.prop(settings, "nr_style")

        column = layout.column(align=True)
        column.scale_y = 1.4
        column.operator("dlss5.render_upscaled", icon="RENDER_STILL")
        column.operator("dlss5.enhance_render", icon="SHADERFX")

        box = layout.box()
        factor = UPSCALING_MODES[settings.upscaling][0]
        if factor > 1.0:
            render = scene_render_sizes(context.scene, factor)
            box.label(text="Render Upscaled: {}x{} -> {}x{}".format(*render),
                      icon="INFO")
            box.label(text="Cycles does {:.0f}% of the pixels".format(
                100.0 / (factor * factor)))
        else:
            box.label(text="Mode is DLAA - Render Upscaled saves nothing",
                      icon="INFO")


class DLSS5_PT_viewport(bpy.types.Panel):
    bl_label = "Live Viewport"
    bl_parent_id = "DLSS5_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw_header(self, context):
        self.layout.prop(context.scene.dlss5, "live_viewport", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.dlss5

        if _last_error is not None:
            # Shown here because a print() to the system console is easy to miss.
            box = layout.box()
            box.label(text="Last failure:", icon="ERROR")
            for chunk in _wrap(_last_error, 46):
                box.label(text=chunk)
            box.label(text="Full traceback: dlss5_error.txt in the runtime")

        column = layout.column()
        column.active = settings.live_viewport
        column.prop(settings, "update_rate")
        row = column.row()
        row.active = resolve_frame_source(
            settings,
            next((a.spaces.active for a in context.screen.areas
                  if a.type == "VIEW_3D"), None),
            context.scene) == "SCREEN"
        row.prop(settings, "idle_refresh")
        column.prop(settings, "always_reset")
        column.prop(settings, "frame_source")
        column.prop(settings, "debug_view")

        # Rendered shading has to read the screen; say so rather than silently
        # handing DLSS a rasterised preview of a path-traced view.
        space = next(
            (a.spaces.active for a in context.screen.areas if a.type == "VIEW_3D"),
            None,
        )
        if space is not None:
            source = resolve_frame_source(settings, space, context.scene)
            box = layout.box()
            if source == "SCREEN":
                box.label(text="Source: screen capture (Cycles)",
                          icon="RESTRICT_RENDER_OFF")
                box.label(text="Full resolution by definition, so Mode saves")
                box.label(text="nothing here. One redraw per capture shows the")
                box.label(text="raw frame, and overlays are processed too.")
                box.label(text="Switch the engine to EEVEE for the better path.")
            else:
                box.label(text="Source: off-screen render", icon="SHADING_RENDERED")
                box.label(text="No flicker, no overlays, no settle wait - and")
                box.label(text="Mode is a real quality/speed trade, because the")
                box.label(text="scene is rendered at the size DLSS wants.")

            if _stream is not None and getattr(_stream, "frame_empty", False):
                warn = box.box()
                warn.label(text="Captured frame is empty", icon="ERROR")
                warn.label(text="This source produced nothing in this scene.")
                warn.label(text="Set Frame Source to Screen Capture.")

        if settings.live_viewport:
            box = layout.box()
            if _stream is None:
                box.label(text="Waiting for the first viewport redraw", icon="TIME")
            else:
                out_w, out_h = _stream.output_size
                box.label(
                    text="{}x{} -> {}x{}".format(
                        _stream.render_width, _stream.render_height, out_w, out_h
                    ),
                    icon="CHECKMARK",
                )
                total = _stream.average_ms + _stream.average_render_ms
                box.label(text="scene {:.1f} ms + dlss {:.1f} ms = {:.1f} ms ({:.0f} fps)".format(
                    _stream.average_render_ms, _stream.average_ms, total,
                    1000.0 / total if total else 0.0,
                ))
                box.label(text="{} frames, {} resets, {} reused".format(
                    _stream.index, _stream.resets, _stream.skipped))

                if _stream.delta_mean is not None:
                    changed = _stream.delta_mean > 0.0
                    box.label(
                        text="delta {:.2f} avg, {} max, {:.0f}% pixels".format(
                            _stream.delta_mean, _stream.delta_max,
                            _stream.delta_pct,
                        ),
                        icon="CHECKMARK" if changed else "ERROR",
                    )
                    if not changed:
                        box.label(text="Worker returned the frame unchanged")
                if _stream.feedback_hits:
                    box.label(
                        text="capture close to last output {}x ({:.2f})".format(
                            _stream.feedback_hits,
                            _stream.feedback_score or 0.0),
                        icon="FILE_REFRESH",
                    )

            layout.label(text="Changing a setting restarts the worker (~4 s)",
                         icon="INFO")

        # DLSS reconstructs, it does not denoise. Point at the setting that
        # actually fixes a noisy preview.
        state = viewport_denoise_state(context.scene)
        if state is not None and settings.live_viewport:
            if not state["denoise"]:
                box = layout.box()
                box.label(text="Viewport denoising is off", icon="ERROR")
                box.label(text="DLSS reconstructs detail, it does not remove")
                box.label(text="path-tracing noise. Enable Sampling >")
                box.label(text="Viewport > Denoise for a usable input.")
                box.operator("dlss5.enable_viewport_denoise", icon="SHADERFX")
            elif state["samples"] and state["samples"] < 16:
                box = layout.box()
                box.label(text="Viewport samples: {}".format(state["samples"]),
                          icon="INFO")
                box.label(text="Very noisy input limits what DLSS can rebuild")


class DLSS5_PT_controls(bpy.types.Panel):
    bl_label = "Neural Controls"
    bl_parent_id = "DLSS5_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.dlss5

        column = layout.column(align=True)
        column.prop(settings, "intensity")
        column.prop(settings, "local_tone")
        column.prop(settings, "local_structure")

        column = layout.column(align=True)
        column.prop(settings, "auto_mask")
        row = column.row()
        row.active = settings.auto_mask   # skin work is gated on the mask
        row.prop(settings, "skin_structure")


class DLSS5_PT_output(bpy.types.Panel):
    bl_label = "Still Render"
    bl_parent_id = "DLSS5_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        settings = context.scene.dlss5

        column = layout.column(align=True)
        column.prop(settings, "auto_run")
        column.prop(settings, "save_beside_output")

        # The result is display-referred, so the scene's view transform would
        # otherwise grade it a second time on screen.
        layout.prop(settings, "match_view_transform")
        transform = context.scene.view_settings.view_transform
        box = layout.box()
        if settings.match_view_transform:
            box.label(text="Compensating for {}".format(transform),
                      icon="CHECKMARK")
        elif transform != "Standard":
            box.label(text="{} will grade the result twice".format(transform),
                      icon="ERROR")
            box.operator("dlss5.use_standard_view", icon="COLOR")
        else:
            box.label(text="View transform is Standard - no compensation needed",
                      icon="CHECKMARK")


class DLSS5_PT_diagnostics(bpy.types.Panel):
    bl_label = "Diagnostics"
    bl_parent_id = "DLSS5_PT_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("dlss5.test_runtime", icon="PLUGIN")
        row.operator("dlss5.diagnostics", icon="INFO")
        layout.label(text="Compare > Split judges the result honestly",
                     icon="INFO")
        layout.label(text="Compare > Tint checks the frame reaches the screen",
                     icon="INFO")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@bpy.app.handlers.persistent
def _render_complete(scene, _depsgraph=None):
    if not getattr(scene, "dlss5", None) or not scene.dlss5.auto_run:
        return

    # Handlers run on the render thread; bounce to the main thread before
    # touching image datablocks or spawning the worker.
    def run():
        try:
            bpy.ops.dlss5.enhance_render()
        except RuntimeError as exc:
            print("DLSS 5: automatic run failed:", exc)
        return None

    bpy.app.timers.register(run, first_interval=0.1)


CLASSES = (
    DLSS5Preferences,
    DLSS5Settings,
    DLSS5_OT_enhance_render,
    DLSS5_OT_render_upscaled,
    DLSS5_OT_enable_viewport_denoise,
    DLSS5_OT_use_standard_view,
    DLSS5_OT_diagnostics,
    DLSS5_OT_test_runtime,
    DLSS5_PT_panel,
    DLSS5_PT_viewport,
    DLSS5_PT_controls,
    DLSS5_PT_output,
    DLSS5_PT_diagnostics,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dlss5 = bpy.props.PointerProperty(type=DLSS5Settings)
    if _render_complete not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(_render_complete)


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    if bpy.app.timers.is_registered(_viewport_timer):
        bpy.app.timers.unregister(_viewport_timer)
    _stop_stream()   # never leave a worker process behind
    if _render_complete in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.remove(_render_complete)
    del bpy.types.Scene.dlss5
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
