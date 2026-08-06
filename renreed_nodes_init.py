# Monkeypatch torchaudio.save to fall back to a direct ffmpeg WAV write when
# the torchcodec backend is unavailable/incompatible with this container's
# ffmpeg build ("Could not load libtorchcodec... core4/5/6/7/8.so" — found
# live 2026-08-06). torchaudio.save is looked up fresh on every call (plain
# attribute access, not bound at import time), so patching it here — in our
# own auto-pulled custom-node package, loaded at ComfyUI startup before any
# scene renders — covers ComfyUI-LatentSyncWrapper's torchaudio.save() call
# regardless of custom-node load order.
def _patch_torchaudio_save():
    import torchaudio
    if getattr(torchaudio.save, "_renreed_patched", False):
        return
    _orig_save = torchaudio.save

    def _safe_save(uri, src, sample_rate, *a, **kw):
        try:
            return _orig_save(uri, src, sample_rate, *a, **kw)
        except Exception as e:
            print(f"  [RenReed] torchaudio.save fell back to ffmpeg WAV write ({e})")
            import numpy as np, subprocess
            wav = src.detach().cpu().numpy()
            if wav.ndim == 2:
                wav = wav[0] if wav.shape[0] == 1 else wav.mean(axis=0)
            pcm16 = (wav * 32767.0).clip(-32768, 32767).astype(np.int16)
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(sample_rate),
                                "-ac", "1", "-i", "pipe:0", str(uri)], input=pcm16.tobytes(), capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg WAV fallback also failed: {r.stderr.decode()[:300]}")

    _safe_save._renreed_patched = True
    torchaudio.save = _safe_save


# Same story, second call: ComfyUI-LatentSyncWrapper also calls
# torchvision.io.write_video(...) (nodes.py line 575) to write a temp video
# from the raw frames — this torchvision build has REMOVED write_video
# entirely ("module 'torchvision.io' has no attribute 'write_video'", found
# live 2026-08-06, right after the torchaudio.save patch cleared the
# previous failure). Same fix shape: provide/replace it with a direct
# ffmpeg encode of the raw RGB frames.
def _patch_torchvision_write_video():
    import torchvision.io as tvio
    _orig_write_video = getattr(tvio, "write_video", None)

    def _safe_write_video(filename, video_array, fps, video_codec="libx264", **kw):
        if _orig_write_video is not None:
            try:
                return _orig_write_video(filename, video_array, fps, video_codec=video_codec, **kw)
            except Exception as e:
                print(f"  [RenReed] torchvision.io.write_video fell back to ffmpeg ({e})")
        import numpy as np, subprocess
        arr = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else video_array
        arr = arr.astype(np.uint8)
        t, h, w, c = arr.shape
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(filename)],
                           input=arr.tobytes(), capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg write_video fallback also failed: {r.stderr.decode()[:300]}")

    tvio.write_video = _safe_write_video


# Third call in the same LatentSyncWrapper pass: it writes a temp video
# (write_video, patched above) then reads it back (read_video) for further
# processing — this torchvision build has removed read_video too ("module
# 'torchvision.io' has no attribute 'read_video'", found live 2026-08-06
# right after write_video cleared, on the first dialogue scene that
# actually reaches the lip-sync overlay). Same fix shape: decode the file
# straight from ffmpeg into a [T,H,W,C] uint8 tensor, matching
# torchvision.io.read_video's normal return shape (video, audio, info).
def _patch_torchvision_read_video():
    import torchvision.io as tvio
    _orig_read_video = getattr(tvio, "read_video", None)

    def _safe_read_video(filename, *a, **kw):
        if _orig_read_video is not None:
            try:
                return _orig_read_video(filename, *a, **kw)
            except Exception as e:
                print(f"  [RenReed] torchvision.io.read_video fell back to ffmpeg ({e})")
        import json as _json, subprocess, numpy as np, torch
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(filename)],
            capture_output=True, text=True)
        info = _json.loads(probe.stdout)["streams"][0]
        w, h = info["width"], info["height"]
        num, den = info["r_frame_rate"].split("/")
        fps = float(num) / float(den or 1)
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(filename),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError(f"ffmpeg read_video fallback failed: {r.stderr.decode()[:300]}")
        frames = np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, h, w, 3)
        video = torch.from_numpy(frames.copy())
        audio = torch.zeros((1, 0))
        return video, audio, {"video_fps": fps}

    tvio.read_video = _safe_read_video


try:
    _patch_torchaudio_save()
except Exception as _e:
    print(f"  [RenReed] torchaudio.save patch skipped: {_e}")

try:
    _patch_torchvision_write_video()
except Exception as _e:
    print(f"  [RenReed] torchvision.io.write_video patch skipped: {_e}")

try:
    _patch_torchvision_read_video()
except Exception as _e:
    print(f"  [RenReed] torchvision.io.read_video patch skipped: {_e}")

from .compiler_node import NODE_CLASS_MAPPINGS as A, NODE_DISPLAY_NAME_MAPPINGS as AD
from .character_loader_node import NODE_CLASS_MAPPINGS as B, NODE_DISPLAY_NAME_MAPPINGS as BD
from .renreed_tts_node import NODE_CLASS_MAPPINGS as C, NODE_DISPLAY_NAME_MAPPINGS as CD
from .postmaster_node import NODE_CLASS_MAPPINGS as D, NODE_DISPLAY_NAME_MAPPINGS as DD
from .scene_data_node import NODE_CLASS_MAPPINGS as E, NODE_DISPLAY_NAME_MAPPINGS as ED
NODE_CLASS_MAPPINGS = {**A, **B, **C, **D, **E}
NODE_DISPLAY_NAME_MAPPINGS = {**AD, **BD, **CD, **DD, **ED}
