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


try:
    _patch_torchaudio_save()
except Exception as _e:
    print(f"  [RenReed] torchaudio.save patch skipped: {_e}")

from .compiler_node import NODE_CLASS_MAPPINGS as A, NODE_DISPLAY_NAME_MAPPINGS as AD
from .character_loader_node import NODE_CLASS_MAPPINGS as B, NODE_DISPLAY_NAME_MAPPINGS as BD
from .renreed_tts_node import NODE_CLASS_MAPPINGS as C, NODE_DISPLAY_NAME_MAPPINGS as CD
from .postmaster_node import NODE_CLASS_MAPPINGS as D, NODE_DISPLAY_NAME_MAPPINGS as DD
from .scene_data_node import NODE_CLASS_MAPPINGS as E, NODE_DISPLAY_NAME_MAPPINGS as ED
NODE_CLASS_MAPPINGS = {**A, **B, **C, **D, **E}
NODE_DISPLAY_NAME_MAPPINGS = {**AD, **BD, **CD, **DD, **ED}
