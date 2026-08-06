"""
RenReedTTS — v17.2. Loud failure on every error path.
Decoder: ffmpeg subprocess (no pydub/torchaudio/torchcodec dependency).
Env: ELEVENLABS_API_KEY, ELEVENLABS_VOICES
"""
import os, io, json, re, subprocess, urllib.request, urllib.error

AUDIO_DIR = os.environ.get("EPISODE_AUDIO_DIR", "/models/input/episode_audio")
API = "https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=mp3_44100_128"
MIN_BYTES = 4000

def _norm_name(name):
    """Multi-word character names ("COACH NIA", "MR PIP") get mangled in
    different ways depending on HOW an instance's env vars were set: Vast's
    API docker-args string splits raw spaces (fixed via quoting in
    rent_instance.py), but a manually-created instance's dashboard env-var
    form was found (2026-08-06) to silently turn spaces into underscores
    instead ("COACH_NIA"). Rather than chase every env-entry path's own
    mangling convention, normalize BOTH sides of the lookup — collapse any
    run of whitespace/underscores to a single space — so the match succeeds
    regardless of which convention a given instance happened to use."""
    return re.sub(r"[_\s]+", " ", name.strip().upper())

def _parse_voice_map(raw=None):
    """A malformed trailing entry (e.g. ELEVENLABS_VOICES truncated by some
    intermediate shell/env-string layer splitting on a space inside a
    multi-word character name like "COACH NIA") is logged and skipped rather
    than failing the whole map — _require_voice() below still fails loudly
    for any speaker that's actually missing a voice id, so a genuinely
    needed voice is never silently substituted; only unrelated/unreachable
    trailing junk is tolerated."""
    raw = raw if raw is not None else os.environ.get("ELEVENLABS_VOICES", "")
    m = {}
    for part in [p for p in raw.split(",") if p.strip()]:
        if ":" not in part:
            print(f"  [RenReedTTS WARN] skipping malformed ELEVENLABS_VOICES entry '{part}' (need NAME:voice_id)")
            continue
        k, v = part.split(":", 1)
        if not v.strip():
            print(f"  [RenReedTTS WARN] skipping empty voice id for '{k.strip()}'")
            continue
        m[_norm_name(k)] = v.strip()
    return m

def _require_voice(speaker, vmap):
    sp = _norm_name(speaker)
    if sp == "NONE": return None
    if sp not in vmap:
        raise RuntimeError(f"RenReedTTS FATAL: no voice id for '{sp}'. ELEVENLABS_VOICES has: {sorted(vmap)}")
    return vmap[sp]

def _tts_bytes(text, vid, key):
    body = json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode()
    req = urllib.request.Request(API.format(vid=vid), data=body, headers={
        "xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RenReedTTS FATAL: ElevenLabs HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        raise RuntimeError(f"RenReedTTS FATAL: ElevenLabs unreachable: {e}")
    if len(data) < MIN_BYTES:
        raise RuntimeError(f"RenReedTTS FATAL: response only {len(data)} bytes — not audio: {data[:80]!r}")
    return data

def _decode_mp3(data):
    """Decode MP3 via ffmpeg subprocess — zero extra packages required."""
    import numpy as _np, torch as _t
    SR = 44100
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=data, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 4:
        raise RuntimeError(f"RenReedTTS FATAL: ffmpeg MP3 decode failed: {r.stderr.decode()[:300]}")
    samples = _np.frombuffer(r.stdout, dtype=_np.float32).copy()
    wav = _t.from_numpy(samples).unsqueeze(0)   # [1, N]
    if wav.numel() <= 1:
        raise RuntimeError("RenReedTTS FATAL: decoded waveform empty — refusing silent output")
    return {"waveform": wav.unsqueeze(0), "sample_rate": SR}

def _ep(episode_json):
    """Accept str OR dict."""
    if isinstance(episode_json, dict): return episode_json
    if isinstance(episode_json, str):  return json.loads(episode_json.strip())
    raise RuntimeError(f"RenReedTTS FATAL: episode_json is {type(episode_json)}, expected str or dict")

class RenReedTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "episode_json": ("STRING", {"forceInput": True}),
            "scene_number": ("INT", {"default": 1, "min": 1, "max": 500, "forceInput": True}),
        }}
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "speak"
    CATEGORY = "RenReed"

    def speak(self, episode_json, scene_number):
        key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not key:
            raise RuntimeError("RenReedTTS FATAL: ELEVENLABS_API_KEY not set")
        ep = _ep(episode_json)
        sc = next((s for s in ep["scenes"] if s["scene_number"] == scene_number), None)
        if sc is None:
            raise RuntimeError(f"RenReedTTS FATAL: scene {scene_number} not in episode JSON")
        os.makedirs(AUDIO_DIR, exist_ok=True)
        path = os.path.join(AUDIO_DIR, f"scene_{scene_number:03d}.mp3")
        vmap = _parse_voice_map()
        vid = _require_voice(sc["speaker"], vmap)
        if vid is None:
            import torch; sr = 44100
            silent = {"waveform": torch.zeros(1,1,sr), "sample_rate": sr}
            open(path, "wb").close(); return (silent, path)
        data = _tts_bytes(sc["dialogue"], vid, key)
        with open(path, "wb") as f: f.write(data)
        audio = _decode_mp3(data)
        dur = audio["waveform"].shape[-1] / audio["sample_rate"]
        if dur > sc["duration_s"] + 2.0:
            print(f"  [RenReedTTS WARN] scene {scene_number}: speech {dur:.2f}s > scene {sc['duration_s']}s — VHS will trim")
        return (audio, path)

NODE_CLASS_MAPPINGS = {"RenReedTTS": RenReedTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"RenReedTTS": "Ren&Reed TTS (ElevenLabs)"}
