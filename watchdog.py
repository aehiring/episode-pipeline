#!/usr/bin/env python3
"""
watchdog.py — v18. Scene orchestrator + observer.

Replaces ComfyUI's in-graph while-loop (buggy past 1 iteration) with a plain
Python loop: watch for a freshly-compiled episode -> render the anchor once ->
render each scene 1..scene_count as its own /prompt submission (retrying a
failed scene on its own, never taking the whole episode down) -> run
PostMaster once every scene file is confirmed present. This is the only thing
that changed vs the original design; rate_check/Governor/checkpoint below are
unchanged from the prior observe-only watchdog.

Env: COMFY_HOST, VAST_RATE_HR, EPISODE_CAP (default 5.50), EPISODE_COMPILED_PATH,
     SCENE_TIMEOUT_MINUTES (default 20), SCENE_MAX_RETRIES (default 2)
"""
import os, sys, json, time, shutil, re, urllib.request, urllib.error, urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v17_graph as G

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
CAP = float(os.environ.get("EPISODE_CAP", "5.50"))
RATE = float(os.environ.get("VAST_RATE_HR") or "0")
RATE_MAX = 1.45
STALL_MIN = float(os.environ.get("STALL_MINUTES", "8"))
STATE = os.environ.get("STATE_DIR", "/models/episode_state")

COMPILED_PATH = os.environ.get("EPISODE_COMPILED_PATH", "/models/input/episode_compiled.json")
COMFY_INPUT = os.environ.get("COMFY_INPUT", "/models/input")
COMFY_OUTPUT = os.environ.get("COMFY_OUTPUT", "/models/output")
SCENES_DIR = os.path.join(COMFY_OUTPUT, "scenes")
ANCHOR_LOCAL_NAME = "anchor_current.png"
MUSIC_PATH = os.environ.get("EPISODE_MUSIC_PATH", os.path.join(COMFY_OUTPUT, "episode_music.mp3"))
MIN_MUSIC_BYTES = 20000
SCENE_TIMEOUT_MIN = float(os.environ.get("SCENE_TIMEOUT_MINUTES", "20"))
MAX_RETRIES = int(os.environ.get("SCENE_MAX_RETRIES", "2"))
POLL_S = 5

MIN_VIDEO_BYTES = 10000
MIN_IMAGE_BYTES = 10000
SFX_DIR = os.environ.get("SFX_ASSET_DIR", "/opt/pipeline/assets/sfx")
MIN_SFX_BYTES = 6000
_MUSIC_NOTE = None  # set by run_music(), surfaced in EPISODE_REPORT.txt instead of a loud console banner


def get(path):
    try:
        with urllib.request.urlopen(HOST + path, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def loud(msg):  print(f"\n{'!'*66}\n!! {msg}\n{'!'*66}\n", flush=True)
def info(msg):  print(f"[watchdog] {msg}", flush=True)


def rate_check():
    if RATE <= 0:
        loud("VAST_RATE_HR not set — cannot guarantee the $%.2f cap. Set it in the template." % CAP)
    elif RATE > RATE_MAX:
        loud(f"RENTAL RATE ${RATE}/hr > ${RATE_MAX}/hr rule — cap guarantee is BROKEN. "
             f"Destroy and re-rent a cheaper RTX PRO 6000 listing.")
    else:
        info(f"rate OK: ${RATE}/hr (rule <= ${RATE_MAX})")


class Governor:
    """Scene-2 gate: measure real min/frame, project, adjust once."""
    def __init__(self, total_frames, rate_hr):
        self.total = total_frames; self.rate = rate_hr
        self.t0 = {}; self.decided = False
    def scene_started(self, n): self.t0[n] = time.time()
    def scene_done(self, n, frames):
        dt_min = (time.time() - self.t0.get(n, time.time())) / 60
        mpf = dt_min / max(frames, 1)
        proj = self.total * mpf / 60 * self.rate
        info(f"scene {n}: {dt_min:.1f} min, {mpf:.4f} min/frame, projected episode ${proj:.2f}")
        if n == 2 and not self.decided:
            self.decided = True
            if proj > CAP - 0.10:
                loud(f"GOVERNOR: projection ${proj:.2f} near/over cap ${CAP}. Steps already at floor 4. "
                     f"Episode will finish but review before next run.")
                return "hold4"
            if proj < CAP - 0.90:
                info(f"GOVERNOR: headroom ${CAP - proj:.2f} noted — steps stay at floor 4 (quality-safe default).")
        return None


def checkpoint(scene, frame_path):
    os.makedirs(STATE, exist_ok=True)
    json.dump({"scene": scene, "last_frame": frame_path, "t": time.time()},
              open(os.path.join(STATE, "checkpoint.json"), "w"))


# ── Model presence / runtime self-heal ─────────────────────────────────
# Mirrors entrypoint.sh's model list in pure Python (huggingface_hub API,
# no shell/CLI dependency) so a missing or corrupt/interrupted download can
# be fixed DURING a run — no instance reboot needed, per explicit
# requirement. Checked once at watchdog startup and again at the top of
# every run_episode() (idempotent — a no-op once everything verifies OK).
MODELS_ROOT = "/models"
_C = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
_CAM = "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera"
_LX2V = "lightx2v/Wan2.2-Lightning"

# (repo, path_in_repo, dest_relpath, min_bytes, required)
# required=True blocks the episode if it can't be fixed; required=False is
# the action/camera upgrade — best-effort, only scenes that need it are hit.
HF_MODELS = [
    (_C, "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", "diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", 5_000_000_000, True),
    (_C, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 3_000_000_000, True),
    (_C, "split_files/vae/wan_2.1_vae.safetensors", "vae/wan_2.1_vae.safetensors", 100_000_000, True),
    (_C, "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", "audio_encoders/wav2vec2_large_english_fp16.safetensors", 300_000_000, True),
    ("Comfy-Org/flux1-schnell", "flux1-schnell-fp8.safetensors", "diffusion_models/flux1-schnell-fp8.safetensors", 10_000_000_000, True),
    ("Comfy-Org/flux1-kontext-dev_ComfyUI", "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", "diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", 10_000_000_000, True),
    ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "text_encoders/clip_l.safetensors", 200_000_000, True),
    ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors", 3_000_000_000, True),
    ("Kim2091/UltraSharp", "4x-UltraSharp.pth", "upscale_models/4x-UltraSharp.pth", 50_000_000, True),
    (_C, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 5_000_000_000, False),
    (_C, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", 5_000_000_000, False),
    (_CAM, "high_noise_model/diffusion_pytorch_model.safetensors", "diffusion_models/wan2.2_fun_camera_high_noise_14B.safetensors", 5_000_000_000, False),
    (_CAM, "low_noise_model/diffusion_pytorch_model.safetensors", "diffusion_models/wan2.2_fun_camera_low_noise_14B.safetensors", 5_000_000_000, False),
    (_LX2V, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors", "loras/wan22_i2v_lightx2v_4steps_high_noise.safetensors", 200_000_000, False),
    (_LX2V, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors", "loras/wan22_i2v_lightx2v_4steps_low_noise.safetensors", 200_000_000, False),
]

# direct-URL fallbacks (entrypoint.sh's non-HF-API curl downloads)
URL_MODELS = [
    ("https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors", "vae/ae.safetensors", 300_000_000, True),
    ("https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
     "loras/wan22_lightning_fallback_high.safetensors", 400_000_000, True),
]

_MODELS_VERIFIED_OK = False
_last_models_check = 0.0
MODELS_RECHECK_THROTTLE_S = 120  # don't hammer HF/network more than once per 2 min while broken
# local_pipeline_test.py sets this — the CPU-only/zero-GPU-cost local test suite
# must never trigger real multi-GB network downloads onto the dev machine.
SKIP_MODEL_CHECK = os.environ.get("WATCHDOG_SKIP_MODEL_CHECK", "").strip() == "1"


def _model_ok(dest, min_bytes):
    return os.path.isfile(dest) and os.path.getsize(dest) >= min_bytes


def _hf_download_one(repo, path_in_repo, dest):
    from huggingface_hub import hf_hub_download  # same package entrypoint.sh already ensures is installed
    tmp_dir = "/tmp/watchdog_hf"
    os.makedirs(tmp_dir, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=path_in_repo, local_dir=tmp_dir)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(got, dest)


def _url_download_one(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_models():
    """Verify every model file entrypoint.sh should have fetched is present
    and correctly sized; re-download anything missing/short right here in
    Python. A no-op (single boolean check) once everything has verified OK,
    and throttled to at most one real network retry pass per 2 minutes
    while something is still broken, so a genuine outage doesn't turn into
    a tight request loop."""
    global _MODELS_VERIFIED_OK, _last_models_check
    if _MODELS_VERIFIED_OK:
        return
    if SKIP_MODEL_CHECK:
        info("ensure_models: WATCHDOG_SKIP_MODEL_CHECK=1 — skipping (local/offline test mode)")
        _MODELS_VERIFIED_OK = True
        return
    now = time.time()
    if now - _last_models_check < MODELS_RECHECK_THROTTLE_S:
        return
    _last_models_check = now
    missing_required = []
    for repo, path_in_repo, rel, min_bytes, required in HF_MODELS:
        dest = os.path.join(MODELS_ROOT, rel)
        if _model_ok(dest, min_bytes):
            continue
        info(f"ensure_models: {rel} missing/short ({'required' if required else 'optional action/camera'}) — fetching from {repo}")
        try:
            _hf_download_one(repo, path_in_repo, dest)
            info(f"ensure_models: {rel} OK ({os.path.getsize(dest)//1024//1024} MB)")
        except Exception as e:
            msg = f"ensure_models: FAILED to fetch {rel} from {repo}: {e}"
            if required:
                loud(msg); missing_required.append(rel)
            else:
                info(msg + " — action/camera path unavailable until fixed; cheap talking-only scenes unaffected")
    for url, rel, min_bytes, required in URL_MODELS:
        dest = os.path.join(MODELS_ROOT, rel)
        if _model_ok(dest, min_bytes):
            continue
        info(f"ensure_models: {rel} missing/short — fetching from {url}")
        try:
            _url_download_one(url, dest)
            info(f"ensure_models: {rel} OK ({os.path.getsize(dest)//1024//1024} MB)")
        except Exception as e:
            msg = f"ensure_models: FAILED to fetch {rel} from {url}: {e}"
            if required:
                loud(msg); missing_required.append(rel)
            else:
                info(msg)
    if missing_required:
        raise RuntimeError(f"watchdog FATAL: required model(s) still missing after runtime retry: {missing_required}")
    _MODELS_VERIFIED_OK = True
    info("ensure_models: all required models verified OK")


# ── Python dependency self-heal ────────────────────────────────────────
# Same idea as ensure_models() but for a missing pip package rather than a
# model file — lets a dependency gap found on a live run (e.g.
# ComfyUI-LatentSyncWrapper needing torchcodec for torchaudio.save(), found
# 2026-08-06) be installed into the running container via a reboot (which
# re-pulls and re-runs this file) instead of requiring a whole new Docker
# image + fresh instance rent, matching the Dockerfile fix that also
# bakes it in for future builds from scratch.
PYTHON_DEPS = ["torchcodec"]
_PY_DEPS_VERIFIED_OK = False


def ensure_python_deps():
    global _PY_DEPS_VERIFIED_OK
    if _PY_DEPS_VERIFIED_OK or SKIP_MODEL_CHECK:
        _PY_DEPS_VERIFIED_OK = True
        return
    import importlib, subprocess
    for pkg in PYTHON_DEPS:
        try:
            importlib.import_module(pkg)
            continue
        except ImportError:
            pass
        info(f"ensure_python_deps: {pkg} missing — installing")
        r = subprocess.run(["pip", "install", "-q", pkg], capture_output=True, text=True)
        if r.returncode != 0:
            loud(f"ensure_python_deps: FAILED to install {pkg}: {r.stderr[-300:]}")
        else:
            info(f"ensure_python_deps: {pkg} installed OK")
    _PY_DEPS_VERIFIED_OK = True


# ── Self-growing SFX library ───────────────────────────────────────────

def _drop_sfx(ep, names):
    """Strip sfx events we couldn't secure a file for, from every scene —
    lets the episode finish (video+dialogue intact) instead of PostMaster
    hard-failing on a missing mux input after all the GPU rendering is
    already done."""
    names = set(names)
    for sc in ep["scenes"]:
        sc["sfx"] = [fx for fx in sc["sfx"] if fx["name"] not in names]


def ensure_sfx_library(ep):
    """This pipeline renders ANY topic, so SFX can't be capped to one fixed
    list: any sfx name this episode's scenes actually use, that isn't
    already baked in SFX_DIR (the seed library or a prior episode's
    on-demand additions), is generated now via ElevenLabs sound-generation
    and added for reuse. A name that truly can't be generated (bad key, API
    down) is dropped from its scene rather than failing the whole episode at
    the final PostMaster mux step."""
    names = sorted({fx["name"] for sc in ep["scenes"] for fx in sc["sfx"]})
    if not names:
        return
    os.makedirs(SFX_DIR, exist_ok=True)
    missing = [n for n in names if not _model_ok(os.path.join(SFX_DIR, n + ".mp3"), MIN_SFX_BYTES)]
    if not missing:
        info(f"sfx library: all {len(names)} sound(s) already present")
        return
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        loud(f"ELEVENLABS_API_KEY not set — cannot generate {len(missing)} new sfx {missing}; "
             f"dropping these sfx events (video/dialogue unaffected)")
        _drop_sfx(ep, missing)
        return
    info(f"sfx library: generating {len(missing)} new sound(s) on demand: {missing}")
    failed = []
    for name in missing:
        prompt = name.replace("_", " ") + ", short cartoon sound effect, clean, family-friendly"
        out = os.path.join(SFX_DIR, name + ".mp3")
        try:
            body = json.dumps({"text": prompt, "duration_seconds": 2.5, "prompt_influence": 0.5}).encode()
            req = urllib.request.Request("https://api.elevenlabs.io/v1/sound-generation", data=body,
                headers={"xi-api-key": key, "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < MIN_SFX_BYTES:
                raise RuntimeError(f"response too small ({len(data)}B)")
            with open(out, "wb") as f:
                f.write(data)
            info(f"sfx generated: {name} ({len(data)//1024} KB)")
        except Exception as e:
            loud(f"sfx generation failed for '{name}': {e} — dropping this sfx event")
            failed.append(name)
    if failed:
        _drop_sfx(ep, failed)


# ── ComfyUI API plumbing ──────────────────────────────────────────────

def _post_prompt(graph):
    body = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(HOST + "/prompt", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"watchdog FATAL: /prompt submit HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:
        raise RuntimeError(f"watchdog FATAL: /prompt submit failed: {e}")
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"watchdog FATAL: /prompt response missing prompt_id: {data}")
    return pid


def _interrupt():
    try:
        req = urllib.request.Request(HOST + "/interrupt", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass


def _wait_history(prompt_id, timeout_min, label):
    """Poll /history until the prompt completes or errors; loud stall message
    (not fatal on its own) if the queue stops changing; hard timeout raises."""
    deadline = time.time() + timeout_min * 60
    last_sig, last_change = None, time.time()
    while time.time() < deadline:
        h = get(f"/history/{prompt_id}")
        if h and prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed") is True:
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(f"watchdog FATAL: {label} errored in ComfyUI: {json.dumps(status)[:500]}")
        q = get("/queue")
        sig = json.dumps(q)[:200] if q else None
        if sig != last_sig:
            last_sig, last_change = sig, time.time()
        elif time.time() - last_change > STALL_MIN * 60:
            loud(f"STALL: {label} — no queue change for {STALL_MIN:.0f} min (still waiting, timeout at {timeout_min:.0f} min)")
            last_change = time.time()
        time.sleep(POLL_S)
    _interrupt()
    raise RuntimeError(f"watchdog FATAL: {label} timed out after {timeout_min:.0f} min")


def _find_output_file(entry, key_hint=None):
    """History 'outputs' holds {node_id: {ui_key: [{filename, subfolder, type}, ...]}}
    for every OUTPUT_NODE that ran (SaveImage->'images', VHS_VideoCombine->'gifs' on
    most installs). key_hint is tried first but not required — different
    ComfyUI-VideoHelperSuite versions have used different key names, so fall back
    to the first file-like entry under any key rather than failing on a name guess."""
    hinted, any_found = None, None
    for node_out in entry.get("outputs", {}).values():
        for k, v in node_out.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "filename" in v[0]:
                if any_found is None:
                    any_found = (v[0]["filename"], v[0].get("subfolder", ""))
                if key_hint and k == key_hint and hinted is None:
                    hinted = (v[0]["filename"], v[0].get("subfolder", ""))
    return hinted or any_found or (None, None)


# ── Episode orchestration ─────────────────────────────────────────────

def run_anchor(ep):
    dest = os.path.join(COMFY_INPUT, ANCHOR_LOCAL_NAME)
    if os.path.isfile(dest) and os.path.getsize(dest) >= MIN_IMAGE_BYTES:
        info(f"ANCHOR already present, skipping re-render: {dest}")
        return
    info("submitting ANCHOR prompt")
    pid = _post_prompt(G.anchor_template(ep))
    entry = _wait_history(pid, SCENE_TIMEOUT_MIN, "ANCHOR")
    fname, subfolder = _find_output_file(entry, key_hint="images")
    if not fname:
        raise RuntimeError("watchdog FATAL: ANCHOR prompt completed but no saved image found in history")
    src = os.path.join(COMFY_OUTPUT, subfolder, fname)
    if not os.path.isfile(src) or os.path.getsize(src) < MIN_IMAGE_BYTES:
        raise RuntimeError(f"watchdog FATAL: ANCHOR image missing/too small: {src}")
    os.makedirs(COMFY_INPUT, exist_ok=True)
    dest = os.path.join(COMFY_INPUT, ANCHOR_LOCAL_NAME)
    shutil.copy(src, dest)
    info(f"ANCHOR ready: {dest}")


def run_scene(ep, scene, governor):
    n = scene["scene_number"]
    dest = os.path.join(SCENES_DIR, f"scene_{n:05d}.mp4")
    if os.path.isfile(dest) and os.path.getsize(dest) >= MIN_VIDEO_BYTES:
        info(f"scene {n}: already rendered, skipping -> {dest}")
        return
    use_action_path = G._needs_action_path(scene)
    build_graph = (lambda: G.scene_template_action(ep, scene, anchor_image=ANCHOR_LOCAL_NAME)) if use_action_path \
        else (lambda: G.scene_template(ep, scene, anchor_image=ANCHOR_LOCAL_NAME))
    if use_action_path:
        info(f"scene {n}: action={scene.get('has_physical_action')} camera={scene.get('camera_motion')} -> full action/camera path")

    last_err = None
    for attempt in range(1, MAX_RETRIES + 2):  # first try + MAX_RETRIES retries
        info(f"scene {n}: submitting (attempt {attempt})")
        governor.scene_started(n)
        try:
            pid = _post_prompt(build_graph())
            entry = _wait_history(pid, SCENE_TIMEOUT_MIN, f"scene {n} (attempt {attempt})")
            fname, subfolder = _find_output_file(entry, key_hint="gifs")
            if not fname:
                raise RuntimeError(f"scene {n}: no saved video found in history outputs")
            src = os.path.join(COMFY_OUTPUT, subfolder, fname)
            if not os.path.isfile(src) or os.path.getsize(src) < MIN_VIDEO_BYTES:
                raise RuntimeError(f"scene {n}: output file missing/too small: {src}")
            os.makedirs(SCENES_DIR, exist_ok=True)
            shutil.move(src, dest)  # deterministic name, retry-safe overwrite
            governor.scene_done(n, scene["chunks"] * 77)
            checkpoint(n, dest)
            info(f"scene {n}: OK -> {dest}")
            return
        except RuntimeError as e:
            last_err = e
            info(f"scene {n}: attempt {attempt} FAILED: {e}")
            if os.path.isfile(dest):
                os.remove(dest)  # never leave a partial/stale file behind for postmaster to trip on
    raise RuntimeError(f"watchdog FATAL: scene {n} failed after {MAX_RETRIES + 1} attempts. Last error: {last_err}")


JAMENDO_API = "https://api.jamendo.com/v3.0/tracks/"


def _jamendo_search(client_id, **params):
    q = urllib.parse.urlencode({"client_id": client_id, "format": "json", "limit": 5,
        "vocalinstrumental": "instrumental", "order": "popularity_total", "audioformat": "mp31", **params})
    with urllib.request.urlopen(f"{JAMENDO_API}?{q}", timeout=30) as r:
        return json.loads(r.read().decode()).get("results", [])


def _jamendo_music(ep):
    """Real royalty-free track, queried by THIS episode's own title/lesson
    text (whatever topic it actually is — never a hardcoded genre/mood), so
    it's relevant per-episode rather than a fixed loop. Falls back to a
    generic pleasant-instrumental search if the topic query has no matches."""
    client_id = os.environ.get("JAMENDO_CLIENT_ID", "").strip()
    if not client_id:
        return None
    query = f"{ep['episode'].get('title', '')} {ep['episode'].get('lesson', '')}".strip()
    try:
        results = _jamendo_search(client_id, search=query) if query else []
        if not results:
            info(f"Jamendo: no match for '{query}', trying generic pleasant-instrumental search")
            results = _jamendo_search(client_id, tags="children happy")
    except Exception as e:
        info(f"Jamendo search failed: {e}")
        return None
    if not results:
        return None
    track = results[0]
    audio_url = track.get("audio")
    if not audio_url:
        return None
    try:
        with urllib.request.urlopen(audio_url, timeout=60) as r:
            audio = r.read()
    except Exception as e:
        info(f"Jamendo track download failed: {e}")
        return None
    if len(audio) < MIN_MUSIC_BYTES:
        return None
    return audio, track.get("name", "?"), track.get("artist_name", "?")


def _elevenlabs_music(ep, duration_ms):
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        return None
    title = ep["episode"].get("title", "")
    lesson = ep["episode"].get("lesson", "")
    prompt = (f"Instrumental children's cartoon background score, warm, bright, and gentle. "
              f"Matches the mood of a story titled '{title}' whose lesson is: {lesson}. "
              f"No vocals, no lyrics — instrumental only, suitable to sit quietly under dialogue.")
    body = json.dumps({"prompt": prompt, "music_length_ms": duration_ms}).encode()
    req = urllib.request.Request("https://api.elevenlabs.io/v1/music", data=body,
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except Exception as e:
        info(f"ElevenLabs Music fallback failed: {e}")
        return None
    if len(data) < MIN_MUSIC_BYTES:
        return None
    return data, "ElevenLabs generated score", "ElevenLabs"


def run_music(ep):
    """One background score per episode. Jamendo (free, real royalty-free
    tracks, searched using THIS episode's own title/lesson so the result is
    relevant to whatever topic it actually is) is tried first; ElevenLabs
    Music is the fallback if Jamendo has no client id or no match. If both
    fail, that's recorded as a plain note in EPISODE_REPORT.txt — NOT a loud
    console failure banner — the episode still finishes with dialogue+SFX
    only, since PostMaster's mix already tolerates a missing music file."""
    global _MUSIC_NOTE
    if os.path.isfile(MUSIC_PATH) and os.path.getsize(MUSIC_PATH) >= MIN_MUSIC_BYTES:
        info(f"episode music already present, skipping: {MUSIC_PATH}")
        return
    result = _jamendo_music(ep)
    source = "Jamendo"
    if result is None:
        duration_ms = min(600000, max(3000, int(ep["totals"]["sum_duration_s"] * 1000)))
        result = _elevenlabs_music(ep, duration_ms)
        source = "ElevenLabs"
    if result is None:
        _MUSIC_NOTE = "No background score: Jamendo and ElevenLabs both unavailable/no match — episode has dialogue+SFX only."
        info(_MUSIC_NOTE)
        return
    data, track_name, artist = result
    os.makedirs(os.path.dirname(MUSIC_PATH), exist_ok=True)
    with open(MUSIC_PATH, "wb") as f:
        f.write(data)
    _MUSIC_NOTE = f"Background score: '{track_name}' by {artist} (via {source}), {len(data)//1024} KB, mixed quietly under dialogue/SFX."
    info(f"episode music ready: {MUSIC_PATH} ({len(data)//1024} KB, via {source})")


def run_postmaster(ep):
    final = os.path.join(COMFY_OUTPUT, "EPISODE_FINAL.mp4")
    if os.path.isfile(final) and os.path.getsize(final) >= MIN_VIDEO_BYTES:
        info(f"EPISODE_FINAL.mp4 already present, skipping PostMaster re-run: {final}")
        return
    info("submitting PostMaster prompt")
    pid = _post_prompt(G.postmaster_template(ep, SCENES_DIR))
    _wait_history(pid, SCENE_TIMEOUT_MIN, "PostMaster")
    if not os.path.isfile(final):
        raise RuntimeError(f"watchdog FATAL: PostMaster reported done but {final} is missing")
    info(f"EPISODE READY: {final} ({os.path.getsize(final)//1024} KB)")


def _write_report(ep, t0, t1, scene_times):
    total_s = t1 - t0
    lines = [
        f"Episode: {ep['episode'].get('title', '')}",
        f"Started:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}",
        f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))}",
        f"Total time: {total_s/60:.1f} min ({total_s:.0f}s)",
        "",
        "Per-scene render time:",
    ]
    for n, dt in scene_times:
        lines.append(f"  scene {n}: {dt/60:.2f} min")
    if RATE > 0:
        lines += ["", f"Estimated GPU cost: ${(total_s/3600)*RATE:.2f} at ${RATE}/hr"]
    if _MUSIC_NOTE:
        lines += ["", _MUSIC_NOTE]
    path = os.path.join(COMFY_OUTPUT, "EPISODE_REPORT.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    info(f"report written: {path}")


def run_episode(ep):
    n_scenes = ep["episode"]["scene_count"]
    if n_scenes != len(ep["scenes"]):
        raise RuntimeError(f"watchdog FATAL: episode.scene_count={n_scenes} but {len(ep['scenes'])} scenes present")
    ensure_models()
    ensure_python_deps()
    ensure_sfx_library(ep)
    governor = Governor(ep["totals"]["total_frames_16fps"], RATE)
    t_start = time.time()
    run_anchor(ep)
    scene_times = []
    for scene in sorted(ep["scenes"], key=lambda s: s["scene_number"]):
        t_sc = time.time()
        run_scene(ep, scene, governor)
        scene_times.append((scene["scene_number"], time.time() - t_sc))
    run_music(ep)
    run_postmaster(ep)
    _write_report(ep, t_start, time.time(), scene_times)


def main():
    info("v18 watchdog starting — orchestrator + observer")
    rate_check()
    try:
        ensure_models()
    except RuntimeError as e:
        loud(str(e) + " — will keep retrying at runtime, no reboot needed once fixed")
    ensure_python_deps()
    seen_mtime = None
    while True:
        try:
            if os.path.isfile(COMPILED_PATH):
                mtime = os.path.getmtime(COMPILED_PATH)
                if seen_mtime is None or mtime > seen_mtime:
                    info(f"new episode detected: {COMPILED_PATH}")
                    ep = json.load(open(COMPILED_PATH))
                    run_episode(ep)
                    seen_mtime = mtime  # only marked seen after a FULL successful run — every
                    # step above (models, anchor, each scene, postmaster) is already idempotent
                    # (skip-if-done), so retrying the whole episode on any failure is cheap/safe
        except RuntimeError as e:
            loud(str(e))
        except Exception as e:
            loud(f"watchdog FATAL: unexpected error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
