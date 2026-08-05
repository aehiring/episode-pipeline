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
import os, sys, json, time, shutil, re, urllib.request, urllib.error

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
SCENE_TIMEOUT_MIN = float(os.environ.get("SCENE_TIMEOUT_MINUTES", "20"))
MAX_RETRIES = int(os.environ.get("SCENE_MAX_RETRIES", "2"))
POLL_S = 5

MIN_VIDEO_BYTES = 10000
MIN_IMAGE_BYTES = 10000


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
    last_err = None
    for attempt in range(1, MAX_RETRIES + 2):  # first try + MAX_RETRIES retries
        info(f"scene {n}: submitting (attempt {attempt})")
        governor.scene_started(n)
        try:
            pid = _post_prompt(G.scene_template(ep, scene, anchor_image=ANCHOR_LOCAL_NAME))
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


def run_postmaster(ep):
    info("submitting PostMaster prompt")
    pid = _post_prompt(G.postmaster_template(ep, SCENES_DIR))
    _wait_history(pid, SCENE_TIMEOUT_MIN, "PostMaster")
    final = os.path.join(COMFY_OUTPUT, "EPISODE_FINAL.mp4")
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
    path = os.path.join(COMFY_OUTPUT, "EPISODE_REPORT.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    info(f"report written: {path}")


def run_episode(ep):
    n_scenes = ep["episode"]["scene_count"]
    if n_scenes != len(ep["scenes"]):
        raise RuntimeError(f"watchdog FATAL: episode.scene_count={n_scenes} but {len(ep['scenes'])} scenes present")
    governor = Governor(ep["totals"]["total_frames_16fps"], RATE)
    t_start = time.time()
    run_anchor(ep)
    scene_times = []
    for scene in sorted(ep["scenes"], key=lambda s: s["scene_number"]):
        t_sc = time.time()
        run_scene(ep, scene, governor)
        scene_times.append((scene["scene_number"], time.time() - t_sc))
    run_postmaster(ep)
    _write_report(ep, t_start, time.time(), scene_times)


def main():
    info("v18 watchdog starting — orchestrator + observer")
    rate_check()
    seen_mtime = None
    while True:
        try:
            if os.path.isfile(COMPILED_PATH):
                mtime = os.path.getmtime(COMPILED_PATH)
                if seen_mtime is None or mtime > seen_mtime:
                    seen_mtime = mtime
                    info(f"new episode detected: {COMPILED_PATH}")
                    ep = json.load(open(COMPILED_PATH))
                    run_episode(ep)
        except RuntimeError as e:
            loud(str(e))
        except Exception as e:
            loud(f"watchdog FATAL: unexpected error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
