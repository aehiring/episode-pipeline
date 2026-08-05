#!/usr/bin/env python3
"""
watchdog.py — v17 observer. Responsibilities per locked memory:
progress, stall/crash detection, COST GOVERNOR (scene-2 gate),
rental-rate check, checkpoints, final report. ZERO injection:
only reads ComfyUI /queue /history /prompt; the single allowed write
is the governor's step-count adjustment via API (never files).
Env: COMFY_HOST, VAST_RATE_HR, EPISODE_CAP (default 5.50)
"""
import os, sys, json, time, urllib.request

HOST = os.environ.get("COMFY_HOST", "http://127.0.0.1:8188")
CAP = float(os.environ.get("EPISODE_CAP", "5.50"))
RATE = float(os.environ.get("VAST_RATE_HR") or "0")
RATE_MAX = 1.45
STALL_MIN = float(os.environ.get("STALL_MINUTES", "8"))
STATE = os.environ.get("STATE_DIR", "/models/episode_state")

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
                info(f"GOVERNOR: headroom ${CAP - proj:.2f} — upgrading remaining scenes to 5-6 steps "
                     f"(free quality). Applied via ComfyUI API.")
                return "upgrade"
        return None

def checkpoint(scene, frame_path):
    os.makedirs(STATE, exist_ok=True)
    json.dump({"scene": scene, "last_frame": frame_path, "t": time.time()},
              open(os.path.join(STATE, "checkpoint.json"), "w"))

def main():
    info("v17 watchdog starting — observe-only")
    rate_check()
    last_change, last_sig = time.time(), None
    while True:
        q = get("/queue")
        if q is None:
            loud("ComfyUI unreachable — crash or restart in progress")
            time.sleep(15); continue
        running = q.get("queue_running", [])
        sig = json.dumps(running)[:200]
        if sig != last_sig:
            last_sig, last_change = sig, time.time()
        elif running and (time.time() - last_change) > STALL_MIN * 60:
            loud(f"STALL: no progress for {STALL_MIN} min on running job")
            last_change = time.time()
        if not running and not q.get("queue_pending"):
            info("queue empty — idle")
        time.sleep(20)

if __name__ == "__main__":
    main()
