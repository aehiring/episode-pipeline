#!/usr/bin/env python3
"""
fetch_final.py — LOCAL-ONLY helper. Run this on your own PC (it is NOT baked
into the Docker image and NOT pulled from GitHub — it has nothing to do with
the render itself). Watches a running ComfyUI instance for EPISODE_FINAL.mp4
and downloads it to a local folder the moment it's ready, so nobody has to
sit and babysit the browser tab to grab the finished episode.

No new credentials needed: this uses the exact same HTTP port you already use
to open ComfyUI in a browser (ComfyUI serves output files over `/view`).

Usage:  python fetch_final.py http://<vast-instance-ip>:<port> [output_dir]
"""
import sys, os, time, urllib.request, urllib.error

POLL_S = 15
STABLE_CHECKS = 2  # require the same byte count twice in a row before downloading,
                   # so we never grab a file mid-write


def _content_length(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return None


def wait_and_fetch(base_url, out_dir, poll_s=POLL_S, stable_checks=STABLE_CHECKS, log=print):
    os.makedirs(out_dir, exist_ok=True)
    url = f"{base_url.rstrip('/')}/view?filename=EPISODE_FINAL.mp4&type=output"
    log(f"[fetch_final] watching {url}")

    stable, last_len = 0, None
    while True:
        n = _content_length(url)
        if n:
            stable = stable + 1 if n == last_len else 1
            last_len = n
            log(f"[fetch_final] EPISODE_FINAL.mp4 visible, {n} bytes ({stable}/{stable_checks} stable checks)")
            if stable >= stable_checks:
                break
        else:
            log("[fetch_final] not ready yet...")
            stable, last_len = 0, None
        time.sleep(poll_s)

    dest = os.path.join(out_dir, "EPISODE_FINAL.mp4")
    log(f"[fetch_final] downloading -> {dest}")
    urllib.request.urlretrieve(url, dest)
    log(f"[fetch_final] done: {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python fetch_final.py http://<host>:<port> [output_dir]")
    base_url = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output")
    wait_and_fetch(base_url, out_dir)


if __name__ == "__main__":
    main()
