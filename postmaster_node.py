"""
EpisodePostMaster — final OUTPUT node (v17).
Input: concatenated 480p/24fps episode file (post per-scene RIFE) + episode JSON.
Does: lanczos 720p + locked grade + SFX mux (from baked library) + crf17,
writes final into ComfyUI output dir, returns inline player + download.
"""
import os, json, subprocess, shutil

SFX_DIR = os.environ.get("SFX_ASSET_DIR", "/opt/pipeline/assets/sfx")
MUSIC_PATH = os.environ.get("EPISODE_MUSIC_PATH", "/models/output/episode_music.mp3")
MUSIC_VOLUME = 0.2  # ducked well under dialogue/SFX
GRADE = "fps=24,scale=1280:720:flags=lanczos,eq=contrast=1.06:saturation=1.14:gamma=0.98," \
        "unsharp=5:5:0.45:5:5:0.0,noise=alls=1.5:allf=t,format=yuv420p"

def build_sfx_cmd(src, ep, out, sfx_dir=None, grade=GRADE, music_path=None):
    sfx_dir = sfx_dir or SFX_DIR
    music_path = music_path if music_path is not None else MUSIC_PATH
    has_music = bool(music_path and os.path.isfile(music_path))
    events = []
    for sc in ep["scenes"]:
        for fx in sc["sfx"]:
            p = os.path.join(sfx_dir, fx["name"] + ".mp3")
            if not os.path.isfile(p):
                raise RuntimeError(f"PostMaster FATAL: baked SFX missing: {p}")
            events.append((round(sc["start_time_s"] + fx["at_s"], 3), p))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    for _, p in events: cmd += ["-i", p]
    music_idx = None
    if has_music:
        music_idx = len(events) + 1
        cmd += ["-i", music_path]
    n_inputs = 1 + len(events) + (1 if has_music else 0)
    if events or has_music:
        parts = [f"[{i+1}:a]adelay={int(t*1000)}|{int(t*1000)}[fx{i}]" for i, (t, _) in enumerate(events)]
        mix_labels = "[0:a]" + "".join(f"[fx{i}]" for i in range(len(events)))
        if has_music:
            # loop (in case shorter than episode) + duck under dialogue/SFX
            parts.append(f"[{music_idx}:a]aloop=loop=-1:size=2e9,volume={MUSIC_VOLUME}[music]")
            mix_labels += "[music]"
        fc = ";".join(parts) + f";{mix_labels}amix=inputs={n_inputs}:duration=first:normalize=0[aout]"
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-map", "0:v", "-map", "0:a?"]
    cmd += ["-vf", grade, "-r", "24", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-c:a", "aac", "-b:a", "192k", out]
    return cmd, len(events)

class EpisodePostMaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "episode_video_path": ("STRING", {"forceInput": True}),
            "episode_json": ("STRING", {"forceInput": True}),
        }}
    RETURN_TYPES = ()
    FUNCTION = "master"
    CATEGORY = "RenReed"
    OUTPUT_NODE = True

    def master(self, episode_video_path, episode_json):
        ep = json.loads(episode_json)
        # Resolve both possible output dir locations
        alts = [episode_video_path,
                episode_video_path.replace('/models/output', '/opt/ComfyUI/output'),
                episode_video_path.replace('/opt/ComfyUI/output', '/models/output')]
        for alt in alts:
            if os.path.isdir(alt) and any(f.endswith('.mp4') for f in os.listdir(alt)):
                episode_video_path = alt
                break
        # episode_video_path may be a DIRECTORY of per-scene files -> concat first
        if os.path.isdir(episode_video_path):
            import re as _re
            found = {}
            for f in os.listdir(episode_video_path):
                m = _re.match(r"scene_(\d+).*\.mp4$", f)
                if m:
                    found.setdefault(int(m.group(1)), f)
            need = ep["episode"]["scene_count"]
            missing = [i for i in range(1, need + 1) if i not in found]
            if missing:
                raise RuntimeError(f"PostMaster FATAL: scene files missing {missing} "
                                   f"(found {sorted(found)}, need 1..{need})")
            # extra numbers beyond `need` are stray artifacts (e.g. a raw
            # VHS auto-numbered file left behind by a retried/re-run scene) —
            # harmless to ignore as long as every scene 1..need is present
            files = [found[i] for i in range(1, need + 1)]
            lst = os.path.join(episode_video_path, "_concat.txt")
            with open(lst, "w") as f:
                for fn in files:
                    f.write("file '" + os.path.join(episode_video_path, fn).replace("'", r"'\''") + "'\n")
            joined = os.path.join(episode_video_path, "_joined.mp4")
            r = subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0",
                                "-i",lst,"-c","copy",joined], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(joined):
                raise RuntimeError(f"PostMaster FATAL: concat failed:\n{r.stderr[:400]}")
            episode_video_path = joined
        if not os.path.isfile(episode_video_path):
            raise RuntimeError(f"PostMaster FATAL: input video missing: {episode_video_path}")
        try:
            import folder_paths
            outdir = folder_paths.get_output_directory()
        except Exception:
            outdir = "/models/output"
        os.makedirs(outdir, exist_ok=True)
        final = os.path.join(outdir, "EPISODE_FINAL.mp4")
        cmd, n = build_sfx_cmd(episode_video_path, ep, final)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(final) or os.path.getsize(final) < 100000:
            raise RuntimeError(f"PostMaster FATAL: ffmpeg failed (rc={r.returncode}):\n{r.stderr[:600]}")
        return {"ui": {"gifs": [{"filename": "EPISODE_FINAL.mp4", "subfolder": "",
                                 "type": "output", "format": "video/mp4"}]},
                "result": ()}

NODE_CLASS_MAPPINGS = {"EpisodePostMaster": EpisodePostMaster}
NODE_DISPLAY_NAME_MAPPINGS = {"EpisodePostMaster": "Episode PostMaster (final + download)"}
