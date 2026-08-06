#!/usr/bin/env python3
"""
pose_build.py — ONE-TIME local script, zero GPU / zero external API cost.
Generates OpenPose-format (COCO-18) skeleton reference images for every
locked pose_library entry in episode_schema.json, into assets/poses/<name>/pose.png.

Why drawn, not generated: FLUX/Wan need a GPU to render a photorealistic
pose reference, but pose-conditioning models (Wan2.2 Fun Control) only need
the SKELETON topology — a plain line-and-dot keypoint drawing is exactly
what OpenPose/ControlNet-style preprocessors normally produce and consume,
so we can build the asset directly with no model, no GPU, no API key.

Usage: python3 pose_build.py [out_dir]
"""
import os, sys, json
from PIL import Image, ImageDraw

W, H = 512, 512
MIN_BYTES = 1500

# COCO-18 keypoint order (standard OpenPose body model)
KP = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
      "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
      "REye", "LEye", "REar", "LEar"]
KI = {name: i for i, name in enumerate(KP)}

# Standard OpenPose COCO-18 skeleton connections
LIMBS = [
    ("Neck", "RShoulder"), ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("Neck", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("Neck", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("Neck", "Nose"), ("Nose", "REye"), ("REye", "REar"),
    ("Nose", "LEye"), ("LEye", "LEar"),
]

def _rainbow(n, i):
    """Cycle hue across limb index, close to OpenPose's own render palette."""
    import colorsys
    h = (i / max(n, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def render_pose(keypoints_norm, out_path):
    """keypoints_norm: dict name -> (x, y) in 0..1 canvas coords, y grows down."""
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = {name: (x * W, y * H) for name, (x, y) in keypoints_norm.items()}
    for i, (a, b) in enumerate(LIMBS):
        if a in pts and b in pts:
            draw.line([pts[a], pts[b]], fill=_rainbow(len(LIMBS), i), width=6)
    for i, name in enumerate(KP):
        if name in pts:
            x, y = pts[name]
            r = 7
            draw.ellipse([x - r, y - r, x + r, y + r], fill=_rainbow(len(KP), i))
    img.save(out_path)

# ── Pose library: hand-authored approximate keypoints (normalized 0..1) ──
# Front-facing figure, y grows downward. These are best-effort approximations
# of each named pose's silhouette — refine after seeing real render results.

def _base_standing():
    return {
        "Nose": (0.50, 0.10), "Neck": (0.50, 0.16),
        "RShoulder": (0.44, 0.17), "LShoulder": (0.56, 0.17),
        "RElbow": (0.42, 0.30), "LElbow": (0.58, 0.30),
        "RWrist": (0.41, 0.42), "LWrist": (0.59, 0.42),
        "RHip": (0.46, 0.48), "LHip": (0.54, 0.48),
        "RKnee": (0.46, 0.68), "LKnee": (0.54, 0.68),
        "RAnkle": (0.46, 0.88), "LAnkle": (0.54, 0.88),
        "REye": (0.48, 0.09), "LEye": (0.52, 0.09),
        "REar": (0.46, 0.10), "LEar": (0.54, 0.10),
    }

def pose_standing_neutral():
    return _base_standing()

def pose_walking():
    p = _base_standing()
    p.update({
        "RHip": (0.46, 0.48), "LHip": (0.54, 0.48),
        "RKnee": (0.40, 0.66), "RAnkle": (0.36, 0.87),   # forward leg
        "LKnee": (0.58, 0.70), "LAnkle": (0.62, 0.86),   # back leg
        "RElbow": (0.58, 0.30), "RWrist": (0.60, 0.40),  # opposite-arm swing
        "LElbow": (0.42, 0.30), "LWrist": (0.40, 0.40),
    })
    return p

def pose_sitting():
    p = _base_standing()
    p.update({
        "RHip": (0.46, 0.55), "LHip": (0.54, 0.55),
        "RKnee": (0.40, 0.62), "LKnee": (0.60, 0.62),   # knees forward
        "RAnkle": (0.40, 0.85), "LAnkle": (0.60, 0.85),
        "RWrist": (0.42, 0.52), "LWrist": (0.58, 0.52),
        "RElbow": (0.42, 0.40), "LElbow": (0.58, 0.40),
    })
    return p

def pose_waving():
    p = _base_standing()
    p.update({
        "RElbow": (0.36, 0.16), "RWrist": (0.32, 0.02),  # raised waving arm
    })
    return p

def pose_pointing():
    p = _base_standing()
    p.update({
        "RElbow": (0.34, 0.24), "RWrist": (0.24, 0.22),  # extended pointing arm
    })
    return p

def pose_mountain_pose():
    p = _base_standing()
    p.update({
        "RWrist": (0.45, 0.46), "LWrist": (0.55, 0.46),  # arms straight at sides
        "RAnkle": (0.48, 0.88), "LAnkle": (0.52, 0.88),  # feet together
    })
    return p

def pose_tree_pose():
    p = _base_standing()
    p.update({
        "RKnee": (0.46, 0.68), "RAnkle": (0.46, 0.88),          # standing leg straight
        "LKnee": (0.62, 0.56), "LAnkle": (0.50, 0.58),          # bent leg, foot at inner knee
        "RWrist": (0.50, 0.20), "LWrist": (0.50, 0.20),         # hands together overhead
        "RElbow": (0.47, 0.24), "LElbow": (0.53, 0.24),
    })
    return p

def pose_warrior_one():
    p = _base_standing()
    p.update({
        "RHip": (0.44, 0.50), "LHip": (0.56, 0.50),
        "RKnee": (0.36, 0.66), "RAnkle": (0.32, 0.88),   # front knee bent
        "LKnee": (0.62, 0.78), "LAnkle": (0.68, 0.90),   # back leg straight, angled
        "RWrist": (0.44, 0.02), "LWrist": (0.56, 0.02),  # arms overhead
        "RElbow": (0.45, 0.10), "LElbow": (0.55, 0.10),
    })
    return p

def pose_cat_cow():
    # quadruped: hands and knees on the ground, torso horizontal
    return {
        "Nose": (0.85, 0.55), "Neck": (0.78, 0.52),
        "RShoulder": (0.78, 0.50), "LShoulder": (0.78, 0.54),
        "RElbow": (0.60, 0.50), "LElbow": (0.60, 0.54),
        "RWrist": (0.42, 0.62), "LWrist": (0.42, 0.66),
        "RHip": (0.30, 0.50), "LHip": (0.30, 0.54),
        "RKnee": (0.16, 0.62), "LKnee": (0.16, 0.66),
        "RAnkle": (0.08, 0.62), "LAnkle": (0.08, 0.66),
        "REye": (0.87, 0.53), "LEye": (0.87, 0.57),
        "REar": (0.84, 0.52), "LEar": (0.84, 0.58),
    }

def pose_butterfly_pose():
    p = _base_standing()
    p.update({
        "RHip": (0.46, 0.55), "LHip": (0.54, 0.55),
        "RKnee": (0.34, 0.60), "LKnee": (0.66, 0.60),     # knees fall outward
        "RAnkle": (0.48, 0.72), "LAnkle": (0.52, 0.72),   # soles together
        "RWrist": (0.46, 0.68), "LWrist": (0.54, 0.68),   # hands holding feet
        "RElbow": (0.40, 0.58), "LElbow": (0.60, 0.58),
    })
    return p

def pose_seated_breathing():
    p = _base_standing()
    p.update({
        "RHip": (0.46, 0.55), "LHip": (0.54, 0.55),
        "RKnee": (0.38, 0.58), "LKnee": (0.62, 0.58),     # cross-legged
        "RAnkle": (0.52, 0.62), "LAnkle": (0.48, 0.62),
        "RWrist": (0.46, 0.50), "LWrist": (0.54, 0.50),   # hands on belly
        "RElbow": (0.42, 0.38), "LElbow": (0.58, 0.38),
    })
    return p

POSES = {
    "standing_neutral": pose_standing_neutral,
    "walking": pose_walking,
    "sitting": pose_sitting,
    "waving": pose_waving,
    "pointing": pose_pointing,
    "mountain_pose": pose_mountain_pose,
    "tree_pose": pose_tree_pose,
    "warrior_one": pose_warrior_one,
    "cat_cow": pose_cat_cow,
    "butterfly_pose": pose_butterfly_pose,
    "seated_breathing": pose_seated_breathing,
}

def schema_names():
    here = os.path.dirname(os.path.abspath(__file__))
    sp = os.path.join(here, "episode_schema.json")
    return json.load(open(sp))["x_constants"]["pose_library"]

def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "assets/poses"
    names = schema_names()
    missing_fns = [n for n in names if n not in POSES]
    if missing_fns:
        sys.exit(f"FATAL: schema pose_library without a generator: {missing_fns}")
    fails = []
    for n in names:
        d = os.path.join(outdir, n)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "pose.png")
        if os.path.isfile(p) and os.path.getsize(p) >= MIN_BYTES:
            print(f"  ok   {n}"); continue
        try:
            render_pose(POSES[n](), p)
            sz = os.path.getsize(p)
            if sz < MIN_BYTES:
                raise RuntimeError(f"rendered file too small ({sz}B)")
            print(f"  done {n} ({sz}B)")
        except Exception as e:
            print(f"  FAIL {n}: {e}"); fails.append(n)

    have = {n for n in names if os.path.isfile(os.path.join(outdir, n, "pose.png"))
            and os.path.getsize(os.path.join(outdir, n, "pose.png")) >= MIN_BYTES}
    need = set(names)
    if have >= need and not fails:
        print(f"\nALL {len(need)} POSES PRESENT — ready to bake into Docker")
        sys.exit(0)
    print(f"\nINCOMPLETE — missing: {sorted(need - have)}")
    sys.exit(1)

if __name__ == "__main__":
    main()
