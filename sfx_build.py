#!/usr/bin/env python3
"""
sfx_build.py — ONE-TIME local script (Muhammad runs with his own key).
Generates all 30 schema SFX via ElevenLabs sound-generation API into
assets/sfx/. Idempotent, size-validated, exits non-zero on ANY miss.
Usage:  ELEVENLABS_API_KEY=sk_... python3 sfx_build.py [out_dir]
"""
import os, sys, json, time, urllib.request, urllib.error

API = "https://api.elevenlabs.io/v1/sound-generation"
MIN_BYTES = 6000

PROMPTS = {
 "door_knock":"three gentle knocks on a wooden door, cartoon, clean",
 "door_open":"wooden door opening with a soft creak, cartoon",
 "door_close":"wooden door closing softly, cartoon",
 "footsteps_grass":"light child footsteps on grass, four steps, cartoon",
 "footsteps_floor":"light child footsteps on wooden floor, four steps, cartoon",
 "dog_bark":"single friendly small dog bark, cartoon",
 "cat_meow":"single soft cat meow, cartoon",
 "birds_chirp":"cheerful small birds chirping, bright morning, cartoon, loopable",
 "wind_soft":"soft gentle wind through leaves, calm",
 "rain_light":"light rain on leaves, gentle, no thunder",
 "thunder_soft":"distant soft thunder rumble, mild, not scary",
 "water_splash":"small playful water splash, cartoon",
 "ball_bounce":"rubber ball bouncing twice on pavement, cartoon",
 "bicycle_bell":"bright bicycle bell ring, single, cheerful",
 "school_bell":"school bell ringing briefly, classic",
 "phone_chime":"soft friendly phone notification chime",
 "pop":"small cartoon pop",
 "whoosh":"quick soft cartoon whoosh",
 "sparkle":"magical sparkle shimmer, bright, short",
 "magic_chime":"gentle magical chime, wonder, short",
 "crowd_kids_laugh":"small group of children laughing happily, brief",
 "yawn":"soft cute cartoon yawn",
 "giggle":"single child giggle, cute",
 "gasp":"soft surprised cartoon gasp",
 "boing":"classic cartoon boing spring",
 "thud_soft":"soft cushioned thud, cartoon, harmless",
 "paper_rustle":"paper pages rustling briefly",
 "zip":"quick zipper zip, cartoon",
 "car_pass_soft":"car passing by softly in distance",
 "kitchen_sizzle":"gentle frying pan sizzle, pleasant kitchen",
}
DUR = {"birds_chirp":4,"wind_soft":5,"rain_light":5,"kitchen_sizzle":4,
       "footsteps_grass":3,"footsteps_floor":3,"crowd_kids_laugh":3}

def schema_names():
    here = os.path.dirname(os.path.abspath(__file__))
    sp = os.path.join(here, "episode_schema.json")
    return json.load(open(sp))["x_constants"]["sfx_library"]

def gen(name, key, out):
    body = json.dumps({"text": PROMPTS[name],
                       "duration_seconds": DUR.get(name, 2.0),
                       "prompt_influence": 0.6}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "xi-api-key": key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) < MIN_BYTES:
        raise RuntimeError(f"{name}: only {len(data)} bytes — API returned junk")
    with open(out, "wb") as f: f.write(data)
    return len(data)

def main():
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        sys.exit("FATAL: ELEVENLABS_API_KEY not set")
    outdir = sys.argv[1] if len(sys.argv) > 1 else "assets/sfx"
    os.makedirs(outdir, exist_ok=True)
    names = schema_names()
    missing_prompts = [n for n in names if n not in PROMPTS]
    if missing_prompts:
        sys.exit(f"FATAL: schema SFX without prompts: {missing_prompts}")
    fails = []
    for n in names:
        p = os.path.join(outdir, n + ".mp3")
        if os.path.isfile(p) and os.path.getsize(p) >= MIN_BYTES:
            print(f"  ok   {n}"); continue
        try:
            b = gen(n, key, p); print(f"  done {n} ({b} bytes)"); time.sleep(0.6)
        except Exception as e:
            print(f"  FAIL {n}: {e}"); fails.append(n)
    # final gate — exact match with schema
    have = {f[:-4] for f in os.listdir(outdir) if f.endswith(".mp3")
            and os.path.getsize(os.path.join(outdir, f)) >= MIN_BYTES}
    need = set(names)
    if have >= need and not fails:
        print(f"\nALL {len(need)} SFX PRESENT — ready to bake into Docker")
        sys.exit(0)
    print(f"\nINCOMPLETE — missing: {sorted(need - have)}")
    sys.exit(1)

if __name__ == "__main__":
    main()
