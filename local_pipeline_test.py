#!/usr/bin/env python3
"""
Local end-to-end pipeline test — CPU only, zero GPU cost.
Tests: JSON parsing, TTS API call, VHS-style file saving, PostMaster.
Mocks: all GPU nodes (KSampler, VAEDecode, S2V) with dummy tensors.
Run: python3 local_pipeline_test.py
"""
import json, os, sys, subprocess, importlib.util, shutil, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EPISODE_SCHEMA_PATH", os.path.join(os.path.dirname(__file__), "episode_schema.json"))
os.environ.setdefault("EPISODE_AUDIO_DIR", "/tmp/rr_test/audio")
os.environ.setdefault("SFX_ASSET_DIR", os.path.join(os.path.dirname(__file__), "assets/sfx"))

WORK = "/tmp/rr_test"; os.makedirs(WORK+"/audio", exist_ok=True)
os.makedirs(WORK+"/scenes", exist_ok=True)

C = json.load(open(os.environ["EPISODE_SCHEMA_PATH"]))["x_constants"]
anchor = C["style_anchor"]

# ── TEST EPISODE ──
def wc(s): return len(s.split())
d1 = "Reed had one plan. It was already falling apart."
d2 = "Stay calm. Find Ren. Tell her the plan. Simple. Easy."
d3 = "Reed. We missed the bus."
EP = {"schema_version":"1.0","episode":{"title":"Reeds One Plan","logline":"test","lesson":"test","total_chunks":5,"scene_count":3,"characters_used":["REN","REED"]},
  "scenes":[
    {"scene_number":1,"chunks":2,"duration_s":9.625,"start_time_s":0.0,"location":"HALLWAY","time_of_day":"DAY","characters":["REED"],"speaker":"NARRATOR","wardrobe":{"carry":True},"shot":"WIDE","keyframe_prompt":anchor+". HALLWAY from above, wide shot.","motion_prompts":["Camera drifts.","Light streams."],"dialogue":d1,"dialogue_word_count":wc(d1),"sfx":[{"name":"footsteps_floor","at_s":2.0}]},
    {"scene_number":2,"chunks":2,"duration_s":9.625,"start_time_s":9.625,"location":"HALLWAY","time_of_day":"DAY","characters":["REED"],"speaker":"REED","wardrobe":{"carry":True},"shot":"MEDIUM","keyframe_prompt":anchor+". REED at locker, medium shot.","motion_prompts":["REED opens locker.","REED nods."],"dialogue":d2,"dialogue_word_count":wc(d2),"sfx":[{"name":"school_bell","at_s":4.0}]},
    {"scene_number":3,"chunks":1,"duration_s":4.8125,"start_time_s":19.25,"location":"HALLWAY","time_of_day":"DAY","characters":["REN","REED"],"speaker":"REN","wardrobe":{"carry":True},"shot":"MEDIUM","keyframe_prompt":anchor+". REN behind REED, medium shot.","motion_prompts":["REED turns slowly."],"dialogue":d3,"dialogue_word_count":wc(d3),"sfx":[{"name":"gasp","at_s":2.0}]}],
  "totals":{"sum_chunks":5,"sum_duration_s":24.0625,"total_frames_16fps":385}}

EP_JSON = json.dumps(EP, separators=(',',':'))
P=F=0
def ok(name): global P; print(f"  ✅ {name}"); P+=1
def fail(name, reason=""): global F; print(f"  ❌ {name}: {reason}"); F+=1

print("\n══════════════════════════════════════")
print("  REN & REED LOCAL PIPELINE TEST")
print("══════════════════════════════════════\n")

# 1. Compiler pass-through
print("▶ STEP 1: Compiler (JSON pass-through)")
try:
    spec=importlib.util.spec_from_file_location("cn","compiler_node.py"); cn=importlib.util.module_from_spec(spec); spec.loader.exec_module(cn)
    ej, rep = cn.EpisodeCompile().compile(EP_JSON)
    parsed = json.loads(ej)
    assert parsed["episode"]["title"] == EP["episode"]["title"]
    ok(f"compiler pass-through: {rep[:50]}")
except Exception as e: fail("compiler", str(e))

# 2. TTS (requires ELEVENLABS_API_KEY + ELEVENLABS_VOICES)
print("\n▶ STEP 2: TTS (ElevenLabs)")
el_key = os.environ.get("ELEVENLABS_API_KEY","").strip()
el_voices = os.environ.get("ELEVENLABS_VOICES","").strip()
if not el_key or not el_voices:
    print("  ⚠️  SKIP: set ELEVENLABS_API_KEY and ELEVENLABS_VOICES to test TTS")
else:
    try:
        spec2=importlib.util.spec_from_file_location("tn","renreed_tts_node.py"); tn=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tn)
        audio_out, path = tn.RenReedTTS().speak(EP_JSON, 3)  # scene 3: "Reed. We missed the bus."
        assert os.path.isfile(path) and os.path.getsize(path) > 1000
        ok(f"TTS scene 3: {audio_out['waveform'].shape}, saved {path}")
    except Exception as e: fail("TTS", str(e))

# 3. Mock VHS — create dummy scene mp4 files (ffmpeg sine+color, 832x480)
print("\n▶ STEP 3: Mock scene files (ffmpeg, no GPU)")
for i in range(1, 4):
    out = f"{WORK}/scenes/scene_{i:05d}_.mp4"
    r = subprocess.run(["ffmpeg","-y","-v","error",
        "-f","lavfi","-i",f"testsrc2=size=832x480:rate=16:duration={EP['scenes'][i-1]['duration_s']}",
        "-f","lavfi","-i",f"sine=frequency={300+i*100}:duration={EP['scenes'][i-1]['duration_s']}",
        "-c:v","libx264","-preset","ultrafast","-c:a","aac", out], capture_output=True)
    if r.returncode==0 and os.path.getsize(out)>1000:
        ok(f"mock scene {i}: {out}")
    else: fail(f"mock scene {i}", r.stderr.decode()[:100])

# 4. PostMaster (real ffmpeg, real SFX if present)
print("\n▶ STEP 4: PostMaster (concat + grade + SFX)")
try:
    spec3=importlib.util.spec_from_file_location("pm","postmaster_node.py"); pm=importlib.util.module_from_spec(spec3); spec3.loader.exec_module(pm)
    pm.SFX_DIR = os.environ["SFX_ASSET_DIR"]
    # Check if SFX files exist, otherwise skip SFX
    sfx_ok = all(os.path.isfile(f"{pm.SFX_DIR}/{s['name']}.mp3") for sc in EP["scenes"] for s in sc["sfx"])
    if not sfx_ok:
        print("  ⚠️  SFX files not found locally — removing SFX from test episode")
        for sc in EP["scenes"]: sc["sfx"] = []
    pm.EpisodePostMaster().master(WORK+"/scenes", json.dumps(EP))
    fin = "/models/output/EPISODE_FINAL.mp4"
    if not os.path.isfile(fin): fin = "/opt/ComfyUI/output/EPISODE_FINAL.mp4"
    if not os.path.isfile(fin):
        import glob; candidates = glob.glob("/tmp/**/EPISODE_FINAL.mp4", recursive=True) + glob.glob("/models/**/*.mp4", recursive=True)
        fin = candidates[0] if candidates else fin
    if os.path.isfile(fin):
        sz = os.path.getsize(fin)
        pr = subprocess.run(["ffprobe","-v","error","-show_entries","stream=width,height,r_frame_rate","-select_streams","v:0","-of","json",fin],capture_output=True,text=True)
        st = json.loads(pr.stdout)["streams"][0]
        ok(f"PostMaster: {st['width']}x{st['height']} {st['r_frame_rate']}fps, {sz//1024}KB → {fin}")
    else:
        fail("PostMaster", f"EPISODE_FINAL.mp4 not found (tried {fin})")
except Exception as e:
    import traceback; fail("PostMaster", traceback.format_exc()[-300:])

# 5. Validation
print("\n▶ STEP 5: Schema + Validator")
try:
    import jsonschema
    S = json.load(open(os.environ["EPISODE_SCHEMA_PATH"]))
    jsonschema.validate(EP, S)
    spec4=importlib.util.spec_from_file_location("vc","validator_core.py"); vc=importlib.util.module_from_spec(spec4); spec4.loader.exec_module(vc)
    errs = vc.validate_semantic(EP, S["x_constants"])
    if not errs: ok("schema + semantic: 0 errors")
    else: fail("validator", str(errs[:2]))
except Exception as e: fail("validator", str(e))

print(f"\n══════════════════════════════════════")
print(f"  RESULT: {P} pass / {F} fail")
print(f"══════════════════════════════════════")
if F == 0:
    print("  🎉 ALL PASS — safe to deploy on GPU")
else:
    print("  🔴 FIXES NEEDED before GPU deploy")
