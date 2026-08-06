#!/usr/bin/env python3
"""
ComfyUI API-graph templates for the Ren & Reed pipeline — v18.

Replaces the single monolithic loop-graph (which depended on ComfyUI-Easy-Use's
`whileLoopStart/End`, buggy past 1 iteration) with four small, independent
templates that `watchdog.py` submits one at a time via the ComfyUI `/prompt`
API. Python owns the scene loop; ComfyUI only ever renders one thing per call.

    trigger_template()                -> the only workflow loaded in the browser
                                          (paste script/JSON, click Run)
    anchor_template(ep)                -> one-time 720p reference keyframe
    scene_template(ep, scene)          -> one scene's full render + save
    postmaster_template(ep, scenes_dir)-> final concat + grade + SFX + output

All per-scene text/number fields (keyframe_prompt, motion_prompts, scene_number,
...) are resolved by Python from the already-compiled episode dict and baked in
as literal widget values — no SceneField/MathExpression/loop-control nodes are
needed anymore, since there's no in-graph loop left to feed.
"""
import json

STYLE_ANCHOR = "3D animated cartoon, Pixar-style, soft rounded shapes, warm colors, gentle lighting, family-friendly"
ANCHOR_FILENAME_PREFIX = "anchor/ANCHOR"
SCENES_DIR = "/models/output/scenes"


def _graph():
    """Fresh node-dict + id counter + N()/L() helpers, scoped per template so
    templates never share or leak node ids between separate /prompt submissions."""
    G = {}
    _i = [0]
    def N(ct, inp, title=""):
        _i[0] += 1
        nid = str(_i[0])
        G[nid] = {"inputs": inp, "class_type": ct, "_meta": {"title": title or ct}}
        return nid
    def L(n, s=0):
        return [n, s]
    return G, N, L


def _wan_loaders(N, L):
    """S2V model stack — identical across anchor/scene templates; ComfyUI caches
    loader nodes by identical input signature across separate /prompt calls, so
    redeclaring these per template costs nothing at runtime."""
    s2v_u = N("UNETLoader", {"unet_name": "wan2.2_s2v_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "S2V UNET")
    s2v_l = N("LoraLoaderModelOnly", {"model": L(s2v_u),
        "lora_name": "wan22_lightning_fallback_high.safetensors", "strength_model": 1.5}, "LoRA 1.5")
    s2v_m = N("ModelSamplingSD3", {"model": L(s2v_l), "shift": 8.0}, "Shift 8")
    wclip = N("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "Wan CLIP")
    wvae  = N("VAELoader", {"vae_name": "wan_2.1_vae.safetensors"}, "Wan VAE")
    aenc  = N("AudioEncoderLoader", {"audio_encoder_name": "wav2vec2_large_english_fp16.safetensors"}, "wav2vec2")
    wneg  = N("CLIPTextEncode", {"clip": L(wclip), "text": "blurry, static, frozen, jerky motion, deformed hands, text, watermark"}, "Wan neg")
    return {"s2v_model": s2v_m, "wclip": wclip, "wvae": wvae, "aenc": aenc, "wneg": wneg}


def _wan_i2v_loaders(N, L):
    """Wan2.2 I2V dual-expert (high-noise/low-noise MoE) stack for the general
    action path — ANY physical action on ANY topic, driven by motion_prompts
    free text (no fixed pose list). Native ComfyUI nodes only; the camera-
    motion variant (WanCameraImageToVideo) reuses this same model/lora pair,
    just swapping which conditioning node feeds the samplers below."""
    hi_u = N("UNETLoader", {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "I2V UNET hi")
    hi_l = N("LoraLoaderModelOnly", {"model": L(hi_u),
        "lora_name": "wan22_i2v_lightx2v_4steps_high_noise.safetensors", "strength_model": 1.0}, "I2V LoRA hi")
    lo_u = N("UNETLoader", {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "I2V UNET lo")
    lo_l = N("LoraLoaderModelOnly", {"model": L(lo_u),
        "lora_name": "wan22_i2v_lightx2v_4steps_low_noise.safetensors", "strength_model": 1.0}, "I2V LoRA lo")
    wclip = N("CLIPLoader", {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "Wan CLIP (i2v)")
    wvae  = N("VAELoader", {"vae_name": "wan_2.1_vae.safetensors"}, "Wan VAE (i2v)")
    wneg  = N("CLIPTextEncode", {"clip": L(wclip), "text": "blurry, static, frozen, jerky motion, deformed hands, text, watermark"}, "Wan neg (i2v)")
    return {"hi": hi_l, "lo": lo_l, "wclip": wclip, "wvae": wvae, "wneg": wneg}


def _kontext_loaders(N, L):
    kxu  = N("UNETLoader", {"unet_name": "flux1-dev-kontext_fp8_scaled.safetensors", "weight_dtype": "default"}, "Kontext UNET")
    kxc  = N("DualCLIPLoader", {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux", "device": "default"}, "FLUX CLIP")
    kxv  = N("VAELoader", {"vae_name": "ae.safetensors"}, "FLUX VAE")
    kneg = N("CLIPTextEncode", {"clip": L(kxc), "text": ""}, "Kontext neg")
    return {"kxu": kxu, "kxc": kxc, "kxv": kxv, "kneg": kneg}


def _kontext_keyframe(N, L, kx, ref_latent, keyframe_prompt, title_suffix=""):
    """Kontext keyframe render: scene's own prompt, referenced against a latent
    (character sheet for the anchor, ANCHOR image for every regular scene)."""
    kp   = N("CLIPTextEncode", {"clip": L(kx["kxc"]), "text": keyframe_prompt}, f"Kontext pos{title_suffix}")
    rf   = N("ReferenceLatent", {"conditioning": L(kp), "latent": ref_latent}, f"ref{title_suffix}")
    g    = N("FluxGuidance", {"conditioning": L(rf), "guidance": 2.5}, "guide 2.5")
    canvas = N("EmptySD3LatentImage", {"width": 1280, "height": 720, "batch_size": 1}, "canvas 720p")
    ks   = N("KSampler", {"model": L(kx["kxu"]), "positive": L(g), "negative": L(kx["kneg"]), "latent_image": L(canvas),
        "seed": 7, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}, f"KF sample{title_suffix}")
    dec  = N("VAEDecode", {"samples": L(ks), "vae": L(kx["kxv"])}, f"KF image{title_suffix}")
    return dec


def anchor_template(ep):
    """One-time 720p reference keyframe, built from the character sheet using
    scene 1's own keyframe_prompt. Saved to disk (SaveImage) — watchdog copies
    the result into ComfyUI's input dir so every scene_template can reload it."""
    G, N, L = _graph()
    ep_json = json.dumps(ep)
    scene1 = next(s for s in ep["scenes"] if s["scene_number"] == 1)

    chars = N("CharacterRefLoader", {"episode_json": ep_json}, "Character refs")
    sheet = N("ImageStitch", {"image1": L(chars, 0), "direction": "right", "match_image_size": True,
        "spacing_width": 0, "spacing_color": "white"}, "Char sheet")
    kx = _kontext_loaders(N, L)
    kxv = kx["kxv"]
    shl = N("VAEEncode", {"pixels": L(sheet), "vae": L(kxv)}, "sheet->lat")
    dec = _kontext_keyframe(N, L, kx, L(shl), scene1["keyframe_prompt"], " 1")
    save = N("SaveImage", {"images": L(dec), "filename_prefix": ANCHOR_FILENAME_PREFIX}, "SAVE ANCHOR")
    return G


def scene_template(ep, scene, anchor_image="anchor_current.png"):
    """Full render for one scene: Kontext keyframe (from the saved ANCHOR) ->
    832x480 -> TTS -> S2V init + (chunks-1) extends -> decode -> first-frame
    fix -> save. Extend count MUST track scene['chunks'] (1-5) — a fixed
    5-segment loop here was the bug that made every scene render a forced 24s
    regardless of its real length (found live on the 59-scene render: 826s
    video for what should have been ~380s of content).
    `anchor_image` must already be sitting in ComfyUI's input dir (watchdog's
    job, after the anchor prompt completes) — LoadImage resolves it from there."""
    G, N, L = _graph()
    ep_json = json.dumps(ep)
    n = scene["scene_number"]
    chunks = scene["chunks"]
    total_frames = chunks * 77

    kx = _kontext_loaders(N, L)
    anc_img = N("LoadImage", {"image": anchor_image}, "ANCHOR (loaded)")
    anc_lat = N("VAEEncode", {"pixels": L(anc_img, 0), "vae": L(kx["kxv"])}, "anchor->lat")
    kf = _kontext_keyframe(N, L, kx, L(anc_lat), scene["keyframe_prompt"], f" {n}")
    k480 = N("ImageScale", {"image": L(kf), "upscale_method": "lanczos", "width": 832, "height": 480, "crop": "center"}, "->832x480")

    tts = N("RenReedTTS", {"episode_json": ep_json, "scene_number": n}, "TTS (loud)")
    wan = _wan_loaders(N, L)
    aemb = N("AudioEncoderEncode", {"audio_encoder": L(wan["aenc"]), "audio": L(tts, 0)}, "audio embed")
    wpos = N("CLIPTextEncode", {"clip": L(wan["wclip"]), "text": " ".join(scene["motion_prompts"])}, "Wan pos")

    def KS(lat, pos, neg, title):
        return N("KSamplerAdvanced", {"model": L(wan["s2v_model"]), "add_noise": "enable", "noise_seed": 3,
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta",
            "positive": pos, "negative": neg, "latent_image": lat,
            "start_at_step": 0, "end_at_step": 10000, "return_with_leftover_noise": "disable"}, title)

    c1 = N("WanSoundImageToVideo", {"positive": L(wpos), "negative": L(wan["wneg"]), "vae": L(wan["wvae"]),
        "width": 832, "height": 480, "length": 77, "batch_size": 1,
        "audio_encoder_output": L(aemb), "ref_image": L(k480)}, "S2V init")
    acc = L(KS(L(c1, 2), L(c1, 0), L(c1, 1), "KSA c1"))
    for i in range(2, chunks + 1):  # (chunks-1) extends; chunks=1 -> no extends, just the init 77 frames
        ex = N("WanSoundImageToVideoExtend", {"positive": L(wpos), "negative": L(wan["wneg"]), "vae": L(wan["wvae"]),
            "length": 77, "video_latent": acc, "audio_encoder_output": L(aemb), "ref_image": L(k480)}, f"S2V ext{i}")
        sN = KS(L(ex, 2), L(ex, 0), L(ex, 1), f"KSA c{i}")
        acc = L(N("LatentConcat", {"samples1": acc, "samples2": L(sN), "dim": "t"}, f"acc{i}"))

    dec = N("VAEDecode", {"samples": acc, "vae": L(wan["wvae"])}, f"decode {total_frames}f")
    f1 = N("ImageFromBatch", {"image": L(dec), "batch_index": 1, "length": 1}, "frame1")
    rest = N("ImageFromBatch", {"image": L(dec), "batch_index": 1, "length": total_frames - 1}, "f1..last")
    fix = N("ImageBatch", {"image1": L(f1), "image2": L(rest)}, f"fixed {total_frames}")
    N("VHS_VideoCombine", {"frame_rate": 16, "loop_count": 0, "filename_prefix": "scenes/scene",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False,
        "trim_to_audio": True, "pingpong": False, "save_output": True,
        "images": L(fix), "audio": L(tts, 0)}, "save scene mp4")
    return G


def _needs_action_path(scene):
    """Cheap-path scenes (just standing, static camera) keep using the plain
    S2V scene_template above — only pay for the heavier general I2V action/
    camera/decoupled-lip-sync pipeline when a scene actually asks for real
    motion. has_physical_action is a free-text-backed flag (motion_prompts
    already carries whatever the script describes, any topic) — NOT a fixed
    pose list."""
    return scene.get("has_physical_action", False) \
        or scene.get("camera_motion", "static") != "static"


def scene_template_action(ep, scene, anchor_image="anchor_current.png"):
    """
    ================================ R&D — READ ME ============================
    Full action/camera path for scenes with has_physical_action and/or a
    non-static camera_motion. Uses Wan2.2's native, general-purpose I2V
    conditioning (WanImageToVideo — ships with core ComfyUI, text-driven via
    motion_prompts, follows ANY described action on ANY topic — no fixed
    pose list), or WanCameraImageToVideo (Kijai's ComfyUI-WanVideoWrapper,
    Fun Camera Control) when camera movement is requested — that node also
    takes the same free-text positive conditioning, so one pass covers both
    "character does X" and "camera does Y" together rather than chaining two
    separate motion passes. Dual high/low-noise MoE KSamplerAdvanced pattern
    (matches the LightX2V 4-step lora pair downloaded in entrypoint.sh).
    ShmuelRonen's ComfyUI-LatentSyncWrapper then overlays the TTS audio as a
    decoupled lip-sync pass (silent motion first, mouth-sync after — the
    video model isn't fighting body motion and mouth accuracy at once).

    The camera-embedding and lip-sync nodes are NOT yet verified against a
    live ComfyUI install — check /object_info on the running instance for
    the ACTUAL registered class_type/input names if these come back red on
    the first real GPU submission; watchdog.ensure_models() cannot fix a
    wrong node/input name, only a missing model file, so a node-shape
    mismatch here is the one thing that would still need a manual code
    patch (not a reboot) mid-run. WanImageToVideo itself is native/core
    ComfyUI, so that half is expected to work as written. Everything
    upstream (Kontext keyframe, anchor loading) is the same proven code as
    scene_template() — only the motion stage differs.
    =============================================================================
    """
    G, N, L = _graph()
    ep_json = json.dumps(ep)
    n = scene["scene_number"]
    cam_motion = scene.get("camera_motion", "static")
    total_frames = scene["chunks"] * 77  # same chunk-accurate length as scene_template — never hardcode 77/385

    kx = _kontext_loaders(N, L)
    anc_img = N("LoadImage", {"image": anchor_image}, "ANCHOR (loaded)")
    anc_lat = N("VAEEncode", {"pixels": L(anc_img, 0), "vae": L(kx["kxv"])}, "anchor->lat")
    kf = _kontext_keyframe(N, L, kx, L(anc_lat), scene["keyframe_prompt"], f" {n}")
    k480 = N("ImageScale", {"image": L(kf), "upscale_method": "lanczos", "width": 832, "height": 480, "crop": "center"}, "->832x480")
    motion_text = " ".join(scene["motion_prompts"])

    i2v = _wan_i2v_loaders(N, L)
    pos = N("CLIPTextEncode", {"clip": L(i2v["wclip"]), "text": motion_text}, "I2V pos")

    if cam_motion != "static":
        cam_embed = N("WanCameraEmbedding", {"camera_motion": cam_motion, "width": 832, "height": 480, "length": total_frames}, "camera embed")
        cond = N("WanCameraImageToVideo", {  # VERIFY: exact WanVideoWrapper input names
            "positive": L(pos), "negative": L(i2v["wneg"]), "vae": L(i2v["wvae"]),
            "width": 832, "height": 480, "length": total_frames, "batch_size": 1,
            "start_image": L(k480, 0), "camera_conditions": L(cam_embed, 0)}, "camera cond")
    else:
        cond = N("WanImageToVideo", {"positive": L(pos), "negative": L(i2v["wneg"]), "vae": L(i2v["wvae"]),
            "width": 832, "height": 480, "length": total_frames, "batch_size": 1,
            "start_image": L(k480, 0)}, "I2V cond")

    def KS(model, lat, add_noise, start, end, leftover, title):
        return N("KSamplerAdvanced", {"model": model, "add_noise": add_noise, "noise_seed": 9,
            "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": L(cond, 0), "negative": L(cond, 1), "latent_image": lat,
            "start_at_step": start, "end_at_step": end, "return_with_leftover_noise": leftover}, title)

    hi_pass = KS(L(i2v["hi"]), L(cond, 2), "enable", 0, 2, "enable", "KSA hi (MoE)")
    lo_pass = KS(L(i2v["lo"]), L(hi_pass), "disable", 2, 10000, "disable", "KSA lo (MoE)")
    dec = N("VAEDecode", {"samples": L(lo_pass), "vae": L(i2v["wvae"])}, f"decode {total_frames}f")

    tts = N("RenReedTTS", {"episode_json": ep_json, "scene_number": n}, "TTS (loud)")
    lipsync = N("LatentSyncNode", {  # VERIFY: exact ComfyUI-LatentSyncWrapper node/input names
        "video": L(dec), "audio": L(tts, 0), "video_frame_rate": 16}, "lip-sync overlay")

    f1 = N("ImageFromBatch", {"image": L(lipsync, 0), "batch_index": 1, "length": 1}, "frame1")
    rest = N("ImageFromBatch", {"image": L(lipsync, 0), "batch_index": 1, "length": total_frames - 1}, "f1..last")
    fix = N("ImageBatch", {"image1": L(f1), "image2": L(rest)}, f"fixed {total_frames}")
    N("VHS_VideoCombine", {"frame_rate": 16, "loop_count": 0, "filename_prefix": "scenes/scene",
        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False,
        "trim_to_audio": True, "pingpong": False, "save_output": True,
        "images": L(fix), "audio": L(tts, 0)}, "save scene mp4")
    return G


def postmaster_template(ep, scenes_dir=SCENES_DIR):
    """Final concat + grade + SFX mux. No PathAfter gate — watchdog only submits
    this after confirming every scene file exists, so ordering is already safe."""
    G, N, L = _graph()
    N("EpisodePostMaster", {"episode_video_path": scenes_dir, "episode_json": json.dumps(ep)}, "FINAL PostMaster")
    return G


def trigger_template():
    """The only workflow the user still loads in the ComfyUI browser: paste
    script/JSON, click Run. EpisodeCompile writes the compiled episode to
    EPISODE_COMPILED_PATH as a side effect — watchdog picks it up from there."""
    G, N, L = _graph()
    script = N("Text Multiline", {"text": "=== EPISODE ===\n(paste episode script or compiled JSON here)\n=== END ==="}, "SCRIPT INPUT")
    N("EpisodeCompile", {"script_text": L(script)}, "Compile (Claude)")
    return G


if __name__ == "__main__":
    # Regenerate the browser-facing trigger workflow, and dump sample
    # anchor/scene/postmaster templates (built from the local test episode) for
    # inspection / the "red nodes = 0" CPU-only sanity check.
    json.dump(trigger_template(), open("workflow_v17_api.json", "w"), indent=1)

    sample_ep = {
        "schema_version": "1.0",
        "episode": {"title": "sample", "logline": "x", "lesson": "x", "total_chunks": 2,
                    "scene_count": 1, "characters_used": ["REN"]},
        "scenes": [{"scene_number": 1, "chunks": 2, "duration_s": 9.625, "start_time_s": 0.0,
                    "location": "X", "time_of_day": "DAY", "characters": ["REN"], "speaker": "NONE",
                    "wardrobe": {"carry": True}, "shot": "WIDE",
                    "keyframe_prompt": STYLE_ANCHOR + ". sample scene, wide shot.",
                    "motion_prompts": ["a", "b"], "dialogue": "NONE", "dialogue_word_count": 0, "sfx": [],
                    "has_physical_action": True, "camera_motion": "zoom_in"}],
        "totals": {"sum_chunks": 2, "sum_duration_s": 9.625, "total_frames_16fps": 154},
    }
    json.dump(anchor_template(sample_ep), open("sample_anchor_api.json", "w"), indent=1)
    json.dump(scene_template(sample_ep, sample_ep["scenes"][0]), open("sample_scene_api.json", "w"), indent=1)
    json.dump(scene_template_action(sample_ep, sample_ep["scenes"][0]), open("sample_scene_action_api.json", "w"), indent=1)
    json.dump(postmaster_template(sample_ep), open("sample_postmaster_api.json", "w"), indent=1)
    for name, g in [("trigger", trigger_template()), ("anchor", anchor_template(sample_ep)),
                    ("scene", scene_template(sample_ep, sample_ep["scenes"][0])),
                    ("scene_action", scene_template_action(sample_ep, sample_ep["scenes"][0])),
                    ("postmaster", postmaster_template(sample_ep))]:
        print(f"{name}: {len(g)} nodes")
