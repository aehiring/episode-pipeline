#!/usr/bin/env python3
"""workflow_v17_api.json — locked-spec graph, proven-shape signatures."""
import json
G={}; _i=[0]
def N(ct, inp, title=""):
    _i[0]+=1; nid=str(_i[0])
    G[nid]={"inputs":inp,"class_type":ct,"_meta":{"title":title or ct}}; return nid
def L(n,s=0): return [n,s]

# 0. INPUT + COMPILE
script  = N("Text Multiline",{"text":"=== EPISODE ===\n(paste episode script here)\n=== END ==="},"SCRIPT INPUT")
comp    = N("EpisodeCompile",{"script_text":L(script)},"Compile (Claude)")
EJ      = L(comp,0)
scount  = N("SceneCount",{"episode_json":EJ},"Scene count")
chars   = N("CharacterRefLoader",{"episode_json":EJ},"Character refs")

# 1. MODELS
s2v_u = N("UNETLoader",{"unet_name":"wan2.2_s2v_14B_fp8_scaled.safetensors","weight_dtype":"default"},"S2V UNET")
s2v_l = N("LoraLoaderModelOnly",{"model":L(s2v_u),
  "lora_name":"lightx2v_t2v_14b_cfg_step_distill_v2_lora_rank256_bf16.safetensors","strength_model":1.5},"LoRA 1.5")
s2v_m = N("ModelSamplingSD3",{"model":L(s2v_l),"shift":8.0},"Shift 8")
wclip = N("CLIPLoader",{"clip_name":"umt5_xxl_fp8_e4m3fn_scaled.safetensors","type":"wan","device":"default"},"Wan CLIP")
wvae  = N("VAELoader",{"vae_name":"wan_2.1_vae.safetensors"},"Wan VAE")
aenc  = N("AudioEncoderLoader",{"audio_encoder_name":"wav2vec2_large_english_fp16.safetensors"},"wav2vec2")
kxu   = N("UNETLoader",{"unet_name":"flux1-dev-kontext_fp8_scaled.safetensors","weight_dtype":"default"},"Kontext UNET")
kxc   = N("DualCLIPLoader",{"clip_name1":"clip_l.safetensors","clip_name2":"t5xxl_fp8_e4m3fn_scaled.safetensors","type":"flux","device":"default"},"FLUX CLIP")
kxv   = N("VAELoader",{"vae_name":"ae.safetensors"},"FLUX VAE")
wneg  = N("CLIPTextEncode",{"clip":L(wclip),"text":"blurry, static, frozen, jerky motion, deformed hands, text, watermark"},"Wan neg")
kneg  = N("CLIPTextEncode",{"clip":L(kxc),"text":""},"Kontext neg")

# 2. ANCHOR (scene 1) from character sheet
sheet = N("ImageStitch",{"image1":L(chars,0),"direction":"right","match_image_size":True,
  "spacing_width":0,"spacing_color":"white"},"Char sheet")
kf1t  = N("SceneField",{"episode_json":EJ,"scene_number":1,"field":"keyframe_prompt"},"KF1 prompt")
kp1   = N("CLIPTextEncode",{"clip":L(kxc),"text":L(kf1t,0)},"Kontext pos 1")
shl   = N("VAEEncode",{"pixels":L(sheet),"vae":L(kxv)},"sheet->lat")
rf1   = N("ReferenceLatent",{"conditioning":L(kp1),"latent":L(shl)},"ref sheet")
g1    = N("FluxGuidance",{"conditioning":L(rf1),"guidance":2.5},"guide 2.5")
canvas= N("EmptySD3LatentImage",{"width":1280,"height":720,"batch_size":1},"canvas 720p")
aks   = N("KSampler",{"model":L(kxu),"positive":L(g1),"negative":L(kneg),"latent_image":L(canvas),
  "seed":7,"steps":20,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0},"anchor sample")
ANC   = N("VAEDecode",{"samples":L(aks),"vae":L(kxv)},"ANCHOR")
ancl  = N("VAEEncode",{"pixels":L(ANC,0),"vae":L(kxv)},"anchor->lat")

# 3. LOOP
one   = N("PrimitiveInt",{"value":1},"const 1")
idx0  = N("PrimitiveInt",{"value":0},"idx0")
ws    = N("easy whileLoopStart",{"condition":True,"initial_value0":L(idx0)},"WHILE start")
FLOW, IDX = L(ws,0), L(ws,1)
sn    = N("MathExpression|pysssss",{"expression":"a+1","a":IDX},"scene_number")
SN    = L(sn,0)

kft   = N("SceneField",{"episode_json":EJ,"scene_number":SN,"field":"keyframe_prompt"},"KF prompt")
kpn   = N("CLIPTextEncode",{"clip":L(kxc),"text":L(kft,0)},"Kontext pos N")
rfn   = N("ReferenceLatent",{"conditioning":L(kpn),"latent":L(ancl)},"ref ANCHOR")
gn    = N("FluxGuidance",{"conditioning":L(rfn),"guidance":2.5},"guide 2.5")
kks   = N("KSampler",{"model":L(kxu),"positive":L(gn),"negative":L(kneg),"latent_image":L(canvas),
  "seed":7,"steps":20,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0},"KF sample")
kfi   = N("VAEDecode",{"samples":L(kks),"vae":L(kxv)},"KF image")
iss1  = N("LayerUtility: BooleanOperatorV2",{"a_value":"","b_value":"","operator":"==","a":SN,"b":L(one)},"scene==1")
key   = N("easy ifElse",{"boolean":L(iss1,0),"on_true":L(ANC,0),"on_false":L(kfi,0)},"keyframe pick")
k480  = N("ImageScale",{"image":L(key,0),"upscale_method":"lanczos","width":832,"height":480,"crop":"center"},"->832x480")

tts   = N("RenReedTTS",{"episode_json":EJ,"scene_number":SN},"TTS (loud)")
aemb  = N("AudioEncoderEncode",{"audio_encoder":L(aenc),"audio":L(tts,0)},"audio embed")
mot   = N("SceneField",{"episode_json":EJ,"scene_number":SN,"field":"motion_all"},"motion text")
wpos  = N("CLIPTextEncode",{"clip":L(wclip),"text":L(mot,0)},"Wan pos")

def KS(lat,pos,neg,t):
    return N("KSamplerAdvanced",{"model":L(s2v_m),"add_noise":"enable","noise_seed":3,"steps":4,"cfg":1.0,
      "sampler_name":"euler","scheduler":"beta","positive":pos,"negative":neg,"latent_image":lat,
      "start_at_step":0,"end_at_step":10000,"return_with_leftover_noise":"disable"},t)

c1  = N("WanSoundImageToVideo",{"positive":L(wpos),"negative":L(wneg),"vae":L(wvae),"width":832,"height":480,
  "length":77,"batch_size":1,"audio_encoder_output":L(aemb),"ref_image":L(k480)},"S2V init")
ACC = L(KS(L(c1,2),L(c1,0),L(c1,1),"KSA c1"))
for i in range(2,6):
    ex = N("WanSoundImageToVideoExtend",{"positive":L(wpos),"negative":L(wneg),"vae":L(wvae),"length":77,
      "video_latent":ACC,"audio_encoder_output":L(aemb),"ref_image":L(k480)},f"S2V ext{i}")
    sN = KS(L(ex,2),L(ex,0),L(ex,1),f"KSA c{i}")
    ACC= L(N("LatentConcat",{"samples1":ACC,"samples2":L(sN),"dim":"t"},f"acc{i}"))

dec  = N("VAEDecode",{"samples":ACC,"vae":L(wvae)},"decode 385f")
f1   = N("ImageFromBatch",{"image":L(dec),"batch_index":1,"length":1},"frame1")
rest = N("ImageFromBatch",{"image":L(dec),"batch_index":1,"length":384},"f1..384")
fix  = N("ImageBatch",{"image1":L(f1),"image2":L(rest)},"fixed 385")
rife = N("RIFE VFI",{"ckpt_name":"rife49.pth","frames":L(fix),"clear_cache_after_n_frames":10,
  "multiplier":3,"fast_mode":True,"ensemble":True,"scale_factor":1.0,"dtype":"float32","torch_compile":False},"RIFE x3")
half = N("ImageIntervalSelect",{"image":L(rife),"interval":2,"start_at":0,"end_at":0},"every 2nd")
snstr= N("easy convertAnything",{"*":SN,"output_type":"string"},"sn->str")
name = N("Text Concatenate",{"delimiter":"","clean_whitespace":"true","text_a":"scenes/scene_","text_b":L(snstr,0)},"filename")
save = N("VHS_VideoCombine",{"frame_rate":24,"loop_count":0,"filename_prefix":L(name,0),
  "format":"video/h264-mp4","pix_fmt":"yuv420p","crf":17,"save_metadata":False,"trim_to_audio":False,
  "pingpong":False,"save_output":True,"images":L(half),"audio":L(tts,0)},"save scene mp4")

cond = N("LayerUtility: BooleanOperatorV2",{"a_value":"","b_value":"","operator":"<","a":SN,"b":L(scount,0)},"scene<count")
we   = N("easy whileLoopEnd",{"flow":FLOW,"condition":L(cond,0),"initial_value0":SN},"WHILE end")

# 4. FINAL (executes only after loop via gate)
gate = N("PathAfter",{"path":"/models/output/scenes","after":L(we,0)},"gate: after loop")
pm   = N("EpisodePostMaster",{"episode_video_path":L(gate,0),"episode_json":EJ},"FINAL PostMaster")

json.dump(G,open("workflow_v17_api.json","w"),indent=1)
print(f"nodes: {len(G)}")
