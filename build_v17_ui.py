#!/usr/bin/env python3
"""Convert workflow_v17_api.json -> UI-format graph with GROUPS + layout."""
import json

API=json.load(open('workflow_v17_api.json'))

# ---- slot type tables (outputs per class) ----
OUT={
 "Text Multiline":[("STRING","STRING")],
 "EpisodeCompile":[("episode_json","STRING"),("compile_report","STRING")],
 "SceneCount":[("scene_count","INT")],
 "SceneField":[("text","STRING"),("int","INT"),("float","FLOAT")],
 "CharacterRefLoader":[("character_refs","IMAGE"),("names_csv","STRING")],
 "RenReedTTS":[("audio","AUDIO"),("audio_path","STRING")],
 "EpisodePostMaster":[],
 "PathAfter":[("path","STRING")],
 "UNETLoader":[("MODEL","MODEL")],
 "LoraLoaderModelOnly":[("MODEL","MODEL")],
 "ModelSamplingSD3":[("MODEL","MODEL")],
 "CLIPLoader":[("CLIP","CLIP")],
 "DualCLIPLoader":[("CLIP","CLIP")],
 "VAELoader":[("VAE","VAE")],
 "AudioEncoderLoader":[("AUDIO_ENCODER","AUDIO_ENCODER")],
 "AudioEncoderEncode":[("AUDIO_ENCODER_OUTPUT","AUDIO_ENCODER_OUTPUT")],
 "CLIPTextEncode":[("CONDITIONING","CONDITIONING")],
 "VAEEncode":[("LATENT","LATENT")],
 "VAEDecode":[("IMAGE","IMAGE")],
 "ReferenceLatent":[("CONDITIONING","CONDITIONING")],
 "FluxGuidance":[("CONDITIONING","CONDITIONING")],
 "EmptySD3LatentImage":[("LATENT","LATENT")],
 "KSampler":[("LATENT","LATENT")],
 "KSamplerAdvanced":[("LATENT","LATENT")],
 "WanSoundImageToVideo":[("positive","CONDITIONING"),("negative","CONDITIONING"),("latent","LATENT")],
 "WanSoundImageToVideoExtend":[("positive","CONDITIONING"),("negative","CONDITIONING"),("latent","LATENT")],
 "LatentConcat":[("LATENT","LATENT")],
 "ImageFromBatch":[("IMAGE","IMAGE")],
 "ImageBatch":[("IMAGE","IMAGE")],
 "ImageStitch":[("IMAGE","IMAGE")],
 "ImageScale":[("IMAGE","IMAGE")],
 "RIFE VFI":[("IMAGE","IMAGE")],
 "ImageIntervalSelect":[("IMAGE","IMAGE")],
 "easy convertAnything":[("output","*")],
 "Text Concatenate":[("STRING","STRING")],
 "VHS_VideoCombine":[("Filenames","VHS_FILENAMES")],
 "MathExpression|pysssss":[("INT","INT"),("FLOAT","FLOAT")],
 "PrimitiveInt":[("INT","INT")],
 "LayerUtility: BooleanOperatorV2":[("boolean","BOOLEAN"),("string","STRING")],
 "easy ifElse":[("*","*")],
 "easy whileLoopStart":[("FLOW_CONTROL","FLOW_CONTROL"),("value0","*"),("value1","*"),("value2","*"),("value3","*"),("value4","*")],
 "easy whileLoopEnd":[("value0","*"),("value1","*"),("value2","*"),("value3","*"),("value4","*")],
}
# widget key order per class (non-link inputs, in ComfyUI's declared order)
WORDER={
 "Text Multiline":["text"],
 "EpisodeCompile":["script_text"],
 "SceneField":["scene_number","field"],
 "RenReedTTS":["scene_number"],
 "PathAfter":["path"],
 "UNETLoader":["unet_name","weight_dtype"],
 "LoraLoaderModelOnly":["lora_name","strength_model"],
 "ModelSamplingSD3":["shift"],
 "CLIPLoader":["clip_name","type","device"],
 "DualCLIPLoader":["clip_name1","clip_name2","type","device"],
 "VAELoader":["vae_name"],
 "AudioEncoderLoader":["audio_encoder_name"],
 "CLIPTextEncode":["text"],
 "FluxGuidance":["guidance"],
 "EmptySD3LatentImage":["width","height","batch_size"],
 "KSampler":["seed","steps","cfg","sampler_name","scheduler","denoise"],
 "KSamplerAdvanced":["add_noise","noise_seed","steps","cfg","sampler_name","scheduler","start_at_step","end_at_step","return_with_leftover_noise"],
 "WanSoundImageToVideo":["width","height","length","batch_size"],
 "WanSoundImageToVideoExtend":["length"],
 "LatentConcat":["dim"],
 "ImageFromBatch":["batch_index","length"],
 "ImageStitch":["direction","match_image_size","spacing_width","spacing_color"],
 "ImageScale":["upscale_method","width","height","crop"],
 "RIFE VFI":["ckpt_name","clear_cache_after_n_frames","multiplier","fast_mode","ensemble","scale_factor","dtype","torch_compile"],
 "ImageIntervalSelect":["interval","start_at","end_at"],
 "easy convertAnything":["output_type"],
 "Text Concatenate":["delimiter","clean_whitespace","text_a","text_b"],
 "VHS_VideoCombine":["frame_rate","loop_count","filename_prefix","format","pix_fmt","crf","save_metadata","trim_to_audio","pingpong","save_output"],
 "MathExpression|pysssss":["expression"],
 "PrimitiveInt":["value"],
 "LayerUtility: BooleanOperatorV2":["a_value","b_value","operator"],
 "easy whileLoopStart":["condition"],
}
# groups: (title, colour, [node ids])  — assigned by title match
def grp_of(nid,n):
    t=n["_meta"]["title"]; c=n["class_type"]
    if nid in ("1","2","3","4"): return "01 · INPUT + COMPILE"
    if c in ("UNETLoader","LoraLoaderModelOnly","ModelSamplingSD3","CLIPLoader","DualCLIPLoader",
             "VAELoader","AudioEncoderLoader") or t in ("Wan neg","Kontext neg"): return "02 · MODELS"
    if t in ("Char sheet","KF1 prompt","Kontext pos 1","sheet->lat","ref sheet","guide 2.5","canvas 720p",
             "anchor sample","ANCHOR","anchor->lat") and nid in ("21","22","23","24","25","26","27","28","29","30"):
        return "03 · ANCHOR (scene 1)"
    if t in ("const 1","idx0","WHILE start","scene_number","WHILE end","scene<count"): return "04 · LOOP CONTROL"
    if t in ("KF prompt","Kontext pos N","ref ANCHOR","guide 2.5","KF sample","KF image","scene==1",
             "keyframe pick","->832x480"): return "05 · KEYFRAME (anchor-based)"
    if t in ("TTS (loud)","audio embed","motion text","Wan pos"): return "06 · AUDIO + PROMPT"
    if c in ("WanSoundImageToVideo","WanSoundImageToVideoExtend","KSamplerAdvanced","LatentConcat"): return "07 · S2V CHUNKS (77×5)"
    if t in ("decode 385f","frame1","f1..384","fixed 385"): return "08 · DECODE + FIRST-FRAME FIX"
    if t in ("RIFE x3","every 2nd","sn->str","filename","save scene mp4"): return "09 · RIFE → SCENE MP4"
    if t in ("gate: after loop","FINAL PostMaster"): return "10 · FINAL (download here)"
    return "02 · MODELS"

COL={"01 · INPUT + COMPILE":("#223","#335"),"02 · MODELS":("#232","#353"),
 "03 · ANCHOR (scene 1)":("#322","#533"),"04 · LOOP CONTROL":("#332","#553"),
 "05 · KEYFRAME (anchor-based)":("#323","#535"),"06 · AUDIO + PROMPT":("#233","#355"),
 "07 · S2V CHUNKS (77×5)":("#422","#633"),"08 · DECODE + FIRST-FRAME FIX":("#242","#464"),
 "09 · RIFE → SCENE MP4":("#224","#446"),"10 · FINAL (download here)":("#442","#664")}
ORDER=["01 · INPUT + COMPILE","02 · MODELS","03 · ANCHOR (scene 1)","04 · LOOP CONTROL",
 "05 · KEYFRAME (anchor-based)","06 · AUDIO + PROMPT","07 · S2V CHUNKS (77×5)",
 "08 · DECODE + FIRST-FRAME FIX","09 · RIFE → SCENE MP4","10 · FINAL (download here)"]

buckets={g:[] for g in ORDER}
for nid,n in API.items(): buckets[grp_of(nid,n)].append(nid)

# layout: columns of groups
nodes=[]; links=[]; lid=[0]; pos={}
GX=0; groups=[]
for g in ORDER:
    ids=buckets[g]
    cols=max(1,(len(ids)+7)//8)
    w=cols*340+40; rows=min(8,len(ids))
    h=rows*130+80
    groups.append({"title":g,"bounding":[GX,0,w,h],"color":COL[g][1],"font_size":24,"flags":{}})
    for k,nid in enumerate(ids):
        cx=k//8; cy=k%8
        pos[nid]=[GX+20+cx*340, 60+cy*130]
    GX+=w+60

def out_slots(c): return OUT.get(c,[])
for nid,n in API.items():
    c=n["class_type"]; ins=n["inputs"]
    inputs=[]; widgets=[]
    worder=WORDER.get(c,[])
    for k,v in ins.items():
        if isinstance(v,list) and len(v)==2 and isinstance(v[0],str) and v[0] in API:
            inputs.append({"name":k,"type":"*","link":None,"_src":v})
    for k in worder:
        if k in ins and not (isinstance(ins[k],list) and len(ins[k])==2 and isinstance(ins[k][0],str)):
            widgets.append(ins[k])
    outs=[{"name":o[0],"type":o[1],"links":[],"slot_index":i} for i,o in enumerate(out_slots(c))]
    nodes.append({"id":int(nid),"type":c,"pos":pos[nid],"size":[300,90],"flags":{},"order":int(nid),
      "mode":0,"inputs":inputs,"outputs":outs,"properties":{"Node name for S&R":c},
      "widgets_values":widgets,"title":n["_meta"]["title"],"color":COL[grp_of(nid,n)][0],
      "bgcolor":COL[grp_of(nid,n)][1]})

byid={n["id"]:n for n in nodes}
for n in nodes:
    for inp in n["inputs"]:
        src=inp.pop("_src"); sid=int(src[0]); sslot=src[1]
        lid[0]+=1
        inp["link"]=lid[0]
        srcn=byid[sid]
        while len(srcn["outputs"])<=sslot:
            srcn["outputs"].append({"name":f"out{len(srcn['outputs'])}","type":"*","links":[],"slot_index":len(srcn["outputs"])})
        srcn["outputs"][sslot]["links"].append(lid[0])
        links.append([lid[0],sid,sslot,n["id"],n["inputs"].index(inp),srcn["outputs"][sslot]["type"]])

ui={"last_node_id":max(byid),"last_link_id":lid[0],"nodes":nodes,"links":links,
    "groups":groups,"config":{},"extra":{},"version":0.4}
json.dump(ui,open("workflow_v17_ui.json","w"),indent=1)
print(f"UI graph: {len(nodes)} nodes, {len(links)} links, {len(groups)} groups")
for g in groups: print(f"  {g['title']}: {len([1 for n in nodes if n['color']==COL[g['title']][0]])} nodes")
