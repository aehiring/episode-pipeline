#!/usr/bin/env bash
# ── Ren & Reed v17 entrypoint ──
set -uo pipefail   # NOT -e: one bad download must not crash-loop the box

M=/models
mkdir -p "$M"/{diffusion_models,text_encoders,vae,upscale_models,audio_encoders,loras,input,output,episode_state}
mkdir -p "$M/input/episode_audio" "$M/output/scenes"

if command -v hf >/dev/null 2>&1; then HF="hf";
elif command -v huggingface-cli >/dev/null 2>&1; then HF="huggingface-cli";
else pip install -q "huggingface_hub[cli]" && HF="hf"; fi
echo "using downloader: $HF"
FAILED=0

get() {  # get <repo> <path-in-repo> <dest-dir>
  local repo="$1" path="$2" dir="$3" name dest
  name="$(basename "$path")"; dest="$dir/$name"
  [ -s "$dest" ] && { echo "  ok   $name"; return 0; }
  echo "  pull $name"
  rm -rf /tmp/hf && mkdir -p /tmp/hf
  if ! $HF download "$repo" "$path" --local-dir /tmp/hf >/dev/null; then
    echo "  FAIL $name (repo=$repo)"; FAILED=$((FAILED+1)); return 1; fi
  local src; src="$(find /tmp/hf -type f -name "$name" | head -n1)"
  [ -z "$src" ] && { echo "  FAIL $name (not found after download)"; FAILED=$((FAILED+1)); return 1; }
  mv "$src" "$dest"; echo "  done $name ($(du -h "$dest" | cut -f1))"
}

echo "== models =="
C=Comfy-Org/Wan_2.2_ComfyUI_Repackaged
get "$C" split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors "$M/diffusion_models"
get "$C" split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors   "$M/text_encoders"
get "$C" split_files/vae/wan_2.1_vae.safetensors                            "$M/vae"
get "$C" split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors "$M/audio_encoders"
get Comfy-Org/flux1-schnell flux1-schnell-fp8.safetensors "$M/diffusion_models"
get Comfy-Org/flux1-kontext-dev_ComfyUI split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors "$M/diffusion_models"
T=comfyanonymous/flux_text_encoders
get "$T" clip_l.safetensors                  "$M/text_encoders"
get "$T" t5xxl_fp8_e4m3fn_scaled.safetensors "$M/text_encoders"
get Kim2091/UltraSharp 4x-UltraSharp.pth "$M/upscale_models"

# ── FLUX VAE (non-gated mirrors + size gate) ──
AE="$M/vae/ae.safetensors"; AE_MIN=300000000
ae_ok(){ [ -f "$AE" ] && [ "$(stat -c%s "$AE" 2>/dev/null || echo 0)" -ge "$AE_MIN" ]; }
if ae_ok; then echo "  ok   ae.safetensors"; else
  rm -f "$AE"; echo "  pull ae.safetensors (ffxvs/vae-flux)"
  curl -fL --retry 3 --retry-delay 5 -o "$AE" \
    "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors" || true
  ae_ok || { rm -f "$AE"; echo "  pull ae.safetensors (fallback fofr/comfyui)";
    curl -fL --retry 3 --retry-delay 5 -o "$AE" \
      "https://huggingface.co/fofr/comfyui/resolve/bd378c497fb9329c1dc61be4a74a2eb2bc4f2b70/vae/ae.safetensors" || true; }
  ae_ok && echo "  done ae.safetensors" || { echo "  FAIL ae.safetensors"; FAILED=$((FAILED+1)); }
fi

# ── v17 LoRAs: primary (S2V-tested) + official fallback, size-gated ──
lget(){ # lget <url> <dest> <min_bytes>
  local url="$1" dest="$2" min="$3"
  [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -ge "$min" ] && { echo "  ok   $(basename "$dest")"; return 0; }
  rm -f "$dest"; echo "  pull $(basename "$dest")"
  curl -fL --retry 3 --retry-delay 5 -o "$dest" "$url" || true
  [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -ge "$min" ] \
    && echo "  done $(basename "$dest")" \
    || { rm -f "$dest"; echo "  FAIL $(basename "$dest")"; FAILED=$((FAILED+1)); }
}
# Primary LoRA (2.38GB) consistently fails mid-download on rental networks — skipped.
# Fallback LoRA is the active default in the graph.
lget "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
     "$M/loras/wan22_lightning_fallback_high.safetensors" 400000000

# ── input dir: LoadAudio-class nodes resolve against ComfyUI/input ──
if [ ! -L /opt/ComfyUI/input ]; then rm -rf /opt/ComfyUI/input; ln -s "$M/input" /opt/ComfyUI/input; fi
echo "  ok   input dir -> $M/input"
if [ ! -L /opt/ComfyUI/output ]; then rm -rf /opt/ComfyUI/output; ln -s "$M/output" /opt/ComfyUI/output; fi
echo "  ok   output dir -> $M/output"


# ── GitHub auto-pull: update code files without rebuild ──
CODE_URL="${PIPELINE_CODE_URL:-https://raw.githubusercontent.com/aehiring/episode-pipeline/main}"
FILES="compiler_node.py character_loader_node.py renreed_tts_node.py postmaster_node.py scene_data_node.py validator_core.py renreed_nodes_init.py"
UPDATED=0; FAILED=0
for f in $FILES; do
    dest="/opt/ComfyUI/custom_nodes/RenReedNodes/$f"
    if curl -fsSL --connect-timeout 10 --max-time 30 "$CODE_URL/$f" -o /tmp/_pull_$f 2>/dev/null; then
        if python3 -c "import ast,sys; ast.parse(open('/tmp/_pull_$f').read()); sys.exit(0)" 2>/dev/null; then
            mv /tmp/_pull_$f "$dest"; UPDATED=$((UPDATED+1))
        else
            rm -f /tmp/_pull_$f; echo "  skip $f (syntax error in pull)"; FAILED=$((FAILED+1))
        fi
    fi
done
# Also pull watchdog + its graph-template module (both live in /opt/pipeline, not the node package dir)
for pf in watchdog.py build_v17_graph.py; do
    if curl -fsSL --connect-timeout 10 --max-time 30 "$CODE_URL/$pf" -o "/tmp/_pull_$pf" 2>/dev/null; then
        python3 -c "import ast; ast.parse(open('/tmp/_pull_$pf').read())" 2>/dev/null && mv "/tmp/_pull_$pf" "/opt/pipeline/$pf"
    fi
done
echo "  code pull: $UPDATED updated, $FAILED skipped (baked fallback in use)"

# ── runtime env asserts (loud, before ComfyUI) ──
[ -n "${ANTHROPIC_API_KEY:-}" ]  || echo "!! ANTHROPIC_API_KEY empty — EpisodeCompile will FAIL LOUDLY"
[ -n "${ELEVENLABS_API_KEY:-}" ] || echo "!! ELEVENLABS_API_KEY empty — RenReedTTS will FAIL LOUDLY"
[ -n "${ELEVENLABS_VOICES:-}" ]  || echo "!! ELEVENLABS_VOICES empty — RenReedTTS will FAIL LOUDLY"
[ -n "${VAST_RATE_HR:-}" ]       || echo "!! VAST_RATE_HR empty — watchdog cannot verify the \$1.45 rate rule"

[ "$FAILED" -gt 0 ] && { echo "!! $FAILED download(s) failed — ComfyUI starts for inspection; FIX BEFORE RENDERING"; }

echo "== model inventory =="; du -sh "$M"/* 2>/dev/null

echo "== starting ComfyUI =="
cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --use-sage-attention &
COMFY_PID=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8188/system_stats >/dev/null && break; sleep 5; done

# ── scheduler probe: beta57 if present, else beta (spec #3) ──
SCHED=$(curl -sf http://127.0.0.1:8188/object_info 2>/dev/null | python3 -c "
import sys, json
try:
    oi = json.load(sys.stdin)
    scheds = oi['KSamplerAdvanced']['input']['required']['scheduler'][0]
    print('beta57' if 'beta57' in scheds else 'beta')
except Exception:
    print('beta')")
echo "== scheduler selected: $SCHED =="
echo "$SCHED" > /models/episode_state/scheduler.txt

echo "== starting watchdog =="
python3 /opt/pipeline/watchdog.py &
echo "== v18 ready =="
wait "$COMFY_PID"
