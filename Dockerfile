# ── Ren & Reed episode pipeline v17 ──
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 aria2 tmux curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /opt/ComfyUI
RUN pip install -r requirements.txt

# ── custom node packs (BEFORE cu128 swap — ordering rule) ──
WORKDIR /opt/ComfyUI/custom_nodes
RUN set -eux; \
    for repo in \
        https://github.com/ltdrdata/ComfyUI-Manager.git \
        https://github.com/kijai/ComfyUI-KJNodes.git \
        https://github.com/yolain/ComfyUI-Easy-Use.git \
        https://github.com/WASasquatch/was-node-suite-comfyui.git \
        https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git \
        https://github.com/chflame163/ComfyUI_LayerStyle.git \
        https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside.git \
        https://github.com/EllangoK/ComfyUI-post-processing-nodes.git \
        https://github.com/set-soft/ComfyUI-AudioBatch.git \
        https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    ; do git clone --depth 1 "$repo"; done; \
    for d in */ ; do \
        if [ -f "$d/requirements.txt" ]; then pip install -r "$d/requirements.txt" || true; fi; \
    done
# NOTE: jerilseb/ComfyUI-ElevenLabs RETIRED — replaced by RenReedTTS (own node)

# ── cu128 torch (Blackwell sm_120) — must come AFTER node reqs ──
RUN pip install --upgrade --force-reinstall \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
RUN pip install sageattention || echo "sageattention unavailable, using default attention"
RUN python -c "import torch; \
    cuda = torch.version.cuda or '0.0'; \
    print('torch', torch.__version__, '| built against CUDA', cuda); \
    assert tuple(int(x) for x in cuda.split('.')[:2]) >= (12, 8), 'cu128 required for Blackwell'"

RUN pip install requests google-api-python-client google-auth jsonschema pydub --break-system-packages 2>/dev/null || pip install requests google-api-python-client google-auth jsonschema pydub

# pydub/ffmpeg sanity (RenReedTTS uses torchaudio; ffmpeg needed by PostMaster)
RUN python -c "import shutil; assert shutil.which('ffmpeg'), 'ffmpeg missing'; print('ffmpeg OK')"

RUN rm -rf /opt/ComfyUI/models && ln -s /models /opt/ComfyUI/models

# ── pipeline: schema, validator, custom nodes, watchdog, assets ──
WORKDIR /opt/pipeline
COPY episode_schema.json      ./
COPY validator_core.py        ./
COPY watchdog.py              ./
COPY assets/characters/       ./assets/characters/
COPY assets/sfx/              ./assets/sfx/

# RenReed nodes live as ONE custom-node package
RUN mkdir -p /opt/ComfyUI/custom_nodes/RenReedNodes
COPY compiler_node.py character_loader_node.py renreed_tts_node.py postmaster_node.py scene_data_node.py \
     /opt/ComfyUI/custom_nodes/RenReedNodes/
COPY renreed_nodes_init.py /opt/ComfyUI/custom_nodes/RenReedNodes/__init__.py
# validator_core importable by compiler node
RUN cp /opt/pipeline/validator_core.py /opt/ComfyUI/custom_nodes/RenReedNodes/

# ── BUILD-TIME ASSET & WIRING ASSERTS (fail here, not on a paid GPU) ──
RUN python - << 'PY'
import json, os, sys
C = json.load(open('/opt/pipeline/episode_schema.json'))["x_constants"]
errs = []
# characters: every roster member baked with ref.png
for name in C["roster"]:
    p = f"/opt/pipeline/assets/characters/{name}/ref.png"
    if not os.path.isfile(p): errs.append(f"character missing: {p}")
# sfx: exact match with schema list
have = {f[:-4] for f in os.listdir('/opt/pipeline/assets/sfx') if f.endswith('.mp3')}
need = set(C["sfx_library"])
if have != need:
    errs.append(f"sfx mismatch: missing={sorted(need-have)} extra={sorted(have-need)}")
for f in sorted(have & need):
    fp = f"/opt/pipeline/assets/sfx/{f}.mp3"
    if os.path.getsize(fp) < 6000: errs.append(f"sfx too small (junk): {fp}")
# nodes importable (logic layer only; torch parts lazy)
sys.path.insert(0, '/opt/ComfyUI/custom_nodes/RenReedNodes')
import validator_core, compiler_node, character_loader_node, renreed_tts_node, postmaster_node, scene_data_node  # noqa
if errs:
    print("BUILD ASSERT FAILED:"); [print("  -", e) for e in errs]; sys.exit(1)
print(f"BUILD ASSERTS OK: {len(C['roster'])} characters, {len(need)} SFX, 6 modules import clean")
PY

COPY workflow_v17_api.json    ./
COPY entrypoint.sh            ./
RUN chmod +x entrypoint.sh

ENV COMFY_HOST=http://127.0.0.1:8188 \
    COMFY_OUTPUT=/models/output \
    COMFY_INPUT=/models/input \
    STATE_DIR=/models/episode_state \
    EPISODE_SCHEMA_PATH=/opt/pipeline/episode_schema.json \
    CHARACTER_ASSET_DIR=/opt/pipeline/assets/characters \
    SFX_ASSET_DIR=/opt/pipeline/assets/sfx \
    EPISODE_AUDIO_DIR=/models/input/episode_audio \
    EPISODE_CAP=5.50

# runtime-only secrets/config — values come from the Vast template
ENV ANTHROPIC_API_KEY="" ELEVENLABS_API_KEY="" ELEVENLABS_VOICES="" VAST_RATE_HR=""

EXPOSE 8188
ENTRYPOINT ["/opt/pipeline/entrypoint.sh"]
