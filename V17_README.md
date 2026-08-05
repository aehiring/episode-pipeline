# REN & REED v17 — FINAL PACKAGE (graph included)

## DOWNLOAD LIST — YES, nodes bhi (Docker unhe COPY karta hai)
Sab 14 files C:\episode-pipeline\ mein rakho:

| File | Rename to | Role |
|---|---|---|
| Dockerfile.v17 | **Dockerfile** | v17 image (build asserts baked) |
| entrypoint.v17.sh | **entrypoint.sh** | models+LoRAs+scheduler probe+watchdog |
| workflow_v17_api.json | (same) | THE GRAPH — 69 nodes, QA passed |
| episode_schema.json | (same) | single source of truth v1.1 |
| validator_core.py | (same) | V1–V13 semantic checks |
| compiler_node.py | (same) | Claude compile + 1 repair + loud stop |
| renreed_tts_node.py | (same) | ElevenLabs direct, loud errors |
| character_loader_node.py | (same) | baked refs, only characters_used |
| scene_data_node.py | (same) | SceneField/SceneCount/PathAfter |
| postmaster_node.py | (same) | concat+720p grade+SFX mux+DOWNLOAD |
| renreed_nodes_init.py | (same) | node package __init__ |
| sfx_build.py | (same) | ONE-TIME: 30 SFX from your key |
| watchdog.py | (same) | observer+governor+rate rule |
| build_v17_graph.py | (same) | graph generator (regen/audit) |

Plus folders (README niche): assets/characters/<NAME>/ref.png ×6, assets/sfx/ (sfx_build se).

## MODEL LINKS — verification status
9 base models: **kal instance par LIVE download ho chuke** (tumhara log) — proven.
ae.safetensors mirrors: pichli chat mein live-verified (335MB, SHA256 match).
2 naye LoRA URLs: Kijai rank256 (repo listing mein 2.92GB confirmed) + Wan2.2-Lightning
Seko-V1.1 fallback (repo tree confirmed) — dono par entrypoint SIZE-GATE hai:
chhota/HTML file = FAIL loud, ComfyUI phir bhi khulti hai inspection ke liye.
Sandbox se HuggingFace direct hit nahi ho sakta — instance par pehla start hi final proof hai,
aur wo render se PEHLE hota hai (paisa nahi jalta).

## PIPELINE (graph ke andar, zero manual)
SCRIPT paste -> EpisodeCompile -> SceneCount/CharacterRefs
-> ANCHOR (Kontext from char sheet, 720p)
-> LOOP per scene: keyframe (scene1=ANCHOR, warna Kontext-from-ANCHOR) -> 832x480
   -> RenReedTTS -> wav2vec2 -> S2V init77 + 4 extends (LatentConcat accumulate,
      audio offset 80/160/240/320 — core math verified)
   -> KSA 4 steps CFG1 euler/beta, LoRA rank256@1.5
   -> decode 397 -> first-frame fix -> EXACT 385 -> RIFE x3 -> /2 -> 578f @24
   -> scene mp4 (480p, crf17, speech audio)
-> PathAfter gate -> EpisodePostMaster: numeric-sort concat -> lanczos 720p
   -> grade -> SFX mux (duration=first) -> crf17 -> PLAYER + DOWNLOAD

## RUN ORDER
1. assets bharo (characters + sfx_build.py)
2. docker build (asserts yahin pakdenge)  3. push
4. FREE CHECK: docker run CPU-only -> ComfyUI kholo -> graph load -> red nodes = 0
5. Vast RTX PRO 6000 <= $1.45/hr, env: ANTHROPIC_API_KEY, ELEVENLABS_API_KEY,
   ELEVENLABS_VOICES, VAST_RATE_HR
6. Script paste -> Run -> watchdog dekho -> EPISODE_FINAL.mp4 download

## QA LEDGER (is package par)
Unit 21 + Integration 11 + System 25 + NewNodes 10+2 + Graph 15 + DecodeMath 1 = **85 checks, 0 fail**
Graph: har input real signatures (v16-proven + source) ke khilaaf lint, links resolve,
loop sim 20 & 21 scenes, PostMaster loop-gated, single terminal.
