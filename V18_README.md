# REN & REED v18 — Python-orchestrated scene loop

## What changed from v17.2
v17's single ComfyUI graph used `easy whileLoopStart/End` (ComfyUI-Easy-Use) to loop
over scenes *inside one graph execution*. That loop only ever ran 1 iteration
regardless of `scene_count` — a third-party node bug outside our control to fix
directly. v18 removes the in-graph loop entirely: `watchdog.py` now submits one
`/prompt` call per scene via the ComfyUI HTTP API and sequences them itself.
A failed scene retries (or halts loudly) on its own — it can no longer take the
whole episode down silently.

## Files
| File | Role |
|---|---|
| `build_v17_graph.py` | Generates 4 independent ComfyUI API templates: `trigger_template()` (Text Multiline → EpisodeCompile, the only thing loaded in the browser), `anchor_template()`, `scene_template()`, `postmaster_template()`. No loop/gate nodes anywhere. |
| `watchdog.py` | Orchestrator + observer. Watches for a freshly-compiled episode, submits anchor → each scene (1..scene_count, with retries) → postmaster, via the ComfyUI API. Reuses the existing cost-governor / rate-check logic. |
| `compiler_node.py` | Unchanged compile/validate logic, plus one addition: writes the compiled episode JSON to `EPISODE_COMPILED_PATH` — this file is the handoff watchdog watches for. |
| `episode_schema.json`, `validator_core.py`, `character_loader_node.py`, `renreed_tts_node.py`, `postmaster_node.py`, `scene_data_node.py`, `renreed_nodes_init.py` | Unchanged. |
| `local_pipeline_test.py` | 6-step local test (was 5) — the new Step 6 is a direct regression test for the loop bug: a mocked ComfyUI server asserts watchdog submits 3 separate scene prompts (not 1) and runs postmaster once, after all 3. Zero GPU cost. |
| `fetch_final.py` | **Local-only**, not baked into Docker, not pulled from GitHub. Run on your own PC: watches a running instance for `EPISODE_FINAL.mp4` and downloads it automatically the moment it's ready — no browser tab-watching needed. |

## Pipeline (zero manual, zero continuous monitoring)
```
Paste script/JSON in ComfyUI (trigger_template) → EpisodeCompile
  → writes episode_compiled.json  ─────────────────────────────┐
                                                                 ▼
watchdog.py (background, already running):                watches file
  → ANCHOR (one 720p Kontext render from the character sheet, saved to disk)
  → for scene in 1..scene_count:
        keyframe (Kontext from ANCHOR) → 832x480 → TTS → S2V init+4 extends
        → KSamplerAdvanced x5 → decode → first-frame fix → save scene mp4
        (retries up to SCENE_MAX_RETRIES on failure; never overwrites another
         scene's file — deterministic scene_NNNNN.mp4 naming)
  → PostMaster (concat + 720p grade + SFX mux) → EPISODE_FINAL.mp4
```
Review in ComfyUI's inline player (unchanged), or run `fetch_final.py` locally
to auto-download the finished file to your own PC without watching anything.

## Run order
1. `python local_pipeline_test.py` — 7 checks, $0 cost, must be all-pass before any GPU spend.
2. `docker build` — build-time asserts now also verify all 4 templates build cleanly with no dangling links (fails at build time, not on a paid GPU).
3. FREE CHECK: `docker run` CPU-only → open ComfyUI → load `workflow_v17_api.json` (the trigger graph) → confirm 0 red nodes.
4. Rent GPU, paste script, watchdog does the rest.

## Explicitly out of scope for this change
- RIFE frame interpolation (separate quality upgrade, unrelated to the loop fix).
- Processing more than one episode per container lifetime (current scope: one at a time).
- Vast.ai instance auto-rent/auto-stop — still manual (or operator-driven) for now.
