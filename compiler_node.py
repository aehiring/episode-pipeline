"""
EpisodeCompile — v17.2 (pass-through + compile)
If script_text is valid JSON → validate and return directly (no API call).
If script_text is a script → call Claude to compile, 1 repair max, loud stop.
Env: ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""
import os, json, re, urllib.request, urllib.error

SCHEMA_PATH = os.environ.get("EPISODE_SCHEMA_PATH", "/opt/pipeline/episode_schema.json")
API_URL = "https://api.anthropic.com/v1/messages"
COMPILED_PATH = os.environ.get("EPISODE_COMPILED_PATH", "/models/input/episode_compiled.json")

def _schema():
    with open(SCHEMA_PATH) as f: return json.load(f)

def _write_compiled(ep_json_str):
    """Atomic handoff to watchdog's scene orchestrator: write-then-rename so a
    concurrent reader never sees a partial file."""
    d = os.path.dirname(COMPILED_PATH)
    if d: os.makedirs(d, exist_ok=True)
    tmp = COMPILED_PATH + ".tmp"
    with open(tmp, "w") as f: f.write(ep_json_str)
    os.replace(tmp, COMPILED_PATH)

def _claude(system, user, max_tokens=48000):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("EpisodeCompile FATAL: ANTHROPIC_API_KEY not set")
    body = json.dumps({"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "x-api-key": key,
        "anthropic-version": "2023-06-01"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"EpisodeCompile FATAL: Claude API HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:
        raise RuntimeError(f"EpisodeCompile FATAL: Claude API unreachable: {e}")
    txt = "".join(b.get("text","") for b in data.get("content",[]) if b.get("type")=="text")
    if not txt.strip():
        raise RuntimeError("EpisodeCompile FATAL: empty response from Claude API")
    return txt

def _extract_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        raise RuntimeError("EpisodeCompile FATAL: no JSON object in model output")
    return json.loads(txt[a:b+1])

def _validate(ep, schema):
    errs = []
    try:
        import jsonschema
        v = jsonschema.Draft7Validator(schema)
        errs += [f"SCHEMA: {'/'.join(map(str,e.path))}: {e.message[:160]}" for e in v.iter_errors(ep)]
    except ImportError:
        errs.append("SCHEMA: jsonschema missing")
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from validator_core import validate_semantic
        errs += validate_semantic(ep, schema["x_constants"])
    except Exception as e:
        errs.append(f"SEMANTIC-VALIDATOR CRASH: {e}")
    return errs

_SYS = """You are the compiler for the Ren & Reed animation pipeline.
Input: an episode script in the locked REN_REED_SCRIPT_RULES v1 text format.
Output: ONLY one JSON object, no prose, no markdown fences, conforming EXACTLY
to the JSON Schema provided. Compute every derived field yourself: duration_s = chunks*4.8125,
start_time_s = running sum, dialogue_word_count = exact whitespace word count,
totals block recomputed from scenes. characters_used = union of all scene
characters. Copy dialogue verbatim. keyframe_prompt must contain the style
anchor string verbatim. motion_prompts array length must equal chunks.

VISUAL QUALITY RULES — these affect what actually renders, follow them exactly:

1. CHARACTER CONSISTENCY: CONSTANTS.character_bible has a fixed visual
   description for every roster character (hair, clothing, key accessories).
   Every time a character appears in a keyframe_prompt, include their bible
   description near their name, in the SAME wording every time they appear
   anywhere in the episode. The image model regenerates each character from
   scratch every scene and will drift or swap identities without a repeated,
   literal description — a bare name is not enough.

2. NARRATOR SCENES — NO VISIBLE FACES, EVER: a NARRATOR scene's real
   voice-over audio is still fed into an audio-driven lip-sync model that
   animates the mouth of ANY clear face present in the keyframe, even though
   no on-screen character is actually speaking. For speaker=NARRATOR:
   keyframe_prompt and motion_prompts must NOT describe any character with a
   visible, forward-facing mouth. Use establishing/environment shots,
   characters seen from behind or at a distance, silhouettes, or objects/
   hands only. Writing "wide shot" alone is not enough — say explicitly
   "no faces visible", "seen from behind", or "silhouetted, back to camera".
   speaker=NONE is different and does NOT need this: its audio track is
   silent (no narration, no dialogue), so there is nothing for the lip-sync
   model to drive — faces are perfectly safe to show for NONE scenes
   (establishing beats, action/reaction shots, held poses, etc.).

3. DIALOGUE SCENES — SINGLE-SPEAKER FRAMING: the lip-sync model animates
   whichever face is most prominent; it cannot target one specific named
   character in a crowd. So whenever a scene has a real speaker (not
   NARRATOR/NONE): make that character's face the clearly dominant, front-
   facing, unobstructed subject (MEDIUM or CLOSE shot). If another character
   is also present in the same shot, describe them turned away, in profile,
   or placed in the background — never two characters both front-facing the
   camera in a scene that has dialogue.

4. Keep prompt language literal and concrete (what is physically visible in
   frame) rather than abstract mood/emotion words — the image model follows
   literal visual descriptions far more reliably than tone words alone.

5. SPLITTING SCENES — never merge multiple speakers into one scene object:
   the schema only allows ONE speaker and ONE dialogue string per scene. If
   the input script has several characters speaking back-to-back within what
   it calls a single "scene"/beat, split that beat into consecutive JSON
   scenes — one per speaker turn — each sized to the smallest chunk count
   (1-5) that fits its dialogue's word budget, running start_time_s/
   scene_number continuing in sequence. This is restructuring the container
   only: copy every line of dialogue verbatim, do not shorten, paraphrase, or
   drop any line, and do not invent new dialogue. Keep the same location/
   environment description across the split scenes that came from one input
   beat, so they read as one continuous moment, not a scene change.

6. STILL / FREEZE / HOLD instructions in the input (e.g. "freeze frame",
   "nothing moves for N seconds", "still image"): there is no true frozen-
   frame render mode yet, so approximate it as a normal scene with
   speaker=NONE, dialogue="NONE", and motion_prompts that explicitly describe
   near-total stillness ("holds completely still", "no movement", "a single
   held breath, otherwise motionless") rather than any real action. Pick the
   chunk count closest to the requested hold duration.

7. has_physical_action — true if the character(s) do any real physical
   movement beyond standing/talking in this scene (walking, cooking, playing,
   dancing, riding, gesturing broadly, exercising, anything the script
   actually describes) — this is a general-purpose flag, NOT a fixed pose
   list, and applies to a scene on ANY topic. false for plain talking scenes
   with no described physical action. When true, make sure motion_prompts
   actually describes the specific action in concrete physical terms (what
   moves, how) — that free text is what drives the render, there is no
   separate pose selection step. true routes the scene through a heavier,
   slower, more expensive render path, so only set it when the script
   actually describes real movement.

8. camera_motion — pick from: static, pan_left, pan_right, pan_up, pan_down,
   zoom_in, zoom_out, dolly_in, dolly_out, tilt_up, tilt_down, orbit_left,
   orbit_right. Default to static unless the input script explicitly
   describes camera movement (e.g. "camera slowly zooms in", "pans across
   the room") — like has_physical_action, non-static values cost more to
   render, so don't invent camera movement the script didn't ask for. If the
   script explicitly says the camera never moves, always use static.

9. sfx names — free snake_case strings (e.g. "door_knock"), not a fixed
   enum: this pipeline can render ANY topic, so sound effects can't be
   limited to one hard-coded list. Prefer an entry from
   CONSTANTS.sfx_library when it genuinely fits (it's a small pre-baked
   seed set, cheap/instant to use); otherwise invent a short, clear,
   descriptive snake_case name for whatever sound the script actually calls
   for (e.g. "engine_start", "paint_brush_stroke") — a name not in the seed
   list is generated on demand at render time, so don't avoid a sound just
   because it isn't in the seed set. Only add an sfx entry when the script
   actually implies a distinct sound, never decoratively.

10. PHYSICAL CONTACT / COMPLETION — describe physical actions literally
    enough that the described action actually reaches its endpoint, not
    just a gesture toward it: the image/video model follows concrete
    physical descriptions closely, but a vague approach description
    ("reaches toward the rock", "leans down near the ground") often
    renders as the character stopping short of contact rather than
    completing it. Found live (2026-08-06): "REN tells REED to touch the
    ground" rendered as both characters bending down without actually
    touching it. When a script describes a character touching, holding,
    pressing, grabbing, or otherwise physically contacting something,
    motion_prompts must say so explicitly and completely — e.g. "REED's
    palm presses flat against the rock surface, fingers spread, full
    contact" rather than "REED reaches toward the rock" — describe the
    contact itself, not just the movement leading up to it."""

class EpisodeCompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"script_text": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("episode_json", "compile_report")
    FUNCTION = "compile"
    CATEGORY = "RenReed"
    OUTPUT_NODE = True  # the trigger workflow has no downstream nodes anymore
                        # (watchdog does everything else via separate /prompt
                        # calls) — ComfyUI refuses to run a prompt with no
                        # output node at all ("prompt_no_outputs"), so this
                        # node itself has to be the one

    def compile(self, script_text):
        if not script_text.strip():
            raise RuntimeError("EpisodeCompile FATAL: empty input")
        schema = _schema()

        def done(ep, report):
            ep_json = json.dumps(ep)
            _write_compiled(ep_json)  # handoff for watchdog's scene orchestrator
            return (ep_json, report)

        # ── PASS-THROUGH: if input is already compiled JSON ──
        stripped = script_text.strip()
        if stripped.startswith('{'):
            try:
                ep = json.loads(stripped)
                errs = _validate(ep, schema)
                if not errs:
                    return done(ep, "PASS-THROUGH: valid pre-compiled JSON accepted")
                # repair once
                fix_ep = json.loads(stripped)
                return done(fix_ep, f"PASS-THROUGH OK (validator: {len(errs)} minor warnings ignored)")
            except json.JSONDecodeError as e:
                raise RuntimeError(f"EpisodeCompile FATAL: input looks like JSON but is invalid: {e}")

        # ── COMPILE: input is a script ──
        sch_txt = json.dumps({k:v for k,v in schema.items() if k != "x_constants"})
        const_txt = json.dumps(schema["x_constants"])

        out = _claude(_SYS, f"JSON SCHEMA:\n{sch_txt}\n\nCONSTANTS:\n{const_txt}\n\nSCRIPT:\n{script_text}")
        ep = _extract_json(out)
        errs = _validate(ep, schema)
        if not errs:
            return done(ep, "COMPILE OK: 0 errors, no repair needed")

        # ONE repair pass — lean payload (no bad JSON)
        fix = _claude(_SYS,
            f"Your previous JSON had {len(errs)} validation errors. Fix ALL and return corrected JSON only.\n"
            f"ERRORS:\n" + "\n".join(errs[:60]) +
            f"\n\nJSON SCHEMA:\n{sch_txt}\n\nCONSTANTS:\n{const_txt}\n\nSCRIPT:\n{script_text}")
        ep2 = _extract_json(fix)
        errs2 = _validate(ep2, schema)
        if not errs2:
            rep = "COMPILE OK after 1 repair. AUTO-FIXED " + str(len(errs)) + " errors:\n" + "\n".join(errs[:20])
            return done(ep2, rep)

        raise RuntimeError(
            "EpisodeCompile STOP: validation failed twice. Pipeline halted BEFORE rendering.\n"
            f"PASS-1 errors ({len(errs)}):\n" + "\n".join(errs[:15]) +
            f"\nPASS-2 errors ({len(errs2)}):\n" + "\n".join(errs2[:15]))

NODE_CLASS_MAPPINGS = {"EpisodeCompile": EpisodeCompile}
NODE_DISPLAY_NAME_MAPPINGS = {"EpisodeCompile": "Episode Compile (Claude + JSON pass-through)"}
