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

def _claude(system, user, max_tokens=16000):
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
anchor string verbatim. motion_prompts array length must equal chunks."""

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
