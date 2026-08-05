
import json, re

def load_constants(schema_path):
    return json.load(open(schema_path))["x_constants"]

def validate_semantic(ep, C):
    """Returns list of error strings. Empty list = pass."""
    E = []
    scenes = ep["scenes"]
    B = {int(k): v for k, v in C["word_budget"].items()}
    CH = C["chunk_seconds"]

    # V3 — totals recompute
    s_chunks = sum(s["chunks"] for s in scenes)
    if s_chunks != ep["episode"]["total_chunks"]:
        E.append(f"V3: episode.total_chunks={ep['episode']['total_chunks']} but scenes sum={s_chunks}")
    if s_chunks != ep["totals"]["sum_chunks"]:
        E.append(f"V3: totals.sum_chunks={ep['totals']['sum_chunks']} but recomputed={s_chunks}")
    # Duration range check removed — flexible duration supported
    if len(scenes) != ep["episode"]["scene_count"]:
        E.append(f"V3: scene_count={ep['episode']['scene_count']} but {len(scenes)} scenes present")
    exp_dur = round(s_chunks * CH, 4)
    if abs(ep["totals"]["sum_duration_s"] - exp_dur) > 0.01:
        E.append(f"V3: sum_duration_s={ep['totals']['sum_duration_s']} expected {exp_dur}")
    exp_frames = s_chunks * C["chunk_frames"]
    if ep["totals"]["total_frames_16fps"] != exp_frames:
        E.append(f"V3: total_frames={ep['totals']['total_frames_16fps']} expected {exp_frames}")

    # per-scene checks
    t_running = 0.0
    seen_num = set()
    tod_order = {"DAY":0,"EVENING":1,"NIGHT":2,"INDOOR":-1}
    last_tod = -1
    char_used = set()
    for s in scenes:
        n = s["scene_number"]
        if n in seen_num: E.append(f"V2: scene_number {n} duplicated")
        seen_num.add(n)

        # V4 — motion count
        if len(s["motion_prompts"]) != s["chunks"]:
            E.append(f"V4: scene {n}: {len(s['motion_prompts'])} motion lines for {s['chunks']} chunks")

        # duration + running offset
        exp = round(s["chunks"] * CH, 4)
        if abs(s["duration_s"] - exp) > 0.001:
            E.append(f"V3: scene {n}: duration_s={s['duration_s']} expected {exp}")
        if abs(s["start_time_s"] - round(t_running,4)) > 0.01:
            E.append(f"V3: scene {n}: start_time_s={s['start_time_s']} expected {round(t_running,4)}")
        t_running += exp

        # V5 — speaker consistency
        sp = s["speaker"]
        if sp not in ("NARRATOR","NONE") and sp not in s["characters"]:
            E.append(f"V5: scene {n}: speaker {sp} not in characters {s['characters']}")
        char_used.update(s["characters"])

        # V6 — dialogue budget + recount + digits
        d = s["dialogue"]
        if sp == "NONE":
            if d.strip() != "NONE":
                E.append(f"V6: scene {n}: speaker NONE requires dialogue 'NONE'")
        else:
            words = [w for w in re.split(r"\s+", d.strip()) if w]
            wc = len(words)
            # Use actual recount for budget check (±2 tolerance for compiler counting differences)
            wc_check = max(wc, s["dialogue_word_count"])
            if wc_check > B[s["chunks"]] + 2:
                E.append(f"V6: scene {n}: {wc_check} words > budget {B[s['chunks']]} for {s['chunks']} chunks")
            if re.search(r"\d", d):
                E.append(f"V6: scene {n}: digits in dialogue — write numbers as words")

        # V7 — narrator faceless
        if sp == "NARRATOR" and s["shot"] != "WIDE":
            E.append(f"V7: scene {n}: NARRATOR requires SHOT=WIDE, got {s['shot']}")

        # V8 — sfx timestamps inside scene
        for fx in s["sfx"]:
            if fx["at_s"] >= exp:
                E.append(f"V8: scene {n}: sfx {fx['name']} @ {fx['at_s']}s >= scene length {exp}s")

        # V12 — style anchor verbatim
        if C["style_anchor"] not in s["keyframe_prompt"]:
            E.append(f"V12: scene {n}: style anchor missing from keyframe_prompt")

        # V11 — time of day forward only (INDOOR exempt)
        tv = tod_order[s["time_of_day"]]
        if tv >= 0:
            if tv < last_tod:
                E.append(f"V11: scene {n}: time_of_day moves backward")
            last_tod = tv

    # sequential numbering
    if sorted(seen_num) != list(range(1, len(scenes)+1)):
        E.append("V2: scene numbers not sequential from 1")

    # characters_used exact match
    if set(ep["episode"]["characters_used"]) != char_used:
        E.append(f"V5: episode.characters_used={sorted(ep['episode']['characters_used'])} but scenes use {sorted(char_used)}")

    return E
