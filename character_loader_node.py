"""
CharacterRefLoader — loads ONLY the episode's required character reference
images from the Docker-baked asset dir. Loud error if any ref missing.
"""
import os, json

CHAR_DIR = os.environ.get("CHARACTER_ASSET_DIR", "/opt/pipeline/assets/characters")

def resolve_refs(characters_used, base=None):
    base = base or CHAR_DIR
    out, missing = {}, []
    for name in characters_used:
        d = os.path.join(base, name)
        ref = os.path.join(d, "ref.png")
        if not os.path.isdir(d):
            missing.append(f"{name}: folder missing ({d})")
        elif not os.path.isfile(ref):
            missing.append(f"{name}: ref.png missing in {d}")
        else:
            out[name] = ref
    if missing:
        raise RuntimeError("CharacterRefLoader FATAL — missing baked assets:\n" + "\n".join(missing))
    return out

class CharacterRefLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"episode_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("character_refs", "names_csv")
    FUNCTION = "load"
    CATEGORY = "RenReed"

    def load(self, episode_json):
        ep = json.loads(episode_json)
        used = ep["episode"]["characters_used"]
        refs = resolve_refs(used)
        import torch, numpy as np
        from PIL import Image
        tensors = []
        for name in used:
            img = Image.open(refs[name]).convert("RGB")
            tensors.append(torch.from_numpy(np.asarray(img).astype("float32") / 255.0))
        batch = torch.stack(tensors)  # all refs are same-size by asset rule
        return (batch, ",".join(used))

NODE_CLASS_MAPPINGS = {"CharacterRefLoader": CharacterRefLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CharacterRefLoader": "Character Refs (baked)"}
