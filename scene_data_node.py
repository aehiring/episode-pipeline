"""SceneField / SceneCount — graph reads per-scene data from episode JSON."""
import json

FIELDS = ["keyframe_prompt","dialogue","location","speaker","shot",
          "motion_all","chunks","duration_s","start_time_s","scene_frames"]

class SceneField:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "episode_json":("STRING",{"forceInput":True}),
            "scene_number":("INT",{"default":1,"min":1,"max":112,"forceInput":True}),
            "field":(FIELDS,)}}
    RETURN_TYPES=("STRING","INT","FLOAT")
    RETURN_NAMES=("text","int","float")
    FUNCTION="get"; CATEGORY="RenReed"
    def get(self, episode_json, scene_number, field):
        ep=json.loads(episode_json)
        sc=next((s for s in ep["scenes"] if s["scene_number"]==scene_number),None)
        if sc is None:
            raise RuntimeError(f"SceneField FATAL: scene {scene_number} missing from episode JSON")
        if field=="motion_all":
            v=" ".join(sc["motion_prompts"])
        elif field=="scene_frames":
            v=sc["chunks"]*77
        else:
            v=sc[field]
        return (str(v), int(v) if isinstance(v,(int,float)) else 0,
                float(v) if isinstance(v,(int,float)) else 0.0)

class SceneCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"episode_json":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("INT",); RETURN_NAMES=("scene_count",)
    FUNCTION="get"; CATEGORY="RenReed"
    def get(self, episode_json):
        ep=json.loads(episode_json)
        n=ep["episode"]["scene_count"]
        if n!=len(ep["scenes"]):
            raise RuntimeError(f"SceneCount FATAL: header {n} vs {len(ep['scenes'])} scenes")
        return (n,)

NODE_CLASS_MAPPINGS={"SceneField":SceneField,"SceneCount":SceneCount}
NODE_DISPLAY_NAME_MAPPINGS={"SceneField":"Scene Field","SceneCount":"Scene Count"}


class PathAfter:
    """Returns a fixed path AFTER an upstream dependency finishes (execution gate)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"path":("STRING",{"default":"/models/output/scenes"}),
                            "after":("*",{"forceInput":True})}}
    RETURN_TYPES=("STRING",); RETURN_NAMES=("path",)
    FUNCTION="go"; CATEGORY="RenReed"
    def go(self, path, after): return (path,)

NODE_CLASS_MAPPINGS["PathAfter"]=PathAfter
NODE_DISPLAY_NAME_MAPPINGS["PathAfter"]="Path (after gate)"
