from .compiler_node import NODE_CLASS_MAPPINGS as A, NODE_DISPLAY_NAME_MAPPINGS as AD
from .character_loader_node import NODE_CLASS_MAPPINGS as B, NODE_DISPLAY_NAME_MAPPINGS as BD
from .renreed_tts_node import NODE_CLASS_MAPPINGS as C, NODE_DISPLAY_NAME_MAPPINGS as CD
from .postmaster_node import NODE_CLASS_MAPPINGS as D, NODE_DISPLAY_NAME_MAPPINGS as DD
from .scene_data_node import NODE_CLASS_MAPPINGS as E, NODE_DISPLAY_NAME_MAPPINGS as ED
NODE_CLASS_MAPPINGS = {**A, **B, **C, **D, **E}
NODE_DISPLAY_NAME_MAPPINGS = {**AD, **BD, **CD, **DD, **ED}
