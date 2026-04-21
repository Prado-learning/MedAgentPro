from .GPT_Decider import GPT_Decider
from .Pro_Decider import Pro_Decider

__all__ = ["GPT_Decider", "Pro_Decider"]


_OPTIONAL_DECIDERS = {
    "Janus_Decider": "Janus_Decider",
    "BioMedClip_Decider": "BioMedClip_Decider",
    "Qwen_Decider": "Qwen_Decider",
    "InternVL_Decider": "InternVL_Decider",
    "Gemma_Decider": "Gemma_Decider",
}

__all__.extend(_OPTIONAL_DECIDERS.keys())


def __getattr__(name):
    module_name = _OPTIONAL_DECIDERS.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
