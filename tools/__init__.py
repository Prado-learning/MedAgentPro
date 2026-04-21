__all__ = ["VQA_Module", "MAIRA", "MedSAM", "VisionUniteModel"]

_LAZY_IMPORTS = {
    "VQA_Module": (".VQA", "VQA_Module"),
    "MAIRA": (".maira", "MAIRA"),
    "MedSAM": (".MedSAM.model", "MedSAM"),
    "VisionUniteModel": (".VisionUnite.model", "VisionUniteModel"),
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = target
    module = __import__(f"{__name__}{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
