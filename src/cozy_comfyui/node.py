"""."""

import sys
import json
import inspect
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from cozy_comfyui import \
    COZY_INTERNAL, \
    logger

# ==============================================================================
# === TYPE ===
# ==============================================================================

class CozyTypeAny(str):
    """AnyType input wildcard trick taken from pythongossss's:

    https://github.com/pythongosssss/ComfyUI-Custom-Scripts
    """
    def __ne__(self, __value: object) -> bool:
        return False

COZY_TYPE_ANY = CozyTypeAny("*")

COZY_TYPE_NUMBER = "BOOLEAN,FLOAT,INT"
COZY_TYPE_VECTOR = "VEC2,VEC3,VEC4,VEC2INT,VEC3INT,VEC4INT,COORD2D,COORD3D"
COZY_TYPE_NUMERICAL = f"{COZY_TYPE_NUMBER},{COZY_TYPE_VECTOR}"
COZY_TYPE_IMAGE = "IMAGE,MASK"
COZY_TYPE_FULL = f"{COZY_TYPE_NUMBER},{COZY_TYPE_IMAGE}"

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def load_module(root:str, name: str) -> None|ModuleType:
    root_module = root.split("/")[-1]
    try:
        route = name.split(f"{root}/")[1]
        route = route.split('.')[0].replace('/', '.')
        module = f"{root_module}.{route}"
    except Exception as e:
        logger.warning(f"module failed {name}")
        logger.warning(str(e))
        return

    try:
        return importlib.import_module(module)
    except Exception as e:
        logger.warning(f"module failed {module}")
        logger.warning(str(e))

def loader(root: str, pack: str, directory: str='',
           category: str="COZY COMFYUI ☕",
           rename: bool=True) -> Tuple[Dict[str, object], Dict[str, str]]:

    """
    rename will force the new name from the existing definition on the old node.
    Currently used to support older Jovimetrix nodes
    """

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    NODE_LIST_MAP = {}

    # package core root
    root = Path(root)
    root_str = str(root).replace("\\", "/")
    core = str(root.parent)
    sys.path.append(core)
    node_root = f"{directory}/**/*.py"

    for fname in root.glob(node_root):
        if fname.stem.startswith('_'):
            continue

        fname = str(fname).replace("\\", "/")
        if (module := load_module(root_str, fname)) is None:
            continue

        # check if there is a dynamic register function....
        try:
            if hasattr(module, 'import_dynamic'):
                for class_name, class_def in module.import_dynamic():
                    setattr(module, class_name, class_def)
        except Exception as e:
            logger.exception(str(e))

        classes = inspect.getmembers(module, inspect.isclass)
        for class_name, class_object in classes:
            if not class_name.endswith('BaseNode') and hasattr(class_object, 'NAME'): # and hasattr(class_object, 'CATEGORY'):
                name = f"{class_object.NAME} ({pack})" if rename else class_object.NAME
                NODE_DISPLAY_NAME_MAPPINGS[name] = name
                NODE_CLASS_MAPPINGS[name] = class_object
                desc = class_object.DESCRIPTION if hasattr(class_object, 'DESCRIPTION') else name
                NODE_LIST_MAP[name] = desc.split('.')[0].strip('\n')
                if not hasattr(class_object, 'CATEGORY'):
                    class_object.CATEGORY = category

    NODE_CLASS_MAPPINGS = {x[0] : x[1] for x in sorted(NODE_CLASS_MAPPINGS.items(),
                                                            key=lambda item: getattr(item[1], 'SORT', 0))}

    keys = NODE_CLASS_MAPPINGS.keys()
    #for name in keys:
    #    logger.debug(f"✅ {name}")
    logger.info(f"{pack} {len(keys)} nodes loaded")

    # only do the list on local runs...
    if COZY_INTERNAL:
        with open(f"{root_str}/node_list.json", "w", encoding="utf-8") as f:
            json.dump(NODE_LIST_MAP, f, sort_keys=True, indent=4 )

    return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ==============================================================================
# === CLASS ===
# ==============================================================================

class Singleton(type):
    """THERE CAN BE ONLY ONE"""
    _instances = {}

    def __call__(cls, *arg, **kw) -> Any:
        # If the instance does not exist, create and store it
        if cls not in cls._instances:
            instance = super().__call__(*arg, **kw)
            cls._instances[cls] = instance
        return cls._instances[cls]

class CozyBaseNode:
    INPUT_IS_LIST = True
    NOT_IDEMPOTENT = True
    RETURN_TYPES = ()
    FUNCTION = "run"

    @classmethod
    def IS_CHANGED(cls, **kw) -> float:
        return float('nan')

    @classmethod
    def VALIDATE_INPUTS(cls, input_types) -> bool:
        return True

    @classmethod
    def INPUT_TYPES(cls, prompt:bool=False, extra_png:bool=False, dynprompt:bool=False) -> Dict[str, str]:
        data = {
            "required": {},
            "hidden": {
                "ident": "UNIQUE_ID"
            }
        }
        if prompt:
            data["hidden"]["prompt"] = "PROMPT"
        if extra_png:
            data["hidden"]["extra_pnginfo"] = "EXTRA_PNGINFO"

        if dynprompt:
            data["hidden"]["dynprompt"] = "DYNPROMPT"
        return data

class CozyImageNode(CozyBaseNode):
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("RGBA", "RGB", "MASK")
    OUTPUT_TOOLTIPS = (
        "Full channel [RGBA] image. If there is an alpha, the image will be masked out with it when using this output.",
        "Three channel [RGB] image. There will be no alpha.",
        "Single channel mask output."
    )
