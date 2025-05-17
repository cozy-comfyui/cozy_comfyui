"""Cozy ComfyUI Node Support Library"""

__version__ = "0.0.26"

import os
import sys
import json
from enum import Enum
from typing import Any, Dict, List, Generator, Optional, Tuple, TypeAlias

from loguru import logger as loguru_logger
import torch

# ==============================================================================
# === TYPE ===
# ==============================================================================

TensorType: TypeAlias = torch.Tensor
RGBAMaskType: TypeAlias = Tuple[TensorType, ...]
InputType: TypeAlias = Dict[str, Any]

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

logger = loguru_logger.bind(cozy=True)
has_cozy_handler = any(
    "cozy" in h.filter_.__code__.co_code
    for h in loguru_logger._core.handlers.values()
    if hasattr(h, 'filter_') and h.filter_ is not None
)

if not has_cozy_handler:
    loguru_logger.add(
        sys.stdout,
        level=os.getenv("COZY_LOG_LEVEL", "INFO"),
        filter=lambda record: "cozy" in record["extra"],
        enqueue=True
)

COZY_INTERNAL = os.getenv("COZY_INTERNAL", 'false').strip().lower() in ('true', '1', 't')

IMAGE_SIZE_MIN: int = 32
IMAGE_SIZE_MAX: int = 8192
IMAGE_SIZE_DEFAULT: int = 512

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumConvertType(Enum):
    BOOLEAN = 0
    INT = 10
    FLOAT = 12
    VEC2 = 20
    VEC3 = 30
    VEC4 = 40
    VEC2INT = 25
    VEC3INT = 35
    VEC4INT = 45
    STRING = 50
    LIST = 60
    DICT = 65
    IMAGE = 70
    MASK = 75
    LATENT = 80
    ANY = 70

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def deep_merge(d1: InputType, d2: InputType) -> InputType:
    """
    Deep merge multiple dictionaries recursively.

    Args:
        *dicts: Variable number of dictionaries to be merged.

    Returns:
        dict: Merged dictionary.
    """
    for key in d2:
        if key in d1:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                deep_merge(d1[key], d2[key])
            else:
                d1[key] = d2[key]
        else:
            d1[key] = d2[key]
    return d1

def load_file(fname: str) -> str | None:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(e)

def parse_dynamic(data: InputType, prefix: str, typ: EnumConvertType, default: Any) -> List[Any]:
    """Convert iterated input field(s) based on a s into a single compound list of entries.

    The default will just look for all keys as integer:

        `#_<field name>` or `#_<prefix>_<field name>`

    This will return N entries in a list based on the prefix pattern or not.

    """
    vals: List[Any] = []
    fail = 0
    keys = data.keys()
    for i in range(100):
        if fail > 2:
            break

        found = None
        for k in keys:
            if k.startswith(f"{i}_") or k.startswith(f"{i}_{prefix}_"):
                val = parse_param(data, k, typ, default)
                vals.extend(val)
                found = True
                break

        if found is None:
            fail += 1

    return vals

def parse_value(val: Any, typ: EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero: int=0) -> List[Any]|None:
    """Convert target value into the new specified type."""

    if typ == EnumConvertType.ANY:
        return val

    if isinstance(default, torch.Tensor) and typ not in [EnumConvertType.IMAGE,
                                                         EnumConvertType.MASK,
                                                         EnumConvertType.LATENT]:
        h, w = default.shape[:2]
        cc = default.shape[2] if len(default.shape) > 2 else 1
        default = (w, h, cc)

    if val is None:
        if default is None:
            return None
        val = default

    if isinstance(val, dict):
        # old jovimetrix index?
        if '0' in val or 0 in val:
            val = [val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4))]
        # coord2d?
        elif 'x' in val:
            val = [val.get(c, 0) for c in 'xyzw']
        # wacky color struct?
        elif 'r' in val:
            val = [val.get(c, 0) for c in 'rgba']
    elif isinstance(val, torch.Tensor) and typ not in [EnumConvertType.IMAGE,
                                                       EnumConvertType.MASK,
                                                       EnumConvertType.LATENT]:
        h, w = val.shape[:2]
        cc = val.shape[2] if len(val.shape) > 2 else 1
        val = (w, h, cc)

    new_val: Any = val
    if typ in [EnumConvertType.FLOAT, EnumConvertType.INT,
            EnumConvertType.VEC2, EnumConvertType.VEC2INT,
            EnumConvertType.VEC3, EnumConvertType.VEC3INT,
            EnumConvertType.VEC4, EnumConvertType.VEC4INT]:
            #EnumConvertType.COORD2D]:

        if not isinstance(val, (list, tuple, torch.Tensor)):
            val = [val]

        size = max(1, int(typ.value / 10))
        new_val = []
        for idx in range(size):
            d: Any = default
            try:
                d = default[idx] if idx < len(default) else 0
            except:
                try:
                    d = default.get(str(idx), 0)
                except:
                    d = default

            v: Any = d if val is None else val[idx] if idx < len(val) else d
            if isinstance(v, (str, )):
                v = v.strip('\n').strip()
                if v == '':
                    v = 0

            try:
                if typ in [EnumConvertType.FLOAT, EnumConvertType.VEC2, EnumConvertType.VEC3, EnumConvertType.VEC4]:
                    v = round(float(v or 0), 16)
                else:
                    v = int(v)
                if clip_min is not None:
                    v = max(v, clip_min)
                if clip_max is not None:
                    v = min(v, clip_max)
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error converting value: {val} -- {v}")
                v = 0

            if v == 0:
                v = zero
            new_val.append(v)
        new_val = new_val[0] if size == 1 else tuple(new_val)
    elif typ == EnumConvertType.DICT:
        try:
            if isinstance(new_val, (str,)):
                try:
                    new_val = json.loads(new_val)
                except json.decoder.JSONDecodeError:
                    new_val = {}
            else:
                if not isinstance(new_val, (list, tuple,)):
                    new_val = [new_val]
                new_val = {i: v for i, v in enumerate(new_val)}
        except Exception as e:
            logger.exception(e)
    elif typ == EnumConvertType.LIST:
        if isinstance(new_val, (str, list, int, float,)):
            new_val = [new_val]
        else:
            new_val = list(new_val)
    elif typ == EnumConvertType.STRING:
        if isinstance(new_val, (str, list, int, float,)):
            new_val = [new_val]
        new_val = ", ".join(map(str, new_val)) if not isinstance(new_val, str) else new_val
    elif typ == EnumConvertType.BOOLEAN:
        if isinstance(new_val, (torch.Tensor,)):
            new_val = True
        elif isinstance(new_val, (dict,)):
            new_val = len(new_val.keys()) > 0
        elif isinstance(new_val, (list, tuple,)) and len(new_val) > 0 and (nv := new_val[0]) is not None:
            if isinstance(nv, (str,)):
                new_val = False if nv in ['', '0', '0.0', 'False', 'false'] else True
            elif isinstance(nv, (int, float,)):
                new_val = nv > 0
        elif isinstance(new_val, (str,)):
            new_val = False if new_val in ['', '0', '0.0', 'False', 'false'] else True
        elif isinstance(new_val, (int, float,)):
            new_val = new_val > 0
    elif typ == EnumConvertType.LATENT:
        # covert image into latent
        if isinstance(new_val, (torch.Tensor,)):
            new_val = {'samples': new_val.unsqueeze(0)}
        else:
            # convert whatever into a latent sample...
            new_val = torch.empty((4, 64, 64), dtype=torch.uint8).unsqueeze(0)
            new_val = {'samples': new_val}
    elif typ == EnumConvertType.IMAGE:
        # covert image into image? just skip if already an image
        if not isinstance(new_val, (torch.Tensor,)):
            color = parse_value(new_val, EnumConvertType.VEC4INT, (0,0,0,255), 0, 255)
            color = torch.tensor(color, dtype=torch.int32).tolist()
            new_val = torch.empty((IMAGE_SIZE_MIN, IMAGE_SIZE_MIN, 4), dtype=torch.uint8)
            new_val[0,:,:] = color[0]
            new_val[1,:,:] = color[1]
            new_val[2,:,:] = color[2]
            new_val[3,:,:] = color[3]

    elif typ == EnumConvertType.MASK:
        if not isinstance(new_val, (torch.Tensor,)):
            color = parse_value(new_val, EnumConvertType.INT, 0, 0, 255)
            color = torch.tensor(color, dtype=torch.int32).tolist()
            new_val = torch.empty((IMAGE_SIZE_MIN, IMAGE_SIZE_MIN, 1), dtype=torch.uint8)
            new_val[0,:,:] = color
        else:
            # if the incomming image is RGB/RGBA, we leave it alone
            # if it is greyscale (a mask) we should invert for "alpha"
            pass
            '''
            if new_val.ndim == 2 or new_val.shape[2] == 1:
                new_val = 1.0 - new_val
            '''

    elif issubclass(typ, Enum):
        new_val = typ[val]

    #if typ == EnumConvertType.COORD2D:
    #    new_val = {'x': new_val[0], 'y': new_val[1]}
    return new_val

def parse_param(data:InputType, key:str, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> List[Any]:
    """Convenience because of the dictionary parameters."""
    values = data.get(key, default)
    if typ == EnumConvertType.ANY:
        if values is None:
            return [default]
    return parse_param_list(values, typ, default, clip_min, clip_max, zero)

def parse_param_list(values:Any, typ:EnumConvertType, default: Any,
                clip_min: Optional[float]=None, clip_max: Optional[float]=None,
                zero:int=0) -> List[Any]:
    """Convert list of values into a list of specified type."""

    if not isinstance(values, (list,)):
        values = [values]

    value_array: List[Any] = []
    for val in values:
        if isinstance(val, (str,)):
            try: val = json.loads(val.replace("'", '"'))
            except json.JSONDecodeError: pass
            value_array.append(val)
        # see if we are a Jovimetrix hacked vector blob... {0:x, 1:y, 2:z, 3:w}
        elif isinstance(val, dict):
            # mixlab layer?
            if (image := val.get('image', None)) is not None:
                ret = image
                if (mask := val.get('mask', None)) is not None:
                    while len(mask.shape) < len(image.shape):
                        mask = mask.unsqueeze(-1)
                    ret = torch.cat((image, mask), dim=-1)
                if ret.ndim == 2:
                    val = [v.unsqueeze(-1) for v in ret]
                val = [t for t in ret]
                value_array.extend(val)
            # vector patch....
            elif 'xyzw' in val:
                val = tuple(x for x in val["xyzw"])
            # latents....
            elif 'samples' in val:
                val = tuple(x for x in val["samples"])
            elif ('0' in val) or (0 in val):
                val = tuple(val.get(i, val.get(str(i), 0)) for i in range(min(len(val), 4)))
            elif 'x' in val and 'y' in val:
                val = tuple(val.get(c, 0) for c in 'xyzw')
            elif 'r' in val and 'g' in val:
                val = tuple(val.get(c, 0) for c in 'rgba')
            elif len(val) == 0:
                val = tuple()
            value_array.append(val)
        elif isinstance(val, (torch.Tensor,)):
            # a batch of Grayscale
            if val.ndim == 2:
                val = [t.unsqueeze(-1) for t in val]
            val = [t for t in val]
            value_array.extend(val)
        elif isinstance(val, (list, tuple, set)):
            if isinstance(val, (tuple, set,)):
                val = list(val)
            value_array.append(val)
        elif issubclass(type(val), (Enum,)):
            val = str(val.name)
            value_array.append(val)
        else:
            value_array.append(val)

    return [parse_value(v, typ, default, clip_min, clip_max, zero) for v in value_array]

def zip_longest_fill(*iterables: Any) -> Generator[Tuple[Any, ...], None, None]:
    """
    Zip longest with fill value.

    This function behaves like itertools.zip_longest, but it fills the values
    of exhausted iterators with their own last values instead of None.
    """
    iterators = [iter(iterable) for iterable in iterables]

    while True:
        values = [next(iterator, None) for iterator in iterators]

        # Check if all iterators are exhausted
        if all(value is None for value in values):
            break

        # Fill in the last values of exhausted iterators with their own last values
        for i, _ in enumerate(iterators):
            if values[i] is None:
                iterator_copy = iter(iterables[i])
                while True:
                    current_value = next(iterator_copy, None)
                    if current_value is None:
                        break
                    values[i] = current_value

        yield tuple(values)
