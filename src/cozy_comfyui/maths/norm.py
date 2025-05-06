""" Maths Normalization Function Support """

import inspect
from enum import Enum

import numpy as np
from numba import jit

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

MODULE = inspect.getmodule(inspect.currentframe())

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumNormalize(Enum):
    MINMAX1 = 0  #  0..1
    MINMAX2 = 1  # -1..1
    ZSCORE  = 2  # Standard Score
    L2      = 3  # Unit Vector

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

@jit(nopython=True, cache=True)
def norm_minmax1(value: np.ndarray) -> np.ndarray:
    return (value - value.min()) / (value.max() - value.min())

@jit(nopython=True, cache=True)
def norm_minmax2(value: np.ndarray) -> np.ndarray:
    return 2 * ((value - value.min()) / (value.max() - value.min())) - 1

@jit(nopython=True, cache=True)
def norm_zscore(value: np.ndarray) -> np.ndarray:
    return (value - value.mean()) / value.std()

@jit(nopython=True, cache=True)
def norm_l2(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)

def norm_op(op: EnumNormalize, value: np.ndarray) -> np.ndarray:
    op = op.name.lower()
    if (func := getattr(MODULE, f"norm_{op}", None)) is None:
        raise Exception(f"Bad Operator: {op.name}")
    return func(value)
