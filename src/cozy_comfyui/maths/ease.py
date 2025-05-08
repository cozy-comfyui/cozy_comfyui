""" Maths Easing Function Support """

import inspect
from enum import Enum
from typing import Union

import numpy as np
from numba import jit

# ==============================================================================
# === TYPE ===
# ==============================================================================

TYPE_NUMBER = Union[int|float|np.ndarray]

# ==============================================================================
# === GLOBAL ===
# ==============================================================================

MODULE = inspect.getmodule(inspect.currentframe())

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumEase(Enum):
    LINEAR = 0

    BOUNCE_IN = 100
    BOUNCE_OUT = 101
    BOUNCE_IN_OUT = 102

    ELASTIC_IN = 80
    ELASTIC_OUT = 81
    ELASTIC_IN_OUT = 82

    BACK_IN = 90
    BACK_OUT = 91
    BACK_IN_OUT = 92

    SIN_IN = 50
    SIN_OUT = 51
    SIN_IN_OUT = 52

    CUBIC_IN = 20
    CUBIC_OUT = 21
    CUBIC_IN_OUT = 22

    QUAD_IN = 10
    QUAD_OUT = 11
    QUAD_IN_OUT = 12

    QUARTIC_IN = 30
    QUARTIC_OUT = 31
    QUARTIC_IN_OUT = 32

    QUINTIC_IN = 40
    QUINTIC_OUT = 41
    QUINTIC_IN_OUT = 42

    CIRCULAR_IN = 60
    CIRCULAR_OUT = 61
    CIRCULAR_IN_OUT = 62

    EXPONENTIAL_IN = 70
    EXPONENTIAL_OUT = 71
    EXPONENTIAL_IN_OUT = 72

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

@jit(nopython=True, cache=True)
def ease_linear(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t

@jit(nopython=True, cache=True)
def ease_quad_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t

@jit(nopython=True, cache=True)
def ease_quad_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return -(t * (t - 2))

@jit(nopython=True, cache=True)
def ease_quad_in_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return np.where(t < 0.5, 2 * t * t, (-2 * t * t) + (4 * t) - 1)

@jit(nopython=True, cache=True)
def ease_cubic_in(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return t * t * t

@jit(nopython=True, cache=True)
def ease_cubic_out(t: TYPE_NUMBER) -> TYPE_NUMBER:
    return (t - 1) * (t - 1) * (t - 1) + 1

@jit(nopython=True, cache=True)
def ease_cubic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 4 * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(nopython=True, cache=True)
def ease_quartic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t

@jit(nopython=True, cache=True)
def ease_quartic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1

@jit(nopython=True, cache=True)
def ease_quartic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 8 * t * t * t * t,
                    -8 * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1)

@jit(nopython=True, cache=True)
def ease_quintic_in(t: np.ndarray) -> np.ndarray:
    return t * t * t * t * t

@jit(nopython=True, cache=True)
def ease_quintic_out(t: np.ndarray) -> np.ndarray:
    return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1

@jit(nopython=True, cache=True)
def ease_quintic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 16 * t * t * t * t * t,
                    0.5 * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) * (2 * t - 2) + 1)

@jit(nopython=True, cache=True)
def ease_sin_in(t: np.ndarray) -> np.ndarray:
    return np.sin((t - 1) * np.pi * 0.5) + 1

@jit(nopython=True, cache=True)
def ease_sin_out(t: np.ndarray) -> np.ndarray:
    return np.sin(t * np.pi * 0.5)

@jit(nopython=True, cache=True)
def ease_sin_in_out(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1 - np.cos(t * np.pi))

@jit(nopython=True, cache=True)
def ease_circular_in(t: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - (t * t))

@jit(nopython=True, cache=True)
def ease_circular_out(t: np.ndarray) -> np.ndarray:
    return np.sqrt((2 - t) * t)

@jit(nopython=True, cache=True)
def ease_circular_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (1 - np.sqrt(1 - 4 * (t * t))),
                    0.5 * (np.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1))

@jit(nopython=True, cache=True)
def ease_exponential_in(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, 0, np.power(2, 10 * (t - 1)))

@jit(nopython=True, cache=True)
def ease_exponential_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 1, 1, 1 - np.power(2, -10 * t))

@jit(nopython=True, cache=True)
def ease_exponential_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t == 0, t, np.where(t < 0.5, 0.5 * np.power(2, (20 * t) - 10),
                                        -0.5 * np.power(2, (-20 * t) + 10) + 1))

@jit(nopython=True, cache=True)
def ease_elastic_in(t: np.ndarray) -> np.ndarray:
    return np.sin(13 * np.pi * 0.5 * t) * np.power(2, 10 * (t - 1))

@jit(nopython=True, cache=True)
def ease_elastic_out(t: np.ndarray) -> np.ndarray:
    return np.sin(-13 * np.pi * 0.5 * (t + 1)) * np.power(2, -10 * t) + 1

@jit(nopython=True, cache=True)
def ease_elastic_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * np.sin(13 * np.pi * 0.5 * (2 * t)) * np.power(2, 10 * ((2 * t) - 1)),
                    0.5 * (np.sin(-13 * np.pi * 0.5 * ((2 * t - 1) + 1)) * np.power(2, -10 * (2 * t - 1)) + 2))

@jit(nopython=True, cache=True)
def ease_back_in(t: np.ndarray) -> np.ndarray:
    return t * t * t - t * np.sin(t * np.pi)

@jit(nopython=True, cache=True)
def ease_back_out(t: np.ndarray) -> np.ndarray:
    p = 1 - t
    return 1 - (p * p * p - p * np.sin(p * np.pi))

@jit(nopython=True, cache=True)
def ease_back_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * (2 * t) * (2 * t) * (2 * t) - (2 * t) * np.sin((2 * t) * np.pi),
                    0.5 * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) * (1 - (2 * t - 1)) - (1 - (2 * t - 1)) * np.sin((1 - (2 * t - 1)) * np.pi) + 0.5)

@jit(nopython=True, cache=True)
def ease_bounce_in(t: np.ndarray) -> np.ndarray:
    return 1 - ease_bounce_out(1 - t)

@jit(nopython=True, cache=True)
def ease_bounce_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 4 / 11, 121 * t * t / 16,
        np.where(t < 8 / 11, (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0,
        np.where(t < 9 / 10, (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0,
                (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0)))

@jit(nopython=True, cache=True)
def ease_bounce_in_out(t: np.ndarray) -> np.ndarray:
    return np.where(t < 0.5, 0.5 * ease_bounce_in(t * 2), 0.5 * ease_bounce_out(t * 2 - 1) + 0.5)

def ease_simple(op: EnumEase, start: float=0, end: float=1, duration: float=1,
            alpha: np.ndarray = np.linspace(0, 1, 100), clip: tuple[int, int]=(0, 1)) -> np.ndarray:
    """
    Compute eased values.

    Parameters:
        op (EaseOP): Easing operator.
        start (float): Starting value.
        end (float): Ending value.
        duration (float): Duration of the easing.
        alpha (np.ndarray): Alpha values.
        clip (tuple[int, int]): Clip range.

    Returns:
        np.ndarray: Eased value(s)
    """
    if (func := getattr(MODULE, f"ease_{op.name.lower()}", None)) is None:
        raise Exception(f"Bad Operator: {op.name}")

    t = clip[0] * (1 - alpha) + clip[1] * alpha
    duration = max(min(duration, 1), 0)
    t /= duration
    a = func(t)
    return end * a + start * (1 - a)

def ease_op(op: EnumEase,
            values: np.ndarray,
            steps: int,
            clip: tuple[float, float] = (0, 1),
            duration: float = 1.0) -> np.ndarray:
    """
    Applies easing across a set of values using a chosen easing function.

    Parameters:
        op (EnumEase): Easing function to use.
        values (np.ndarray): The values to ease through.
        steps (int): Number of output steps (interpolated points).
        clip (tuple[float, float]): Clip range for easing curve.
        duration (float): Normalized duration (default 1.0).

    Returns:
        np.ndarray: Eased values interpolated from the original values.
    """

    if (func := getattr(MODULE, f"ease_{op.name.lower()}", None)) is None:
        raise Exception(f"Bad Operator: {op.name}")

    # Normalized time samples (0..1), remapped by clip and duration
    alpha = np.linspace(0, 1, steps)
    t = clip[0] * (1 - alpha) + clip[1] * alpha
    t /= max(min(duration, 1), 1e-8)

    eased = func(t)  # eased alpha

    # Map eased alpha to float indices into `values`
    indices = eased * (len(values) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    frac = indices - lower

    # Clamp to valid range
    lower = np.clip(lower, 0, len(values) - 1)
    upper = np.clip(upper, 0, len(values) - 1)

    return values[lower] * (1 - frac) + values[upper] * frac
