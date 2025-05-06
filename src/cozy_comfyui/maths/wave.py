""" Maths Wave Function Support """

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

class EnumWave(Enum):
    SIN = 0
    COS = 3
    TAN = 6
    SAWTOOTH = 30
    TRIANGLE = 32
    SQUARE = 34
    PULSE = 36
    RAMP = 40
    STEP = 41
    EXPONENTIAL = 50
    LOGARITHMIC = 55
    NOISE = 60
    HAVERSINE = 70
    RECTANGULAR_PULSE = 80
    GAUSSIAN = 90
    CHIRP = 100

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

@jit(nopython=True, cache=True)
def wave_sin(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.sin(frequency * np.pi * 2 * timestep + phase) + offset

@jit(nopython=True, cache=True)
def wave_cos(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.cos(frequency * np.pi * 2 * timestep + phase) + offset

@jit(nopython=True, cache=True)
def wave_tan(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.tan(frequency * np.pi * 2 * timestep + phase) + offset

@jit(nopython=True, cache=True)
def wave_sawtooth(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * (2 * (frequency * timestep + phase) % 1 - 0.5) + offset

@jit(nopython=True, cache=True)
def wave_triangle(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * (4 * np.abs((frequency * timestep + phase) % 1 - 0.5) - 1) + offset

@jit(nopython=True, cache=True)
def wave_ramp(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * (frequency * timestep + phase % 1) + offset

@jit(nopython=True, cache=True)
def wave_step(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.heaviside(frequency * timestep + phase, 1) + offset

@jit(nopython=True, cache=True)
def wave_haversine(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * (1 - np.cos(frequency * np.pi * 2 * (timestep + phase))) + offset

@jit(nopython=True, cache=True)
def wave_noise(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.random.uniform(-1, 1) + offset

@jit(nopython=True, cache=True)
def wave_square(phase: float, frequency: float, amplitude: float, offset: float,
                timestep: float) -> float:
    return amplitude * np.sign(np.sin(np.pi * 2 * timestep + phase) - frequency) + offset

@jit(nopython=True, cache=True)
def wave_exponential(phase: float, frequency: float, amplitude: float,
                    offset: float, timestep: float) -> float:
    return amplitude * np.exp(-frequency * (timestep + phase)) + offset

@jit(nopython=True, cache=True)
def wave_rectangular_pulse(phase: float, frequency: float, amplitude: float,
                        offset: float, timestep: float) -> float:
    return amplitude * np.heaviside(timestep + phase, 1) * np.heaviside(-(timestep + phase) + frequency, 1) + offset

@jit(nopython=True, cache=True)
def wave_logarithmic(phase: float, frequency: float, amplitude: float, offset: float,
                    timestep: float) -> float:
    return amplitude * np.log10(timestep + phase) / np.max(1, np.log10(frequency)) + offset

@jit(nopython=True, cache=True)
def wave_chirp(phase: float, frequency: float, amplitude: float, offset: float,
            timestep: float) -> float:
    return amplitude * np.sin(np.pi * 2 * frequency * (timestep + phase)**2) + offset

@jit(nopython=True, cache=True)
def wave_gaussian(phase: float, mean: float, amplitude: float, offset: float,
                timestep: float, std_dev: float = 1) -> float:
    return amplitude * np.exp(-0.5 * ((timestep + phase - mean) / std_dev)**2) + offset

def wave_op(op: EnumWave, phase: float, frequency: float, amplitude: float,
            offset: float, timestep: float, std_dev: float=1) -> np.ndarray:

    op = op.name.lower()
    if (func := getattr(MODULE, f"wave_{op}", None)) is None:
        raise Exception(f"Bad Operator: {op.name}")
    """
    phase = float(phase)
    frequency = float(frequency)
    amplitude = float(amplitude)
    offset = float(offset)
    timestep = float(timestep)
    std_dev = float(std_dev)
    """
    if op.endswith('gaussian'):
        return func(phase, frequency, amplitude, offset, timestep, std_dev)
    return func(phase, frequency, amplitude, offset, timestep)
