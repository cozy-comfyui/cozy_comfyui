# cozy_comfyui
Base class supports for writing comfyui nodes.

**2025/05/02** @0.0.20:
* EnumConvertType re-rodered for use in VALUE, LERP and OP BINARY nodes (Jovimetrix)

**2025/05/01** @0.0.18:
* EnumConvertType values reset to be parsable for VECTOR types

**2025/05/01** @0.0.17:
* vectors unified to just be float types

**2025/04/30** @0.0.16:
* better boolean conversion for widget defaults
* deprecated COORD2D support

**2025/04/19** @0.0.15:
* realigned categories

**2025/04/19** @0.0.14:
* cleanedup all JS message functions

**2025/04/19** @0.0.13:
* return explict zero from parse_reset instead of None

**2025/04/19** @0.0.12:
* absorb image io functions

**2025/04/19** @0.0.11:
* reduced convoluted logic for parsing tensors
* contemplating removing mixlab support

**2025/04/14** @0.0.10:
* updated matte function

**2025/04/14** @0.0.9:
* migrated old routes from Jovimetrix

**2025/04/14** @0.0.8:
* force fixed import problem in comfyui

**2025/04/14** @0.0.7:
* merged more conversion functions
* added more TYPES

**2025/04/12** @0.0.6:
* allow category change per module using loader
* cleanup parse functions for full list support