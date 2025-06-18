# cozy_comfyui
Base class supports for writing comfyui nodes.

**2025/06/18** @0.0.36:
* support for hex code parsing into VEC4

**2025/06/07** @0.0.35:
* cleaned up image_convert for grayscale/mask

**2025/06/06** @0.0.34:
* default width, height to 1
* better image convert with full mask support

**2025/05/31** @0.0.33:
* patch around data returned in widgets waiting for ComfyUI PR 8127

**2025/05/29** @0.0.32:
* single tensors read as single tensors

**2025/05/29** @0.0.31:
* fixed bug in tensors being read as a batch when only a single tensor

**2025/05/27** @0.0.29:
* optimize numerical checks for clip min/max

**2025/05/25** @0.0.28:
* loosened restriction for python 3.11+ to allow for 3.10+
* * I make zero guarantee that will actually let 3.10 work and I will not support 3.10

**2025/05/18** @0.0.27:
* fall thru for tooltips to use their Lexicon entry if none provided

**2025/05/16** @0.0.26:
* fixed double loads

**2025/05/16** @0.0.25:
* equalize, brightness, pixelate, pixelscale
* migrated bulk of Jovimetrix image library
* list conversion updates
* flatten fix
* ease NONE type

**2025/05/08** @0.0.24:
* adjusted for masks being inverted
* new LEXICON `INPUT`

**2025/05/05** @0.0.23:
* lexicon updates

**2025/05/05** @0.0.22:
* migrated maths functions

**2025/05/05** @0.0.21:
* remove force cache refreshes
* migrate Lexicon into cozy

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