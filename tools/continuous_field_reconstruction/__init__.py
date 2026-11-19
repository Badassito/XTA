"""Continuous-field reconstruction of an edited preview Global at native resolution.

Credit: originally named continuous_b3 (B3 = cubic cardinal B-spline). Field
method: Team A, 2026-11-06 upscaling-mechanism experiment. Launcher contract
(FIELD_MEMORY_MIB/FIELD_CACHE, now prepare_pta_input.py's --field-memory-mib and
--field-cache): Team B, 2026-11-06 upscaler pipeline bundle. o7 Team B.
"""
from .policy import POLICY, ALGORITHM, ALGORITHM_VERSION
from .field_reference import build_field_reference, validate_field_reference
from .radius_upscaler import FieldReference, resolve_reference, sample_reference, iter_native_blocks, block_plan
from .nrrd_writer import write_upscaled_nrrd

__all__=['POLICY','ALGORITHM','ALGORITHM_VERSION','build_field_reference','validate_field_reference',
         'FieldReference','resolve_reference','sample_reference','iter_native_blocks','block_plan',
         'write_upscaled_nrrd']
