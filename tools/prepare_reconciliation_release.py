"""Prepare an append-only release review from an authenticated Git predecessor.

The default invocation writes review evidence outside the repository. Use --write
only after the reviewed runtime and validation tools are frozen.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import verify_package_inventory as inventory


RELEASES = {
    '22.3.1': dict(token='22_3_1', previous_token='22_3',
                  feature='efficient-union-and-native-confidence',
                  validation_tools=(
                      'tools/compare_reconciliation.py',
                      'tools/qualify_tta_reconciliation.py',
                      'tools/export_reconciliation_evidence.py')),
    '22.3.2': dict(token='22_3_2', previous_token='22_3_1',
                  feature='bounded-confidence-publication-throughput',
                  validation_tools=(
                      'tools/compare_reconciliation.py',
                      'tools/qualify_tta_reconciliation.py',
                      'tools/export_reconciliation_evidence.py',
                      'tools/qualify_confidence_consolidation.py',
                      'tools/qualify_d1_confidence_bounds.py',
                      'tools/analyze_pipeline_trace.py')),
    '23.0.1': dict(token='23', previous_token='22_3_2',
                   feature='yolo-semantic-segmentation',
                   validation_tools=(
                       'tools/compare_reconciliation.py',
                       'tools/qualify_tta_reconciliation.py',
                       'tools/export_reconciliation_evidence.py',
                       'tools/qualify_confidence_consolidation.py',
                       'tools/qualify_d1_confidence_bounds.py',
                       'tools/analyze_pipeline_trace.py',
                       'tools/export_semantic_logits.py',
                       'tools/qualify_semantic_trt.py',
                       'tools/qualify_pta_classification.py',
                       'tools/qualify_pta_gpu_masks.py',
                       'tools/qualify_pta_gpu_render.py')),
    '23.0.2': dict(token='23_0_2', previous_token='23',
                   feature='repository-review-corrections',
                   validation_tools=(
                       'tools/compare_reconciliation.py',
                       'tools/qualify_tta_reconciliation.py',
                       'tools/export_reconciliation_evidence.py',
                       'tools/qualify_confidence_consolidation.py',
                       'tools/qualify_d1_confidence_bounds.py',
                       'tools/analyze_pipeline_trace.py',
                       'tools/export_semantic_logits.py',
                       'tools/qualify_semantic_trt.py',
                       'tools/qualify_pta_classification.py',
                       'tools/qualify_pta_gpu_masks.py',
                       'tools/qualify_pta_gpu_render.py')),
    '23.0.3': dict(token='23_0_3', previous_token='23_0_2',
                   feature='tta-throughput-restoration',
                   predecessor_inventory_path='release/_package_inventory.json',
                   validation_tools=(
                       'tools/compare_reconciliation.py',
                       'tools/qualify_tta_reconciliation.py',
                       'tools/export_reconciliation_evidence.py',
                       'tools/qualify_confidence_consolidation.py',
                       'tools/qualify_d1_confidence_bounds.py',
                       'tools/analyze_pipeline_trace.py',
                       'tools/export_semantic_logits.py',
                       'tools/qualify_semantic_trt.py',
                       'tools/qualify_pta_classification.py',
                       'tools/qualify_pta_gpu_masks.py',
                       'tools/qualify_pta_gpu_render.py',
                       'tools/qualify_radial_bitset_compaction.py',
                       'tools/qualify_d1_confidence_masked_transfer.py',
                       'tools/qualify_release.py')),
    '23.0.4': dict(token='23_0_4', previous_token='23_0_3',
                   feature='legacy-configuration-and-compiled-cpu-policy',
                   predecessor_inventory_path='release/_package_inventory.json',
                   validation_tools=(
                       'tools/compare_reconciliation.py',
                       'tools/qualify_tta_reconciliation.py',
                       'tools/export_reconciliation_evidence.py',
                       'tools/qualify_confidence_consolidation.py',
                       'tools/qualify_d1_confidence_bounds.py',
                       'tools/analyze_pipeline_trace.py',
                       'tools/export_semantic_logits.py',
                       'tools/qualify_semantic_trt.py',
                       'tools/qualify_pta_classification.py',
                       'tools/qualify_pta_gpu_masks.py',
                       'tools/qualify_pta_gpu_render.py',
                       'tools/qualify_radial_bitset_compaction.py',
                       'tools/qualify_d1_confidence_masked_transfer.py',
                       'tools/qualify_release.py')),
}
RELEASES['23.0.5'] = dict(
    token='23_0_5', previous_token='23_0_4',
    feature='tta-pta-throughput',
    predecessor_inventory_path='release/_package_inventory.json',
    validation_tools=RELEASES['23.0.4']['validation_tools'] + (
        'tools/qualify_native_trt_lease.py',
    ),
)
RELEASES['24.0.0'] = dict(
    token='24_0_0', previous_token='23_0_5',
    feature='single-backend-sam-interpolation',
    predecessor_inventory_path='release/_package_inventory.json',
    validation_tools=RELEASES['23.0.5']['validation_tools'] + (
        'tools/diagnose_sam_interpolation.py',
        'tools/lta_dynamic_crop_diagnostic.py',
        'tools/analyze_lta_dynamic_crop_diagnostic.py',
        'tools/study_sam_fusion.py',
        'tools/qualify_sdf_alignment.py',
        'tools/prepare_sam_holdout.py',
        'tools/heatsoak_sam_benchmark.py',
        'tools/qualify_sam_feature_cache.py',
        'tools/evaluate_sam_holdout.py',
        'tools/audit_sam_context.py',
        'tools/compare_sam_crop_strategies.py',
        'tools/sam_crop_strategy_geometry.py',
        'tools/report_sam_crop_strategies.py',
        'tools/sam_crop_seed_diagnostics.py',
        'tools/analyze_sam_crop_strategies.py',
        'tools/sam_crop_quality.py',
        'tools/prepare_sam_crop_two_tile.py',
        'tools/compare_sam_crop_zoom.py',
        'tools/qualify_sam_tiled_integration.py',
    ),
)
SAM_INTERPOLATION_REASONS = {
    '__init__': 'Publish the package release identity as {release}.',
    'cli': 'Validate the selected TTA interpolation contract before importing detector runtimes.',
    'config': 'Resolve separate detector/SAM model roles and devices, a single interpolation backend, and unchanged interpolation controls.',
    'assembly': 'Route full-frame and consolidated tile interpolation through the selected generator while preserving the existing component gate.',
    'pipeline': 'Own the selected SAM image/provider runtime and preserve resource admission and failure publication.',
    'lta_worker_adapter': 'Expose raw masks before publication filtering for SAM proposal evidence.',
    'lta_experimental': 'Retain raw independent endpoint-seeded tracker observations for SAM interpolation.',
    'sam_bridge_planning': 'Plan bounded original-observation families with fixed context, acceptance, branch write domains, and explicit unresolved limits.',
    'sam_tracker_runtime': 'Dispatch bounded independent endpoint jobs across admitted persistent SAM workers, retain exact cross-session features, and transfer attributable raw artifacts without dense queues.',
    'sam_interpolation': 'Generate complete original-seeded whole or experimental tiled SAM proposals, retain halo/core ownership, select quality, and publish directional slots without SDF fallback.',
    'sam_integration': 'Plan before image/model/GPU work, retain exact compact demanded canvas images, and admit SAM through role-aware GPU leases with verified residency retirement.',
    'sam_evidence': 'Store and verify indexed raw/candidate masks, optional full tile halos and child identities/scores, and unknown core availability while preserving legacy whole evidence.',
    'sam_policy': 'Preserve historical whole v2 and resolve explicit tiled v3 core/halo/availability quality with unchanged thresholds, bounded replay, and upstream dependency checks.',
    'sam_filtering': 'Remove undersized full-raw SAM slice components before quality decisions and reproduce filtered selected masks from an immutable versioned receipt.',
    'sam_replay': 'Export selected fixed proposals as bounded view-native directional NRRDs without loading model runtimes or claiming source-grid pipeline equivalence.',
    'lta_config': 'Expose opt-in bounded dynamic LTA crop controls while retaining default tiled behavior.',
    'lta_dynamic_crops': 'Plan deterministic full-seed native crops and explicit model transforms with bounded split and interior-guard geometry.',
    'lta_dynamic_execution': 'Advance sealed predecessor crop batches with finite independent directional tracking and nonrecursive window patches.',
    'lta_execution': 'Route opt-in dynamic LTA through the existing worker/publication ownership with explicit bounded crop receipts.',
    'lta_runtime': 'Retain selected LTA crop identity and settings in immutable execution plans.',
    'lta_feature_cache': 'Retain bounded exact frame features across independent tracker sessions with model/source/transform identity, shared tensor-storage accounting, and headroom admission.',
    'sam_mask_reader': 'Reuse bounded immutable raw, filtered, and candidate masks within one verified proposal transaction without changing measurements or policy decisions.',
    'sam_crop_tiling': 'Plan independent original-seeded 1008-side tracking tiles with 128 halos, fixed ownership, and explicit unavailable seed coverage on the existing working canvas.',
    'reconciliation_policy': 'Add versioned proposal selection alongside existing source-slab policies.',
    'reconciliation_runtime': 'Select SAM proposals before source-union reuse and retain separate voting provenance.',
}
REASONS = {
    '__init__': 'Publish the package release identity as {release}.',
    'cli': 'Use the sole {release} launcher and current release identity.',
    'config': 'Publish the current release identity and reconciliation configuration.',
    'pipeline': 'Reuse the assembled additive union, release replaced buffer owners, defer unused confidence projection, and drain owned evidence futures with completion progress and propagated failures.',
    'reconciliation_runtime': 'Reuse borrowed union buffers without component rescans and publish truthful metadata while closing explicit reader owners.',
    'confidence_evidence': 'Persist bounded block and native confidence with publication start/completion progress, without unnecessary source-grid projection.',
    'confidence_storage': 'Store validated, bounded confidence blocks with checked codec metadata and periodic write progress.',
    'confidence_native': 'Preserve native confidence pieces and make bounded source-grid conversion an explicit operation.',
    'confidence_export': 'Export retained native confidence through explicit source-grid conversion and checked output metadata.',
    'confidence_projection': 'Project observed confidence only when requested, preserving geometry and reporting progress during tilted-azimuthal staging.',
    'confidence_tiles': 'Preserve gated tile confidence as native evidence with explicit parent support provenance.',
    'cuda_d1': 'Retain native D1 confidence shards through bounded storage and retirement.',
    'cuda_backend': 'Fuse confidence composition in an optional compiled 2D loop with conservative layout and alias admission, safe compile preflight, and unchanged NumPy fallback semantics.',
    'packed_publication': 'Accelerate packed metadata scans with LLVM population count, bit scans, and vectorizable interior counting while preserving the existing payload encoder.',
    'assembly': 'Preserve immutable confidence publication and release score workspaces on success or failure.',
    'reconciliation_io': 'Read persisted source or native confidence references without implicit projection.',
    'outputs': 'Preserve output publication and reconciliation evidence metadata.',
    'workers': 'Carry current confidence transport and ownership configuration into inference workers.',
    'publication_memory': 'Account for bounded confidence and publication ownership.',
    'inference': 'Preserve confidence transport without changing inference masks or morphology.',
    'backprojection': 'Preserve native confidence evidence and the existing binary projection contract.',
    'examples/external_reconciliation/_confidence_core': 'Provide the curated confidence core and spatial rescue decision with the validated two-dimensional behavior.',
    'examples/external_reconciliation/_hybrid': 'Provide the curated hybrid consensus, rescue and enclosed-hole filling decision.',
    'examples/external_reconciliation/confidence_core_rescue': 'Publish the confidence core rescue preset selected through visual review.',
    'examples/external_reconciliation/quorum3': 'Publish the independent section quorum preset as a compact voting alternative.',
    'examples/external_reconciliation/hybrid_with_fill': 'Publish the hybrid consensus and limited enclosed-hole filling preset.',
    'confidence_consolidation': 'Consolidate native confidence pieces into a bounded shared payload without changing scores or known support.',
    'confidence_publication': 'Bound pending confidence publications and preserve owner lifetimes, completion progress and failures.',
    'd1_confidence_retirement': 'Retire immutable D1 confidence shards with bounded deferred publication and explicit cleanup ownership.',
    'spherical_projection': 'Avoid redundant empty spherical CPU work while preserving mixed CPU and CUDA projection semantics.',
    'spherical_projection_cpu': 'Skip provably empty spherical ranges and retain exact projection values for occupied regions.',
}
THROUGHPUT_REASONS = {
    'pipeline': 'Bound confidence publication and retirement queues, preserve completion acknowledgements independently of lease retirement order, overlap assembled-union reconciliation with independent pending exports while retaining their backing owners, cooperatively drain background work around worker-result credits, service GPU stage admission changes on the scheduler thread, and report bounded wait-state memory and stage ownership diagnostics.',
    'confidence_evidence': 'Publish consolidated native confidence with bounded ownership and restrict observed-score capture to trusted pre-interpolation support bounds.',
    'confidence_storage': 'Support bounded consolidation of compressed confidence blocks without decoding or altering retained scores.',
    'confidence_native': 'Publish copied confidence pieces transactionally and admit source conversion workspaces against explicit memory budgets while preserving scores, known support and geometry.',
    'confidence_projection': 'Bound numeric confidence projection strips and output-plane workspaces before allocation while preserving projection arithmetic and explicit progress.',
    'assembly': 'Forward trusted pre-interpolation slice support metadata into native confidence capture while preserving score ownership and cleanup.',
    'cuda_d1': 'Defer native confidence shard retirement through bounded publication and derive missing device-mask support bounds for exact cropped capture, preserving immutable scores and conservative metadata fallbacks.',
    'inference': 'Preserve confidence transport and derive compact device-mask support metadata after the existing producer fence without changing inference masks or morphology.',
    'runtime': 'Reuse acknowledged persistent source descriptors, preserve atomic GPU sample identity, exclude auxiliary interpolation during a claimed main-process CUDA stage, and measure memory-map advice. Coalesce scheduler timings, counters and gauges under independent short locks, preserve numeric and mixed ordinary-gauge ordering, bound trace capture, notify the background writer without blocking compute or credit paths, and preserve complete explicit and final telemetry flushes.',
    'tta_scheduler': 'Track acknowledged worker source ownership, memoize repeated selector evaluations, prioritize bounded compute-credit draining ahead of final-result callbacks while preserving failure fences and admission semantics, time refill selection, admission, workspace and transport steps, coalesce GPU stage wakeups into owner-thread admission retries with a bounded timer when an idle worker remains reserved, and route scheduler counters and gauges through the isolated telemetry channel with compatible fallback.',
    'backprojection': 'Reserve main-process GPU stages with epoch and lease tokens, performing CUDA memory probes and auxiliary claims outside the global admission lock while fencing stale releases and provisional devices.',
    'tta_background': 'Bound and rotate main-thread background completion categories, yielding after atomic callbacks to process worker credits promptly.',
    'scheduler_diagnostics': 'Measure scheduler operations and nested refill steps with wall and calling-thread CPU durations, coalescing timings without taking the diagnostic writer lock and emitting bounded slow-operation events without storage access on the measured path.',
    'mmap_advice': 'Call memory-map advice without holding the Python GIL while a zero-copy view pins the mapping, preserving the native error and portable fallback behavior.',
    'publication_memory': 'Account for bounded pending confidence publication and retirement buffers in admission.',
    'outputs': 'Batch repeated cached-zero gzip members without changing their byte sequence, preserve bounded ordered writes and pending export reference lifetimes, bound completed-output reaping, and report writer waits, atomic publication durability, and memory-map advice timings. Report the selected NRRD codec and imported-module provenance once, reject known pre-0.9 python-deflate bindings that hold the GIL with an actionable explicit-selection error, and allow automatic CPU selection to continue to compatible ISA-L or zlib without misclassifying unknown custom bindings.',
    'reconciliation_runtime': 'Reuse an assembled union while guarding export overlap against shared writable storage and preserve evidence ownership, metadata and cleanup.',
}
SEMANTIC_REASONS = {
    '__init__': 'Publish the 23.0.1 package release identity.',
    'cli': 'Use the sole 23.0.1 launcher and current release identity, clarifying mode and dependency-light dispatch comments.',
    'config': 'Expose semantic task and output choices for TTA and PTA with the 23.0.1 identity and clarify angle-variant documentation.',
    'geometry': 'Clarify current TTA geometry documentation and explicitly retire the unused private augmentation-angle parser.',
    'lta_worker_adapter': 'Clarify that parent-side LTA code imports artifact helpers while predictor construction remains inside isolated workers.',
    'inference': 'Consume semantic model logits and confidence without requiring instance masks, using the CUDA TensorRT ring when eligible, and clarify device-union admission.',
    'semantic_inference': 'Decode raw semantic logits into foreground probability and preserve confidence for thresholding and reconciliation.',
    'semantic_cuda': 'Run semantic probability, filtering and mask publication on CUDA while preserving CPU reference behavior.',
    'semantic_trt': 'Reuse fixed-shape TensorRT bindings and captured CUDA work across semantic inference batches.',
    'outputs': 'Publish semantic class-index masks with a reserved ignore value.',
    'confidence_evidence': 'Record task-neutral prediction scores in retained confidence evidence.',
    'confidence_storage': 'Store task-neutral prediction scores with their quantization metadata.',
    'pta': 'Preserve partial semantic coverage and route eligible categorical rendering and publication through resident CUDA work, recording actual CPU and CUDA dispatch counts.',
    'pta_classification': 'Reuse exact native-plane semantic occupancy and affine metadata to classify only requested PTA outputs.',
    'pta_config': 'Validate semantic task and save choices for pretraining.',
    'pta_gpu_publication': 'Assemble semantic labels and ignore regions on CUDA, then publish bounded image batches and report the selected concurrent host PNG encoder.',
    'pta_augmentation': 'Preserve semantic label and ignore values through pretraining augmentation.',
    'pta_publication': 'Write semantic labels with the unknown-region ignore value using a qualified PNG filter and run-length configuration with compatible fallback.',
    'pta_runtime': 'Keep semantic candidate confidence available for pretraining publication.',
    'pta_workers': 'Carry semantic task and coverage settings through pretraining workers, keeping eligible categorical rendering resident on CUDA while clarifying fallback and publication ownership.',
    'pta_cuda_masks': 'Render categorical foreground and annotation coverage on CUDA with nearest sampling and checked CPU fallback.',
    'pta_cuda_cartesian': 'Render Cartesian and tilted Cartesian categorical planes on CUDA while preserving bounded geometric parity.',
    'pta_cuda_azimuthal': 'Render upright and tilted azimuthal categorical planes on CUDA with wrap and mirror geometry.',
    'pta_cuda_shells': 'Render radial and spherical categorical planes on CUDA using resident source volumes.',
    'unification/sampling': 'Version the forward sampling policy and record qualified resident CUDA categorical sampling with bounded parity tolerance.',
    'pipeline': 'Route semantic predictions through TTA confidence and reconciliation.',
    'tta_mode': 'Accept semantic task and save modes for test-time augmentation.',
    'tta_outputs': 'Publish semantic masks from test-time augmentation.',
    'tta_prediction': 'Apply semantic confidence handling to test-time predictions.',
    'runtime': 'Record semantic task identity and preserve prediction confidence.',
    'workers': 'Carry semantic task settings through test-time workers, retire TensorRT ring resources safely, and clarify current task errors and documentation.',
    'examples/external_augmentations/GPU_baseline': 'Declare that baseline geometric augmentation does not require an instance mask.',
    'examples/external_augmentations/GPU_light': 'Declare that light geometric augmentation does not require an instance mask.',
    'examples/external_augmentations/GPU_heavy': 'Declare that heavy geometric augmentation does not require an instance mask.',
    'examples/external_augmentations/GPU_superheavy': 'Declare that superheavy geometric augmentation does not require an instance mask.',
}
PATCH_REASONS = {
    '__init__': 'Bump package identity after the tagged 23.0.1 release.',
    'cli': 'Read the release identity from the inert package initializer and derive the launcher name.',
    'config': 'Read the release identity from the inert package initializer and derive compact and launcher names.',
    'geometry': 'Guard the supported Ultralytics loader signature, clarify current geometry behavior, and retire the unused private angle parser.',
    'inference': 'Fail clearly when Ultralytics predictor signatures differ from the supported family and clarify current inference behavior.',
    'workers': 'Propagate predictor patch-installation failures instead of continuing with unpatched inference.',
    'pta_augmentation': 'Compile the exact augmentation policy bytes that were hashed before execution.',
    'pta': 'Decode child-process output predictably on Windows and clarify current PTA diagnostics.',
    'media': 'Clarify current media diagnostics while preserving payload handling.',
    'lta_inputs': 'Decode input-probe subprocess output as UTF-8 with replacement for invalid bytes.',
    'finalization': 'Retire owned temporary binary volumes after borrowed views release and clarify current diagnostics without changing output arithmetic.',
    'backprojection': 'Retire tilted-azimuthal backing paths after borrowed views release and clarify current diagnostics without changing projection arithmetic.',
    'assembly': 'Preserve mapped volume owners until all assembled views retire.',
    'lta_rendering': 'Preserve mapped LTA render owners until borrowed views retire.',
    'lta_scheduler': 'Clarify the current scheduler admission and relay bounds.',
    'lta_worker_adapter': 'Clarify that predictor construction remains isolated in workers.',
    'pta_workers': 'Clarify categorical CUDA fallback and publication ownership.',
    'runtime': 'Guard scratch memmap unmapping against live views while retaining explicit ownership cleanup.',
    'reconciliation_runtime': 'Preserve mapped evidence ownership until all borrowed views retire.',
    'pipeline': 'Extract admitted-view preparation from the TTA orchestrator and retain existing stage admission behavior.',
    'tta_augmentation_runtime': 'Keep mapped augmentation roots owned while shared views are live.',
    'tta_augmentation_cpu_runtime': 'Keep mapped CPU augmentation roots owned while shared views are live.',
    'outputs': 'Publish JSON manifests with durable atomic replacement.',
    'lta_outputs': 'Use the shared durable JSON publication helper for LTA manifests.',
    'confidence_evidence': 'Use the shared durable JSON publication helper and guard borrowed memmap lifetimes in confidence evidence.',
    'confidence_native': 'Preserve mapped native confidence owners while borrowed score views remain live.',
    'confidence_projection': 'Release borrowed memmap ownership only after all confidence projection views retire.',
    'confidence_tiles': 'Preserve parent memmap ownership while derived confidence tile views remain live.',
    'lta_execution': 'Keep LTA memmap owners alive through asynchronous view use and release them at terminal completion.',
    'lta_sam': 'Restore process CUDA precision flags after SAM imports and model construction.',
    'json_publication': 'Consolidate atomic JSON writers with unique temporary names, strict finite encoding and durable file and directory sync.',
    'view_prepare': 'Move admitted-view preparation into an explicit run-state object with preserved scheduler ownership.',
    'cuda_finalization': 'Retire unsafe direct memmap-close fallback while preserving CUDA finalization ownership.',
    'cuda_d1': 'Retire owned D1 archive and delete backings after borrowed mappings release.',
    'sparse_projection': 'Retain mapped sparse-projection inputs until dependent views retire.',
    'union_artifacts': 'Retain mapped union-artifact inputs until dependent views retire.',
    'lta_union_artifacts': 'Retain mapped LTA union-artifact inputs until dependent views retire.',
    'interpolation': 'Retire exact owned scratch paths only after borrowed views and readers release their mappings.',
    'tta_outputs': 'Drop completed output artifact aliases before owned scratch cleanup and manifest publication.',
    'tta_scheduler': 'Defer exact owned tile-result backing cleanup until outstanding mapped views retire.',
    'unification/manifest': 'Publish complete run manifests through a durable atomic JSON writer.',
}
TTA_THROUGHPUT_REASONS = {
    '__init__': 'Advance the package identity after the tagged 23.0.2 release.',
    'geometry_quality': 'Restore the qualified compiled Spherical CPU policy without enabling unrelated fast geometry.',
    'spherical_projection': 'Let compiled Spherical readers use the compact-memory worker admission policy.',
    'cylindrical_bitset_compaction': 'Publish Radial bitsets through bounded GPU packing with exact slice encoding.',
    'cylindrical_owner': 'Export packed Radial slices and elide proven-empty owner downloads.',
    'cuda_d1': 'Carry encoded Radial publication, empty-owner completion, and exact masked confidence crops through D1.',
    'workers': 'Report the restored Radial and Spherical worker paths.',
    'outputs': 'Drain completed zero-descriptor gzip members without blocking unrelated publication.',
    'tta_scheduler': 'Avoid repeated immutable task calculations while preserving dynamic admission decisions.',
}
TTA_PTA_THROUGHPUT_REASONS = {
    '__init__': 'Advance the package identity after the tagged 23.0.4 release.',
    'cuda_backend': 'Make the native TensorRT ring admission family-specific while preserving generic rendering for ineligible tasks.',
    'pipeline': 'Carry the reviewed native-ring policy through TTA task execution and reporting.',
    'confidence_storage': 'Reduce confidence evidence publication overhead while preserving encoded score and index bytes.',
    'outputs': 'Improve bounded NRRD publication diagnostics without changing layer contents.',
    'pta_publication': 'Reduce PTA image publication filesystem work while retaining checked publication semantics.',
    'pta_workers': 'Use the reviewed PTA publication and image-tree verification paths.',
    'runtime': 'Sample bounded NRRD publication diagnostics on the existing telemetry cadence and at shutdown.',
}
REMOVAL_REASONS = {
    'examples/external_reconciliation/' + name:
        'Retire this preset from the curated package selection while preserving its authenticated release history.'
    for name in ('baseline', 'confidence_voxel', 'cross_sections', 'provenance')
}
REMOVED_DEFINITION_REASONS = {
    ('geometry', '_angle_from_aug_id'):
        'Retire the unused private augmentation-angle parser while retaining its authenticated v22.3.2 source hash.',
    ('assembly', '_SparseComponentKernelUnavailable'):
        'Remove the private signal used to silently switch sparse-component work to Python.',
    ('assembly', '_run_sparse_component_kernel'):
        'Call the mandatory compiled sparse-component kernel directly and report faults explicitly.',
    ('cylindrical_owner', '_bucket_shell_pixels_numpy'):
        'Remove the slow NumPy Radial bucket fallback from production.',
    ('cylindrical_projection', '_occurrence_rows'):
        'Keep inverse-shear sampling arithmetic in the independent Radial test oracle.',
    ('cylindrical_projection', '_pull_radial_chunk'):
        'Retire the bounded NumPy Radial pull from production after admitting the compiled plan-free path.',
    ('geometry_quality', 'spherical_cpu_compiled_requested'):
        'Require the exact compiled Spherical CPU path instead of a process-wide reference opt-out.',
    ('interpolation', '_disable_planning_kernels'):
        'Fail explicitly on compiled planner faults instead of disabling Numba for the rest of the process.',
    ('interpolation', '_planning_kernels_active'):
        'Use the mandatory compiled planner without a Python fallback activation switch.',
    ('interpolation', 'compiled_interpolation_kernels_enabled'):
        'Require Numba interpolation kernels instead of exposing a Python fallback selector.',
    ('interpolation', 'compiled_topology_kernels_enabled'):
        'Require Numba topology kernels instead of exposing a Python fallback selector.',
    ('interpolation', '_find_slice_projection_candidates_python'):
        'Keep the independent Python projection planner as a test reference, outside production XTA.',
    **{
        ('spherical_projection', name):
            'Move the independent vectorized Spherical CPU reference into tests while retaining compiled production geometry.'
        for name in ('_nearest_global_shell', '_processing_index', '_pull_spherical_chunk')
    },
    **{
        (module, name): 'Retire this compatibility accessor after preserving its current default behavior and fallback admission.'
        for module, names in {
            'backprojection': (
                'main_process_gpu_stage_inference_overlap_enabled',
                'main_process_gpu_stage_inference_priority_enabled',
            ),
            'cuda_backend': ('gpu_cube_resize_enabled',),
            'cuda_d1': ('raw_bbox_nrrd_layers_enabled',),
            'finalization': (
                'fused_final_native_sparse_cpu_enabled',
                'fused_final_restore_geometry_groups_enabled',
                'fused_final_view_union_enabled',
                'scheduler_push_drain_enabled',
            ),
            'inference': (
                'cpu_retina_roi_only_enabled',
                'gpu_retina_flatten_enabled',
                'gpu_retina_proto_union_enabled',
                'gpu_retina_warp_enabled',
                'gpu_worker_chunk_hole_fill_enabled',
            ),
            'interpolation': ('interpolation_fused_bridge_merge_enabled',),
            'outputs': ('nrrd_extent_zero_skip_enabled', 'nrrd_live_global_layer_enabled'),
            'runtime': ('gpu_worker_direct_union_enabled', 'hybrid_gpu_stealback_enabled'),
            'spherical_projection': ('spherical_cpu_compact_enabled',),
            'topology': ('interpolation_skip_compact_relabel_enabled', 'interpolation_sparse_labels_enabled'),
            'workspace': ('tilted_inplane_linear_enabled',),
        }.items()
        for name in names
    },
}
REMOVED_STATEMENT_REASONS = {
    ('_deps', 'from typing import Optional'): 'Remove unused Optional type import after requiring the Numba dependency.',
    ('interpolation', 'from .inference import _cv2_connected_components, _fill_holes_2d_opencv'):
        'Remove image helpers used only by the Python interpolation reference moved into tests.',
    ('spherical_projection', 'from .qsc import qsc_forward_face'):
        'Move the vectorized Spherical reference and its QSC import into tests.',
    ('spherical_projection', 'from .geometry_quality import spherical_cpu_compiled_requested'):
        'Remove the compiled CPU opt-out after requiring its exact kernel.',
    ('lta_outputs', 'import json'): 'The shared JSON writer now owns encoding.',
    ('lta_outputs', 'import os'): 'The shared JSON writer now owns fsync and replacement.',
    ('lta_outputs', 'import tempfile'): 'The shared JSON writer now owns unique temporary files.',
    ('unification/manifest', 'import json'): 'The shared JSON writer now owns encoding.',
    ('unification/manifest', 'import os'): 'The shared JSON writer now owns fsync and replacement.',
    ('unification/manifest', 'import threading'): 'The shared JSON writer now owns temporary names.',
}
REMOVED_STATEMENT_HASH_REASONS = {
    ('assembly', '8b8e9eb230fe2b63c9496772d6bb7ad340050d700bd52aa82f5ebe5fb999d5e7'):
        'Bind the required Numba sparse-component kernels without optional import branching.',
    ('assembly', '14726e24984d9df4f8db59388a57016387e1e658052a8f551afa8af37648ce67'):
        'Remove the process-wide sparse-component kernel failure latch.',
    ('backprojection', '138fa3acc84bf3ba73c42ed3853ad547e780d51cf25c6107fef73e51a9266351'):
        'Bind the required packed-bit coordinate kernel without optional import branching.',
    ('cuda_backend', '2de8f9f12f3d1012fbdc7e18f12ddc8b636d37e529b8a10de20d6367202565d4'):
        'Bind the required compiled union-confidence kernel unconditionally.',
    ('cylindrical_projection', 'a00fd59c3a0ef57a76f093e34ec03927219de47f7f0ad44fe7c22b21bc713347'):
        'Compile the exact Radial CPU gather and projection kernels unconditionally.',
    ('spherical_projection_cpu', '62dfc204b213a46d14451a67a59b2d9ac8d1625a929df9f22a601b16be349147'):
        'Remove the optional compiled-dispatcher placeholder after requiring Numba.',
    ('spherical_projection_cpu', 'c2ac2fee3ddb3649c691ab29da24c51cca922f57e65586eeacb81b759994965b'):
        'Remove the fallback reason for a missing Numba installation.',
    ('spherical_projection_cpu', '63020afc675eee8a36e437c13a59a4b57c6a23a77655ef87114867bd4a20ada3'):
        'Bind the required exact compiled Spherical CPU kernel unconditionally.',
    ('interpolation', 'd24f27cb632054fb67662096b8c4070083fc9c3b985370cccd7d26337306a9a9'):
        'Remove the process-wide projection-kernel failure latch.',
    ('interpolation', '3ab30c336abd35d22525a428c7d84454e0105acfd26b72e24434d8616cc10208'):
        'Remove the process-wide planning-kernel failure latch.',
    ('interpolation', '4e01060dae0bb88826bf2b113bcb0f5e9d10f2ade407861176dd712774bb430e'):
        'Bind the required compiled nearest-pixel planner without optional import branching.',
    ('interpolation', '42b6144e5eaac69f1d042276d509eb9d9e470d16f848cee29f0c01a51a8b2253'):
        'Bind the required compiled candidate-search planner without optional import branching.',
    ('outputs', 'e7dff3fea3e8f03b78266fb443ee7ff6a55559fcefce4003e4ea192bfe2bf6d9'):
        'Remove the process-wide sparse-area Numba failure latch.',
    ('outputs', '29265bc1808725d5378e1c5865e0bc9e926d57d3ba33f0cee556ee002570c45e'):
        'Remove the sparse-area fallback announcement latch.',
    ('outputs', 'e2c64bae2e96424a93052b292875c466a1a6c500d60a1e7bd51c4342098f31dc'):
        'Bind the required compiled sparse-area integration kernels unconditionally.',
    ('packed_publication', 'db93f1d46cb61813da7a6de5c765048e75a2feafc0c52bb16feff796e25e999c'):
        'Bind the required Numba packed-publication intrinsics without optional import branching.',
    ('sparse_projection', '1d8caaec63bd240e132cc12a2f9a591faa7f6974047c46ef91d7d76a91e3dc33'):
        'Compile the required sparse-projection kernels unconditionally.',
    ('topology', '34e3971f0db977a8caac8cdee85f257d108adb8fe05c301f0f5ac657a0ce2238'):
        'Remove the process-wide union-find Numba failure latch.',
    ('topology', '550c6f6ee03b1d2ed4339c862cd934c6650182ba4eb44a2d3a27792df064d4f3'):
        'Remove the process-wide adjacency Numba failure latch.',
    ('topology', 'e7fffed290409ac21b8ea3f3619a087ce96d51576973e806684dbcda5e43ca8b'):
        'Bind the required compiled union-find kernel without optional import branching.',
    ('topology', 'e788bfa27be8f551f0df192472371bcb65e128ed9921e5f6f11e93655941e274'):
        'Bind the required compiled topology kernels without optional import branching.',
    ('topology', 'eaffedf3ccea2cc63cf7173a998341012d7c8f78b44f57a3c871a1e7ad3edd2e'):
        'Bind the required compiled adjacency kernel without optional import branching.',
    ('topology_runs', '6824f9609888b53198dac49de1d9ce860e0894d8c36a2d16c926c7a75a872ddc'):
        'Bind the required compiled run-adjacency kernels without optional import branching.',
}
TOOL_REASONS = {
    'tools/study_sam_fusion.py': 'Compare frozen fusion variants over retained selected SAM proposals without inference, separating development diagnostics from independently withheld labels.',
    'tools/qualify_sdf_alignment.py': 'Evaluate predeclared SDF anchor/transport/area experiments without changing the production SDF generator or using labels before predictions are frozen.',
    'tools/prepare_sam_holdout.py': 'Pre-register bounded detector-only interpolation cases and input identities before held-out annotations are opened.',
    'tools/heatsoak_sam_benchmark.py': 'Reserve the local GPU and heatsoak CPU/GPU before scheduling sanity benchmarks, retaining thermal receipts without claiming target-system performance.',
    'tools/qualify_sam_feature_cache.py': 'Run real-model cache/dispatch equivalence against fixed complete raw masks and scores with bounded warm local timing and explicit source identity.',
    'tools/evaluate_sam_holdout.py': 'Freeze pre-registered retained-proposal fusion outputs before reading held-out labels and score matched no-interpolation, SAM, and SDF variants in the declared source coordinates.',
    'tools/audit_sam_context.py': 'Audit original observation masks without labels to identify cases censored by artificial diagnostic ROI boundaries, preserving frozen candidate methods and production policy.',
    'tools/compare_sam_crop_strategies.py': 'Run paired whole-native-crop resize and independent overlapping-tile SAM experiments on matched observed anchors without cross-tile propagation or production backend changes.',
    'tools/sam_crop_strategy_geometry.py': 'Define deterministic research crop, model-canvas, and inverse-stitch geometry for paired resize and independent-tile strategies without using evaluation labels.',
    'tools/report_sam_crop_strategies.py': 'Report matched paired crop-strategy outputs, held-out measurements, limitations, and reviewable overlays without presenting an experimental strategy as a production default.',
    'tools/sam_crop_seed_diagnostics.py': 'Measure original native seed coverage and model-canvas seed survival for paired crop strategies without introducing missing seed support or changing model predictions.',
    'tools/analyze_sam_crop_strategies.py': 'Analyze frozen paired crop-strategy outputs offline with matched geometry and explicit research limitations while preserving immutable observations and predictions.',
    'tools/sam_crop_quality.py': 'Compute diagnostic eligibility for paired crop-strategy evidence without changing or claiming stock production v2 proposal acceptance.',
    'tools/prepare_sam_crop_two_tile.py': 'Freeze the requested two 1260-by-659 native footprints from the existing family plan and original seeds without altering previous plans or production geometry.',
    'tools/compare_sam_crop_zoom.py': 'Compare retained whole-crop, three-tile, and requested two-tile research evidence with overlap-safe recomposition and matched source/seed/context identities without new inference.',
    'tools/qualify_sam_tiled_integration.py': 'Qualify the actual SAM assembly/runtime seam on a bounded native-source fixture under explicit GPU reservation while distinguishing it from ordinary CLI processing-cube behavior.',
    'tools/analyze_lta_dynamic_crop_diagnostic.py': 'Verify completed native/scaled dynamic LTA output manifests and authoritative seed preservation, and render measured comparison overlays without model loading or independent quality claims.',
    'tools/lta_dynamic_crop_diagnostic.py': 'Prepare bounded lossless real-data LTA fixtures and reproducible native/scaled dynamic crop commands; the caller owns GPU reservation and execution.',
    'tools/diagnose_sam_interpolation.py': 'Run bounded detector-derived SDF/SAM comparisons, publish reviewable overlays, and retain exact evaluation commands without claiming independent label truth.',
    'tools/compare_reconciliation.py': 'Compare persisted source evidence and fixed SAM proposals with bounded readers, dependency invalidation, and unchanged input artifacts.',
    'tools/qualify_tta_reconciliation.py': 'Qualify unchanged masks and complete source or native confidence across CPU, GPU and hybrid inference while isolating Ultralytics settings outside the repository.',
    'tools/export_reconciliation_evidence.py': 'Export retained confidence into a checked source-grid companion or copy a verified portable SAM proposal bundle without inference.',
    'tools/qualify_confidence_consolidation.py': 'Qualify consolidated native confidence against original pieces with exact score and known-support parity.',
    'tools/qualify_d1_confidence_bounds.py': 'Qualify cropped and dense confidence capture from the same real generic Radial prediction, preserving exact encoded score/index bytes and device source tensors.',
    'tools/analyze_pipeline_trace.py': 'Interpret bounded task traces with incomplete-capture warnings and GPU compute-credit timing that distinguishes prefetch and result-first ambiguity.',
    'tools/export_semantic_logits.py': 'Export semantic model logits for confidence-aware OpenVINO, ONNX and TensorRT inference.',
    'tools/qualify_semantic_trt.py': 'Compare the semantic TensorRT CUDA ring with the direct reference path and record performance.',
    'tools/qualify_pta_classification.py': 'Qualify semantic PTA classification bypass and exact native-plane occupancy on representative plans.',
    'tools/qualify_pta_gpu_masks.py': 'Qualify resident CUDA categorical geometry and publication against CPU references, including boundary and ignored-region cases.',
    'tools/qualify_pta_gpu_render.py': 'Measure the integrated PTA CUDA render, augmentation, nvJPEG and semantic PNG path against a CPU categorical fallback with identical inputs.',
    'tools/qualify_radial_bitset_compaction.py': 'Qualify exact packed Radial output and bounded device-to-host traffic against the original path.',
    'tools/qualify_d1_confidence_masked_transfer.py': 'Qualify exact cropped confidence payloads and reduced device-to-host calls for masked transfer.',
    'tools/qualify_release.py': 'Route qualification temporary files and GPU compiler caches into task Scratch while preserving the source and test gates.',
    'tools/qualify_native_trt_lease.py': 'Qualify explicit native TensorRT lease modes, execution and output parity.',
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git_file(root, predecessor, relative):
    result = subprocess.run(['git', 'show', predecessor + ':' + relative], cwd=root, capture_output=True)
    return None if result.returncode else result.stdout.decode('utf-8')


def identity(node):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return ('definition', node.name)
    if isinstance(node, ast.Assign):
        return ('binding', tuple(n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)))
    if isinstance(node, ast.AnnAssign):
        return ('binding', (getattr(node.target, 'id', ast.dump(node.target)),))
    if isinstance(node, ast.ImportFrom):
        return ('from', node.level, node.module)
    if isinstance(node, ast.Import):
        return ('import', tuple(alias.name for alias in node.names))
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return ('docstring',)
    return (type(node).__name__,)


def statement_label(node):
    key = identity(node)
    if key[0] == 'binding':
        return 'binding_' + '_'.join(key[1])
    if key[0] == 'from':
        return 'import_' + '.' * key[1] + (key[2] or '')
    if key[0] == 'import':
        return 'import_' + '_'.join(key[1])
    if key[0] == 'docstring':
        return 'module_docstring'
    return f'{type(node).__name__}:{node.lineno}'


def review_module(module, old_source, new_source, *, complete, labels_by_hash, reason):
    """Account for every predecessor statement without silently dropping one."""
    old = ast.parse(old_source) if old_source is not None else None
    new = ast.parse(new_source)
    old_hashes = [inventory.digest(node) for node in old.body] if old is not None else []
    new_hashes = [inventory.digest(node) for node in new.body]
    pin = dict(ast_sha256=inventory.digest(old) if old is not None else None,
               statements_sha256=hashlib.sha256(json.dumps(old_hashes, separators=(',', ':')).encode()).hexdigest())
    snapshot = dict(module=module, previous_ast_sha256=pin['ast_sha256'], ast_sha256=inventory.digest(new),
                    previous_top_level=old_hashes, top_level=new_hashes, reason=reason)
    records = dict(definitions=[], statements=[], local_import_seam_updates=[],
                   removed_definitions=[], removed_statements=[])
    unmatched, used_labels = set(range(len(old_hashes))), set()
    for index, node in enumerate(new.body):
        previous = None
        if old is not None:
            previous = next((i for i in sorted(unmatched) if identity(old.body[i]) == identity(node)
                             and old_hashes[i] == new_hashes[index]), None)
            if previous is None:
                previous = next((i for i in sorted(unmatched) if identity(old.body[i]) == identity(node)), None)
        if previous is not None:
            unmatched.remove(previous)
        previous_hash = old_hashes[previous] if previous is not None else None
        if previous_hash == new_hashes[index] and not complete:
            continue
        item = dict(module=module, previous_sha256=previous_hash, sha256=new_hashes[index],
                    previous_index=previous, current_index=index, reason=reason)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            records['definitions'].append({**item, 'name': node.name})
        else:
            label = labels_by_hash.get((module, previous_hash), statement_label(node))
            if label in used_labels:
                label += f':{index}'
            used_labels.add(label)
            item['label'] = label
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                item['binding'] = node.targets[0].id
            records['statements'].append(item)
    for index in sorted(unmatched):
        node = old.body[index]
        key = (module, getattr(node, 'name', None))
        retirement_reason = REMOVED_DEFINITION_REASONS.get(key)
        if retirement_reason and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            records['removed_definitions'].append(dict(
                module=module, name=node.name, previous_index=index,
                previous_sha256=old_hashes[index], reason=retirement_reason,
            ))
            continue
        retirement_reason = (REMOVED_STATEMENT_REASONS.get((module, ast.unparse(node)))
                             or REMOVED_STATEMENT_HASH_REASONS.get((module, old_hashes[index])))
        if not retirement_reason:
            raise ValueError(f'{module}: unaccounted predecessor statements: '
                             f'{[(i, identity(old.body[i])) for i in sorted(unmatched)]}')
        records['removed_statements'].append(dict(
            module=module, previous_index=index,
            previous_sha256=old_hashes[index], reason=retirement_reason,
        ))
    old_seams = inventory.reviewed_local_import_seams(module, old_source, old) if old is not None else {}
    new_seams = inventory.reviewed_local_import_seams(module, new_source, new)
    retired_seams = {(module, item['name']) for item in records['removed_definitions']}
    if set(new_seams) - set(old_seams) or set(old_seams) - set(new_seams) - retired_seams:
        raise ValueError(f'{module}: local-import seam ownership changed')
    for key, current in new_seams.items():
        previous = old_seams[key]
        if current != previous:
            records['local_import_seam_updates'].append(dict(module=module, name=key[1],
                previous_definition_sha256=previous[0], previous_seam_sha256=previous[1],
                definition_sha256=current[0], seam_sha256=current[1], reason=reason))
    return pin, snapshot, records


def review_removed_module(module, old_source, *, reason):
    """Record an explicitly reviewed module retirement with its exact predecessor."""
    if module not in REMOVAL_REASONS or old_source is None or not reason:
        raise ValueError('Module deletion needs a separate review: ' + module)
    old = ast.parse(old_source)
    historical = [inventory.digest(node) for node in old.body]
    pin = dict(ast_sha256=inventory.digest(old),
               statements_sha256=hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest())
    snapshot = dict(module=module, previous_ast_sha256=pin['ast_sha256'], ast_sha256=None,
                    previous_top_level=historical, top_level=[], removed=True, reason=reason)
    records = dict(definitions=[], statements=[], local_import_seam_updates=[])
    return pin, snapshot, records


def _qualified_definition(source, name):
    node = ast.parse(source)
    for part in name.split('.'):
        node = next(value for value in node.body if getattr(value, 'name', None) == part)
    return node


def _update_verifier_pins(source, prefix, digest, pins, removals=None):
    updates = {prefix + '_SHA256': repr(digest),
               prefix + '_PREDECESSOR_MODULES': pprint.pformat(pins, width=110, sort_dicts=True)}
    if removals is not None:
        updates[prefix + '_REMOVALS'] = pprint.pformat(removals, width=110, sort_dicts=True)
    lines = source.splitlines(keepends=True)
    changes = [(node, target.id) for node in ast.parse(source).body if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Name) and target.id in updates]
    if len(changes) != len(updates) or {name for _node, name in changes} != set(updates):
        raise ValueError('Verifier is missing the release digest or predecessor-module pin binding')
    for node, name in sorted(changes, key=lambda pair: pair[0].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [name + ' = ' + updates[name] + '\n']
    return ''.join(lines)


def prepare(*, output_dir, release='24.0.0', write=False):
    root, output_dir = ROOT, Path(output_dir).resolve()
    if output_dir.is_relative_to(root):
        raise ValueError('Generated release-review evidence belongs outside the repository')
    spec = RELEASES[release]
    reasons = (SAM_INTERPOLATION_REASONS if release == '24.0.0' else
               TTA_PTA_THROUGHPUT_REASONS if release == '23.0.5' else
               {} if release == '23.0.4' else
               {**REASONS, **(THROUGHPUT_REASONS if release == '22.3.2' else {}),
                **(SEMANTIC_REASONS if release == '23.0.1' else {}),
                **(PATCH_REASONS if release == '23.0.2' else {}),
                **(TTA_THROUGHPUT_REASONS if release == '23.0.3' else {})})
    fallback_reason = ('Integrate bounded SAM interpolation, retained proposal evidence, or dynamic LTA crop support in this source module.'
                       if release == '24.0.0' else
                       'Record the reviewed v23.0.5 TTA/PTA throughput changes in this module.'
                       if release == '23.0.5' else
                       'Record the reviewed v23.0.4 cleanup and compiled CPU backend changes in this module.'
                       if release == '23.0.4' else
                       'Implement the reviewed {release} TTA throughput contract in this source module.')
    prefix = 'REVIEWED_V' + spec['token'] + '_RELEASE'
    key = 'v' + spec['token'] + '_release_review'
    predecessor_commit = getattr(inventory, prefix + '_PREDECESSOR_COMMIT')
    predecessor_path = spec.get('predecessor_inventory_path', 'XTA/_package_inventory.json')
    predecessor = json.loads(git_file(root, predecessor_commit, predecessor_path))
    if canonical(predecessor) != getattr(inventory, prefix + '_PREDECESSOR_SHA256'):
        raise ValueError('Git predecessor differs from the independently authenticated inventory')
    current = json.loads(inventory.MANIFEST.read_text(encoding='utf-8'))
    if {k: v for k, v in current.items() if k != key} != predecessor:
        raise ValueError('Preserve every predecessor inventory record before adding this review')
    audited = {item['module'] for item in predecessor['statements']}
    labels_by_hash = {}
    for value in predecessor.values():
        if isinstance(value, dict):
            for category in ('definitions', 'statements'):
                for item in value.get(category, ()):
                    audited.add(item['module'])
                    if category == 'statements' and 'label' in item:
                        labels_by_hash[item['module'], item['sha256']] = item['label']
    for (module, label), (value, _reason) in inventory.REVIEWED_V20_ADDED_STATEMENTS.items():
        labels_by_hash.setdefault((module, value), label)
    paths = subprocess.check_output(['git', 'diff', '--name-only', predecessor_commit, '--', 'XTA'], cwd=root, text=True).splitlines()
    paths += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'XTA'], cwd=root, text=True).splitlines()
    review = dict(release=release, feature=spec['feature'],
        previous_review_sha256=getattr(inventory, 'REVIEWED_V' + spec['previous_token'] + '_RELEASE_SHA256'),
        predecessor_commit=predecessor_commit, predecessor_inventory_sha256=canonical(predecessor),
        definitions=[], statements=[], removed_definitions=[], removed_statements=[],
        local_import_seam_updates=[], preserved_radial_definition_updates=[],
        preserved_radial_module_updates=[], complete_modules=[], module_snapshots=[])
    source_pins = {}
    for relative in sorted(set(paths)):
        if not relative.endswith('.py'):
            continue
        path = root / relative
        module = relative.removeprefix('XTA/').removesuffix('.py')
        old_source = git_file(root, predecessor_commit, relative)
        if not path.is_file():
            pin, snapshot, records = review_removed_module(module, old_source,
                reason=REMOVAL_REASONS.get(module, ''))
            complete = False
        else:
            new_source = path.read_text(encoding='utf-8')
            if old_source is not None and inventory.digest(ast.parse(old_source)) == inventory.digest(ast.parse(new_source)):
                continue
            reason = reasons.get(module, fallback_reason).format(release=release)
            complete = module not in audited
            pin, snapshot, records = review_module(module, old_source, new_source,
                complete=complete, labels_by_hash=labels_by_hash, reason=reason)
        source_pins[module] = pin
        review['module_snapshots'].append(snapshot)
        if complete:
            review['complete_modules'].append(module)
        for category, values in records.items():
            review[category].extend(values)
    patches = [value for name, value in predecessor.items() if name != 'v21_review'
               and isinstance(value, dict) and 'release' in value and 'definitions' in value]
    for module, previous_hash in inventory.reviewed_radial_module_hashes(predecessor['v21_review'], patches).items():
        text = (root / 'XTA' / f'{module}.py').read_text(encoding='utf-8')
        new_hash = hashlib.sha256(text.encode()).hexdigest()
        if new_hash != previous_hash:
            if hashlib.sha256(git_file(root, predecessor_commit, f'XTA/{module}.py').encode()).hexdigest() != previous_hash:
                raise ValueError(f'Preserved module predecessor differs: {module}')
            review['preserved_radial_module_updates'].append(dict(module=module,
                previous_sha256=previous_hash, sha256=new_hash,
                reason=reasons.get(module, fallback_reason).format(release=release)))
    for (module, name), previous_hash in inventory.reviewed_radial_definition_hashes(predecessor['v21_review'], patches).items():
        text = (root / 'XTA' / f'{module}.py').read_text(encoding='utf-8')
        new_hash = inventory.digest(_qualified_definition(text, name))
        if new_hash != previous_hash:
            old_text = git_file(root, predecessor_commit, f'XTA/{module}.py')
            if inventory.digest(_qualified_definition(old_text, name)) != previous_hash:
                raise ValueError(f'Preserved definition predecessor differs: {module}.{name}')
            review['preserved_radial_definition_updates'].append(dict(module=module, qualified_name=name,
                previous_sha256=previous_hash, sha256=new_hash,
                reason=reasons.get(module, fallback_reason).format(release=release)))
    previous_review = predecessor['v' + spec['previous_token'] + '_release_review']
    previous_tools = {item['path']: item['sha256'] for item in previous_review.get('validation_tools', ())}

    def previous_tool_sha(path):
        if path in previous_tools:
            return previous_tools[path]
        source = git_file(root, predecessor_commit, path)
        return hashlib.sha256(source.encode()).hexdigest() if source is not None else None

    review['validation_tools'] = [dict(
        path=path,
        previous_sha256=previous_tool_sha(path),
        sha256=hashlib.sha256((root / path).read_text(encoding='utf-8').encode()).hexdigest(),
        reason=TOOL_REASONS[path],
    ) for path in spec['validation_tools']]
    payload = {**predecessor, key: review}
    digest = canonical(review)
    # Validate draft structure using its proposed pins without publishing them.
    removals = {
        'definitions': tuple(sorted((item['module'], item['name'], item['previous_index'],
                                     item['previous_sha256']) for item in review['removed_definitions'])),
        'statements': tuple(sorted((item['module'], item['previous_index'], item['previous_sha256'])
                                   for item in review['removed_statements'])),
    }
    original_digest = getattr(inventory, prefix + '_SHA256')
    original_pins = getattr(inventory, prefix + '_PREDECESSOR_MODULES')
    pins_removals = release in ('23.0.4', '23.0.5', '24.0.0')
    original_removals = getattr(inventory, prefix + '_REMOVALS') if pins_removals else None
    try:
        setattr(inventory, prefix + '_SHA256', digest)
        setattr(inventory, prefix + '_PREDECESSOR_MODULES', source_pins)
        if pins_removals:
            setattr(inventory, prefix + '_REMOVALS', removals)
        getattr(inventory, 'reviewed_v' + spec['token'] + '_release_contract')(payload, predecessor['v21_review'])
    finally:
        setattr(inventory, prefix + '_SHA256', original_digest)
        setattr(inventory, prefix + '_PREDECESSOR_MODULES', original_pins)
        if pins_removals:
            setattr(inventory, prefix + '_REMOVALS', original_removals)
    if write:
        if subprocess.check_output(['git', 'tag', '--list', f'v{release}'], cwd=root).strip():
            raise ValueError(f'v{release} is already tagged; its authenticated review cannot be rewritten')
        trees = {item['module']: ast.parse((root / 'XTA' / (item['module'] + '.py')).read_text(encoding='utf-8'))
                 for item in review['module_snapshots'] if not item.get('removed')}
        inventory.verify_v22_3_source_snapshots(review, trees)
        inventory.verify_v22_3_validation_tools(review)
        verifier = root / 'tools/verify_package_inventory.py'
        verifier_source = _update_verifier_pins(verifier.read_text(encoding='utf-8'), prefix, digest, source_pins,
                                                removals if pins_removals else None)
        inventory.MANIFEST.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
        verifier.write_text(verifier_source, encoding='utf-8', newline='\n')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'release_review_draft.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'release_predecessor_source_pins.json').write_text(json.dumps(source_pins, indent=2) + '\n', encoding='utf-8')
    summary = dict(written=write, release=release, modules=len(source_pins), definitions=len(review['definitions']),
                   statements=len(review['statements']), seams=len(review['local_import_seam_updates']),
                   removed_modules=sum(item.get('removed') is True for item in review['module_snapshots']), sha256=digest)
    (output_dir / 'release_review_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', choices=tuple(RELEASES), default='24.0.0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(output_dir=args.output_dir, release=args.release, write=args.write)))


if __name__ == '__main__':
    main()
