from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import math
import os
import struct
import time
from typing import BinaryIO, Callable, Optional

import bpy
from mathutils import Vector
import numpy as np

from .collision import clear_static_collision_cache, collect_static_triangles
from .cache_names import safe_cache_stem
from .force_fields import EMPTY_FORCE_FIELD_BATCH, ForceFieldBatch, collect_force_fields, has_force_field_sources
from .native_backend import NativeGlobalDynamicScene, NativeStepDiagnostics, NativeXpbdSolver, status as native_status
from .xpbd_core import (
    ClothBuildData,
    PIN_HARD_WEIGHT_THRESHOLD,
    PinAttachmentBatch,
    SELF_COLLISION_FAST,
    build_cloth_data,
    clear_cloth_topology_cache,
    effective_pin_weights_from_settings,
    make_pin_attachment_batch,
    mesh_local_positions,
    settings_to_options,
    to_local,
    to_world,
    triangulated_faces,
    world_positions_from_object,
)


_SCENE_SESSIONS: dict[str, "SceneSession"] = {}
_OBJECT_TO_SCENE_SESSION: dict[str, str] = {}
_STATUS: dict[str, str] = {}
_LAST_DIAGNOSTICS: dict[str, NativeStepDiagnostics] = {}
_FREE_SIM_VERTEX_CACHE: dict[str, tuple[tuple, bool]] = {}
_CACHE_PATH_PROP = "_ssbl_xpbd_cache_path"
_CACHE_MODIFIER_NAME = "SSBL XPBD Cache"
_UNSUPPORTED_INPUT_TYPES = {"solid", "rod", "stitch", "tet"}
_OBJECT_COLLISION_LAYER_PROP = "ssbl_collision_layer"
_INTERACTIVE_PIN_PROP = "_ssbl_interactive_pin"
_INTERACTIVE_PIN_PREFIX = "SSBL_Interactive_Pin"
_IDENTITY_4X4 = np.eye(4, dtype=np.float32)
_AUTO_WRITEBACK_INTERVAL = 0
_MIN_WRITEBACK_INTERVAL = 1
_MAX_AUTO_WRITEBACK_INTERVAL = 8
_WRITEBACK_EWMA_ALPHA = 0.25
_PREVIEW_SELF_COLLISION_BUDGET_VERTEX_THRESHOLD = 4096
_PREVIEW_SELF_COLLISION_BUDGET_TRIANGLE_THRESHOLD = 8192
STATUS_IDLE = "Idle"
STATUS_PREVIEW_RUNNING = "Preview Running"
STATUS_PREVIEW_PAUSED = "Preview Paused"
STATUS_PREVIEW_STOPPED = "Preview Stopped"
STATUS_BAKING = "Baking"
STATUS_FINISHED = "Finished"
STATUS_ERROR = "Error"
STATUS_NO_SIMULATED_VERTICES = "No Simulated Vertices"


def _set_bake_progress_state(
    obj: bpy.types.Object | None,
    current: int,
    total: int,
    *,
    in_progress: bool,
) -> None:
    if obj is None or not hasattr(obj, "ssbl_cloth"):
        return
    settings = obj.ssbl_cloth
    total = max(int(total), 0)
    current = max(0, min(int(current), total if total > 0 else int(current)))
    settings.bake_in_progress = bool(in_progress)
    settings.bake_progress_current = current
    settings.bake_progress_total = total
    settings.bake_progress_percent = (float(current) / float(total) * 100.0) if total > 0 else 0.0


@dataclass
class FramePerf:
    frame_ms: float = 0.0
    frame_set_ms: float = 0.0
    input_refresh_ms: float = 0.0
    pin_upload_ms: float = 0.0
    runtime_upload_ms: float = 0.0
    static_upload_ms: float = 0.0
    dynamic_upload_ms: float = 0.0
    dynamic_collider_pack_ms: float = 0.0
    dynamic_collider_cache_hits: int = 0
    dynamic_collider_cache_misses: int = 0
    dynamic_pair_cache_hits: int = 0
    dynamic_pair_cache_misses: int = 0
    dynamic_pair_cache_reused_triangles: int = 0
    dynamic_pair_cache_reused_particles: int = 0
    dynamic_triangle_count: int = 0
    dynamic_particle_count: int = 0
    cuda_step_call_ms: float = 0.0
    download_ms: float = 0.0
    writeback_ms: float = 0.0
    writeback_to_local_ms: float = 0.0
    writeback_foreach_set_ms: float = 0.0
    writeback_mesh_update_ms: float = 0.0
    frame_input_upload_ms: float = 0.0
    writeback_performed: bool = False
    diagnostics_ms: float = 0.0
    viewport_tag_ms: float = 0.0


@dataclass
class CrossClothPairColliderPackage:
    key: tuple = ()
    cover_lower: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    cover_upper: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    source_motion_stamp: float = 0.0
    full_vertex_selection: bool = False
    full_triangle_selection: bool = False
    selected_vertices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    selected_triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3), dtype=np.float32))
    particle_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    particle_radii: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    triangle_buffer: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3), dtype=np.float32))
    triangle_index_buffer: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    particle_position_buffer: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    particle_radius_buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class CrossClothColliderCache:
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3, 3), dtype=np.float32))
    triangle_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    particle_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    particle_radii: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    particle_inv_mass: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    particle_slot_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    particle_phases: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    triangle_signature: tuple = ()
    particle_signature: tuple = ()
    swept_min: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    swept_max: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    max_motion: float = 0.0
    max_edge_length: float = 0.0
    motion_delta: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    vertex_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    vertex_mask_tmp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    triangle_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    triangle_mask_tmp: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    motion_sq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    positions_generation: int = 0
    motion_accumulated: float = 0.0
    pair_packages: dict[tuple[str, str], CrossClothPairColliderPackage] = field(default_factory=dict)


@dataclass
class InteractivePinState:
    object_name: str
    vertex_index: int
    pin_group_name: str
    hook_group_name: str
    hook_modifier_name: str
    empty_name: str
    previous_weight: float | None
    previous_preview_weight: float | None
    previous_active_name: str
    previous_selection_names: list[str]
    previous_use_evaluated_mesh: bool


@dataclass
class RealtimeCacheWriter:
    path: str
    temp_path: str
    handle: BinaryIO
    start_frame: int
    vertex_count: int
    sample_count: int = 0


@dataclass
class ClothSlot:
    object_name: str
    cloth: ClothBuildData
    native: NativeXpbdSolver
    original_mesh: bpy.types.Mesh
    preview_mesh: bpy.types.Mesh
    suspended_modifiers: list[tuple[str, bool, bool]]
    use_evaluated_mesh: bool
    static_collider_signature: tuple[tuple[str, int, int], ...]
    static_triangles: np.ndarray
    static_runtime_signature: tuple
    pin_attachment_pairs: np.ndarray
    pin_targets_world: np.ndarray
    runtime_options_signature: tuple
    solver_options_signature: tuple
    collision_layer: int
    external_contact_distance: float
    current_positions_world: np.ndarray
    previous_positions_world: np.ndarray
    writeback_local_buffer: np.ndarray
    writeback_flat_buffer: np.ndarray
    substeps: int
    iterations: int
    writeback_interval: int
    frame_count: int
    use_object_settings: bool
    force_fields_active: bool
    auto_sphere_object_name: str = ""
    auto_cache_realtime: bool = False
    realtime_cache: RealtimeCacheWriter | None = None
    pin_settings_signature: tuple = ()
    runtime_refresh_signature: tuple = ()
    force_next_writeback: bool = False
    interactive_pin: InteractivePinState | None = None
    dynamic_collider_cache: CrossClothColliderCache = field(default_factory=CrossClothColliderCache)


@dataclass
class DynamicCollisionSource:
    object_name: str
    target_names: set[str] = field(default_factory=set)
    current_positions_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    previous_positions_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    local_positions_buffer: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    triangle_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int32))
    topology_signature: tuple = ()
    collision_layer: int = 0
    external_contact_distance: float = 0.02
    max_edge_length: float = 0.08
    dynamic_collider_cache: CrossClothColliderCache = field(default_factory=CrossClothColliderCache)


@dataclass
class SceneSession:
    scene_name: str
    object_name: str
    slots: dict[str, ClothSlot]
    solve_order: list[str]
    frame_index: int
    frame_count: int
    start_frame: int
    substeps: int
    iterations: int
    writeback_interval: int
    cross_cloth_mode: str
    last_fps_time: float
    fps_sample_frames: int
    actual_fps: float
    adaptive_writeback_interval: int = 1
    frame_ms_ewma: float = 0.0
    writeback_ms_ewma: float = 0.0
    target_fps: float = 30.0
    last_diagnostics: NativeStepDiagnostics = field(default_factory=NativeStepDiagnostics)
    stop_requested: bool = False
    closed: bool = False
    playback_driven: bool = False
    paused: bool = False
    last_scene_frame: int = 0
    dynamic_collision_sources: dict[str, DynamicCollisionSource] = field(default_factory=dict)
    global_dynamic_scene: NativeGlobalDynamicScene | None = None
    global_dynamic_source_ids: dict[str, int] = field(default_factory=dict)
    global_dynamic_scene_enabled: bool = False

    @property
    def cloth(self) -> ClothBuildData:
        return self.slots[self.object_name].cloth

    @property
    def native(self) -> NativeXpbdSolver:
        return self.slots[self.object_name].native


def backend_status_text() -> str:
    info = native_status()
    return info.message


def session_status(obj: Optional[bpy.types.Object]) -> str:
    if obj is None:
        return STATUS_IDLE
    session = _session_for_object_name(obj.name)
    if session is not None:
        if bool(getattr(session, "paused", False)):
            return STATUS_PREVIEW_PAUSED
        return STATUS_PREVIEW_RUNNING
    return _STATUS.get(obj.name, STATUS_IDLE)


def session_fps(obj: Optional[bpy.types.Object]) -> float:
    if obj is None:
        return 0.0
    session = _session_for_object_name(obj.name)
    if session is None:
        return 0.0
    return float(session.actual_fps)


def session_diagnostics(obj: Optional[bpy.types.Object]) -> NativeStepDiagnostics:
    if obj is None:
        return NativeStepDiagnostics()
    session = _session_for_object_name(obj.name)
    if session is not None:
        return session.last_diagnostics
    return _LAST_DIAGNOSTICS.get(obj.name, NativeStepDiagnostics())


def record_viewport_tag_ms(object_name: str, elapsed_ms: float) -> None:
    session = _session_for_object_name(object_name)
    if session is None:
        return
    diag = session.last_diagnostics
    session.last_diagnostics = NativeStepDiagnostics(
        step_ms=diag.step_ms,
        hash_build_ms=diag.hash_build_ms,
        constraints_ms=diag.constraints_ms,
        volume_ms=diag.volume_ms,
        analytic_collision_ms=diag.analytic_collision_ms,
        static_collision_ms=diag.static_collision_ms,
        dynamic_collision_ms=diag.dynamic_collision_ms,
        dynamic_particle_collision_ms=diag.dynamic_particle_collision_ms,
        self_hash_ms=diag.self_hash_ms,
        self_solve_ms=diag.self_solve_ms,
        self_probe_ms=diag.self_probe_ms,
        self_recovery_ms=diag.self_recovery_ms,
        sync_ms=diag.sync_ms,
        diagnostics_fetch_ms=diag.diagnostics_fetch_ms,
        candidate_count=diag.candidate_count,
        resolved_contacts=diag.resolved_contacts,
        min_gap=diag.min_gap,
        ccd_clamp_count=diag.ccd_clamp_count,
        recovery_passes=diag.recovery_passes,
        local_retry_count=diag.local_retry_count,
        jitter_stabilized_vertices=diag.jitter_stabilized_vertices,
        jitter_rejected_vertices=diag.jitter_rejected_vertices,
        jitter_max_correction=diag.jitter_max_correction,
        external_contact_cache_hits=diag.external_contact_cache_hits,
        external_contact_cache_misses=diag.external_contact_cache_misses,
        external_contact_cache_count=diag.external_contact_cache_count,
        external_contact_cache_overflow=diag.external_contact_cache_overflow,
        external_friction_corrections=diag.external_friction_corrections,
        force_field_count=diag.force_field_count,
        unsupported_force_field_count=diag.unsupported_force_field_count,
        dynamic_particle_count=diag.dynamic_particle_count,
        dynamic_particle_candidate_count=diag.dynamic_particle_candidate_count,
        dynamic_particle_contacts=diag.dynamic_particle_contacts,
        dynamic_particle_overflow=diag.dynamic_particle_overflow,
        dynamic_triangle_count=diag.dynamic_triangle_count,
        dynamic_triangle_candidate_count=diag.dynamic_triangle_candidate_count,
        dynamic_triangle_bucket_overflow=diag.dynamic_triangle_bucket_overflow,
        dynamic_triangle_large_primitive_count=diag.dynamic_triangle_large_primitive_count,
        dynamic_triangle_aabb_reject_count=diag.dynamic_triangle_aabb_reject_count,
        dynamic_triangle_max_bucket_occupancy=diag.dynamic_triangle_max_bucket_occupancy,
        global_dynamic_scene_pack_ms=diag.global_dynamic_scene_pack_ms,
        global_dynamic_scene_upload_ms=diag.global_dynamic_scene_upload_ms,
        global_dynamic_hash_ms=diag.global_dynamic_hash_ms,
        global_dynamic_particle_count=diag.global_dynamic_particle_count,
        global_dynamic_triangle_count=diag.global_dynamic_triangle_count,
        global_dynamic_hash_overflow=diag.global_dynamic_hash_overflow,
        static_triangle_count=diag.static_triangle_count,
        finite=diag.finite,
        fast_exact_vt_candidates=diag.fast_exact_vt_candidates,
        fast_exact_vt_projected=diag.fast_exact_vt_projected,
        fast_exact_vt_guarded=diag.fast_exact_vt_guarded,
        fast_exact_vt_skipped_rest=diag.fast_exact_vt_skipped_rest,
        fast_soft_repulsion_candidates=diag.fast_soft_repulsion_candidates,
        fast_soft_repulsion_applied=diag.fast_soft_repulsion_applied,
        fast_soft_repulsion_max_push=diag.fast_soft_repulsion_max_push,
        fast_hard_projection_count=diag.fast_hard_projection_count,
        fast_manifold_contacts=diag.fast_manifold_contacts,
        fast_manifold_reused=diag.fast_manifold_reused,
        fast_barrier_projected=diag.fast_barrier_projected,
        fast_barrier_smoothed_vertices=diag.fast_barrier_smoothed_vertices,
        fast_barrier_overflow=diag.fast_barrier_overflow,
        fast_barrier_max_delta=diag.fast_barrier_max_delta,
        fast_edge_edge_candidates=diag.fast_edge_edge_candidates,
        fast_edge_edge_contacts=diag.fast_edge_edge_contacts,
        fast_triangle_pair_candidates=diag.fast_triangle_pair_candidates,
        fast_triangle_pair_contacts=diag.fast_triangle_pair_contacts,
        fast_triangle_pair_skipped_rest=diag.fast_triangle_pair_skipped_rest,
        fast_contact_classification_guarded=diag.fast_contact_classification_guarded,
        fast_region_cluster_candidates=diag.fast_region_cluster_candidates,
        fast_region_cluster_contacts=diag.fast_region_cluster_contacts,
        fast_region_cluster_guarded=diag.fast_region_cluster_guarded,
        fast_overlap_island_candidates=diag.fast_overlap_island_candidates,
        fast_overlap_island_clusters=diag.fast_overlap_island_clusters,
        fast_overlap_island_vertex_refs=diag.fast_overlap_island_vertex_refs,
        fast_overlap_island_applied_vertices=diag.fast_overlap_island_applied_vertices,
        fast_overlap_island_guarded=diag.fast_overlap_island_guarded,
        fast_overlap_island_max_delta=diag.fast_overlap_island_max_delta,
        fast_cc_overlap_components=diag.fast_cc_overlap_components,
        fast_cc_overlap_seed_triangles=diag.fast_cc_overlap_seed_triangles,
        fast_cc_overlap_owned_vertices=diag.fast_cc_overlap_owned_vertices,
        fast_cc_overlap_union_edges=diag.fast_cc_overlap_union_edges,
        fast_cc_overlap_guarded=diag.fast_cc_overlap_guarded,
        fast_cc_overlap_applied_vertices=diag.fast_cc_overlap_applied_vertices,
        fast_cc_overlap_max_delta=diag.fast_cc_overlap_max_delta,
        abi41_soft_contact_count=diag.abi41_soft_contact_count,
        abi41_exact_impulse_contact_count=diag.abi41_exact_impulse_contact_count,
        abi41_edge_edge_contact_count=diag.abi41_edge_edge_contact_count,
        abi41_max_smoothed_delta=diag.abi41_max_smoothed_delta,
        abi41_hard_projection_fallbacks=diag.abi41_hard_projection_fallbacks,
        static_sdf_rebuild_count=diag.static_sdf_rebuild_count,
        static_sdf_voxel_count=diag.static_sdf_voxel_count,
        static_sdf_grid_x=diag.static_sdf_grid_x,
        static_sdf_grid_y=diag.static_sdf_grid_y,
        static_sdf_grid_z=diag.static_sdf_grid_z,
        static_sdf_build_ms=diag.static_sdf_build_ms,
        static_sdf_contact_count=diag.static_sdf_contact_count,
        static_sdf_unsigned_fallback_count=diag.static_sdf_unsigned_fallback_count,
        abi41_pcg_iterations=diag.abi41_pcg_iterations,
        abi41_pcg_guarded=diag.abi41_pcg_guarded,
        abi41_pcg_csr_nnz=diag.abi41_pcg_csr_nnz,
        abi41_pcg_texture_ready=diag.abi41_pcg_texture_ready,
        abi41_pcg_initial_residual=diag.abi41_pcg_initial_residual,
        abi41_pcg_final_residual=diag.abi41_pcg_final_residual,
        abi41_pcg_max_delta=diag.abi41_pcg_max_delta,
        abi41_pcg_solve_ms=diag.abi41_pcg_solve_ms,
        abi41_pcg_system_ms=diag.abi41_pcg_system_ms,
        abi41_pcg_ad_ms=diag.abi41_pcg_ad_ms,
        abi41_direct_stretch_ms=diag.abi41_direct_stretch_ms,
        abi41_lra_tack_count=diag.abi41_lra_tack_count,
        abi41_bending_wing_count=diag.abi41_bending_wing_count,
        abi41_bending_texture_ready=diag.abi41_bending_texture_ready,
        abi41_tack_jitter_guarded=diag.abi41_tack_jitter_guarded,
        abi41_bending_guarded=diag.abi41_bending_guarded,
        dynamic_collider_pack_ms=diag.dynamic_collider_pack_ms,
        dynamic_triangle_upload_ms=diag.dynamic_triangle_upload_ms,
        dynamic_particle_upload_ms=diag.dynamic_particle_upload_ms,
        dynamic_collider_cache_hits=diag.dynamic_collider_cache_hits,
        dynamic_collider_cache_misses=diag.dynamic_collider_cache_misses,
        dynamic_pair_cache_hits=diag.dynamic_pair_cache_hits,
        dynamic_pair_cache_misses=diag.dynamic_pair_cache_misses,
        dynamic_pair_cache_reused_triangles=diag.dynamic_pair_cache_reused_triangles,
        dynamic_pair_cache_reused_particles=diag.dynamic_pair_cache_reused_particles,
        dynamic_collision_skipped_launches=diag.dynamic_collision_skipped_launches,
        self_collision_skipped_launches=diag.self_collision_skipped_launches,
        self_candidate_count=diag.self_candidate_count,
        self_filter_seen=diag.self_filter_seen,
        self_filter_accepted_vv=diag.self_filter_accepted_vv,
        self_filter_accepted_vt=diag.self_filter_accepted_vt,
        self_filter_accepted_ee=diag.self_filter_accepted_ee,
        self_filter_rejected_rest=diag.self_filter_rejected_rest,
        self_filter_rejected_duplicate=diag.self_filter_rejected_duplicate,
        self_filter_rejected_ownership=diag.self_filter_rejected_ownership,
        self_filter_cache_hits=diag.self_filter_cache_hits,
        self_filter_cache_misses=diag.self_filter_cache_misses,
        self_cluster_count=diag.self_cluster_count,
        self_cluster_owned_contacts=diag.self_cluster_owned_contacts,
        frame_ms=diag.frame_ms,
        frame_set_ms=diag.frame_set_ms,
        input_refresh_ms=diag.input_refresh_ms,
        pin_upload_ms=diag.pin_upload_ms,
        runtime_upload_ms=diag.runtime_upload_ms,
        static_upload_ms=diag.static_upload_ms,
        dynamic_upload_ms=diag.dynamic_upload_ms,
        cuda_step_call_ms=diag.cuda_step_call_ms,
        download_ms=diag.download_ms,
        writeback_ms=diag.writeback_ms,
        writeback_to_local_ms=diag.writeback_to_local_ms,
        writeback_foreach_set_ms=diag.writeback_foreach_set_ms,
        writeback_mesh_update_ms=diag.writeback_mesh_update_ms,
        frame_input_upload_ms=diag.frame_input_upload_ms,
        writeback_performed=diag.writeback_performed,
        diagnostics_ms=diag.diagnostics_ms,
        viewport_tag_ms=float(elapsed_ms),
    )
    for slot_name in session.solve_order:
        _LAST_DIAGNOSTICS[slot_name] = session.last_diagnostics
    for source_name in session.dynamic_collision_sources:
        _LAST_DIAGNOSTICS[source_name] = session.last_diagnostics


def has_session(obj: Optional[bpy.types.Object]) -> bool:
    return obj is not None and _session_for_object_name(obj.name) is not None


def session_object_names(scene: bpy.types.Scene | None = None) -> list[str]:
    if scene is not None:
        session = _SCENE_SESSIONS.get(_scene_key(scene))
        if session is None or session.closed or session.paused:
            return []
        return list(session.solve_order)
    names: list[str] = []
    for session in _SCENE_SESSIONS.values():
        if session.closed or session.paused:
            continue
        names.extend(session.solve_order)
    return names


def begin_interactive_pin(
    context: bpy.types.Context,
    object_name: str,
    vertex_index: int,
    world_position,
) -> bpy.types.Object | None:
    session = _session_for_object_name(object_name)
    if session is None or session.closed or session.paused:
        return None
    slot = session.slots.get(object_name)
    obj = bpy.data.objects.get(object_name)
    if slot is None or obj is None or obj.type != "MESH" or not _rna_alive(slot.original_mesh):
        return None
    vertex_index = int(vertex_index)
    if vertex_index < 0 or vertex_index >= len(slot.original_mesh.vertices):
        return None

    _clear_interactive_pin_state(slot, obj, restore_selection=False)
    settings = _settings_for_slot(context, slot)
    pin_group_name = str(getattr(settings, "pin_vertex_group", "")).strip()
    if not pin_group_name:
        pin_group_name = "ssbl_pin"
        settings.pin_vertex_group = pin_group_name

    previous_active = getattr(context.view_layer.objects, "active", None)
    previous_selection = [
        selected.name
        for selected in getattr(context, "selected_objects", [])
        if selected is not None and _rna_alive(selected)
    ]
    handle = _make_interactive_pin_empty(context, obj, vertex_index, Vector(world_position))
    state: InteractivePinState | None = None
    previous_preview_weight = (
        _vertex_group_weight_on_mesh(obj, slot.preview_mesh, pin_group_name, vertex_index)
        if _rna_alive(slot.preview_mesh) and not _same_mesh(slot.preview_mesh, slot.original_mesh)
        else None
    )
    with _with_preview_source_state(slot, obj):
        pin_group = obj.vertex_groups.get(pin_group_name) or obj.vertex_groups.new(name=pin_group_name)
        previous_weight = _vertex_group_weight(pin_group, vertex_index)
        pin_group.add([vertex_index], 1.0, "ADD")

        hook_group_name = _unique_vertex_group_name(obj, f"{_INTERACTIVE_PIN_PREFIX}_Hook_{vertex_index}")
        hook_group = obj.vertex_groups.new(name=hook_group_name)
        hook_group.add([vertex_index], 1.0, "REPLACE")

        hook_modifier_name = _unique_modifier_name(obj, f"{_INTERACTIVE_PIN_PREFIX}_Hook")
        modifier = obj.modifiers.new(hook_modifier_name, "HOOK")
        modifier.object = handle
        modifier.vertex_group = hook_group_name
        modifier.matrix_inverse = handle.matrix_world.inverted()
        modifier.strength = 1.0
        modifier.show_viewport = True
        modifier.show_render = True
        slot.suspended_modifiers.append((modifier.name, True, True))
        state = InteractivePinState(
            object_name=obj.name,
            vertex_index=vertex_index,
            pin_group_name=pin_group_name,
            hook_group_name=hook_group_name,
            hook_modifier_name=modifier.name,
            empty_name=handle.name,
            previous_weight=previous_weight,
            previous_preview_weight=previous_preview_weight,
            previous_active_name=previous_active.name if previous_active is not None and _rna_alive(previous_active) else "",
            previous_selection_names=previous_selection,
            previous_use_evaluated_mesh=bool(slot.use_evaluated_mesh),
        )
        slot.interactive_pin = state
        slot.use_evaluated_mesh = True
        _refresh_slot_pin_weights_from_group(obj, slot, settings, force=True)

    if _rna_alive(slot.preview_mesh) and not _same_mesh(slot.preview_mesh, slot.original_mesh):
        _add_vertex_group_weight_on_mesh(obj, slot.preview_mesh, pin_group_name, vertex_index, 1.0, "ADD")
        _add_vertex_group_weight_on_mesh(obj, slot.preview_mesh, hook_group_name, vertex_index, 1.0, "REPLACE")

    _select_interactive_pin_handle(context, handle)
    slot.force_next_writeback = True
    return handle


def move_interactive_pin(object_name: str, world_position) -> bool:
    session = _session_for_object_name(object_name)
    if session is None:
        return False
    slot = session.slots.get(object_name)
    state = slot.interactive_pin if slot is not None else None
    if state is None:
        return False
    handle = bpy.data.objects.get(state.empty_name)
    if handle is None:
        return False
    handle.location = Vector(world_position)
    slot.force_next_writeback = True
    return True


def end_interactive_pin(object_name: str, *, restore_selection: bool = True) -> bool:
    session = _session_for_object_name(object_name)
    if session is None:
        return False
    slot = session.slots.get(object_name)
    obj = bpy.data.objects.get(object_name)
    if slot is None or obj is None:
        return False
    return _clear_interactive_pin_state(slot, obj, restore_selection=restore_selection)


def cleanup_interactive_pins(scene: bpy.types.Scene | None = None) -> None:
    sessions = [_SCENE_SESSIONS.get(_scene_key(scene))] if scene is not None else list(_SCENE_SESSIONS.values())
    for session in sessions:
        if session is None:
            continue
        for slot in session.slots.values():
            obj = bpy.data.objects.get(slot.object_name)
            if obj is not None:
                _clear_interactive_pin_state(slot, obj)


def preview_warnings(obj: bpy.types.Object, settings) -> list[str]:
    warnings: list[str] = []
    if obj is None or obj.type != "MESH":
        return warnings
    scene = bpy.context.scene
    if len(obj.data.polygons) > 10000:
        warnings.append("Large meshes may need optimized preview settings for stable realtime playback.")
    if bool(getattr(settings, "use_ground", False)):
        bbox_min_z = min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box)
        ground_limit = float(getattr(settings, "ground_height", 0.0)) + float(getattr(settings, "collision_margin", 0.0))
        if bbox_min_z < ground_limit - 1.0e-4:
            warnings.append("The mesh starts below the ground collision plane; expect an upward correction on the first frame.")
    auto_sphere_names = {
        item.name for item in _auto_sphere_collider_candidates(bpy.context, obj, settings)
    }
    if auto_sphere_names:
        names = ", ".join(sorted(auto_sphere_names)[:3])
        warnings.append(f"{names} will be treated as an analytic sphere collider during preview.")
    selected_sphere_cloths = [
        selected_obj.name
        for selected_obj in bpy.context.selected_objects
        if selected_obj is not None
        and selected_obj != obj
        and selected_obj.type == "MESH"
        and selected_obj.name not in auto_sphere_names
        and _object_cloth_settings(selected_obj) is not None
        and not _is_collision_only_object(scene, selected_obj, active_obj=obj, active_settings=settings)
        and _mesh_is_probably_sphere_like(selected_obj)
    ]
    if selected_sphere_cloths:
        names = ", ".join(selected_sphere_cloths[:3])
        warnings.append(
            f"{names} is cloth-enabled and selected, so preview treats it as another cloth object, not a cheap sphere collider."
        )
    return warnings


def request_stop(obj: bpy.types.Object) -> bool:
    session = _session_for_object_name(obj.name if obj else "")
    if session is None:
        return False
    _finish_session(session, STATUS_PREVIEW_STOPPED, finalize_realtime_cache=True)
    return True


def reset_preview_object(obj: bpy.types.Object) -> bool:
    session = _session_for_object_name(obj.name if obj else "")
    if session is None:
        return False
    _finish_session(session, STATUS_IDLE, finalize_realtime_cache=True)
    return True


def reset_timeline_preview_if_endpoint(scene: bpy.types.Scene | None = None) -> bool:
    scene = scene or bpy.context.scene
    if scene is None:
        return False
    current_frame = int(scene.frame_current)
    start_frame = int(scene.frame_start)
    end_frame = int(scene.frame_end)
    if current_frame != start_frame and current_frame != end_frame:
        return False
    session = _SCENE_SESSIONS.get(_scene_key(scene))
    if session is None or not bool(session.playback_driven):
        return False
    _finish_session(session, STATUS_IDLE, finalize_realtime_cache=_session_has_realtime_cache(session))
    return True


def cleanup_all_sessions() -> None:
    for session in list(_SCENE_SESSIONS.values()):
        _finish_session(session, STATUS_IDLE)


def clear_startup_build_caches() -> None:
    clear_cloth_topology_cache()
    clear_static_collision_cache()


def _object_cloth_settings(obj: bpy.types.Object | None):
    if obj is None or not hasattr(obj, "ssbl_cloth"):
        return None
    settings = obj.ssbl_cloth
    if bool(getattr(settings, "enabled", False)):
        return settings
    return None


def _settings_for_object(
    context: bpy.types.Context,
    obj: bpy.types.Object | None,
    fallback=None,
    require_enabled: bool = False,
):
    object_settings = _object_cloth_settings(obj)
    if object_settings is not None:
        return object_settings
    if require_enabled:
        return None
    return fallback if fallback is not None else context.scene.ssbl_preview


def _settings_for_slot(context: bpy.types.Context, slot: ClothSlot):
    obj = bpy.data.objects.get(slot.object_name)
    if bool(slot.use_object_settings):
        object_settings = _object_cloth_settings(obj)
        if object_settings is not None:
            return object_settings
    return context.scene.ssbl_preview


def _iter_collection_mesh_objects(collection: bpy.types.Collection | None):
    if collection is None:
        return
    try:
        objects = collection.all_objects
    except AttributeError:
        objects = collection.objects
    for item in objects:
        if item is not None and item.type == "MESH":
            yield item


def _collection_contains_object(collection: bpy.types.Collection | None, obj: bpy.types.Object | None) -> bool:
    if collection is None or obj is None:
        return False
    for item in _iter_collection_mesh_objects(collection):
        if item == obj or item.name == obj.name:
            return True
    return False


def _settings_reference_collision_object(settings, candidate: bpy.types.Object, owner: bpy.types.Object | None = None) -> bool:
    if settings is None or candidate is None:
        return False
    if owner is not None and (candidate == owner or candidate.name == owner.name):
        return False
    sphere_obj = getattr(settings, "sphere_object", None)
    if bool(getattr(settings, "use_sphere", False)) and sphere_obj is not None:
        if sphere_obj == candidate or sphere_obj.name == candidate.name:
            return True
    static_collection = getattr(settings, "static_collider_collection", None)
    if _collection_contains_object(static_collection, candidate):
        return True
    dynamic_collection = getattr(settings, "dynamic_collider_collection", None)
    return _collection_contains_object(dynamic_collection, candidate)


def _has_free_simulation_vertices(obj: bpy.types.Object | None, settings) -> bool:
    if obj is None or obj.type != "MESH" or settings is None:
        return False
    mesh = getattr(obj, "data", None)
    vertex_count = int(len(mesh.vertices)) if mesh is not None else 0
    if vertex_count <= 0:
        return False
    group_name = str(getattr(settings, "pin_vertex_group", "") or "").strip()
    group = obj.vertex_groups.get(group_name) if group_name else None
    signature = (
        int(id(mesh)),
        int(vertex_count),
        group_name,
        int(group.index) if group is not None else -1,
        float(getattr(settings, "pin_hardness", 1.0)),
    )
    cached = _FREE_SIM_VERTEX_CACHE.get(obj.name)
    if cached is not None and cached[0] == signature:
        return bool(cached[1])
    weights = effective_pin_weights_from_settings(obj, settings, vertex_count)
    if len(weights) != vertex_count:
        result = True
    else:
        result = not bool(np.all(weights >= PIN_HARD_WEIGHT_THRESHOLD))
    _FREE_SIM_VERTEX_CACHE[obj.name] = (signature, result)
    return result


def _invalidate_free_simulation_vertex_cache(obj: bpy.types.Object | None) -> None:
    if obj is not None:
        _FREE_SIM_VERTEX_CACHE.pop(obj.name, None)


def _is_enabled_simulatable_cloth_object(obj: bpy.types.Object | None) -> bool:
    settings = _object_cloth_settings(obj)
    return bool(
        settings is not None
        and _has_simulatable_cloth_source_geometry(obj)
        and _declared_input_type(obj) not in _UNSUPPORTED_INPUT_TYPES
        and _has_free_simulation_vertices(obj, settings)
    )


def _collision_only_object_names_for_scene(scene: bpy.types.Scene) -> set[str]:
    collision_only: set[str] = set()
    if scene is None:
        return collision_only
    scene_settings = getattr(scene, "ssbl_preview", None)
    if scene_settings is not None:
        for candidate in scene.objects:
            if _is_enabled_simulatable_cloth_object(candidate):
                continue
            if candidate is not None and candidate.type == "MESH" and _settings_reference_collision_object(scene_settings, candidate):
                collision_only.add(candidate.name)
    for owner in scene.objects:
        owner_settings = _object_cloth_settings(owner)
        if owner_settings is None:
            continue
        for candidate in scene.objects:
            if candidate is None or candidate.type != "MESH":
                continue
            if _is_enabled_simulatable_cloth_object(candidate):
                continue
            if _settings_reference_collision_object(owner_settings, candidate, owner):
                collision_only.add(candidate.name)
    return collision_only


def _is_collision_only_object(
    scene: bpy.types.Scene,
    candidate: bpy.types.Object,
    active_obj: bpy.types.Object | None = None,
    active_settings=None,
) -> bool:
    if candidate is None:
        return False
    if active_obj is not None and (candidate == active_obj or candidate.name == active_obj.name):
        return False
    if _settings_reference_collision_object(active_settings, candidate, active_obj):
        return True
    scene_settings = getattr(scene, "ssbl_preview", None)
    if _settings_reference_collision_object(scene_settings, candidate, active_obj):
        return True
    return candidate.name in _collision_only_object_names_for_scene(scene)


def _auto_sphere_collider_candidates(
    context: bpy.types.Context,
    active_obj: bpy.types.Object | None,
    active_settings=None,
) -> list[bpy.types.Object]:
    if context is None:
        return []
    explicit_sphere = getattr(active_settings, "sphere_object", None) if active_settings is not None else None
    if bool(getattr(active_settings, "use_sphere", False)) and explicit_sphere is not None:
        return []
    candidates: list[bpy.types.Object] = []
    for selected_obj in context.selected_objects:
        if selected_obj is None or selected_obj.type != "MESH":
            continue
        if active_obj is not None and (selected_obj == active_obj or selected_obj.name == active_obj.name):
            continue
        if _declared_input_type(selected_obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        if _mesh_is_probably_sphere_like(selected_obj):
            candidates.append(selected_obj)
    return sorted(candidates, key=lambda item: (len(item.data.polygons), item.name.casefold()))


def _auto_sphere_collider_for_preview(
    context: bpy.types.Context,
    active_obj: bpy.types.Object | None,
    active_settings=None,
) -> bpy.types.Object | None:
    candidates = _auto_sphere_collider_candidates(context, active_obj, active_settings)
    return candidates[0] if candidates else None


def _has_simulatable_cloth_source_geometry(obj: bpy.types.Object | None) -> bool:
    mesh = getattr(obj, "data", None) if obj is not None and obj.type == "MESH" else None
    return bool(mesh is not None and len(mesh.vertices) > 0 and len(mesh.polygons) > 0)


def _enabled_playback_cloth_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    collision_only_names = _collision_only_object_names_for_scene(scene)
    for obj in scene.objects:
        if obj is None or obj.type != "MESH":
            continue
        if obj.name in collision_only_names:
            continue
        settings = _object_cloth_settings(obj)
        if settings is None:
            continue
        if not _has_simulatable_cloth_source_geometry(obj):
            continue
        if _declared_input_type(obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        if not _has_free_simulation_vertices(obj, settings):
            _STATUS[obj.name] = STATUS_NO_SIMULATED_VERTICES
            continue
        objects.append(obj)
    return sorted(objects, key=lambda item: (int(getattr(item, _OBJECT_COLLISION_LAYER_PROP, item.get(_OBJECT_COLLISION_LAYER_PROP, 1))), item.name.casefold()))


def _auto_cross_cloth_mode(slot_count: int) -> str:
    return "all_selected" if int(slot_count) > 1 else "off"


def _cross_cloth_mode_from_settings(settings, slot_count: int) -> str:
    if int(slot_count) <= 1:
        return "off"
    configured = str(getattr(settings, "cross_cloth_collision", "off") or "off").lower()
    if configured in {"all_selected", "lower_layers"}:
        return configured
    return _auto_cross_cloth_mode(slot_count)


def start_preview(context: bpy.types.Context, obj: bpy.types.Object) -> SceneSession:
    try:
        if context.mode != "OBJECT":
            raise ValueError("Preview must be started in Object mode.")
        settings = _settings_for_object(context, obj, context.scene.ssbl_preview)
        auto_sphere_obj = _auto_sphere_collider_for_preview(context, obj, settings)
        cloth_objects = _preview_cloth_objects(context, obj, settings)
        if not cloth_objects:
            raise ValueError("No simulated SSBL cloth vertices found. Put animated character meshes in Dynamic Collider Collection instead of enabling them as cloth.")
        for cloth_obj in cloth_objects:
            _ensure_supported_cloth_object(cloth_obj)

        for existing in _sessions_for_objects(cloth_objects):
            _finish_session(existing, STATUS_IDLE)
        scene_key = _scene_key(context.scene)
        existing_scene_session = _SCENE_SESSIONS.get(scene_key)
        if existing_scene_session is not None:
            _finish_session(existing_scene_session, STATUS_IDLE)

        slots: dict[str, ClothSlot] = {}
        depsgraph = context.evaluated_depsgraph_get()
        preview_slot_count = len(cloth_objects)
        for cloth_obj in cloth_objects:
            slot_settings = _settings_for_object(context, cloth_obj, settings)
            slot = _create_cloth_slot(
                context,
                cloth_obj,
                slot_settings,
                depsgraph,
                auto_sphere_obj,
                preview_slot_count=preview_slot_count,
            )
            slots[slot.object_name] = slot

        solve_order = sorted(
            slots.keys(),
            key=lambda name: (slots[name].collision_layer, name.casefold()),
        )
        active_name = obj.name if obj and obj.name in slots else solve_order[0]
        session = SceneSession(
            scene_name=context.scene.name,
            object_name=active_name,
            slots=slots,
            solve_order=solve_order,
            frame_index=0,
            frame_count=max(int(settings.frame_count), 1),
            start_frame=int(context.scene.frame_current),
            substeps=max(int(settings.substeps), 1),
            iterations=max(int(settings.iterations), 1),
            writeback_interval=_preview_writeback_interval(settings),
            cross_cloth_mode=_cross_cloth_mode_from_settings(settings, len(slots)),
            last_fps_time=time.perf_counter(),
            fps_sample_frames=0,
            actual_fps=0.0,
            target_fps=_target_fps_from_settings(settings),
            last_scene_frame=int(context.scene.frame_current),
        )
        _SCENE_SESSIONS[scene_key] = session
        slot_settings = {name: _settings_for_slot(context, slot) for name, slot in slots.items()}
        _sync_session_dynamic_collision_sources(context, session, depsgraph, slot_settings)
        for name, slot in slots.items():
            _OBJECT_TO_SCENE_SESSION[name] = scene_key
            _STATUS[name] = STATUS_PREVIEW_RUNNING
            _apply_world_positions(
                bpy.data.objects[name],
                slot.current_positions_world,
                slot.cloth.matrix_world_inv,
                local_buffer=slot.writeback_local_buffer,
                flat_buffer=slot.writeback_flat_buffer,
            )
        try:
            _begin_realtime_cache_for_session(session)
        except Exception:
            _finish_session(session, STATUS_ERROR, finalize_realtime_cache=False)
            raise
        return session
    except Exception:
        if obj is not None and _STATUS.get(obj.name) != STATUS_NO_SIMULATED_VERTICES:
            _STATUS[obj.name] = STATUS_ERROR
        raise


def start_timeline_preview(context: bpy.types.Context, scene: bpy.types.Scene | None = None) -> SceneSession | None:
    scene = scene or context.scene
    cloth_objects = _enabled_playback_cloth_objects(scene)
    if not cloth_objects:
        return None

    scene_key = _scene_key(scene)
    existing = _SCENE_SESSIONS.get(scene_key)
    enabled_names = {obj.name for obj in cloth_objects}
    if existing is not None and bool(existing.playback_driven) and set(existing.solve_order) == enabled_names:
        current_frame = int(scene.frame_current)
        if existing.frame_index == 0 or current_frame >= int(existing.last_scene_frame):
            existing.paused = False
            for name in existing.solve_order:
                _STATUS[name] = STATUS_PREVIEW_RUNNING
            return existing
        _finish_session(existing, STATUS_IDLE)
    elif existing is not None:
        _finish_session(existing, STATUS_IDLE)

    slots: dict[str, ClothSlot] = {}
    depsgraph = context.evaluated_depsgraph_get()
    preview_slot_count = len(cloth_objects)
    for cloth_obj in cloth_objects:
        _ensure_supported_cloth_object(cloth_obj)
        settings = _settings_for_object(context, cloth_obj, require_enabled=True)
        if settings is None:
            continue
        slot = _create_cloth_slot(context, cloth_obj, settings, depsgraph, preview_slot_count=preview_slot_count)
        slots[slot.object_name] = slot
    if not slots:
        return None

    solve_order = sorted(
        slots.keys(),
        key=lambda name: (slots[name].collision_layer, name.casefold()),
    )
    active_name = solve_order[0]
    session = SceneSession(
        scene_name=scene.name,
        object_name=active_name,
        slots=slots,
        solve_order=solve_order,
        frame_index=0,
        frame_count=max(int(scene.frame_end) - int(scene.frame_current) + 1, 1),
        start_frame=int(scene.frame_current),
        substeps=max(slot.substeps for slot in slots.values()),
        iterations=max(slot.iterations for slot in slots.values()),
        writeback_interval=min(slot.writeback_interval for slot in slots.values()),
        cross_cloth_mode=_cross_cloth_mode_from_settings(scene.ssbl_preview, len(slots)),
        last_fps_time=time.perf_counter(),
        fps_sample_frames=0,
        actual_fps=0.0,
        target_fps=_target_fps_from_settings(scene.ssbl_preview),
        playback_driven=True,
        paused=False,
        last_scene_frame=int(scene.frame_current),
    )
    _SCENE_SESSIONS[scene_key] = session
    slot_settings = {name: _settings_for_slot(context, slot) for name, slot in slots.items()}
    _sync_session_dynamic_collision_sources(context, session, depsgraph, slot_settings)
    for name, slot in slots.items():
        _OBJECT_TO_SCENE_SESSION[name] = scene_key
        _STATUS[name] = STATUS_PREVIEW_RUNNING
        _apply_world_positions(
            bpy.data.objects[name],
            slot.current_positions_world,
            slot.cloth.matrix_world_inv,
            local_buffer=slot.writeback_local_buffer,
            flat_buffer=slot.writeback_flat_buffer,
        )
    try:
        _begin_realtime_cache_for_session(session)
    except Exception:
        _finish_session(session, STATUS_ERROR, finalize_realtime_cache=False)
        raise
    return session


def pause_timeline_preview(scene: bpy.types.Scene | None = None) -> None:
    scene = scene or bpy.context.scene
    session = _SCENE_SESSIONS.get(_scene_key(scene))
    if session is None or not bool(session.playback_driven):
        return
    session.paused = True
    for name in session.solve_order:
        slot = session.slots.get(name)
        obj = bpy.data.objects.get(name)
        if slot is not None and obj is not None:
            _clear_interactive_pin_state(slot, obj)
        _STATUS[name] = STATUS_PREVIEW_PAUSED


def stop_timeline_preview(scene: bpy.types.Scene | None = None) -> bool:
    scene = scene or bpy.context.scene
    session = _SCENE_SESSIONS.get(_scene_key(scene))
    if session is None or not bool(session.playback_driven):
        return False
    if not _session_has_realtime_cache(session):
        return False
    _finish_session(session, STATUS_PREVIEW_STOPPED, finalize_realtime_cache=True)
    return True


def _effective_writeback_interval(session: SceneSession, slot: ClothSlot) -> int:
    if slot.writeback_interval > _AUTO_WRITEBACK_INTERVAL:
        return max(int(slot.writeback_interval), _MIN_WRITEBACK_INTERVAL)
    return max(int(session.adaptive_writeback_interval), _MIN_WRITEBACK_INTERVAL)


def _has_auto_writeback_slot(session: SceneSession) -> bool:
    return any(slot.writeback_interval <= _AUTO_WRITEBACK_INTERVAL for slot in session.slots.values())


def _writeback_decisions(session: SceneSession, next_frame: int, scene_end: int) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    for slot_name, slot in session.slots.items():
        interval = _effective_writeback_interval(session, slot)
        decisions[slot_name] = bool(
            session.frame_index == 0
            or bool(slot.force_next_writeback)
            or ((session.frame_index + 1) % interval) == 0
            or next_frame >= scene_end
        )
    return decisions


def _clear_forced_writeback_flags(session: SceneSession, decisions: dict[str, bool]) -> None:
    for slot_name, did_writeback in decisions.items():
        if did_writeback and slot_name in session.slots:
            session.slots[slot_name].force_next_writeback = False


def _session_has_realtime_cache(session: SceneSession | None) -> bool:
    if session is None:
        return False
    return any(slot.realtime_cache is not None for slot in session.slots.values())


def _realtime_cache_download_decisions(session: SceneSession, writeback_by_slot: dict[str, bool]) -> dict[str, bool]:
    decisions = dict(writeback_by_slot)
    for slot_name, slot in session.slots.items():
        if slot.realtime_cache is not None:
            decisions[slot_name] = True
    return decisions


def _update_adaptive_writeback_interval(session: SceneSession, perf: FramePerf) -> None:
    if not _has_auto_writeback_slot(session):
        return
    frame_ms = max(float(perf.frame_ms), 0.0)
    writeback_ms = max(float(perf.writeback_ms), 0.0)
    if frame_ms <= 0.0:
        return

    alpha = _WRITEBACK_EWMA_ALPHA
    if session.frame_ms_ewma <= 0.0:
        session.frame_ms_ewma = frame_ms
    else:
        session.frame_ms_ewma = session.frame_ms_ewma * (1.0 - alpha) + frame_ms * alpha
    if perf.writeback_performed:
        if session.writeback_ms_ewma <= 0.0:
            session.writeback_ms_ewma = writeback_ms
        else:
            session.writeback_ms_ewma = session.writeback_ms_ewma * (1.0 - alpha) + writeback_ms * alpha

    budget_ms = 1000.0 / max(float(session.target_fps), 1.0)
    interval = max(int(session.adaptive_writeback_interval), _MIN_WRITEBACK_INTERVAL)
    writeback_pressure = session.writeback_ms_ewma > max(0.25, budget_ms * 0.12)
    over_budget = session.frame_ms_ewma > budget_ms * 1.05
    comfortably_under = session.frame_ms_ewma < budget_ms * 0.72
    if over_budget and writeback_pressure and interval < _MAX_AUTO_WRITEBACK_INTERVAL:
        session.adaptive_writeback_interval = interval + 1
    elif comfortably_under and interval > _MIN_WRITEBACK_INTERVAL:
        session.adaptive_writeback_interval = interval - 1


def step_timeline_preview(context: bpy.types.Context, scene: bpy.types.Scene | None = None) -> bool:
    scene = scene or context.scene
    session = _SCENE_SESSIONS.get(_scene_key(scene))
    enabled_names = {obj.name for obj in _enabled_playback_cloth_objects(scene)}
    if not enabled_names:
        if session is not None and bool(session.playback_driven):
            _finish_session(session, STATUS_IDLE)
        return True
    if session is None or not bool(session.playback_driven) or set(session.solve_order) != enabled_names:
        session = start_timeline_preview(context, scene)
        if session is None:
            return True

    current_frame = int(scene.frame_current)
    if session.frame_index > 0 and (current_frame <= int(session.last_scene_frame) or current_frame - int(session.last_scene_frame) > 1):
        _finish_session(session, STATUS_IDLE)
        session = start_timeline_preview(context, scene)
        if session is None:
            return True
        return False

    step_started = time.perf_counter()
    perf = FramePerf()
    try:
        _refresh_session_runtime_inputs(context, session, perf)
        writeback_by_slot = _writeback_decisions(session, current_frame, scene_end=int(scene.frame_end))
        download_by_slot = _realtime_cache_download_decisions(session, writeback_by_slot)
        perf.writeback_performed = any(writeback_by_slot.values())
        _step_session_slots(session, download_by_slot, perf)
        _write_realtime_cache_samples(session)
        if perf.writeback_performed:
            started = time.perf_counter()
            for slot_name, should_writeback in writeback_by_slot.items():
                if not should_writeback:
                    continue
                slot = session.slots[slot_name]
                obj = bpy.data.objects.get(slot.object_name)
                if obj is None or obj.type != "MESH":
                    raise ValueError(f"Missing preview object during timeline writeback: {slot.object_name}")
                _apply_world_positions(
                    obj,
                    slot.current_positions_world,
                    slot.cloth.matrix_world_inv,
                    local_buffer=slot.writeback_local_buffer,
                    flat_buffer=slot.writeback_flat_buffer,
                    perf=perf,
                )
            perf.writeback_ms += _elapsed_ms(started)
            _clear_forced_writeback_flags(session, writeback_by_slot)
    except Exception:
        _finish_session(session, STATUS_ERROR)
        raise

    session.paused = False
    session.frame_index += 1
    session.last_scene_frame = current_frame
    perf.frame_ms = _elapsed_ms(step_started)
    session.last_diagnostics = _aggregate_session_diagnostics(session, perf)
    for slot_name in session.solve_order:
        _LAST_DIAGNOSTICS[slot_name] = session.last_diagnostics
        _STATUS[slot_name] = STATUS_PREVIEW_RUNNING
    for source_name in session.dynamic_collision_sources:
        _LAST_DIAGNOSTICS[source_name] = session.last_diagnostics
    _update_adaptive_writeback_interval(session, perf)
    _update_session_fps(session, step_started)
    return False


def step_preview(context: bpy.types.Context, object_name: str) -> bool:
    session = _session_for_object_name(object_name)
    if session is None:
        return True
    scene = bpy.data.scenes.get(session.scene_name)
    if scene is None:
        _finish_session(session, STATUS_ERROR)
        return True
    if session.stop_requested:
        _finish_session(session, STATUS_PREVIEW_STOPPED, finalize_realtime_cache=True)
        return True
    next_frame = session.start_frame + session.frame_index + 1
    if session.frame_index >= session.frame_count or next_frame > int(scene.frame_end):
        _finish_session(session, STATUS_FINISHED, finalize_realtime_cache=True)
        return True

    step_started = time.perf_counter()
    perf = FramePerf()
    try:
        started = time.perf_counter()
        scene.frame_set(next_frame)
        perf.frame_set_ms += _elapsed_ms(started)
        _refresh_session_runtime_inputs(context, session, perf)
        writeback_by_slot = _writeback_decisions(session, next_frame, scene_end=int(scene.frame_end))
        download_by_slot = _realtime_cache_download_decisions(session, writeback_by_slot)
        perf.writeback_performed = any(writeback_by_slot.values())
        _step_session_slots(session, download_by_slot, perf)
        _write_realtime_cache_samples(session)
        if perf.writeback_performed:
            started = time.perf_counter()
            for slot_name, should_writeback in writeback_by_slot.items():
                if not should_writeback:
                    continue
                slot = session.slots[slot_name]
                obj = bpy.data.objects.get(slot.object_name)
                if obj is None or obj.type != "MESH":
                    raise ValueError(f"Missing preview object during writeback: {slot.object_name}")
                _apply_world_positions(
                    obj,
                    slot.current_positions_world,
                    slot.cloth.matrix_world_inv,
                    local_buffer=slot.writeback_local_buffer,
                    flat_buffer=slot.writeback_flat_buffer,
                    perf=perf,
                )
            perf.writeback_ms += _elapsed_ms(started)
            _clear_forced_writeback_flags(session, writeback_by_slot)
    except Exception:
        _finish_session(session, STATUS_ERROR)
        raise

    session.frame_index += 1
    perf.frame_ms = _elapsed_ms(step_started)
    session.last_diagnostics = _aggregate_session_diagnostics(session, perf)
    for slot_name in session.solve_order:
        _LAST_DIAGNOSTICS[slot_name] = session.last_diagnostics
    for source_name in session.dynamic_collision_sources:
        _LAST_DIAGNOSTICS[source_name] = session.last_diagnostics
    _update_adaptive_writeback_interval(session, perf)
    _update_session_fps(session, step_started)
    return False


def bake_xpbd_cache(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    progress_callback: Callable[[int, int], None] | None = None,
) -> str:
    _ensure_supported_cloth_object(obj)

    settings = _settings_for_object(context, obj, context.scene.ssbl_preview)
    start = int(settings.bake_start)
    end = int(settings.bake_end)
    if end < start:
        raise ValueError("Bake end frame must be greater than or equal to bake start frame.")

    native = None
    sample_count = end - start + 1
    _STATUS[obj.name] = STATUS_BAKING
    _set_bake_progress_state(obj, 0, sample_count, in_progress=True)
    if progress_callback is not None:
        progress_callback(0, sample_count)
    original_frame = int(context.scene.frame_current)
    try:
        context.scene.frame_set(start)
        cloth, native, static_signature, _static_tris = _create_native_solver(
            context,
            obj,
            settings,
            runtime_mode_override="bake",
        )
        path = _cache_path_for_object(obj)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as handle:
            _write_pc2_header(handle, len(cloth.positions_world), start, sample_count)
            _write_pc2_sample(handle, cloth.positions_world, cloth.matrix_world_inv)
            _set_bake_progress_state(obj, 1, sample_count, in_progress=True)
            if progress_callback is not None:
                progress_callback(1, sample_count)
            for frame in range(start + 1, end + 1):
                _refresh_bake_runtime_inputs(context, obj, cloth, native, frame, static_signature, settings)
                native.step(max(int(settings.substeps), 1), max(int(settings.iterations), 1))
                world_positions = native.download_positions()
                _write_pc2_sample(handle, world_positions, cloth.matrix_world_inv)
                completed = frame - start + 1
                _set_bake_progress_state(obj, completed, sample_count, in_progress=True)
                if progress_callback is not None:
                    progress_callback(completed, sample_count)
        _bind_mesh_cache(obj, path, start)
        obj[_CACHE_PATH_PROP] = path
        _STATUS[obj.name] = STATUS_FINISHED
        return path
    except Exception:
        _STATUS[obj.name] = STATUS_ERROR
        raise
    finally:
        _set_bake_progress_state(obj, 0, 0, in_progress=False)
        context.scene.frame_set(original_frame)
        if native is not None:
            native.close()


def clear_xpbd_cache(obj: bpy.types.Object) -> bool:
    if obj is None or obj.type != "MESH":
        return False
    removed = False
    modifier = obj.modifiers.get(_CACHE_MODIFIER_NAME)
    if modifier is not None:
        obj.modifiers.remove(modifier)
        removed = True
    path = obj.get(_CACHE_PATH_PROP, "")
    if path and os.path.exists(path):
        os.remove(path)
        removed = True
    obj.pop(_CACHE_PATH_PROP, None)
    _STATUS[obj.name] = STATUS_IDLE
    return removed


def _scene_key(scene: bpy.types.Scene) -> str:
    return scene.name


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _rna_alive(value) -> bool:
    if value is None:
        return False
    try:
        value.as_pointer()
        _ = value.name
    except (ReferenceError, RuntimeError, AttributeError):
        return False
    return True


def _same_mesh(a: bpy.types.Mesh | None, b: bpy.types.Mesh | None) -> bool:
    if not _rna_alive(a) or not _rna_alive(b):
        return False
    try:
        return int(a.as_pointer()) == int(b.as_pointer())
    except (ReferenceError, RuntimeError, AttributeError):
        return False


def _safe_remove_mesh(mesh: bpy.types.Mesh | None) -> None:
    if not _rna_alive(mesh):
        return
    try:
        if int(mesh.users) != 0:
            return
        name = mesh.name
        current = bpy.data.meshes.get(name)
        if not _same_mesh(current, mesh):
            return
        bpy.data.meshes.remove(mesh)
    except (ReferenceError, RuntimeError, AttributeError):
        return


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0 and b.size == 0:
        return True
    return bool(np.allclose(a, b, rtol=0.0, atol=1.0e-6))


def _matrix_signature(matrix) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for row in matrix for value in row)


def _mesh_coordinate_digest(mesh: bpy.types.Mesh) -> str:
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    if len(coords) > 0:
        mesh.vertices.foreach_get("co", coords)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(coords.shape, dtype=np.int64).tobytes())
    digest.update(coords.tobytes())
    return digest.hexdigest()


def _vector_signature(values) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in values)


def _preview_writeback_interval(settings) -> int:
    return max(int(getattr(settings, "preview_writeback_interval", _AUTO_WRITEBACK_INTERVAL)), _AUTO_WRITEBACK_INTERVAL)


def _target_fps_from_settings(settings) -> float:
    return max(float(getattr(settings, "preview_target_fps", 30.0)), 1.0)


def _raw_preview_options_signature(settings, preview_slot_count: int) -> tuple:
    return (
        int(preview_slot_count),
        os.environ.get("SSBL_PREVIEW_SELF_COLLISION_BUDGET", ""),
        os.environ.get("SSBL_JITTER_STABILIZER_ENABLED", ""),
        int(getattr(settings, "substeps", 1)),
        int(getattr(settings, "iterations", 1)),
        round(float(getattr(settings, "dt", 1.0 / 60.0)), 8),
        round(float(getattr(settings, "damping", 0.0)), 6),
        _vector_signature(getattr(settings, "gravity", (0.0, 0.0, -9.81))),
        round(float(getattr(settings, "hardness", 0.5)), 6),
        bool(getattr(settings, "hardness_initialized", False)),
        round(float(getattr(settings, "stretch_compliance", 0.0)), 12),
        round(float(getattr(settings, "bend_compliance", 0.0)), 12),
        bool(getattr(settings, "use_lra", False)),
        round(float(getattr(settings, "lra_compliance", 0.0)), 12),
        round(float(getattr(settings, "lra_slack", 0.0)), 6),
        round(float(getattr(settings, "density", 1.0)), 6),
        round(float(getattr(settings, "pin_hardness", 1.0)), 6),
        str(getattr(settings, "pin_vertex_group", "")),
        round(float(getattr(settings, "collision_margin", 0.0)), 6),
        bool(getattr(settings, "use_ground", False)),
        round(float(getattr(settings, "ground_height", 0.0)), 6),
        bool(getattr(settings, "use_wall", False)),
        _vector_signature(getattr(settings, "wall_origin", (0.0, 0.0, 0.0))),
        _vector_signature(getattr(settings, "wall_normal", (0.0, 0.0, 1.0))),
        bool(getattr(settings, "use_sphere", False)),
        bool(getattr(settings, "self_collision", False)),
        str(getattr(settings, "self_collision_mode", "fast")),
        round(float(getattr(settings, "cloth_thickness", 0.02)), 6),
        round(float(getattr(settings, "self_collision_distance", 0.0)), 6),
        int(getattr(settings, "self_collision_interval", 2)),
        int(getattr(settings, "max_self_collision_neighbors", 32)),
        int(getattr(settings, "fast_self_collision_passes", 4)),
        bool(getattr(settings, "use_volume_pressure", False)),
        round(float(getattr(settings, "volume_compliance", 1.0e-6)), 12),
        round(float(getattr(settings, "pressure_strength", 0.02)), 6),
        round(float(getattr(settings, "volume_target_scale", 1.0)), 6),
        int(getattr(settings, "volume_solve_interval", 1)),
        int(getattr(settings, "self_probe_interval", 1)),
        int(getattr(settings, "self_surface_pair_interval", 1)),
        bool(getattr(settings, "jitter_stabilizer_enabled", True)),
        round(float(getattr(settings, "contact_friction", 0.35)), 6),
        round(float(getattr(settings, "contact_tangent_damping", 0.2)), 6),
        round(float(getattr(settings, "contact_compliance", 0.0)), 12),
        round(float(getattr(settings, "static_sdf_voxel_size", 0.0)), 6),
        int(getattr(settings, "static_sdf_band_voxels", 4)),
        int(getattr(settings, "static_sdf_max_resolution", 160)),
    )


def _static_slot_refresh_validation_interval() -> int:
    try:
        return max(int(os.environ.get("SSBL_STATIC_SLOT_REFRESH_VALIDATE_INTERVAL", "30")), 1)
    except ValueError:
        return 30


def _static_slot_refresh_signature(
    context: bpy.types.Context,
    session: SceneSession,
    slot: ClothSlot,
    obj: bpy.types.Object,
    settings,
    auto_sphere_object: bpy.types.Object | None,
) -> tuple | None:
    if not _env_enabled("SSBL_STATIC_SLOT_REFRESH_FAST_PATH", False):
        return None
    if slot.use_evaluated_mesh or slot.interactive_pin is not None:
        return None
    if len(slot.cloth.pin_indices) > 0:
        return None

    matrix_signature = _matrix_signature(obj.matrix_world)
    cached_signature = slot.runtime_refresh_signature
    should_validate = (
        not cached_signature
        or int(session.frame_index) % _static_slot_refresh_validation_interval() == 0
    )
    if not should_validate:
        return (matrix_signature, cached_signature[1])

    if getattr(settings, "static_collider_collection", None) is not None:
        return None
    if getattr(settings, "dynamic_collider_collection", None) is not None:
        return None
    if has_force_field_sources(context.scene, settings):
        return None
    if bool(getattr(settings, "use_sphere", False)) or auto_sphere_object is not None:
        return None
    return (
        matrix_signature,
        _raw_preview_options_signature(settings, len(session.slots)),
    )


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _apply_preview_self_collision_budget(options, settings, cloth: ClothBuildData | None, preview_slot_count: int):
    if not _env_enabled("SSBL_PREVIEW_SELF_COLLISION_BUDGET", True):
        return options
    if int(preview_slot_count) < 2:
        return options
    if not bool(getattr(options, "self_collision", False)):
        return options
    if int(getattr(options, "self_collision_mode", 0)) != SELF_COLLISION_FAST:
        return options
    vertex_count = int(len(getattr(cloth, "positions_world", ()))) if cloth is not None else 0
    triangle_count = int(len(getattr(cloth, "triangles", ()))) if cloth is not None else 0
    if (
        vertex_count < _PREVIEW_SELF_COLLISION_BUDGET_VERTEX_THRESHOLD
        and triangle_count < _PREVIEW_SELF_COLLISION_BUDGET_TRIANGLE_THRESHOLD
    ):
        return options
    substeps = max(int(getattr(settings, "substeps", getattr(options, "self_collision_interval", 1))), 1)
    interval = max(int(getattr(options, "self_collision_interval", 1)), substeps)
    passes = min(int(getattr(options, "fast_self_collision_passes", 1)), 1)
    if interval == int(getattr(options, "self_collision_interval", 1)) and passes == int(getattr(options, "fast_self_collision_passes", 1)):
        return options
    return replace(options, self_collision_interval=interval, fast_self_collision_passes=passes)


def _options_from_settings(
    settings,
    runtime_mode_override: str | None = None,
    auto_sphere_object=None,
    cloth: ClothBuildData | None = None,
    preview_slot_count: int = 1,
):
    if auto_sphere_object is None:
        options = settings_to_options(settings, runtime_mode_override=runtime_mode_override)
    else:
        explicit_sphere = getattr(settings, "sphere_object", None)
        if bool(getattr(settings, "use_sphere", False)) and explicit_sphere is not None:
            options = settings_to_options(settings, runtime_mode_override=runtime_mode_override)
        else:
            with _temporary_setting(settings, "use_sphere", True):
                with _temporary_setting(settings, "sphere_object", auto_sphere_object):
                    options = settings_to_options(settings, runtime_mode_override=runtime_mode_override)
    if str(runtime_mode_override or "").lower() == "preview":
        options = _apply_preview_self_collision_budget(options, settings, cloth, preview_slot_count)
    return options


def _external_contact_distance_from_options(options) -> float:
    return max(
        float(getattr(options, "cloth_thickness", 0.02)),
        float(getattr(options, "collision_margin", 0.0)),
        1.0e-5,
    )


def _runtime_options_signature(options) -> tuple:
    return (
        bool(options.use_ground),
        round(float(options.ground_height), 6),
        bool(options.use_wall),
        tuple(round(float(value), 6) for value in options.wall_origin),
        tuple(round(float(value), 6) for value in options.wall_normal),
        bool(options.use_sphere),
        tuple(round(float(value), 6) for value in options.sphere_center),
        round(float(options.sphere_radius), 6),
    )


def _solver_options_signature(options, settings=None) -> tuple:
    return (
        round(float(options.dt), 8),
        round(float(options.damping), 6),
        _vector_signature(options.gravity),
        round(float(options.stretch_compliance), 12),
        round(float(options.bend_compliance), 12),
        round(float(options.lra_compliance), 12),
        round(float(options.collision_margin), 6),
        bool(options.self_collision),
        int(options.self_collision_mode),
        round(float(options.cloth_thickness), 6),
        int(options.self_collision_interval),
        int(options.max_self_collision_neighbors),
        bool(options.use_volume_pressure),
        round(float(options.pressure_strength), 6),
        int(options.self_probe_interval),
        int(options.self_surface_pair_interval),
        bool(options.jitter_stabilizer_enabled),
        round(float(getattr(options, "contact_friction", 0.35)), 6),
        round(float(getattr(options, "contact_tangent_damping", 0.2)), 6),
        round(float(getattr(options, "contact_compliance", 0.0)), 12),
        int(getattr(options, "fast_self_collision_passes", 4)),
        round(float(getattr(options, "static_sdf_voxel_size", 0.0)), 6),
        int(getattr(options, "static_sdf_band_voxels", 4)),
        int(getattr(options, "static_sdf_max_resolution", 160)),
        round(float(getattr(settings, "density", 1.0)), 6) if settings is not None else 1.0,
        str(getattr(settings, "pin_vertex_group", "")) if settings is not None else "",
        round(float(getattr(settings, "pin_hardness", 1.0)), 6) if settings is not None else 1.0,
    )


def _static_collider_runtime_signature(
    collection: bpy.types.Collection | None,
    exclude_obj: bpy.types.Object | None,
    depsgraph: bpy.types.Depsgraph | None,
    use_evaluated_mesh: bool,
    exclude_names: set[str] | frozenset[str] | None = None,
) -> tuple:
    if collection is None:
        return ()
    excluded = frozenset(exclude_names or ())
    entries = []
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    for obj in sorted(collection.objects, key=lambda item: item.name):
        if obj is None or obj == exclude_obj or obj.type != "MESH" or obj.name in excluded:
            continue
        if use_evaluated_mesh:
            source = obj.evaluated_get(depsgraph)
            mesh = source.to_mesh()
            try:
                entries.append(
                    (
                        obj.name,
                        len(mesh.vertices),
                        len(mesh.polygons),
                        _matrix_signature(source.matrix_world),
                        _mesh_coordinate_digest(mesh),
                    )
                )
            finally:
                source.to_mesh_clear()
        else:
            mesh = obj.data
            entries.append(
                (
                    obj.name,
                    len(mesh.vertices),
                    len(mesh.polygons),
                    _matrix_signature(obj.matrix_world),
                    _mesh_coordinate_digest(mesh),
                )
            )
    return tuple(entries)


def _pin_targets_from_object(
    obj: bpy.types.Object,
    pin_indices: np.ndarray,
    use_evaluated_mesh: bool,
    depsgraph: bpy.types.Depsgraph | None = None,
    expected_vertex_count: int | None = None,
) -> tuple[np.ndarray, bpy.types.Matrix]:
    if use_evaluated_mesh:
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            if expected_vertex_count is not None and len(mesh.vertices) != expected_vertex_count:
                raise ValueError("Evaluated mesh vertex count changed during preview input refresh.")
            return _pin_targets_from_mesh(mesh, eval_obj.matrix_world.copy(), pin_indices), eval_obj.matrix_world.copy()
        finally:
            eval_obj.to_mesh_clear()

    mesh = obj.data
    if expected_vertex_count is not None and len(mesh.vertices) != expected_vertex_count:
        raise ValueError("Mesh vertex count changed during preview input refresh.")
    return _pin_targets_from_mesh(mesh, obj.matrix_world.copy(), pin_indices), obj.matrix_world.copy()


def _pin_attachment_batch_from_object(
    obj: bpy.types.Object,
    pin_indices: np.ndarray,
    use_evaluated_mesh: bool,
    depsgraph: bpy.types.Depsgraph | None = None,
    expected_vertex_count: int | None = None,
) -> tuple[PinAttachmentBatch, bpy.types.Matrix]:
    targets, matrix_world = _pin_targets_from_object(
        obj,
        pin_indices,
        use_evaluated_mesh,
        depsgraph=depsgraph,
        expected_vertex_count=expected_vertex_count,
    )
    return make_pin_attachment_batch(pin_indices, targets), matrix_world


def _pin_settings_signature(obj: bpy.types.Object | None, settings, vertex_count: int) -> tuple:
    group_name = str(getattr(settings, "pin_vertex_group", "") or "").strip()
    group = obj.vertex_groups.get(group_name) if obj is not None and group_name else None
    return (
        group_name,
        int(group.index) if group is not None else -1,
        float(getattr(settings, "pin_hardness", 1.0)),
        int(vertex_count),
    )


def _pin_attachment_batch_from_slot_object(
    obj: bpy.types.Object,
    slot: ClothSlot,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> tuple[PinAttachmentBatch, bpy.types.Matrix]:
    _refresh_slot_pin_weights_from_group(obj, slot, settings)
    if len(slot.cloth.pin_indices) > 0:
        targets, matrix_world = _pin_targets_from_object(
            obj,
            slot.cloth.pin_indices,
            slot.use_evaluated_mesh,
            depsgraph=depsgraph,
            expected_vertex_count=len(slot.cloth.positions_world),
        )
    else:
        targets = np.empty((0, 3), dtype=np.float32)
        matrix_world = obj.matrix_world.copy()
    batch = make_pin_attachment_batch(slot.cloth.pin_indices, targets, slot.cloth.pin_weights)
    batch = _apply_interactive_pin_target_override(slot, batch)
    slot.cloth.pin_attachment_pairs = np.array(batch.pairs, dtype=np.int32, copy=True)
    slot.cloth.pin_targets_world = np.array(batch.targets_world, dtype=np.float32, copy=True)
    slot.pin_attachment_pairs = np.array(batch.pairs, dtype=np.int32, copy=True)
    return batch, matrix_world


def _apply_interactive_pin_target_override(slot: ClothSlot, batch: PinAttachmentBatch) -> PinAttachmentBatch:
    state = slot.interactive_pin
    if state is None or len(batch.pairs) == 0:
        return batch
    handle = bpy.data.objects.get(state.empty_name)
    if handle is None:
        return batch
    indices = np.asarray(batch.pairs[:, 0], dtype=np.int32)
    matches = np.flatnonzero(indices == int(state.vertex_index))
    if len(matches) == 0:
        return batch
    targets = np.array(batch.targets_world, dtype=np.float32, copy=True)
    targets[int(matches[0])] = np.asarray(handle.location, dtype=np.float32)
    return make_pin_attachment_batch(slot.cloth.pin_indices, targets, slot.cloth.pin_weights)


def _refresh_slot_pin_weights_from_group(
    obj: bpy.types.Object,
    slot: ClothSlot,
    settings,
    *,
    force: bool = False,
) -> bool:
    if force:
        _invalidate_free_simulation_vertex_cache(obj)
    signature = _pin_settings_signature(obj, settings, len(slot.cloth.positions_world))
    if not force and slot.pin_settings_signature == signature:
        return False
    weights_by_vertex = effective_pin_weights_from_settings(obj, settings, len(slot.cloth.positions_world))
    pin_indices = np.flatnonzero(weights_by_vertex > 0.0).astype(np.int32)
    pin_weights = weights_by_vertex[pin_indices].astype(np.float32, copy=True)
    if np.array_equal(pin_indices, slot.cloth.pin_indices) and _array_equal(pin_weights, slot.cloth.pin_weights):
        slot.pin_settings_signature = signature
        return False
    slot.cloth.pin_indices = np.ascontiguousarray(pin_indices, dtype=np.int32)
    slot.cloth.pin_weights = np.ascontiguousarray(pin_weights, dtype=np.float32)
    empty_targets = np.zeros((len(pin_indices), 3), dtype=np.float32)
    batch = make_pin_attachment_batch(pin_indices, empty_targets, pin_weights)
    slot.cloth.pin_attachment_pairs = np.array(batch.pairs, dtype=np.int32, copy=True)
    slot.pin_attachment_pairs = np.array(batch.pairs, dtype=np.int32, copy=True)
    slot.pin_targets_world = np.empty((0, 3), dtype=np.float32)
    slot.pin_settings_signature = signature
    slot.force_next_writeback = True
    _invalidate_free_simulation_vertex_cache(obj)
    return True


def _pin_targets_from_mesh(mesh: bpy.types.Mesh, matrix_world, pin_indices: np.ndarray) -> np.ndarray:
    if len(pin_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    indices = np.asarray(pin_indices, dtype=np.intp)
    coords_flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    if "position" in mesh.attributes:
        mesh.attributes["position"].data.foreach_get("vector", coords_flat)
    else:
        mesh.vertices.foreach_get("co", coords_flat)
    coords = coords_flat.reshape((-1, 3))[indices]
    mat = np.array(matrix_world, dtype=np.float32)
    return coords @ mat[:3, :3].T + mat[:3, 3]


def _session_for_object_name(object_name: str) -> SceneSession | None:
    scene_key = _OBJECT_TO_SCENE_SESSION.get(object_name)
    if scene_key is None:
        return None
    return _SCENE_SESSIONS.get(scene_key)


def _unique_vertex_group_name(obj: bpy.types.Object, base: str) -> str:
    name = base
    index = 1
    while obj.vertex_groups.get(name) is not None:
        name = f"{base}.{index:03d}"
        index += 1
    return name


def _unique_modifier_name(obj: bpy.types.Object, base: str) -> str:
    name = base
    index = 1
    while obj.modifiers.get(name) is not None:
        name = f"{base}.{index:03d}"
        index += 1
    return name


def _vertex_group_weight(group: bpy.types.VertexGroup, vertex_index: int) -> float | None:
    try:
        return float(group.weight(int(vertex_index)))
    except RuntimeError:
        return None


@contextmanager
def _with_object_mesh_data(obj: bpy.types.Object, mesh: bpy.types.Mesh):
    old_mesh = obj.data
    obj.data = mesh
    try:
        yield
    finally:
        if _rna_alive(obj) and _rna_alive(old_mesh):
            obj.data = old_mesh


def _ensure_vertex_group_on_mesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    group_name: str,
) -> bpy.types.VertexGroup | None:
    if not _rna_alive(obj) or not _rna_alive(mesh):
        return None
    with _with_object_mesh_data(obj, mesh):
        return obj.vertex_groups.get(group_name) or obj.vertex_groups.new(name=group_name)


def _vertex_group_weight_on_mesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    group_name: str,
    vertex_index: int,
) -> float | None:
    if not _rna_alive(obj) or not _rna_alive(mesh):
        return None
    with _with_object_mesh_data(obj, mesh):
        group = obj.vertex_groups.get(group_name)
        if group is None:
            return None
        return _vertex_group_weight(group, vertex_index)


def _add_vertex_group_weight_on_mesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    group_name: str,
    vertex_index: int,
    weight: float,
    mode: str,
) -> None:
    group = _ensure_vertex_group_on_mesh(obj, mesh, group_name)
    if group is None:
        return
    with _with_object_mesh_data(obj, mesh):
        group = obj.vertex_groups.get(group_name)
        if group is not None:
            group.add([int(vertex_index)], float(weight), mode)


def _restore_vertex_group_weight_on_mesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    group_name: str,
    vertex_index: int,
    previous_weight: float | None,
) -> None:
    if not _rna_alive(obj) or not _rna_alive(mesh):
        return
    with _with_object_mesh_data(obj, mesh):
        group = obj.vertex_groups.get(group_name)
        if group is None:
            return
        if previous_weight is None:
            try:
                group.remove([int(vertex_index)])
            except RuntimeError:
                pass
        else:
            group.add([int(vertex_index)], float(previous_weight), "REPLACE")


def _remove_vertex_group_on_mesh(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    group_name: str,
) -> None:
    if not _rna_alive(obj) or not _rna_alive(mesh):
        return
    with _with_object_mesh_data(obj, mesh):
        group = obj.vertex_groups.get(group_name)
        if group is not None:
            obj.vertex_groups.remove(group)


def _make_interactive_pin_empty(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    vertex_index: int,
    world_position: Vector,
) -> bpy.types.Object:
    name = f"{_INTERACTIVE_PIN_PREFIX}_{obj.name}_{int(vertex_index)}"
    handle = bpy.data.objects.new(name, None)
    handle.empty_display_type = "SPHERE"
    handle.empty_display_size = max(max(float(value) for value in obj.dimensions) * 0.04, 0.04)
    handle.location = world_position
    handle[_INTERACTIVE_PIN_PROP] = True
    handle["_ssbl_source_object"] = obj.name
    collection = getattr(context, "collection", None)
    if collection is None:
        collection = context.scene.collection
    collection.objects.link(handle)
    return handle


def _select_interactive_pin_handle(context: bpy.types.Context, handle: bpy.types.Object) -> None:
    for selected in list(getattr(context, "selected_objects", [])):
        if selected is not None and _rna_alive(selected):
            selected.select_set(False)
    handle.select_set(True)
    context.view_layer.objects.active = handle


def _restore_interactive_selection(context: bpy.types.Context, state: InteractivePinState) -> None:
    for selected in list(getattr(context, "selected_objects", [])):
        if selected is not None and _rna_alive(selected):
            selected.select_set(False)
    for name in state.previous_selection_names:
        obj = bpy.data.objects.get(name)
        if obj is not None and _rna_alive(obj):
            obj.select_set(True)
    active = bpy.data.objects.get(state.previous_active_name)
    if active is not None and _rna_alive(active):
        context.view_layer.objects.active = active


def _remove_interactive_pin_empty(state: InteractivePinState) -> None:
    handle = bpy.data.objects.get(state.empty_name)
    if handle is not None and bool(handle.get(_INTERACTIVE_PIN_PROP, False)):
        bpy.data.objects.remove(handle, do_unlink=True)


def _clear_interactive_pin_state(
    slot: ClothSlot,
    obj: bpy.types.Object,
    *,
    restore_selection: bool = True,
) -> bool:
    state = slot.interactive_pin
    if state is None:
        return False

    def restore_weight_and_remove_hook() -> None:
        _restore_vertex_group_weight_on_mesh(
            obj,
            slot.original_mesh,
            state.pin_group_name,
            state.vertex_index,
            state.previous_weight,
        )
        if _rna_alive(slot.preview_mesh) and not _same_mesh(slot.preview_mesh, slot.original_mesh):
            _restore_vertex_group_weight_on_mesh(
                obj,
                slot.preview_mesh,
                state.pin_group_name,
                state.vertex_index,
                state.previous_preview_weight,
            )

        modifier = obj.modifiers.get(state.hook_modifier_name)
        if modifier is not None:
            obj.modifiers.remove(modifier)

        _remove_vertex_group_on_mesh(obj, slot.original_mesh, state.hook_group_name)
        if _rna_alive(slot.preview_mesh) and not _same_mesh(slot.preview_mesh, slot.original_mesh):
            _remove_vertex_group_on_mesh(obj, slot.preview_mesh, state.hook_group_name)

    try:
        if _rna_alive(obj) and _rna_alive(slot.original_mesh) and _rna_alive(slot.preview_mesh):
            with _with_preview_source_state(slot, obj):
                restore_weight_and_remove_hook()
                settings = _settings_for_slot(bpy.context, slot)
                _refresh_slot_pin_weights_from_group(obj, slot, settings, force=True)
        elif _rna_alive(obj):
            restore_weight_and_remove_hook()
    finally:
        slot.suspended_modifiers = [
            item for item in slot.suspended_modifiers if item[0] != state.hook_modifier_name
        ]
        slot.use_evaluated_mesh = bool(state.previous_use_evaluated_mesh)
        slot.interactive_pin = None
        slot.force_next_writeback = True
        _remove_interactive_pin_empty(state)
        if restore_selection:
            _restore_interactive_selection(bpy.context, state)
    return True


def _sessions_for_objects(objects: list[bpy.types.Object]) -> list[SceneSession]:
    sessions: list[SceneSession] = []
    seen: set[str] = set()
    for obj in objects:
        session = _session_for_object_name(obj.name)
        if session is None or session.scene_name in seen:
            continue
        sessions.append(session)
        seen.add(session.scene_name)
    return sessions


def _preview_cloth_objects(context: bpy.types.Context, obj: bpy.types.Object, settings) -> list[bpy.types.Object]:
    if obj is None:
        raise ValueError("A mesh object is required to start preview.")
    auto_sphere_names = {
        item.name for item in _auto_sphere_collider_candidates(context, obj, settings)
    }
    selected = []
    for selected_obj in context.selected_objects:
        if selected_obj is None or selected_obj == obj or selected_obj.type != "MESH":
            continue
        if selected_obj.name in auto_sphere_names:
            continue
        if _object_cloth_settings(selected_obj) is None:
            continue
        if not _has_simulatable_cloth_source_geometry(selected_obj):
            continue
        if _declared_input_type(selected_obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        if not _has_free_simulation_vertices(selected_obj, _object_cloth_settings(selected_obj)):
            _STATUS[selected_obj.name] = STATUS_NO_SIMULATED_VERTICES
            continue
        if _is_collision_only_object(context.scene, selected_obj, active_obj=obj, active_settings=settings):
            continue
        selected.append(selected_obj)
    if not _has_free_simulation_vertices(obj, settings):
        _STATUS[obj.name] = STATUS_NO_SIMULATED_VERTICES
        return []
    return [obj] + sorted(
        selected,
        key=lambda item: (
            int(getattr(item, _OBJECT_COLLISION_LAYER_PROP, item.get(_OBJECT_COLLISION_LAYER_PROP, 1))),
            item.name.casefold(),
        ),
    )


def _create_cloth_slot(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph,
    auto_sphere_object: bpy.types.Object | None = None,
    preview_slot_count: int = 1,
) -> ClothSlot:
    original_mesh = obj.data
    use_evaluated_mesh = _effective_use_evaluated_mesh(obj, settings)
    cloth, native, static_signature, static_tris = _create_native_solver(
        context,
        obj,
        settings,
        depsgraph=depsgraph,
        use_evaluated_mesh_override=use_evaluated_mesh,
        runtime_mode_override="preview",
        auto_sphere_object=auto_sphere_object,
        preview_slot_count=preview_slot_count,
    )
    suspended_modifiers = _suspend_preview_modifiers(obj, suspend_all=use_evaluated_mesh)
    preview_mesh = original_mesh.copy()
    preview_mesh.name = f"{original_mesh.name}_SSBL_XPBD_Preview"
    obj.data = preview_mesh
    writeback_flat_buffer = np.empty(cloth.positions_world.size, dtype=np.float32)
    writeback_local_buffer = writeback_flat_buffer.reshape(cloth.positions_world.shape)
    options = _options_from_settings(
        settings,
        runtime_mode_override="preview",
        auto_sphere_object=auto_sphere_object,
        cloth=cloth,
        preview_slot_count=preview_slot_count,
    )
    static_exclude_names = _dynamic_collider_object_names(settings, obj)
    return ClothSlot(
        object_name=obj.name,
        cloth=cloth,
        native=native,
        original_mesh=original_mesh,
        preview_mesh=preview_mesh,
        suspended_modifiers=suspended_modifiers,
        use_evaluated_mesh=use_evaluated_mesh,
        static_collider_signature=static_signature,
        static_triangles=np.array(static_tris, dtype=np.float32, copy=True),
        static_runtime_signature=_static_collider_runtime_signature(
            settings.static_collider_collection,
            obj,
            depsgraph,
            use_evaluated_mesh,
            exclude_names=static_exclude_names,
        ),
        pin_attachment_pairs=np.array(cloth.pin_attachment_pairs, dtype=np.int32, copy=True),
        pin_targets_world=np.array(cloth.pin_targets_world, dtype=np.float32, copy=True),
        runtime_options_signature=_runtime_options_signature(options),
        solver_options_signature=_solver_options_signature(options, settings),
        collision_layer=_object_collision_layer(obj),
        external_contact_distance=_external_contact_distance_from_options(options),
        current_positions_world=np.array(cloth.positions_world, dtype=np.float32, copy=True),
        previous_positions_world=np.array(cloth.positions_world, dtype=np.float32, copy=True),
        writeback_local_buffer=writeback_local_buffer,
        writeback_flat_buffer=writeback_flat_buffer,
        substeps=max(int(getattr(settings, "substeps", 1)), 1),
        iterations=max(int(getattr(settings, "iterations", 1)), 1),
        writeback_interval=_preview_writeback_interval(settings),
        frame_count=max(int(getattr(settings, "frame_count", 1)), 1),
        use_object_settings=bool(_object_cloth_settings(obj) is not None),
        force_fields_active=False,
        auto_sphere_object_name=auto_sphere_object.name if auto_sphere_object is not None else "",
        auto_cache_realtime=bool(getattr(settings, "auto_cache_realtime", False)),
        pin_settings_signature=_pin_settings_signature(obj, settings, len(cloth.positions_world)),
    )


def _create_native_solver(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
    use_evaluated_mesh_override: bool | None = None,
    runtime_mode_override: str | None = None,
    auto_sphere_object: bpy.types.Object | None = None,
    preview_slot_count: int = 1,
) -> tuple[ClothBuildData, NativeXpbdSolver, tuple[tuple[str, int, int], ...], np.ndarray]:
    try:
        depsgraph = depsgraph or context.evaluated_depsgraph_get()
        use_evaluated_mesh = (
            bool(use_evaluated_mesh_override)
            if use_evaluated_mesh_override is not None
            else _effective_use_evaluated_mesh(obj, settings)
        )
        with _temporary_setting(settings, "use_evaluated_mesh", use_evaluated_mesh):
            cloth = build_cloth_data(obj, settings, depsgraph=depsgraph)
        options = _options_from_settings(
            settings,
            runtime_mode_override=runtime_mode_override,
            auto_sphere_object=auto_sphere_object,
            cloth=cloth,
            preview_slot_count=preview_slot_count,
        )
        static_exclude_names = (
            _dynamic_collider_object_names(settings, obj)
            if str(runtime_mode_override or "preview").lower() == "preview"
            else set()
        )
        static_tris, static_signature = collect_static_triangles(
            settings.static_collider_collection,
            obj,
            depsgraph=depsgraph,
            use_evaluated_mesh=use_evaluated_mesh,
            exclude_names=static_exclude_names,
        )
        native = NativeXpbdSolver(cloth, options, static_tris)
        native.update_runtime_colliders(options)
    except Exception:
        _STATUS[obj.name] = STATUS_ERROR
        raise
    return cloth, native, static_signature, static_tris


def _update_slot_live_settings(slot: ClothSlot, obj: bpy.types.Object, settings) -> bool:
    changed = False
    substeps = max(int(getattr(settings, "substeps", 1)), 1)
    iterations = max(int(getattr(settings, "iterations", 1)), 1)
    frame_count = max(int(getattr(settings, "frame_count", 1)), 1)
    writeback_interval = _preview_writeback_interval(settings)
    collision_layer = _object_collision_layer(obj)
    if slot.substeps != substeps:
        slot.substeps = substeps
        changed = True
    if slot.iterations != iterations:
        slot.iterations = iterations
        changed = True
    if slot.frame_count != frame_count:
        slot.frame_count = frame_count
        changed = True
    if slot.writeback_interval != writeback_interval:
        slot.writeback_interval = writeback_interval
        changed = True
    if slot.collision_layer != collision_layer:
        slot.collision_layer = collision_layer
        changed = True
    if changed:
        slot.force_next_writeback = True
    return changed


def _refresh_session_live_settings(
    context: bpy.types.Context,
    session: SceneSession,
    slot_settings: dict[str, object],
) -> None:
    if not session.slots:
        return
    target_values = [_target_fps_from_settings(settings) for settings in slot_settings.values()]
    if target_values:
        session.target_fps = min(target_values)
    session.frame_count = max(slot.frame_count for slot in session.slots.values())
    session.substeps = max(slot.substeps for slot in session.slots.values())
    session.iterations = max(slot.iterations for slot in session.slots.values())
    session.writeback_interval = min(slot.writeback_interval for slot in session.slots.values())

    previous_order = tuple(session.solve_order)
    session.solve_order = sorted(
        session.slots.keys(),
        key=lambda name: (session.slots[name].collision_layer, name.casefold()),
    )
    previous_mode = str(session.cross_cloth_mode or "off")
    session.cross_cloth_mode = _cross_cloth_mode_from_settings(context.scene.ssbl_preview, len(session.slots))
    if tuple(session.solve_order) != previous_order or str(session.cross_cloth_mode or "off") != previous_mode:
        for slot in session.slots.values():
            slot.force_next_writeback = True


def _rebuild_slot_native(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    slot: ClothSlot,
    settings,
    depsgraph: bpy.types.Depsgraph | None,
    options,
    solver_signature: tuple,
    perf: FramePerf | None = None,
    auto_sphere_object: bpy.types.Object | None = None,
    preview_slot_count: int = 1,
) -> None:
    started = time.perf_counter()
    download_started = time.perf_counter()
    current_positions = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
    if perf is not None:
        perf.download_ms += _elapsed_ms(download_started)

    new_native = None
    try:
        cloth, new_native, static_signature, static_tris = _create_native_solver(
            context,
            obj,
            settings,
            depsgraph=depsgraph,
            use_evaluated_mesh_override=slot.use_evaluated_mesh,
            runtime_mode_override="preview",
            auto_sphere_object=auto_sphere_object,
            preview_slot_count=preview_slot_count,
        )
        if len(cloth.positions_world) != len(slot.cloth.positions_world) or len(cloth.positions_world) != len(current_positions):
            raise ValueError("Cloth topology changed during preview tuning; restart the preview after topology changes.")
        new_native.update_positions(current_positions)
        old_native = slot.native
        slot.native = new_native
        new_native = None
        try:
            old_native.close()
        except Exception:
            pass

        slot.cloth = cloth
        slot.static_triangles = np.array(static_tris, dtype=np.float32, copy=True)
        slot.static_collider_signature = static_signature
        static_collection = getattr(settings, "static_collider_collection", None)
        static_exclude_names = _dynamic_collider_object_names(settings, obj)
        slot.static_runtime_signature = (
            _static_collider_runtime_signature(
                static_collection,
                obj,
                depsgraph,
                slot.use_evaluated_mesh,
                exclude_names=static_exclude_names,
            )
            if static_collection is not None
            else ()
        )
        slot.previous_positions_world = np.array(current_positions, dtype=np.float32, copy=True)
        slot.current_positions_world = current_positions
        slot.runtime_options_signature = _runtime_options_signature(options)
        slot.solver_options_signature = solver_signature
        slot.runtime_refresh_signature = ()
        slot.pin_attachment_pairs = np.array(cloth.pin_attachment_pairs, dtype=np.int32, copy=True)
        slot.pin_targets_world = np.empty((0, 3), dtype=np.float32)
        slot.pin_settings_signature = _pin_settings_signature(obj, settings, len(cloth.positions_world))
        slot.force_next_writeback = True
    finally:
        if new_native is not None:
            try:
                new_native.close()
            except Exception:
                pass
        if perf is not None:
            perf.frame_input_upload_ms += _elapsed_ms(started)


def _refresh_session_runtime_inputs(context: bpy.types.Context, session: SceneSession, perf: FramePerf | None = None) -> None:
    refresh_started = time.perf_counter()
    slot_settings = {name: _settings_for_slot(context, slot) for name, slot in session.slots.items()}
    needs_depsgraph = any(
        getattr(slot_settings[name], "static_collider_collection", None) is not None
        or getattr(slot_settings[name], "dynamic_collider_collection", None) is not None
        or slot.use_evaluated_mesh
        or has_force_field_sources(context.scene, slot_settings[name])
        for name, slot in session.slots.items()
    )
    with _with_session_source_state(session):
        if not bool(session.playback_driven):
            context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get() if needs_depsgraph else None
        for slot in list(session.slots.values()):
            obj = bpy.data.objects.get(slot.object_name)
            if obj is None or obj.type != "MESH":
                raise ValueError(f"Missing preview object during input refresh: {slot.object_name}")
            settings = slot_settings[slot.object_name]
            _update_slot_live_settings(slot, obj, settings)
            auto_sphere_object = (
                bpy.data.objects.get(slot.auto_sphere_object_name)
                if slot.auto_sphere_object_name
                else None
            )
            fast_refresh_signature = _static_slot_refresh_signature(
                context,
                session,
                slot,
                obj,
                settings,
                auto_sphere_object,
            )
            if fast_refresh_signature is not None:
                cached_signature = slot.runtime_refresh_signature
                raw_signature_unchanged = (
                    cached_signature == fast_refresh_signature
                    or (
                        bool(cached_signature)
                        and len(cached_signature) == len(fast_refresh_signature)
                        and cached_signature[1:] == fast_refresh_signature[1:]
                    )
                )
                if raw_signature_unchanged:
                    if cached_signature != fast_refresh_signature:
                        slot.cloth.matrix_world_inv = np.array(obj.matrix_world.inverted(), dtype=np.float32)
                    slot.runtime_refresh_signature = fast_refresh_signature
                    continue
            options = _options_from_settings(
                settings,
                runtime_mode_override="preview",
                auto_sphere_object=auto_sphere_object,
                cloth=slot.cloth,
                preview_slot_count=len(session.slots),
            )
            slot.external_contact_distance = _external_contact_distance_from_options(options)
            runtime_signature = _runtime_options_signature(options)
            solver_signature = _solver_options_signature(options, settings)
            if solver_signature != slot.solver_options_signature:
                _rebuild_slot_native(
                    context,
                    obj,
                    slot,
                    settings,
                    depsgraph,
                    options,
                    solver_signature,
                    perf,
                    auto_sphere_object,
                    preview_slot_count=len(session.slots),
                )

            static_collection = settings.static_collider_collection
            has_static_collection = static_collection is not None
            static_exclude_names = _dynamic_collider_object_names(settings, obj)
            pin_attachment_batch, matrix_world = _pin_attachment_batch_from_slot_object(
                obj,
                slot,
                settings,
                depsgraph=depsgraph,
            )
            pin_targets = pin_attachment_batch.targets_world
            static_runtime_signature = (
                _static_collider_runtime_signature(
                    static_collection,
                    obj,
                    depsgraph,
                    slot.use_evaluated_mesh,
                    exclude_names=static_exclude_names,
                )
                if has_static_collection
                else ()
            )
            static_tris = None
            static_signature = slot.static_collider_signature
            if static_runtime_signature != slot.static_runtime_signature:
                static_tris, static_signature = collect_static_triangles(
                    static_collection,
                    obj,
                    depsgraph=depsgraph,
                    use_evaluated_mesh=slot.use_evaluated_mesh,
                    exclude_names=static_exclude_names,
                )
            force_fields_enabled = has_force_field_sources(context.scene, settings)
            force_fields = (
                collect_force_fields(context.scene, depsgraph, settings)
                if force_fields_enabled
                else (EMPTY_FORCE_FIELD_BATCH if slot.force_fields_active else None)
            )
            _apply_runtime_inputs(
                slot,
                options,
                runtime_signature,
                pin_targets,
                matrix_world,
                static_tris,
                static_signature,
                static_runtime_signature,
                force_fields,
                force_fields_enabled,
                perf,
            )
            slot.runtime_refresh_signature = fast_refresh_signature or ()
        _sync_session_dynamic_collision_sources(context, session, depsgraph, slot_settings)
        _refresh_session_live_settings(context, session, slot_settings)
    if perf is not None:
        perf.input_refresh_ms += _elapsed_ms(refresh_started)


def _slot_should_download(download_positions, slot_name: str) -> bool:
    if isinstance(download_positions, dict):
        return bool(download_positions.get(slot_name, False))
    return bool(download_positions)


def _download_slot_positions(slot: ClothSlot, perf: FramePerf | None = None) -> None:
    started = time.perf_counter()
    downloaded_positions = np.asarray(slot.native.download_positions(), dtype=np.float32)
    if slot.previous_positions_world.shape == slot.current_positions_world.shape:
        np.copyto(slot.previous_positions_world, slot.current_positions_world, casting="unsafe")
    else:
        slot.previous_positions_world = np.array(slot.current_positions_world, dtype=np.float32, copy=True)
    if slot.current_positions_world.shape == downloaded_positions.shape:
        np.copyto(slot.current_positions_world, downloaded_positions, casting="unsafe")
    else:
        slot.current_positions_world = np.array(downloaded_positions, dtype=np.float32, copy=True)
    if perf is not None:
        perf.download_ms += _elapsed_ms(started)


def _step_session_slots(session: SceneSession, download_positions, perf: FramePerf | None = None) -> None:
    dynamic_colliders_enabled = _session_dynamic_colliders_enabled(session)
    sample_all_dynamic_slot_diagnostics = _env_enabled("SSBL_DYNAMIC_SLOT_DIAGNOSTICS_ALL", False)
    global_dynamic_scene_enabled = False
    if dynamic_colliders_enabled:
        _prepare_cross_cloth_collider_caches(session, perf)
        global_dynamic_scene_enabled = _update_global_dynamic_scene(session, perf)
    deferred_download_slots: list[ClothSlot] = []
    for slot_name in session.solve_order:
        slot = session.slots[slot_name]
        should_download = _slot_should_download(download_positions, slot_name)
        if dynamic_colliders_enabled and not global_dynamic_scene_enabled:
            started = time.perf_counter()
            pack_started = time.perf_counter()
            dynamic_triangles, dynamic_indexed_triangles, dynamic_particles = _collect_cross_cloth_colliders(session, slot, perf)
            dynamic_triangle_count = int(len(dynamic_triangles)) + int(len(dynamic_indexed_triangles.get("indices", ())))
            if perf is not None:
                perf.dynamic_collider_pack_ms += _elapsed_ms(pack_started)
                perf.dynamic_triangle_count += dynamic_triangle_count
                perf.dynamic_particle_count += int(len(dynamic_particles.get("positions", ())))
            did_upload_dynamic = slot.native.update_frame_inputs(
                pin_indices=None,
                pin_positions=None,
                pin_weights=None,
                update_pin=False,
                options=None,
                update_runtime=False,
                static_triangles=None,
                update_static=False,
                dynamic_triangles=dynamic_triangles,
                update_dynamic=True,
                dynamic_triangle_vertices=(
                    dynamic_indexed_triangles.get("vertices")
                    if len(dynamic_indexed_triangles.get("indices", ())) > 0
                    else None
                ),
                dynamic_triangle_indices=(
                    dynamic_indexed_triangles.get("indices")
                    if len(dynamic_indexed_triangles.get("indices", ())) > 0
                    else None
                ),
                dynamic_triangle_topology_signature=dynamic_indexed_triangles.get("signature"),
                dynamic_particles=dynamic_particles,
                update_dynamic_particles=True,
                force_fields=None,
                update_force_fields=False,
            )
            if perf is not None and did_upload_dynamic:
                elapsed = _elapsed_ms(started)
                perf.dynamic_upload_ms += elapsed
                perf.frame_input_upload_ms += elapsed
        started = time.perf_counter()
        is_last_slot = slot_name == session.solve_order[-1]
        sample_diagnostics = bool(
            should_download
            and (not dynamic_colliders_enabled or sample_all_dynamic_slot_diagnostics)
        ) or bool(dynamic_colliders_enabled and is_last_slot)
        slot.native.step(
            slot.substeps,
            slot.iterations,
            diagnostics=sample_diagnostics,
            synchronize=sample_diagnostics,
        )
        if perf is not None:
            perf.cuda_step_call_ms += _elapsed_ms(started)
        if should_download or dynamic_colliders_enabled:
            if dynamic_colliders_enabled:
                deferred_download_slots.append(slot)
            else:
                _download_slot_positions(slot, perf)
    for slot in deferred_download_slots:
        _download_slot_positions(slot, perf)


def _empty_dynamic_particles() -> dict[str, np.ndarray]:
    return {
        "positions": np.empty((0, 3), dtype=np.float32),
        "radii": np.empty(0, dtype=np.float32),
        "inv_mass": np.empty(0, dtype=np.float32),
        "slot_ids": np.empty(0, dtype=np.int32),
        "phases": np.empty(0, dtype=np.int32),
    }


def _empty_dynamic_indexed_triangles() -> dict[str, np.ndarray]:
    return {
        "vertices": np.empty((0, 3), dtype=np.float32),
        "indices": np.empty((0, 3), dtype=np.int32),
    }


def _single_or_concat_float32(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    if not arrays:
        return np.empty(empty_shape, dtype=np.float32)
    if len(arrays) == 1:
        return np.ascontiguousarray(arrays[0], dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32)


def _single_or_concat_int32(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0, dtype=np.int32)
    if len(arrays) == 1:
        return np.ascontiguousarray(arrays[0], dtype=np.int32).reshape((-1,))
    return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.int32).reshape((-1,))


def _single_or_concat_int32_rows(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    if not arrays:
        return np.empty(empty_shape, dtype=np.int32)
    if len(arrays) == 1:
        return np.ascontiguousarray(arrays[0], dtype=np.int32)
    return np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.int32)


def _ensure_bool_buffer(buffer: np.ndarray, count: int) -> np.ndarray:
    if buffer.shape != (count,):
        return np.empty(count, dtype=bool)
    return buffer


def _ensure_float32_capacity(buffer: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return np.empty(shape, dtype=np.float32)
    if buffer.ndim == len(shape) and buffer.shape[1:] == shape[1:] and buffer.shape[0] >= shape[0]:
        return buffer
    return np.empty(shape, dtype=np.float32)


def _ensure_int32_capacity(buffer: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 0:
        return np.empty(shape, dtype=np.int32)
    if buffer.ndim == len(shape) and buffer.shape[1:] == shape[1:] and buffer.shape[0] >= shape[0]:
        return buffer
    return np.empty(shape, dtype=np.int32)


def _cross_cloth_pair_key(
    mode: str,
    target: ClothSlot,
    source: ClothSlot,
    expanded: tuple[np.ndarray, np.ndarray],
) -> tuple:
    target_cache = target.dynamic_collider_cache
    source_cache = source.dynamic_collider_cache
    lower = np.asarray(expanded[0], dtype=np.float32).reshape((3,))
    upper = np.asarray(expanded[1], dtype=np.float32).reshape((3,))
    return (
        str(mode),
        str(target.object_name),
        str(source.object_name),
        int(target_cache.positions_generation),
        int(source_cache.positions_generation),
        tuple(float(v) for v in lower),
        tuple(float(v) for v in upper),
        tuple(source_cache.triangle_signature),
        tuple(source_cache.particle_signature),
        float(target.external_contact_distance),
        float(source.external_contact_distance),
        int(target.collision_layer),
        int(source.collision_layer),
    )


def _cross_pair_cover_padding(target: ClothSlot, source: ClothSlot) -> float:
    _ = target
    _ = source
    return 0.0


def _cross_pair_key_static_prefix(package_key: tuple) -> tuple:
    # The first seven fields contain mode, target/source names, position generations, and AABB.
    return tuple(package_key[:3]) + tuple(package_key[7:])


def _cross_pair_cache_can_reuse(
    package: CrossClothPairColliderPackage,
    package_key: tuple,
    lower: np.ndarray,
    upper: np.ndarray,
    source_motion_accumulated: float,
) -> bool:
    if package.cover_lower.size != 3 or package.cover_upper.size != 3 or not package.key:
        return False
    if _cross_pair_key_static_prefix(package.key) != _cross_pair_key_static_prefix(package_key):
        return False
    motion_since_filter = max(0.0, float(source_motion_accumulated) - float(package.source_motion_stamp))
    padded_lower = np.asarray(lower, dtype=np.float32).reshape((3,)) - motion_since_filter
    padded_upper = np.asarray(upper, dtype=np.float32).reshape((3,)) + motion_since_filter
    return bool(np.all(padded_lower >= package.cover_lower) and np.all(padded_upper <= package.cover_upper))


def _refresh_pair_collider_package(
    package: CrossClothPairColliderPackage,
    cache: CrossClothColliderCache,
    positions: np.ndarray,
) -> None:
    if package.full_vertex_selection:
        package.particle_positions = positions
        package.particle_radii = cache.particle_radii
    else:
        selected_vertices = package.selected_vertices.reshape((-1,))
        particle_count = int(len(selected_vertices))
        if particle_count <= 0:
            package.particle_positions = np.empty((0, 3), dtype=np.float32)
            package.particle_radii = np.empty(0, dtype=np.float32)
        else:
            package.particle_position_buffer = _ensure_float32_capacity(
                package.particle_position_buffer,
                (particle_count, 3),
            )
            package.particle_radius_buffer = _ensure_float32_capacity(
                package.particle_radius_buffer,
                (particle_count,),
            )
            position_view = package.particle_position_buffer[:particle_count]
            radius_view = package.particle_radius_buffer[:particle_count]
            np.take(positions, selected_vertices, axis=0, out=position_view)
            np.take(cache.particle_radii, selected_vertices, axis=0, out=radius_view)
            package.particle_positions = position_view
            package.particle_radii = radius_view

    if package.full_triangle_selection:
        selected_triangles = cache.triangle_indices
    else:
        selected_triangles = package.selected_triangles
    triangle_count = int(len(selected_triangles))
    if triangle_count <= 0:
        package.triangles = np.empty((0, 3, 3), dtype=np.float32)
        return
    package.triangle_buffer = _ensure_float32_capacity(
        package.triangle_buffer,
        (triangle_count, 3, 3),
    )
    triangle_view = package.triangle_buffer[:triangle_count]
    np.take(positions, selected_triangles.reshape((-1,)), axis=0, out=triangle_view.reshape((-1, 3)))
    package.triangles = triangle_view


def _build_pair_collider_package(
    package: CrossClothPairColliderPackage,
    cache: CrossClothColliderCache,
    positions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cover_padding: float,
    source_motion_accumulated: float,
) -> None:
    cover_lower = np.asarray(lower, dtype=np.float32).reshape((3,)) - float(cover_padding)
    cover_upper = np.asarray(upper, dtype=np.float32).reshape((3,)) + float(cover_padding)
    if (
        cache.swept_min.size == 3
        and cache.swept_max.size == 3
        and bool(np.all(cover_lower <= cache.swept_min))
        and bool(np.all(cover_upper >= cache.swept_max))
    ):
        package.full_vertex_selection = True
        package.full_triangle_selection = True
        package.selected_vertices = np.empty(0, dtype=np.int32)
        package.selected_triangles = np.empty((0, 3), dtype=np.int32)
        package.cover_lower = cover_lower
        package.cover_upper = cover_upper
        package.source_motion_stamp = float(source_motion_accumulated)
        _refresh_pair_collider_package(package, cache, positions)
        return

    vertex_mask = _cross_source_vertex_mask(cache, positions, cover_lower, cover_upper)
    if not bool(np.any(vertex_mask)):
        package.full_vertex_selection = False
        package.full_triangle_selection = False
        package.selected_vertices = np.empty(0, dtype=np.int32)
        package.selected_triangles = np.empty((0, 3), dtype=np.int32)
        package.triangles = np.empty((0, 3, 3), dtype=np.float32)
        package.particle_positions = np.empty((0, 3), dtype=np.float32)
        package.particle_radii = np.empty(0, dtype=np.float32)
        package.cover_lower = cover_lower
        package.cover_upper = cover_upper
        package.source_motion_stamp = float(source_motion_accumulated)
        return

    if bool(np.all(vertex_mask)):
        package.full_vertex_selection = True
        package.selected_vertices = np.empty(0, dtype=np.int32)
        package.particle_positions = positions
        package.particle_radii = cache.particle_radii
    else:
        selected_vertices = np.flatnonzero(vertex_mask)
        particle_count = int(len(selected_vertices))
        package.full_vertex_selection = False
        package.selected_vertices = np.ascontiguousarray(selected_vertices, dtype=np.int32)
        package.particle_position_buffer = _ensure_float32_capacity(
            package.particle_position_buffer,
            (particle_count, 3),
        )
        package.particle_radius_buffer = _ensure_float32_capacity(
            package.particle_radius_buffer,
            (particle_count,),
        )
        position_view = package.particle_position_buffer[:particle_count]
        radius_view = package.particle_radius_buffer[:particle_count]
        np.take(positions, selected_vertices, axis=0, out=position_view)
        np.take(cache.particle_radii, selected_vertices, axis=0, out=radius_view)
        package.particle_positions = position_view
        package.particle_radii = radius_view

    triangles = cache.triangle_indices
    if len(triangles) == 0:
        package.full_triangle_selection = False
        package.selected_triangles = np.empty((0, 3), dtype=np.int32)
        package.triangles = np.empty((0, 3, 3), dtype=np.float32)
        package.cover_lower = cover_lower
        package.cover_upper = cover_upper
        package.source_motion_stamp = float(source_motion_accumulated)
        return
    tri_mask = _cross_source_triangle_mask(cache, triangles, vertex_mask)
    if not bool(np.any(tri_mask)):
        package.full_triangle_selection = False
        package.selected_triangles = np.empty((0, 3), dtype=np.int32)
        package.triangles = np.empty((0, 3, 3), dtype=np.float32)
        package.cover_lower = cover_lower
        package.cover_upper = cover_upper
        package.source_motion_stamp = float(source_motion_accumulated)
        return

    if bool(np.all(tri_mask)):
        package.full_triangle_selection = True
        package.selected_triangles = np.empty((0, 3), dtype=np.int32)
        selected_triangles = triangles
        triangle_count = int(len(triangles))
    else:
        selected_triangle_ids = np.flatnonzero(tri_mask)
        triangle_count = int(len(selected_triangle_ids))
        package.full_triangle_selection = False
        package.triangle_index_buffer = _ensure_int32_capacity(
            package.triangle_index_buffer,
            (triangle_count, 3),
        )
        selected_triangles = package.triangle_index_buffer[:triangle_count]
        np.take(triangles, selected_triangle_ids, axis=0, out=selected_triangles)
        package.selected_triangles = selected_triangles

    package.triangle_buffer = _ensure_float32_capacity(
        package.triangle_buffer,
        (triangle_count, 3, 3),
    )
    triangle_view = package.triangle_buffer[:triangle_count]
    np.take(positions, selected_triangles.reshape((-1,)), axis=0, out=triangle_view.reshape((-1, 3)))
    package.triangles = triangle_view
    package.cover_lower = cover_lower
    package.cover_upper = cover_upper
    package.source_motion_stamp = float(source_motion_accumulated)


def _cross_source_vertex_mask(
    cache: CrossClothColliderCache,
    positions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    count = int(len(positions))
    cache.vertex_mask = _ensure_bool_buffer(cache.vertex_mask, count)
    cache.vertex_mask_tmp = _ensure_bool_buffer(cache.vertex_mask_tmp, count)
    mask = cache.vertex_mask
    tmp = cache.vertex_mask_tmp
    np.greater_equal(positions[:, 0], lower[0], out=mask)
    np.less_equal(positions[:, 0], upper[0], out=tmp)
    np.logical_and(mask, tmp, out=mask)
    np.greater_equal(positions[:, 1], lower[1], out=tmp)
    np.logical_and(mask, tmp, out=mask)
    np.less_equal(positions[:, 1], upper[1], out=tmp)
    np.logical_and(mask, tmp, out=mask)
    np.greater_equal(positions[:, 2], lower[2], out=tmp)
    np.logical_and(mask, tmp, out=mask)
    np.less_equal(positions[:, 2], upper[2], out=tmp)
    np.logical_and(mask, tmp, out=mask)
    return mask


def _cross_source_triangle_mask(
    cache: CrossClothColliderCache,
    triangles: np.ndarray,
    vertex_mask: np.ndarray,
) -> np.ndarray:
    count = int(len(triangles))
    cache.triangle_mask = _ensure_bool_buffer(cache.triangle_mask, count)
    cache.triangle_mask_tmp = _ensure_bool_buffer(cache.triangle_mask_tmp, count)
    tri_mask = cache.triangle_mask
    tmp = cache.triangle_mask_tmp
    np.take(vertex_mask, triangles[:, 0], out=tri_mask)
    np.take(vertex_mask, triangles[:, 1], out=tmp)
    np.logical_or(tri_mask, tmp, out=tri_mask)
    np.take(vertex_mask, triangles[:, 2], out=tmp)
    np.logical_or(tri_mask, tmp, out=tri_mask)
    return tri_mask


def _positions_aabb_fast(positions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if positions is None:
        return None
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        return None
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    if bool(np.isfinite(minimum).all()) and bool(np.isfinite(maximum).all()):
        return minimum, maximum
    return _positions_aabb(array)


def _slot_max_motion_cached(cache: CrossClothColliderCache, current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape or current.ndim != 2 or current.shape[1] != 3 or len(current) == 0:
        return 0.0
    count = int(len(current))
    if cache.motion_delta.shape != current.shape:
        cache.motion_delta = np.empty_like(current, dtype=np.float32)
    if cache.motion_sq.shape != (count,):
        cache.motion_sq = np.empty(count, dtype=np.float32)
    np.subtract(current, previous, out=cache.motion_delta, casting="unsafe")
    delta = cache.motion_delta
    sq = cache.motion_sq
    np.multiply(delta[:, 0], delta[:, 0], out=sq)
    np.add(sq, delta[:, 1] * delta[:, 1], out=sq)
    np.add(sq, delta[:, 2] * delta[:, 2], out=sq)
    max_sq = float(np.max(sq)) if count > 0 else 0.0
    if not math.isfinite(max_sq) or max_sq <= 0.0:
        return 0.0 if max_sq == 0.0 else _slot_max_motion_arrays(current, previous)
    return float(math.sqrt(max_sq))


def _slot_max_motion_arrays(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape or current.ndim != 2 or current.shape[1] != 3 or len(current) == 0:
        return 0.0
    delta = current - previous
    distances = np.linalg.norm(delta, axis=1)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return 0.0
    return float(np.max(finite))


def _refresh_full_dynamic_triangle_cache(cache: CrossClothColliderCache, positions: np.ndarray) -> None:
    triangles = cache.triangle_indices
    if (
        len(positions) == 0
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
    ):
        cache.triangles = np.empty((0, 3, 3), dtype=np.float32)
        return
    triangle_count = int(len(triangles))
    cache.triangles = _ensure_float32_capacity(cache.triangles, (triangle_count, 3, 3))
    triangle_view = cache.triangles[:triangle_count]
    np.take(positions, triangles.reshape((-1,)), axis=0, out=triangle_view.reshape((-1, 3)))
    cache.triangles = triangle_view


def _mesh_topology_signature(mesh: bpy.types.Mesh) -> tuple:
    loop_indices = np.empty(len(mesh.loops), dtype=np.int32)
    if len(loop_indices) > 0:
        mesh.loops.foreach_get("vertex_index", loop_indices)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(loop_indices.shape, dtype=np.int64).tobytes())
    digest.update(loop_indices.tobytes())
    return (
        int(len(mesh.vertices)),
        int(len(mesh.polygons)),
        int(len(mesh.loops)),
        digest.hexdigest(),
    )


def _mesh_topology_count_signature(mesh: bpy.types.Mesh) -> tuple:
    return (
        int(len(mesh.vertices)),
        int(len(mesh.polygons)),
        int(len(mesh.loops)),
    )


def _triangulated_faces_fast(mesh: bpy.types.Mesh) -> np.ndarray:
    polygon_count = int(len(mesh.polygons))
    loop_count = int(len(mesh.loops))
    if polygon_count > 0 and loop_count == polygon_count * 3:
        loop_totals = np.empty(polygon_count, dtype=np.int32)
        mesh.polygons.foreach_get("loop_total", loop_totals)
        if bool(np.all(loop_totals == 3)):
            loop_indices = np.empty(loop_count, dtype=np.int32)
            mesh.loops.foreach_get("vertex_index", loop_indices)
            return np.ascontiguousarray(loop_indices.reshape((-1, 3)), dtype=np.int32)
    return np.ascontiguousarray(triangulated_faces(mesh), dtype=np.int32)


def _dynamic_source_world_positions_into(
    source: DynamicCollisionSource,
    mesh: bpy.types.Mesh,
    matrix_world,
) -> None:
    count = int(len(mesh.vertices))
    shape = (count, 3)
    if source.local_positions_buffer.shape != shape:
        source.local_positions_buffer = np.empty(shape, dtype=np.float32)
    initialize_previous = False
    if source.current_positions_world.shape == shape:
        if source.previous_positions_world.shape != shape:
            source.previous_positions_world = np.empty(shape, dtype=np.float32)
        np.copyto(source.previous_positions_world, source.current_positions_world, casting="unsafe")
    else:
        source.current_positions_world = np.empty(shape, dtype=np.float32)
        source.previous_positions_world = np.empty(shape, dtype=np.float32)
        initialize_previous = True

    local = source.local_positions_buffer
    flat = local.reshape((-1,))
    if "position" in mesh.attributes:
        mesh.attributes["position"].data.foreach_get("vector", flat)
    else:
        mesh.vertices.foreach_get("co", flat)

    matrix = np.asarray(matrix_world, dtype=np.float32)
    world = source.current_positions_world
    np.multiply(local[:, 0], matrix[0, 0], out=world[:, 0])
    np.add(world[:, 0], local[:, 1] * matrix[0, 1], out=world[:, 0])
    np.add(world[:, 0], local[:, 2] * matrix[0, 2], out=world[:, 0])
    np.add(world[:, 0], matrix[0, 3], out=world[:, 0])

    np.multiply(local[:, 0], matrix[1, 0], out=world[:, 1])
    np.add(world[:, 1], local[:, 1] * matrix[1, 1], out=world[:, 1])
    np.add(world[:, 1], local[:, 2] * matrix[1, 2], out=world[:, 1])
    np.add(world[:, 1], matrix[1, 3], out=world[:, 1])

    np.multiply(local[:, 0], matrix[2, 0], out=world[:, 2])
    np.add(world[:, 2], local[:, 1] * matrix[2, 1], out=world[:, 2])
    np.add(world[:, 2], local[:, 2] * matrix[2, 2], out=world[:, 2])
    np.add(world[:, 2], matrix[2, 3], out=world[:, 2])

    if initialize_previous:
        np.copyto(source.previous_positions_world, world, casting="unsafe")


def _triangle_max_edge_length(positions: np.ndarray, triangles: np.ndarray) -> float:
    if (
        len(positions) == 0
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
    ):
        return 0.08
    if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= len(positions):
        return 0.08
    a = positions[triangles[:, 0]]
    b = positions[triangles[:, 1]]
    c = positions[triangles[:, 2]]
    ab = b - a
    bc = c - b
    ca = a - c
    max_sq = max(
        float(np.max(np.einsum("ij,ij->i", ab, ab))),
        float(np.max(np.einsum("ij,ij->i", bc, bc))),
        float(np.max(np.einsum("ij,ij->i", ca, ca))),
    )
    if not math.isfinite(max_sq) or max_sq <= 0.0:
        return 0.08
    return float(math.sqrt(max_sq))


def _refresh_dynamic_collision_source(
    source: DynamicCollisionSource,
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph | None,
) -> None:
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        topology_signature = _mesh_topology_count_signature(mesh)
        _dynamic_source_world_positions_into(source, mesh, eval_obj.matrix_world)
        if topology_signature != source.topology_signature:
            source.triangle_indices = _triangulated_faces_fast(mesh)
            source.topology_signature = topology_signature
            source.dynamic_collider_cache.triangle_signature = ()
            source.dynamic_collider_cache.particle_signature = ()
            source.dynamic_collider_cache.pair_packages.clear()
            source.max_edge_length = _triangle_max_edge_length(source.current_positions_world, source.triangle_indices)
    finally:
        eval_obj.to_mesh_clear()


def _dynamic_collider_collection_for_settings(settings) -> bpy.types.Collection | None:
    return getattr(settings, "dynamic_collider_collection", None) if settings is not None else None


def _iter_dynamic_collider_objects(
    settings,
    target_obj: bpy.types.Object | None,
) -> list[bpy.types.Object]:
    collection = _dynamic_collider_collection_for_settings(settings)
    if collection is None:
        return []
    objects: list[bpy.types.Object] = []
    seen: set[str] = set()
    for obj in _iter_collection_mesh_objects(collection):
        if obj is None or obj.type != "MESH":
            continue
        if target_obj is not None and (obj == target_obj or obj.name == target_obj.name):
            continue
        if obj.name in seen or not _has_simulatable_cloth_source_geometry(obj):
            continue
        seen.add(obj.name)
        objects.append(obj)
    return objects


def _dynamic_collider_object_names(
    settings,
    target_obj: bpy.types.Object | None,
) -> set[str]:
    return {obj.name for obj in _iter_dynamic_collider_objects(settings, target_obj)}


def _sync_session_dynamic_collision_sources(
    context: bpy.types.Context,
    session: SceneSession,
    depsgraph: bpy.types.Depsgraph | None,
    slot_settings: dict[str, object],
) -> None:
    desired_targets: dict[str, set[str]] = {}
    desired_distance: dict[str, float] = {}
    for slot_name, slot in session.slots.items():
        obj = bpy.data.objects.get(slot.object_name)
        settings = slot_settings.get(slot_name)
        for source_obj in _iter_dynamic_collider_objects(settings, obj):
            target_names = desired_targets.setdefault(source_obj.name, set())
            target_names.add(slot_name)
            desired_distance[source_obj.name] = max(
                float(desired_distance.get(source_obj.name, 0.0)),
                float(slot.external_contact_distance),
            )

    for stale_name in set(session.dynamic_collision_sources.keys()) - set(desired_targets.keys()):
        session.dynamic_collision_sources.pop(stale_name, None)
        _OBJECT_TO_SCENE_SESSION.pop(stale_name, None)

    scene_key = session.scene_name
    for source_name, target_names in desired_targets.items():
        obj = bpy.data.objects.get(source_name)
        if obj is None or obj.type != "MESH":
            continue
        source = session.dynamic_collision_sources.get(source_name)
        if source is None:
            source = DynamicCollisionSource(object_name=source_name)
            session.dynamic_collision_sources[source_name] = source
        source.target_names = set(target_names)
        source.external_contact_distance = max(float(desired_distance.get(source_name, 0.0)), 1.0e-6)
        source.collision_layer = _object_collision_layer(obj)
        _refresh_dynamic_collision_source(source, obj, depsgraph)
        _OBJECT_TO_SCENE_SESSION[source_name] = scene_key


def _dynamic_source_enabled_for_target(target: ClothSlot, source: DynamicCollisionSource) -> bool:
    if target.object_name not in source.target_names:
        return False
    source_obj = bpy.data.objects.get(source.object_name)
    if source_obj is not None and not bool(
        getattr(
            source_obj,
            "ssbl_enable_cross_cloth_collision",
            source_obj.get("ssbl_enable_cross_cloth_collision", True),
        )
    ):
        return False
    return True


def _iter_all_dynamic_collision_sources(session: SceneSession):
    yield from session.slots.values()
    yield from session.dynamic_collision_sources.values()


def _session_has_dynamic_collision_sources(session: SceneSession) -> bool:
    return any(source.target_names for source in session.dynamic_collision_sources.values())


def _cross_cloth_slots_enabled(session: SceneSession) -> bool:
    return len(session.slots) > 1 and str(session.cross_cloth_mode or "off").lower() != "off"


def _session_dynamic_colliders_enabled(session: SceneSession) -> bool:
    return _session_has_dynamic_collision_sources(session) or _cross_cloth_slots_enabled(session)


def _session_can_use_global_dynamic_scene(session: SceneSession) -> bool:
    if not _cross_cloth_slots_enabled(session):
        return False
    if _session_has_dynamic_collision_sources(session):
        return False
    if os.environ.get("SSBL_DISABLE_GLOBAL_DYNAMIC_SCENE", "").lower() in {"1", "true", "yes", "on"}:
        return False
    return all(bool(getattr(slot.native, "supports_global_dynamic_scene", False)) for slot in session.slots.values())


def _ensure_global_dynamic_scene(session: SceneSession) -> NativeGlobalDynamicScene | None:
    if not _session_can_use_global_dynamic_scene(session):
        if session.global_dynamic_scene is not None:
            try:
                session.global_dynamic_scene.close()
            except Exception:
                pass
        session.global_dynamic_scene = None
        session.global_dynamic_scene_enabled = False
        return None
    if session.global_dynamic_scene is None:
        session.global_dynamic_scene = NativeGlobalDynamicScene()
    if not session.global_dynamic_source_ids or set(session.global_dynamic_source_ids.keys()) != set(session.solve_order):
        session.global_dynamic_source_ids = {
            name: index + 1
            for index, name in enumerate(session.solve_order)
        }
    session.global_dynamic_scene_enabled = True
    return session.global_dynamic_scene


def _update_global_dynamic_scene(session: SceneSession, perf: FramePerf | None = None) -> bool:
    scene = _ensure_global_dynamic_scene(session)
    if scene is None:
        for slot in session.slots.values():
            slot.native.clear_global_dynamic_scene()
        return False
    started = time.perf_counter()
    positions_list: list[np.ndarray] = []
    previous_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    source_id_list: list[np.ndarray] = []
    layer_list: list[np.ndarray] = []
    triangle_list: list[np.ndarray] = []
    triangle_source_id_list: list[np.ndarray] = []
    triangle_layer_list: list[np.ndarray] = []
    collision_margin = 0.0
    cloth_thickness = 0.0
    for slot_name in session.solve_order:
        slot = session.slots[slot_name]
        cache = slot.dynamic_collider_cache
        positions = np.ascontiguousarray(cache.particle_positions, dtype=np.float32).reshape((-1, 3))
        if len(positions) == 0:
            continue
        previous = np.ascontiguousarray(slot.previous_positions_world, dtype=np.float32).reshape((-1, 3))
        if previous.shape != positions.shape:
            previous = positions
        source_id = int(session.global_dynamic_source_ids.get(slot_name, 0))
        layer = int(slot.collision_layer)
        positions_list.append(positions)
        previous_list.append(previous)
        radii_list.append(np.ascontiguousarray(cache.particle_radii, dtype=np.float32).reshape((-1,)))
        source_id_list.append(np.full(len(positions), source_id, dtype=np.int32))
        layer_list.append(np.full(len(positions), layer, dtype=np.int32))
        triangles = np.ascontiguousarray(cache.triangles, dtype=np.float32).reshape((-1, 3, 3))
        if len(triangles) > 0:
            triangle_list.append(triangles)
            triangle_source_id_list.append(np.full(len(triangles), source_id, dtype=np.int32))
            triangle_layer_list.append(np.full(len(triangles), layer, dtype=np.int32))
        collision_margin = max(collision_margin, float(slot.external_contact_distance))
        cloth_thickness = max(cloth_thickness, float(getattr(slot.native.cached_diagnostics(), "min_gap", 0.0) or 0.0))
    if positions_list:
        particle_positions = _single_or_concat_float32(positions_list, (0, 3))
        particle_previous = _single_or_concat_float32(previous_list, (0, 3))
        particle_radii = _single_or_concat_float32(radii_list, (0,)).reshape((-1,))
        particle_source_ids = _single_or_concat_int32(source_id_list)
        particle_layers = _single_or_concat_int32(layer_list)
    else:
        particle_positions = np.empty((0, 3), dtype=np.float32)
        particle_previous = np.empty((0, 3), dtype=np.float32)
        particle_radii = np.empty(0, dtype=np.float32)
        particle_source_ids = np.empty(0, dtype=np.int32)
        particle_layers = np.empty(0, dtype=np.int32)
    if triangle_list:
        triangle_vertices = _single_or_concat_float32(triangle_list, (0, 3, 3))
        triangle_source_ids = _single_or_concat_int32(triangle_source_id_list)
        triangle_layers = _single_or_concat_int32(triangle_layer_list)
    else:
        triangle_vertices = np.empty((0, 3, 3), dtype=np.float32)
        triangle_source_ids = np.empty(0, dtype=np.int32)
        triangle_layers = np.empty(0, dtype=np.int32)
    scene.update(
        particle_positions=particle_positions,
        particle_prev_positions=particle_previous,
        particle_radii=particle_radii,
        particle_source_ids=particle_source_ids,
        particle_collision_layers=particle_layers,
        triangle_vertices=triangle_vertices,
        triangle_source_ids=triangle_source_ids,
        triangle_collision_layers=triangle_layers,
        collision_margin=collision_margin,
        cloth_thickness=collision_margin,
    )
    if perf is not None:
        perf.dynamic_collider_pack_ms += _elapsed_ms(started)
        perf.dynamic_triangle_count += int(len(triangle_vertices))
        perf.dynamic_particle_count += int(len(particle_positions))
    for slot_name, slot in session.slots.items():
        slot.native.attach_global_dynamic_scene(
            scene,
            target_source_id=int(session.global_dynamic_source_ids.get(slot_name, 0)),
            target_collision_layer=int(slot.collision_layer),
            cross_mode=session.cross_cloth_mode,
        )
    return True


def _prepare_cross_cloth_collider_caches(session: SceneSession, perf: FramePerf | None = None) -> None:
    started = time.perf_counter()
    source_names = list(session.solve_order) + sorted(session.dynamic_collision_sources.keys())
    slot_ids = {name: index + 1 for index, name in enumerate(source_names)}
    for source in _iter_all_dynamic_collision_sources(session):
        positions = np.asarray(source.current_positions_world, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            positions = np.empty((0, 3), dtype=np.float32)
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        cache = source.dynamic_collider_cache
        previous = np.asarray(source.previous_positions_world, dtype=np.float32)
        current_aabb = _positions_aabb_fast(positions)
        previous_aabb = _positions_aabb_fast(previous)
        if current_aabb is None:
            swept = previous_aabb
        elif previous_aabb is None:
            swept = current_aabb
        else:
            swept = (np.minimum(current_aabb[0], previous_aabb[0]), np.maximum(current_aabb[1], previous_aabb[1]))
        if swept is None:
            cache.swept_min = np.empty(0, dtype=np.float32)
            cache.swept_max = np.empty(0, dtype=np.float32)
        else:
            cache.swept_min = np.asarray(swept[0], dtype=np.float32)
            cache.swept_max = np.asarray(swept[1], dtype=np.float32)
        cache.max_motion = _slot_max_motion_cached(cache, positions, previous)
        if cache.particle_positions.shape != positions.shape or cache.max_motion > 1.0e-8:
            cache.positions_generation += 1
        cache.motion_accumulated += float(max(cache.max_motion, 0.0))

        if isinstance(source, DynamicCollisionSource):
            triangles = np.asarray(source.triangle_indices, dtype=np.int32)
            topology_token = tuple(source.topology_signature)
        else:
            triangles = np.asarray(source.cloth.triangles, dtype=np.int32)
            topology_token = (id(source.cloth.triangles), id(source.cloth.edge_rest_lengths))
        triangles_valid = (
            triangles.ndim == 2
            and triangles.shape[1] == 3
            and len(positions) > 0
            and (len(triangles) == 0 or (int(np.min(triangles)) >= 0 and int(np.max(triangles)) < len(positions)))
        )
        triangle_signature = (
            int(len(triangles)) if triangles_valid else 0,
            int(len(positions)),
            topology_token,
        )
        if cache.triangle_signature != triangle_signature:
            cache.triangles = np.empty((0, 3, 3), dtype=np.float32)
            cache.triangle_indices = (
                np.ascontiguousarray(triangles, dtype=np.int32)
                if triangles_valid
                else np.empty((0, 3), dtype=np.int32)
            )
            cache.triangle_signature = triangle_signature
            cache.pair_packages.clear()
            cache.motion_accumulated = 0.0
            if isinstance(source, DynamicCollisionSource):
                rest_max = float(source.max_edge_length)
            else:
                rest_lengths = np.asarray(source.cloth.edge_rest_lengths, dtype=np.float32).reshape((-1,))
                finite_rest = rest_lengths[np.isfinite(rest_lengths)]
                rest_max = float(np.max(finite_rest)) if len(finite_rest) else 0.0
            cache.max_edge_length = max(
                rest_max * 3.0,
                float(source.external_contact_distance) * 8.0,
                0.08,
            )
            if perf is not None:
                perf.dynamic_collider_cache_misses += 1
        else:
            if perf is not None:
                perf.dynamic_collider_cache_hits += 1
        if not isinstance(source, DynamicCollisionSource):
            _refresh_full_dynamic_triangle_cache(cache, positions)

        count = int(len(positions))
        slot_id = int(slot_ids.get(source.object_name, 0))
        particle_radius = float(source.external_contact_distance)
        if isinstance(source, DynamicCollisionSource):
            motion_padding = min(
                max(float(cache.max_motion), 0.0),
                max(particle_radius * 1.5, 0.01),
            )
            particle_radius += motion_padding
        particle_signature = (
            count,
            float(particle_radius),
            int(source.collision_layer),
            slot_id,
            id(source.cloth.inv_mass) if not isinstance(source, DynamicCollisionSource) else tuple(source.topology_signature),
        )
        if cache.particle_signature != particle_signature or cache.particle_positions.shape != (count, 3):
            cache.particle_radii = np.full(count, float(particle_radius), dtype=np.float32)
            source_inv_mass = (
                np.zeros(count, dtype=np.float32)
                if isinstance(source, DynamicCollisionSource)
                else np.asarray(source.cloth.inv_mass, dtype=np.float32).reshape((-1,))
            )
            if len(source_inv_mass) == count:
                cache.particle_inv_mass = np.ascontiguousarray(source_inv_mass, dtype=np.float32)
            else:
                cache.particle_inv_mass = np.ones(count, dtype=np.float32)
            cache.particle_slot_ids = np.full(count, slot_id, dtype=np.int32)
            cache.particle_phases = np.full(count, int(source.collision_layer), dtype=np.int32)
            cache.particle_signature = particle_signature
            cache.pair_packages.clear()
            cache.motion_accumulated = 0.0
            if perf is not None:
                perf.dynamic_collider_cache_misses += 1
        else:
            if perf is not None:
                perf.dynamic_collider_cache_hits += 1
        cache.particle_positions = positions
    if perf is not None:
        perf.dynamic_collider_pack_ms += _elapsed_ms(started)


def _cross_source_expanded_target_aabb(
    target: ClothSlot,
    source: ClothSlot,
) -> tuple[np.ndarray, np.ndarray] | None:
    target_cache = target.dynamic_collider_cache
    source_cache = source.dynamic_collider_cache
    if (
        target_cache.swept_min.size != 3
        or target_cache.swept_max.size != 3
        or source_cache.swept_min.size != 3
        or source_cache.swept_max.size != 3
    ):
        return None
    target_aabb = (target_cache.swept_min, target_cache.swept_max)
    source_aabb = (source_cache.swept_min, source_cache.swept_max)
    contact_radius = max(float(target.external_contact_distance), float(source.external_contact_distance))
    motion_padding = max(
        contact_radius * 4.0,
        float(target_cache.max_motion) * 2.0,
        float(source_cache.max_motion) * 2.0,
        float(source_cache.max_edge_length),
        0.08,
    )
    padding = contact_radius + motion_padding
    if _aabb_distance(target_aabb, source_aabb) > padding:
        return None
    return target_aabb[0] - padding, target_aabb[1] + padding


def _collect_cross_cloth_triangles(session: SceneSession, target: ClothSlot) -> np.ndarray:
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off" and not _session_has_dynamic_collision_sources(session):
        return np.empty((0, 3, 3), dtype=np.float32)
    all_tris: list[np.ndarray] = []
    for source in session.slots.values():
        if not _cross_cloth_source_enabled(mode, target, source):
            continue
        triangles = source.dynamic_collider_cache.triangles
        if len(triangles) > 0:
            all_tris.append(triangles)
    for source in session.dynamic_collision_sources.values():
        if not _dynamic_source_enabled_for_target(target, source):
            continue
        triangles = source.dynamic_collider_cache.triangles
        if len(triangles) > 0:
            all_tris.append(triangles)
    if not all_tris:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(all_tris, axis=0), dtype=np.float32)


def _collect_cross_cloth_colliders(
    session: SceneSession,
    target: ClothSlot,
    perf: FramePerf | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off" and not _session_has_dynamic_collision_sources(session):
        return np.empty((0, 3, 3), dtype=np.float32), _empty_dynamic_indexed_triangles(), _empty_dynamic_particles()
    triangles_list: list[np.ndarray] = []
    vertex_list: list[np.ndarray] = []
    triangle_index_list: list[np.ndarray] = []
    triangle_index_signature_items: list[tuple] = []
    positions_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    particle_signature_items: list[tuple] = []
    vertex_offset = 0
    for source in session.slots.values():
        if not _cross_cloth_source_enabled(mode, target, source):
            continue
        expanded = _cross_source_expanded_target_aabb(target, source)
        if expanded is None:
            continue
        cache = source.dynamic_collider_cache
        positions = cache.particle_positions
        if len(positions) == 0:
            continue
        package_id = (str(target.object_name), str(source.object_name))
        package = cache.pair_packages.get(package_id)
        if package is None:
            package = CrossClothPairColliderPackage()
            cache.pair_packages[package_id] = package
        target_cache = target.dynamic_collider_cache
        can_attempt_reuse = (
            float(cache.max_motion) <= 1.0e-8
            and float(target_cache.max_motion) <= 1.0e-8
        )
        if can_attempt_reuse:
            package_key = _cross_cloth_pair_key(mode, target, source, expanded)
            if _cross_pair_cache_can_reuse(
                package,
                package_key,
                expanded[0],
                expanded[1],
                cache.motion_accumulated,
            ):
                _refresh_pair_collider_package(package, cache, positions)
                if perf is not None:
                    perf.dynamic_pair_cache_hits += 1
                    perf.dynamic_pair_cache_reused_triangles += int(len(package.triangles))
                    perf.dynamic_pair_cache_reused_particles += int(len(package.particle_positions))
            else:
                _build_pair_collider_package(
                    package,
                    cache,
                    positions,
                    expanded[0],
                    expanded[1],
                    _cross_pair_cover_padding(target, source),
                    cache.motion_accumulated,
                )
                package.key = package_key
                if perf is not None:
                    perf.dynamic_pair_cache_misses += 1
        else:
            _build_pair_collider_package(
                package,
                cache,
                positions,
                expanded[0],
                expanded[1],
                _cross_pair_cover_padding(target, source),
                cache.motion_accumulated,
            )
            package.key = ()
            if perf is not None:
                perf.dynamic_pair_cache_misses += 1
        if len(package.particle_positions) == 0:
            continue
        positions_list.append(package.particle_positions)
        radii_list.append(package.particle_radii)
        particle_signature_items.append(
            (
                "cloth",
                str(source.object_name),
                int(source.dynamic_collider_cache.positions_generation),
                int(target.dynamic_collider_cache.positions_generation),
                int(len(package.particle_positions)),
                bool(package.full_vertex_selection),
                tuple(source.dynamic_collider_cache.particle_signature),
            )
        )
        if len(package.triangles) > 0:
            triangles_list.append(package.triangles)
    for source in session.dynamic_collision_sources.values():
        if not _dynamic_source_enabled_for_target(target, source):
            continue
        cache = source.dynamic_collider_cache
        positions = cache.particle_positions
        if len(positions) == 0:
            continue
        vertex_list.append(positions)
        positions_list.append(positions)
        radii_list.append(cache.particle_radii)
        particle_signature_items.append(
            (
                "dynamic_source",
                str(source.object_name),
                int(cache.positions_generation),
                int(target.dynamic_collider_cache.positions_generation),
                int(len(positions)),
                tuple(source.topology_signature),
                tuple(cache.particle_signature),
            )
        )
        if len(cache.triangle_indices) > 0:
            if int(vertex_offset) == 0:
                triangle_index_list.append(cache.triangle_indices)
            else:
                triangle_index_list.append(
                    np.ascontiguousarray(cache.triangle_indices + int(vertex_offset), dtype=np.int32)
                )
            triangle_index_signature_items.append(
                (
                    str(source.object_name),
                    int(len(positions)),
                    int(len(cache.triangle_indices)),
                    int(vertex_offset),
                    tuple(source.topology_signature),
                )
            )
        vertex_offset += int(len(positions))

    dynamic_triangles = _single_or_concat_float32(triangles_list, (0, 3, 3))
    dynamic_vertices = _single_or_concat_float32(vertex_list, (0, 3))
    dynamic_indices = _single_or_concat_int32_rows(triangle_index_list, (0, 3))
    if len(dynamic_triangles) > 0 and len(dynamic_indices) > 0:
        indexed_triangles = np.ascontiguousarray(dynamic_vertices[dynamic_indices], dtype=np.float32)
        dynamic_triangles = _single_or_concat_float32([dynamic_triangles, indexed_triangles], (0, 3, 3))
        dynamic_vertices = np.empty((0, 3), dtype=np.float32)
        dynamic_indices = np.empty((0, 3), dtype=np.int32)
        triangle_index_signature_items.clear()

    if not positions_list:
        return dynamic_triangles, {
            "vertices": dynamic_vertices,
            "indices": dynamic_indices,
            "signature": tuple(triangle_index_signature_items),
        }, _empty_dynamic_particles()
    return dynamic_triangles, {
        "vertices": dynamic_vertices,
        "indices": dynamic_indices,
        "signature": tuple(triangle_index_signature_items),
    }, {
        "positions": _single_or_concat_float32(positions_list, (0, 3)),
        "radii": _single_or_concat_float32(radii_list, (0,)).reshape((-1,)),
        "signature": tuple(particle_signature_items),
    }


def _collect_cross_cloth_particles(session: SceneSession, target: ClothSlot) -> dict[str, np.ndarray]:
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off" and not _session_has_dynamic_collision_sources(session):
        return _empty_dynamic_particles()
    positions_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    inv_mass_list: list[np.ndarray] = []
    slot_ids_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    for source in session.slots.values():
        if not _cross_cloth_source_enabled(mode, target, source):
            continue
        cache = source.dynamic_collider_cache
        positions = cache.particle_positions
        if len(positions) == 0:
            continue
        positions_list.append(positions)
        radii_list.append(cache.particle_radii)
        inv_mass_list.append(cache.particle_inv_mass)
        slot_ids_list.append(cache.particle_slot_ids)
        phase_list.append(cache.particle_phases)
    for source in session.dynamic_collision_sources.values():
        if not _dynamic_source_enabled_for_target(target, source):
            continue
        cache = source.dynamic_collider_cache
        positions = cache.particle_positions
        if len(positions) == 0:
            continue
        positions_list.append(positions)
        radii_list.append(cache.particle_radii)
        inv_mass_list.append(cache.particle_inv_mass)
        slot_ids_list.append(cache.particle_slot_ids)
        phase_list.append(cache.particle_phases)
    if not positions_list:
        return _empty_dynamic_particles()
    return {
        "positions": np.ascontiguousarray(np.concatenate(positions_list, axis=0), dtype=np.float32),
        "radii": np.ascontiguousarray(np.concatenate(radii_list, axis=0), dtype=np.float32),
        "inv_mass": np.ascontiguousarray(np.concatenate(inv_mass_list, axis=0), dtype=np.float32),
        "slot_ids": np.ascontiguousarray(np.concatenate(slot_ids_list, axis=0), dtype=np.int32),
        "phases": np.ascontiguousarray(np.concatenate(phase_list, axis=0), dtype=np.int32),
    }


def _cross_cloth_source_enabled(mode: str, target: ClothSlot, source: ClothSlot) -> bool:
    if source.object_name == target.object_name:
        return False
    source_obj = bpy.data.objects.get(source.object_name)
    if source_obj is not None and not bool(
        getattr(
            source_obj,
            "ssbl_enable_cross_cloth_collision",
            source_obj.get("ssbl_enable_cross_cloth_collision", True),
        )
    ):
        return False
    normalized = str(mode or "off").lower()
    if normalized == "off":
        return False
    if normalized == "lower_layers":
        return int(source.collision_layer) < int(target.collision_layer)
    return True


def _swept_positions_aabb(slot: ClothSlot) -> tuple[np.ndarray, np.ndarray] | None:
    current = _positions_aabb(slot.current_positions_world)
    previous = _positions_aabb(slot.previous_positions_world)
    if current is None:
        return previous
    if previous is None:
        return current
    return np.minimum(current[0], previous[0]), np.maximum(current[1], previous[1])


def _slot_max_motion(slot: ClothSlot) -> float:
    current = np.asarray(slot.current_positions_world, dtype=np.float32)
    previous = np.asarray(slot.previous_positions_world, dtype=np.float32)
    if current.shape != previous.shape or current.ndim != 2 or current.shape[1] != 3 or len(current) == 0:
        return 0.0
    delta = current - previous
    distances = np.linalg.norm(delta, axis=1)
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return 0.0
    return float(np.max(finite))


def _positions_aabb(positions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    if positions is None:
        return None
    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        return None
    finite = np.isfinite(array).all(axis=1)
    if not bool(np.any(finite)):
        return None
    finite_positions = array[finite]
    return np.min(finite_positions, axis=0), np.max(finite_positions, axis=0)


def _aabb_distance(
    aabb_a: tuple[np.ndarray, np.ndarray],
    aabb_b: tuple[np.ndarray, np.ndarray],
) -> float:
    a_min, a_max = aabb_a
    b_min, b_max = aabb_b
    gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(gap))


def _refresh_bake_runtime_inputs(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    cloth: ClothBuildData,
    native: NativeXpbdSolver,
    frame: int,
    expected_static_signature: tuple[tuple[str, int, int], ...],
    settings,
) -> None:
    context.scene.frame_set(frame)
    depsgraph = context.evaluated_depsgraph_get()
    use_evaluated_mesh = _effective_use_evaluated_mesh(obj, settings)
    world_positions, matrix_world = world_positions_from_object(
        obj,
        use_evaluated_mesh,
        depsgraph=depsgraph,
        expected_vertex_count=len(cloth.positions_world),
    )
    static_tris, static_signature = collect_static_triangles(
        settings.static_collider_collection,
        obj,
        depsgraph=depsgraph,
        use_evaluated_mesh=use_evaluated_mesh,
    )
    cloth.matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    pin_attachment_batch = make_pin_attachment_batch(
        cloth.pin_indices,
        np.ascontiguousarray(world_positions[cloth.pin_indices], dtype=np.float32),
    )
    pin_targets = pin_attachment_batch.targets_world
    force_fields = collect_force_fields(context.scene, depsgraph, settings) if has_force_field_sources(context.scene, settings) else None
    native.update_frame_inputs(
        pin_indices=cloth.pin_indices,
        pin_positions=pin_targets,
        pin_weights=cloth.pin_weights,
        update_pin=True,
        options=settings_to_options(settings, runtime_mode_override="bake"),
        update_runtime=True,
        static_triangles=static_tris,
        update_static=True,
        dynamic_triangles=None,
        update_dynamic=False,
        force_fields=force_fields,
        update_force_fields=force_fields is not None,
    )


def _apply_runtime_inputs(
    slot: ClothSlot,
    options,
    runtime_signature: tuple,
    pin_targets: np.ndarray,
    matrix_world,
    static_tris: np.ndarray | None,
    static_signature: tuple[tuple[str, int, int], ...],
    static_runtime_signature: tuple,
    force_fields: ForceFieldBatch | None,
    force_fields_enabled: bool,
    perf: FramePerf | None = None,
) -> None:
    slot.cloth.matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    pin_targets = np.ascontiguousarray(pin_targets, dtype=np.float32)
    update_pin = not _array_equal(pin_targets, slot.pin_targets_world)
    update_runtime = runtime_signature != slot.runtime_options_signature
    update_static = static_tris is not None
    update_force_fields = force_fields is not None
    force_fields_active = bool(force_fields_enabled)
    update_force_field_state = update_force_fields and slot.force_fields_active != force_fields_active
    if update_pin or update_runtime or update_static or update_force_fields:
        started = time.perf_counter()
        slot.native.update_frame_inputs(
            pin_indices=slot.cloth.pin_indices,
            pin_positions=pin_targets,
            pin_weights=slot.cloth.pin_weights,
            update_pin=update_pin,
            options=options,
            update_runtime=update_runtime,
            static_triangles=static_tris,
            update_static=update_static,
            dynamic_triangles=None,
            update_dynamic=False,
            force_fields=force_fields,
            update_force_fields=update_force_fields,
        )
        slot.pin_targets_world = np.array(pin_targets, dtype=np.float32, copy=True)
        if update_force_fields:
            slot.force_fields_active = force_fields_active
        if update_pin or update_runtime or update_static or update_force_field_state:
            slot.force_next_writeback = True
        if perf is not None:
            elapsed = _elapsed_ms(started)
            perf.frame_input_upload_ms += elapsed
            if update_pin:
                perf.pin_upload_ms += elapsed
            if update_runtime:
                perf.runtime_upload_ms += elapsed
            if update_static:
                perf.static_upload_ms += elapsed
        slot.runtime_options_signature = runtime_signature
    if update_static:
        slot.static_triangles = np.array(static_tris, dtype=np.float32, copy=True)
        slot.static_collider_signature = static_signature
        slot.static_runtime_signature = static_runtime_signature


def _effective_use_evaluated_mesh(obj: bpy.types.Object, settings) -> bool:
    if bool(getattr(settings, "use_evaluated_mesh", False)):
        return True
    # Hook modifiers are often used as animated handles for pin targets. If we
    # read the raw mesh in that case, the handles move in Blender but the solver
    # receives unmoved pins, producing the "only raw Hook deformation" look.
    return any(
        modifier.type == "HOOK" and bool(modifier.show_viewport)
        for modifier in obj.modifiers
    )


def _ensure_supported_cloth_object(obj: bpy.types.Object) -> None:
    if obj is None:
        raise ValueError("A mesh object is required.")
    declared_type = _declared_input_type(obj)
    if declared_type in _UNSUPPORTED_INPUT_TYPES:
        raise ValueError(f"SSBL v2 only supports cloth MESH input, not {declared_type}.")



    if obj.type != "MESH":
        raise ValueError("SSBL v2 currently only supports cloth MESH objects.")


def _declared_input_type(obj: bpy.types.Object) -> str:
    for key in ("ssbl_type", "ssbl_kind", "ppf_type", "simulation_type"):
        value = obj.get(key)
        if isinstance(value, str):
            return value.strip().lower()
    return "cloth"


def _object_collision_layer(obj: bpy.types.Object) -> int:
    return int(getattr(obj, _OBJECT_COLLISION_LAYER_PROP, obj.get(_OBJECT_COLLISION_LAYER_PROP, 1)))


def _mesh_is_probably_closed(mesh: bpy.types.Mesh) -> bool:
    if mesh is None or len(mesh.polygons) == 0:
        return False
    edge_use: dict[tuple[int, int], int] = {}
    for poly in mesh.polygons:
        vertices = list(poly.vertices)
        count = len(vertices)
        for index in range(count):
            a = int(vertices[index])
            b = int(vertices[(index + 1) % count])
            edge = (a, b) if a < b else (b, a)
            edge_use[edge] = edge_use.get(edge, 0) + 1
    if not edge_use:
        return False
    return all(use_count == 2 for use_count in edge_use.values())


def _mesh_is_probably_sphere_like(obj: bpy.types.Object) -> bool:
    if obj is None or obj.type != "MESH" or obj.data is None:
        return False
    mesh = obj.data
    name = obj.name.casefold()
    dimensions = [abs(float(value)) for value in obj.dimensions]
    max_dim = max(dimensions) if dimensions else 0.0
    min_dim = min(dimensions) if dimensions else 0.0
    roughly_round = max_dim > 1.0e-5 and (min_dim / max_dim) >= 0.72
    low_to_mid_poly = len(mesh.polygons) <= 4096
    name_hint = "sphere" in name or "ball" in name or "ico" in name or "uv" in name
    return low_to_mid_poly and roughly_round and (name_hint or _mesh_is_probably_closed(mesh))


def _finish_session(session: SceneSession, status: str, *, finalize_realtime_cache: bool = False) -> None:
    if session.closed:
        return
    session.closed = True
    _SCENE_SESSIONS.pop(session.scene_name, None)
    for slot_name in list(session.slots.keys()):
        _OBJECT_TO_SCENE_SESSION.pop(slot_name, None)
    for source_name in list(session.dynamic_collision_sources.keys()):
        _OBJECT_TO_SCENE_SESSION.pop(source_name, None)
        _LAST_DIAGNOSTICS[source_name] = session.last_diagnostics
        if _STATUS.get(source_name) == STATUS_PREVIEW_RUNNING:
            _STATUS[source_name] = status
    if session.global_dynamic_scene is not None:
        for slot in session.slots.values():
            try:
                slot.native.clear_global_dynamic_scene()
            except Exception:
                pass
        try:
            session.global_dynamic_scene.close()
        except Exception:
            pass
        session.global_dynamic_scene = None
        session.global_dynamic_scene_enabled = False

    for slot in list(session.slots.values()):
        _LAST_DIAGNOSTICS[slot.object_name] = session.last_diagnostics
        obj = bpy.data.objects.get(slot.object_name)
        if obj is not None and obj.type == "MESH" and _rna_alive(obj):
            _clear_interactive_pin_state(slot, obj)
            try:
                if _same_mesh(obj.data, slot.preview_mesh) and _rna_alive(slot.original_mesh):
                    obj.data = slot.original_mesh
            except (ReferenceError, RuntimeError, AttributeError):
                pass
            _restore_preview_modifiers(obj, slot.suspended_modifiers)
        slot_status = status
        if not _finish_realtime_cache_for_slot(
            slot,
            obj if obj is not None and _rna_alive(obj) else None,
            bool(finalize_realtime_cache) and status != STATUS_ERROR,
        ):
            slot_status = STATUS_ERROR
        try:
            slot.native.close()
        except Exception:
            pass
        finally:
            _safe_remove_mesh(slot.preview_mesh)
        _STATUS[slot.object_name] = slot_status
    scene = bpy.data.scenes.get(session.scene_name)
    if scene is not None and not bool(session.playback_driven):
        try:
            scene.frame_set(session.start_frame)
        except Exception:
            pass


def _suspend_preview_modifiers(obj: bpy.types.Object, suspend_all: bool = False) -> list[tuple[str, bool, bool]]:
    suspended: list[tuple[str, bool, bool]] = []
    for modifier in obj.modifiers:
        if not suspend_all and modifier.name != _CACHE_MODIFIER_NAME:
            continue
        suspended.append((modifier.name, bool(modifier.show_viewport), bool(modifier.show_render)))
        modifier.show_viewport = False
        modifier.show_render = False
    return suspended


def _restore_preview_modifiers(obj: bpy.types.Object, suspended: list[tuple[str, bool, bool]]) -> None:
    for name, show_viewport, show_render in suspended:
        modifier = obj.modifiers.get(name)
        if modifier is None:
            continue
        modifier.show_viewport = show_viewport
        modifier.show_render = show_render


def _disable_suspended_modifiers(obj: bpy.types.Object, suspended: list[tuple[str, bool, bool]]) -> None:
    for name, _show_viewport, _show_render in suspended:
        modifier = obj.modifiers.get(name)
        if modifier is None:
            continue
        modifier.show_viewport = False
        modifier.show_render = False


@contextmanager
def _temporary_setting(settings, name: str, value):
    old_value = getattr(settings, name)
    setattr(settings, name, value)
    try:
        yield
    finally:
        setattr(settings, name, old_value)


@contextmanager
def _with_preview_source_state(slot: ClothSlot, obj: bpy.types.Object):
    if not _rna_alive(obj) or not _rna_alive(slot.original_mesh):
        raise ValueError(f"Missing preview source mesh: {slot.object_name}")
    if slot.use_evaluated_mesh:
        _restore_preview_modifiers(obj, slot.suspended_modifiers)
    obj.data = slot.original_mesh
    try:
        yield
    finally:
        if _rna_alive(obj):
            if _rna_alive(slot.preview_mesh):
                obj.data = slot.preview_mesh
                _disable_suspended_modifiers(obj, slot.suspended_modifiers)
            elif _rna_alive(slot.original_mesh):
                obj.data = slot.original_mesh


@contextmanager
def _with_session_source_state(session: SceneSession):
    owned: list[tuple[ClothSlot, bpy.types.Object]] = []
    try:
        for slot in session.slots.values():
            obj = bpy.data.objects.get(slot.object_name)
            if obj is None or obj.type != "MESH":
                raise ValueError(f"Missing preview object: {slot.object_name}")
            if not _rna_alive(obj) or not _rna_alive(slot.original_mesh):
                raise ValueError(f"Missing preview source mesh: {slot.object_name}")
            if slot.use_evaluated_mesh:
                _restore_preview_modifiers(obj, slot.suspended_modifiers)
            obj.data = slot.original_mesh
            owned.append((slot, obj))
        yield
    finally:
        for slot, obj in owned:
            if not _rna_alive(obj):
                continue
            if not session.closed and _rna_alive(slot.preview_mesh):
                obj.data = slot.preview_mesh
                _disable_suspended_modifiers(obj, slot.suspended_modifiers)
            elif _rna_alive(slot.original_mesh):
                obj.data = slot.original_mesh


def _apply_world_positions(
    obj: bpy.types.Object,
    world_positions: np.ndarray,
    matrix_world_inv: np.ndarray,
    local_buffer: np.ndarray | None = None,
    flat_buffer: np.ndarray | None = None,
    perf: FramePerf | None = None,
) -> None:
    try:
        mesh = obj.data if obj is not None and obj.type == "MESH" else None
    except (ReferenceError, RuntimeError, AttributeError):
        mesh = None
    if not _rna_alive(mesh):
        raise ValueError("Preview mesh is no longer valid; restart preview.")

    started = time.perf_counter()
    world = np.asarray(world_positions, dtype=np.float32)
    matrix_inv = np.asarray(matrix_world_inv, dtype=np.float32)
    flat = None
    if flat_buffer is not None and flat_buffer.size == world.size:
        candidate = np.asarray(flat_buffer, dtype=np.float32)
        if candidate.ndim == 1 and candidate.flags.c_contiguous:
            flat = candidate
    local = local_buffer if local_buffer is not None and local_buffer.shape == world.shape else None
    if local is None:
        if flat is not None:
            local = flat.reshape(world.shape)
        else:
            local = np.empty_like(world, dtype=np.float32)
    if matrix_inv.shape == (4, 4) and np.allclose(matrix_inv, _IDENTITY_4X4, rtol=0.0, atol=1.0e-7):
        np.copyto(local, world, casting="unsafe")
    else:
        np.matmul(world, matrix_inv[:3, :3].T, out=local)
        local += matrix_inv[:3, 3]
    if flat is None:
        flat = np.ravel(local)
    elif not np.shares_memory(flat, local):
        np.copyto(flat, np.ravel(local), casting="unsafe")
    if perf is not None:
        perf.writeback_to_local_ms += _elapsed_ms(started)

    started = time.perf_counter()
    if "position" in mesh.attributes:
        mesh.attributes["position"].data.foreach_set("vector", flat)
    else:
        mesh.vertices.foreach_set("co", flat)
    if perf is not None:
        perf.writeback_foreach_set_ms += _elapsed_ms(started)

    started = time.perf_counter()
    mesh.update()
    if perf is not None:
        perf.writeback_mesh_update_ms += _elapsed_ms(started)


def _update_session_fps(session: SceneSession, _step_started: float) -> None:
    session.fps_sample_frames += 1
    now = time.perf_counter()
    elapsed = now - session.last_fps_time
    if elapsed < 0.25:
        return
    sample_fps = session.fps_sample_frames / max(elapsed, 1.0e-6)
    if session.actual_fps <= 0.0:
        session.actual_fps = sample_fps
    else:
        session.actual_fps = session.actual_fps * 0.65 + sample_fps * 0.35
    session.fps_sample_frames = 0
    session.last_fps_time = now


def _aggregate_session_diagnostics(session: SceneSession, perf: FramePerf | None = None) -> NativeStepDiagnostics:
    step_ms = 0.0
    hash_build_ms = 0.0
    constraints_ms = 0.0
    volume_ms = 0.0
    analytic_collision_ms = 0.0
    static_collision_ms = 0.0
    dynamic_collision_ms = 0.0
    dynamic_particle_collision_ms = 0.0
    self_hash_ms = 0.0
    self_solve_ms = 0.0
    self_probe_ms = 0.0
    self_recovery_ms = 0.0
    sync_ms = 0.0
    diagnostics_fetch_ms = 0.0
    frame_input_upload_ms = 0.0
    candidate_count = 0
    resolved_contacts = 0
    min_gap: float | None = None
    ccd_clamp_count = 0
    recovery_passes = 0
    local_retry_count = 0
    jitter_stabilized_vertices = 0
    jitter_rejected_vertices = 0
    jitter_max_correction = 0.0
    external_contact_cache_hits = 0
    external_contact_cache_misses = 0
    external_contact_cache_count = 0
    external_contact_cache_overflow = 0
    external_friction_corrections = 0
    force_field_count = 0
    unsupported_force_field_count = 0
    dynamic_particle_count = 0
    dynamic_particle_candidate_count = 0
    dynamic_particle_contacts = 0
    dynamic_particle_overflow = 0
    dynamic_triangle_count = 0
    dynamic_triangle_candidate_count = 0
    dynamic_triangle_bucket_overflow = 0
    dynamic_triangle_large_primitive_count = 0
    dynamic_triangle_aabb_reject_count = 0
    dynamic_triangle_max_bucket_occupancy = 0
    global_dynamic_scene_pack_ms = 0.0
    global_dynamic_scene_upload_ms = 0.0
    global_dynamic_hash_ms = 0.0
    global_dynamic_particle_count = 0
    global_dynamic_triangle_count = 0
    global_dynamic_hash_overflow = 0
    static_triangle_count = 0
    fast_exact_vt_candidates = 0
    fast_exact_vt_projected = 0
    fast_exact_vt_guarded = 0
    fast_exact_vt_skipped_rest = 0
    fast_soft_repulsion_candidates = 0
    fast_soft_repulsion_applied = 0
    fast_soft_repulsion_max_push = 0.0
    fast_hard_projection_count = 0
    fast_manifold_contacts = 0
    fast_manifold_reused = 0
    fast_barrier_projected = 0
    fast_barrier_smoothed_vertices = 0
    fast_barrier_overflow = 0
    fast_barrier_max_delta = 0.0
    fast_edge_edge_candidates = 0
    fast_edge_edge_contacts = 0
    fast_triangle_pair_candidates = 0
    fast_triangle_pair_contacts = 0
    fast_triangle_pair_skipped_rest = 0
    fast_contact_classification_guarded = 0
    fast_region_cluster_candidates = 0
    fast_region_cluster_contacts = 0
    fast_region_cluster_guarded = 0
    fast_overlap_island_candidates = 0
    fast_overlap_island_clusters = 0
    fast_overlap_island_vertex_refs = 0
    fast_overlap_island_applied_vertices = 0
    fast_overlap_island_guarded = 0
    fast_overlap_island_max_delta = 0.0
    fast_cc_overlap_components = 0
    fast_cc_overlap_seed_triangles = 0
    fast_cc_overlap_owned_vertices = 0
    fast_cc_overlap_union_edges = 0
    fast_cc_overlap_guarded = 0
    fast_cc_overlap_applied_vertices = 0
    fast_cc_overlap_max_delta = 0.0
    abi41_soft_contact_count = 0
    abi41_exact_impulse_contact_count = 0
    abi41_edge_edge_contact_count = 0
    abi41_max_smoothed_delta = 0.0
    abi41_hard_projection_fallbacks = 0
    static_sdf_rebuild_count = 0
    static_sdf_voxel_count = 0
    static_sdf_grid_x = 0
    static_sdf_grid_y = 0
    static_sdf_grid_z = 0
    static_sdf_build_ms = 0.0
    static_sdf_contact_count = 0
    static_sdf_unsigned_fallback_count = 0
    abi41_pcg_iterations = 0
    abi41_pcg_guarded = 0
    abi41_pcg_csr_nnz = 0
    abi41_pcg_texture_ready = 0
    abi41_pcg_initial_residual = 0.0
    abi41_pcg_final_residual = 0.0
    abi41_pcg_max_delta = 0.0
    abi41_pcg_solve_ms = 0.0
    abi41_pcg_system_ms = 0.0
    abi41_pcg_ad_ms = 0.0
    abi41_direct_stretch_ms = 0.0
    abi41_lra_tack_count = 0
    abi41_bending_wing_count = 0
    abi41_bending_texture_ready = 0
    abi41_tack_jitter_guarded = 0
    abi41_bending_guarded = 0
    dynamic_collider_pack_ms = 0.0
    dynamic_triangle_upload_ms = 0.0
    dynamic_particle_upload_ms = 0.0
    dynamic_collider_cache_hits = 0
    dynamic_collider_cache_misses = 0
    dynamic_pair_cache_hits = 0
    dynamic_pair_cache_misses = 0
    dynamic_pair_cache_reused_triangles = 0
    dynamic_pair_cache_reused_particles = 0
    dynamic_collision_skipped_launches = 0
    self_collision_skipped_launches = 0
    self_candidate_count = 0
    self_filter_seen = 0
    self_filter_accepted_vv = 0
    self_filter_accepted_vt = 0
    self_filter_accepted_ee = 0
    self_filter_rejected_rest = 0
    self_filter_rejected_duplicate = 0
    self_filter_rejected_ownership = 0
    self_filter_cache_hits = 0
    self_filter_cache_misses = 0
    self_cluster_count = 0
    self_cluster_owned_contacts = 0
    finite = True
    writeback_performed = False
    diag_started = time.perf_counter()
    for slot in session.slots.values():
        diag = slot.native.cached_diagnostics()
        step_ms += float(diag.step_ms)
        hash_build_ms += float(diag.hash_build_ms)
        constraints_ms += float(diag.constraints_ms)
        volume_ms += float(diag.volume_ms)
        analytic_collision_ms += float(diag.analytic_collision_ms)
        static_collision_ms += float(diag.static_collision_ms)
        dynamic_collision_ms += float(diag.dynamic_collision_ms)
        dynamic_particle_collision_ms += float(diag.dynamic_particle_collision_ms)
        self_hash_ms += float(diag.self_hash_ms)
        self_solve_ms += float(diag.self_solve_ms)
        self_probe_ms += float(diag.self_probe_ms)
        self_recovery_ms += float(diag.self_recovery_ms)
        sync_ms += float(diag.sync_ms)
        diagnostics_fetch_ms += float(diag.diagnostics_fetch_ms)
        frame_input_upload_ms += float(diag.frame_input_upload_ms)
        candidate_count += int(diag.candidate_count)
        resolved_contacts += int(diag.resolved_contacts)
        ccd_clamp_count += int(diag.ccd_clamp_count)
        recovery_passes += int(diag.recovery_passes)
        local_retry_count += int(diag.local_retry_count)
        jitter_stabilized_vertices += int(diag.jitter_stabilized_vertices)
        jitter_rejected_vertices += int(diag.jitter_rejected_vertices)
        jitter_max_correction = max(jitter_max_correction, float(diag.jitter_max_correction))
        external_contact_cache_hits += int(diag.external_contact_cache_hits)
        external_contact_cache_misses += int(diag.external_contact_cache_misses)
        external_contact_cache_count += int(diag.external_contact_cache_count)
        external_contact_cache_overflow += int(diag.external_contact_cache_overflow)
        external_friction_corrections += int(diag.external_friction_corrections)
        force_field_count += int(diag.force_field_count)
        unsupported_force_field_count += int(diag.unsupported_force_field_count)
        dynamic_particle_count += int(diag.dynamic_particle_count)
        dynamic_particle_candidate_count += int(diag.dynamic_particle_candidate_count)
        dynamic_particle_contacts += int(diag.dynamic_particle_contacts)
        dynamic_particle_overflow += int(diag.dynamic_particle_overflow)
        dynamic_triangle_count += int(diag.dynamic_triangle_count)
        dynamic_triangle_candidate_count += int(diag.dynamic_triangle_candidate_count)
        dynamic_triangle_bucket_overflow += int(diag.dynamic_triangle_bucket_overflow)
        dynamic_triangle_large_primitive_count += int(diag.dynamic_triangle_large_primitive_count)
        dynamic_triangle_aabb_reject_count += int(diag.dynamic_triangle_aabb_reject_count)
        dynamic_triangle_max_bucket_occupancy = max(
            dynamic_triangle_max_bucket_occupancy,
            int(diag.dynamic_triangle_max_bucket_occupancy)
        )
        global_dynamic_scene_pack_ms = max(global_dynamic_scene_pack_ms, float(diag.global_dynamic_scene_pack_ms))
        global_dynamic_scene_upload_ms = max(global_dynamic_scene_upload_ms, float(diag.global_dynamic_scene_upload_ms))
        global_dynamic_hash_ms = max(global_dynamic_hash_ms, float(diag.global_dynamic_hash_ms))
        global_dynamic_particle_count = max(global_dynamic_particle_count, int(diag.global_dynamic_particle_count))
        global_dynamic_triangle_count = max(global_dynamic_triangle_count, int(diag.global_dynamic_triangle_count))
        global_dynamic_hash_overflow = max(global_dynamic_hash_overflow, int(diag.global_dynamic_hash_overflow))
        static_triangle_count += int(diag.static_triangle_count)
        fast_exact_vt_candidates += int(diag.fast_exact_vt_candidates)
        fast_exact_vt_projected += int(diag.fast_exact_vt_projected)
        fast_exact_vt_guarded += int(diag.fast_exact_vt_guarded)
        fast_exact_vt_skipped_rest += int(diag.fast_exact_vt_skipped_rest)
        fast_soft_repulsion_candidates += int(diag.fast_soft_repulsion_candidates)
        fast_soft_repulsion_applied += int(diag.fast_soft_repulsion_applied)
        fast_soft_repulsion_max_push = max(fast_soft_repulsion_max_push, float(diag.fast_soft_repulsion_max_push))
        fast_hard_projection_count += int(diag.fast_hard_projection_count)
        fast_manifold_contacts += int(diag.fast_manifold_contacts)
        fast_manifold_reused += int(diag.fast_manifold_reused)
        fast_barrier_projected += int(diag.fast_barrier_projected)
        fast_barrier_smoothed_vertices += int(diag.fast_barrier_smoothed_vertices)
        fast_barrier_overflow += int(diag.fast_barrier_overflow)
        fast_barrier_max_delta = max(fast_barrier_max_delta, float(diag.fast_barrier_max_delta))
        fast_edge_edge_candidates += int(diag.fast_edge_edge_candidates)
        fast_edge_edge_contacts += int(diag.fast_edge_edge_contacts)
        fast_triangle_pair_candidates += int(diag.fast_triangle_pair_candidates)
        fast_triangle_pair_contacts += int(diag.fast_triangle_pair_contacts)
        fast_triangle_pair_skipped_rest += int(diag.fast_triangle_pair_skipped_rest)
        fast_contact_classification_guarded += int(diag.fast_contact_classification_guarded)
        fast_region_cluster_candidates += int(diag.fast_region_cluster_candidates)
        fast_region_cluster_contacts += int(diag.fast_region_cluster_contacts)
        fast_region_cluster_guarded += int(diag.fast_region_cluster_guarded)
        fast_overlap_island_candidates += int(diag.fast_overlap_island_candidates)
        fast_overlap_island_clusters += int(diag.fast_overlap_island_clusters)
        fast_overlap_island_vertex_refs += int(diag.fast_overlap_island_vertex_refs)
        fast_overlap_island_applied_vertices += int(diag.fast_overlap_island_applied_vertices)
        fast_overlap_island_guarded += int(diag.fast_overlap_island_guarded)
        fast_overlap_island_max_delta = max(fast_overlap_island_max_delta, float(diag.fast_overlap_island_max_delta))
        fast_cc_overlap_components += int(diag.fast_cc_overlap_components)
        fast_cc_overlap_seed_triangles += int(diag.fast_cc_overlap_seed_triangles)
        fast_cc_overlap_owned_vertices += int(diag.fast_cc_overlap_owned_vertices)
        fast_cc_overlap_union_edges += int(diag.fast_cc_overlap_union_edges)
        fast_cc_overlap_guarded += int(diag.fast_cc_overlap_guarded)
        fast_cc_overlap_applied_vertices += int(diag.fast_cc_overlap_applied_vertices)
        fast_cc_overlap_max_delta = max(fast_cc_overlap_max_delta, float(diag.fast_cc_overlap_max_delta))
        abi41_soft_contact_count += int(diag.abi41_soft_contact_count)
        abi41_exact_impulse_contact_count += int(diag.abi41_exact_impulse_contact_count)
        abi41_edge_edge_contact_count += int(diag.abi41_edge_edge_contact_count)
        abi41_max_smoothed_delta = max(abi41_max_smoothed_delta, float(diag.abi41_max_smoothed_delta))
        abi41_hard_projection_fallbacks += int(diag.abi41_hard_projection_fallbacks)
        static_sdf_rebuild_count += int(diag.static_sdf_rebuild_count)
        static_sdf_voxel_count = max(static_sdf_voxel_count, int(diag.static_sdf_voxel_count))
        static_sdf_grid_x = max(static_sdf_grid_x, int(diag.static_sdf_grid_x))
        static_sdf_grid_y = max(static_sdf_grid_y, int(diag.static_sdf_grid_y))
        static_sdf_grid_z = max(static_sdf_grid_z, int(diag.static_sdf_grid_z))
        static_sdf_build_ms += float(diag.static_sdf_build_ms)
        static_sdf_contact_count += int(diag.static_sdf_contact_count)
        static_sdf_unsigned_fallback_count += int(diag.static_sdf_unsigned_fallback_count)
        abi41_pcg_iterations += int(diag.abi41_pcg_iterations)
        abi41_pcg_guarded += int(diag.abi41_pcg_guarded)
        abi41_pcg_csr_nnz = max(abi41_pcg_csr_nnz, int(diag.abi41_pcg_csr_nnz))
        abi41_pcg_texture_ready = max(abi41_pcg_texture_ready, int(diag.abi41_pcg_texture_ready))
        abi41_pcg_initial_residual = max(abi41_pcg_initial_residual, float(diag.abi41_pcg_initial_residual))
        abi41_pcg_final_residual = max(abi41_pcg_final_residual, float(diag.abi41_pcg_final_residual))
        abi41_pcg_max_delta = max(abi41_pcg_max_delta, float(diag.abi41_pcg_max_delta))
        abi41_pcg_solve_ms += float(diag.abi41_pcg_solve_ms)
        abi41_pcg_system_ms += float(diag.abi41_pcg_system_ms)
        abi41_pcg_ad_ms += float(diag.abi41_pcg_ad_ms)
        abi41_direct_stretch_ms += float(diag.abi41_direct_stretch_ms)
        abi41_lra_tack_count += int(diag.abi41_lra_tack_count)
        abi41_bending_wing_count += int(diag.abi41_bending_wing_count)
        abi41_bending_texture_ready = max(abi41_bending_texture_ready, int(diag.abi41_bending_texture_ready))
        abi41_tack_jitter_guarded += int(diag.abi41_tack_jitter_guarded)
        abi41_bending_guarded += int(diag.abi41_bending_guarded)
        dynamic_collider_pack_ms += float(diag.dynamic_collider_pack_ms)
        dynamic_triangle_upload_ms += float(diag.dynamic_triangle_upload_ms)
        dynamic_particle_upload_ms += float(diag.dynamic_particle_upload_ms)
        dynamic_collider_cache_hits += int(diag.dynamic_collider_cache_hits)
        dynamic_collider_cache_misses += int(diag.dynamic_collider_cache_misses)
        dynamic_pair_cache_hits += int(diag.dynamic_pair_cache_hits)
        dynamic_pair_cache_misses += int(diag.dynamic_pair_cache_misses)
        dynamic_pair_cache_reused_triangles += int(diag.dynamic_pair_cache_reused_triangles)
        dynamic_pair_cache_reused_particles += int(diag.dynamic_pair_cache_reused_particles)
        dynamic_collision_skipped_launches += int(diag.dynamic_collision_skipped_launches)
        self_collision_skipped_launches += int(diag.self_collision_skipped_launches)
        self_candidate_count += int(diag.self_candidate_count)
        self_filter_seen += int(diag.self_filter_seen)
        self_filter_accepted_vv += int(diag.self_filter_accepted_vv)
        self_filter_accepted_vt += int(diag.self_filter_accepted_vt)
        self_filter_accepted_ee += int(diag.self_filter_accepted_ee)
        self_filter_rejected_rest += int(diag.self_filter_rejected_rest)
        self_filter_rejected_duplicate += int(diag.self_filter_rejected_duplicate)
        self_filter_rejected_ownership += int(diag.self_filter_rejected_ownership)
        self_filter_cache_hits += int(diag.self_filter_cache_hits)
        self_filter_cache_misses += int(diag.self_filter_cache_misses)
        self_cluster_count += int(diag.self_cluster_count)
        self_cluster_owned_contacts += int(diag.self_cluster_owned_contacts)
        finite = finite and bool(diag.finite)
        writeback_performed = writeback_performed or bool(diag.writeback_performed)
        if diag.min_gap is not None:
            min_gap = float(diag.min_gap) if min_gap is None else min(min_gap, float(diag.min_gap))
    if perf is not None:
        perf.diagnostics_ms += _elapsed_ms(diag_started)
    return NativeStepDiagnostics(
        step_ms=step_ms,
        hash_build_ms=hash_build_ms,
        constraints_ms=constraints_ms,
        volume_ms=volume_ms,
        analytic_collision_ms=analytic_collision_ms,
        static_collision_ms=static_collision_ms,
        dynamic_collision_ms=dynamic_collision_ms,
        dynamic_particle_collision_ms=dynamic_particle_collision_ms,
        self_hash_ms=self_hash_ms,
        self_solve_ms=self_solve_ms,
        self_probe_ms=self_probe_ms,
        self_recovery_ms=self_recovery_ms,
        sync_ms=sync_ms,
        diagnostics_fetch_ms=diagnostics_fetch_ms,
        frame_input_upload_ms=perf.frame_input_upload_ms if perf is not None else frame_input_upload_ms,
        candidate_count=candidate_count,
        resolved_contacts=resolved_contacts,
        min_gap=min_gap,
        ccd_clamp_count=ccd_clamp_count,
        recovery_passes=recovery_passes,
        local_retry_count=local_retry_count,
        jitter_stabilized_vertices=jitter_stabilized_vertices,
        jitter_rejected_vertices=jitter_rejected_vertices,
        jitter_max_correction=jitter_max_correction,
        external_contact_cache_hits=external_contact_cache_hits,
        external_contact_cache_misses=external_contact_cache_misses,
        external_contact_cache_count=external_contact_cache_count,
        external_contact_cache_overflow=external_contact_cache_overflow,
        external_friction_corrections=external_friction_corrections,
        force_field_count=force_field_count,
        unsupported_force_field_count=unsupported_force_field_count,
        dynamic_particle_count=(
            perf.dynamic_particle_count if perf is not None and perf.dynamic_particle_count > 0 else dynamic_particle_count
        ),
        dynamic_particle_candidate_count=dynamic_particle_candidate_count,
        dynamic_particle_contacts=dynamic_particle_contacts,
        dynamic_particle_overflow=dynamic_particle_overflow,
        dynamic_triangle_count=(
            perf.dynamic_triangle_count if perf is not None and perf.dynamic_triangle_count > 0 else dynamic_triangle_count
        ),
        dynamic_triangle_candidate_count=dynamic_triangle_candidate_count,
        dynamic_triangle_bucket_overflow=dynamic_triangle_bucket_overflow,
        dynamic_triangle_large_primitive_count=dynamic_triangle_large_primitive_count,
        dynamic_triangle_aabb_reject_count=dynamic_triangle_aabb_reject_count,
        dynamic_triangle_max_bucket_occupancy=dynamic_triangle_max_bucket_occupancy,
        global_dynamic_scene_pack_ms=(
            perf.dynamic_collider_pack_ms
            if perf is not None and session.global_dynamic_scene_enabled
            else global_dynamic_scene_pack_ms
        ),
        global_dynamic_scene_upload_ms=global_dynamic_scene_upload_ms,
        global_dynamic_hash_ms=global_dynamic_hash_ms,
        global_dynamic_particle_count=global_dynamic_particle_count,
        global_dynamic_triangle_count=global_dynamic_triangle_count,
        global_dynamic_hash_overflow=global_dynamic_hash_overflow,
        static_triangle_count=static_triangle_count,
        fast_exact_vt_candidates=fast_exact_vt_candidates,
        fast_exact_vt_projected=fast_exact_vt_projected,
        fast_exact_vt_guarded=fast_exact_vt_guarded,
        fast_exact_vt_skipped_rest=fast_exact_vt_skipped_rest,
        fast_soft_repulsion_candidates=fast_soft_repulsion_candidates,
        fast_soft_repulsion_applied=fast_soft_repulsion_applied,
        fast_soft_repulsion_max_push=fast_soft_repulsion_max_push,
        fast_hard_projection_count=fast_hard_projection_count,
        fast_manifold_contacts=fast_manifold_contacts,
        fast_manifold_reused=fast_manifold_reused,
        fast_barrier_projected=fast_barrier_projected,
        fast_barrier_smoothed_vertices=fast_barrier_smoothed_vertices,
        fast_barrier_overflow=fast_barrier_overflow,
        fast_barrier_max_delta=fast_barrier_max_delta,
        fast_edge_edge_candidates=fast_edge_edge_candidates,
        fast_edge_edge_contacts=fast_edge_edge_contacts,
        fast_triangle_pair_candidates=fast_triangle_pair_candidates,
        fast_triangle_pair_contacts=fast_triangle_pair_contacts,
        fast_triangle_pair_skipped_rest=fast_triangle_pair_skipped_rest,
        fast_contact_classification_guarded=fast_contact_classification_guarded,
        fast_region_cluster_candidates=fast_region_cluster_candidates,
        fast_region_cluster_contacts=fast_region_cluster_contacts,
        fast_region_cluster_guarded=fast_region_cluster_guarded,
        fast_overlap_island_candidates=fast_overlap_island_candidates,
        fast_overlap_island_clusters=fast_overlap_island_clusters,
        fast_overlap_island_vertex_refs=fast_overlap_island_vertex_refs,
        fast_overlap_island_applied_vertices=fast_overlap_island_applied_vertices,
        fast_overlap_island_guarded=fast_overlap_island_guarded,
        fast_overlap_island_max_delta=fast_overlap_island_max_delta,
        fast_cc_overlap_components=fast_cc_overlap_components,
        fast_cc_overlap_seed_triangles=fast_cc_overlap_seed_triangles,
        fast_cc_overlap_owned_vertices=fast_cc_overlap_owned_vertices,
        fast_cc_overlap_union_edges=fast_cc_overlap_union_edges,
        fast_cc_overlap_guarded=fast_cc_overlap_guarded,
        fast_cc_overlap_applied_vertices=fast_cc_overlap_applied_vertices,
        fast_cc_overlap_max_delta=fast_cc_overlap_max_delta,
        abi41_soft_contact_count=abi41_soft_contact_count,
        abi41_exact_impulse_contact_count=abi41_exact_impulse_contact_count,
        abi41_edge_edge_contact_count=abi41_edge_edge_contact_count,
        abi41_max_smoothed_delta=abi41_max_smoothed_delta,
        abi41_hard_projection_fallbacks=abi41_hard_projection_fallbacks,
        static_sdf_rebuild_count=static_sdf_rebuild_count,
        static_sdf_voxel_count=static_sdf_voxel_count,
        static_sdf_grid_x=static_sdf_grid_x,
        static_sdf_grid_y=static_sdf_grid_y,
        static_sdf_grid_z=static_sdf_grid_z,
        static_sdf_build_ms=static_sdf_build_ms,
        static_sdf_contact_count=static_sdf_contact_count,
        static_sdf_unsigned_fallback_count=static_sdf_unsigned_fallback_count,
        abi41_pcg_iterations=abi41_pcg_iterations,
        abi41_pcg_guarded=abi41_pcg_guarded,
        abi41_pcg_csr_nnz=abi41_pcg_csr_nnz,
        abi41_pcg_texture_ready=abi41_pcg_texture_ready,
        abi41_pcg_initial_residual=abi41_pcg_initial_residual,
        abi41_pcg_final_residual=abi41_pcg_final_residual,
        abi41_pcg_max_delta=abi41_pcg_max_delta,
        abi41_pcg_solve_ms=abi41_pcg_solve_ms,
        abi41_pcg_system_ms=abi41_pcg_system_ms,
        abi41_pcg_ad_ms=abi41_pcg_ad_ms,
        abi41_direct_stretch_ms=abi41_direct_stretch_ms,
        abi41_lra_tack_count=abi41_lra_tack_count,
        abi41_bending_wing_count=abi41_bending_wing_count,
        abi41_bending_texture_ready=abi41_bending_texture_ready,
        abi41_tack_jitter_guarded=abi41_tack_jitter_guarded,
        abi41_bending_guarded=abi41_bending_guarded,
        dynamic_collider_pack_ms=(
            perf.dynamic_collider_pack_ms if perf is not None else dynamic_collider_pack_ms
        ),
        dynamic_triangle_upload_ms=dynamic_triangle_upload_ms,
        dynamic_particle_upload_ms=dynamic_particle_upload_ms,
        dynamic_collider_cache_hits=(
            perf.dynamic_collider_cache_hits if perf is not None else dynamic_collider_cache_hits
        ),
        dynamic_collider_cache_misses=(
            perf.dynamic_collider_cache_misses if perf is not None else dynamic_collider_cache_misses
        ),
        dynamic_pair_cache_hits=(
            perf.dynamic_pair_cache_hits if perf is not None else dynamic_pair_cache_hits
        ),
        dynamic_pair_cache_misses=(
            perf.dynamic_pair_cache_misses if perf is not None else dynamic_pair_cache_misses
        ),
        dynamic_pair_cache_reused_triangles=(
            perf.dynamic_pair_cache_reused_triangles if perf is not None else dynamic_pair_cache_reused_triangles
        ),
        dynamic_pair_cache_reused_particles=(
            perf.dynamic_pair_cache_reused_particles if perf is not None else dynamic_pair_cache_reused_particles
        ),
        dynamic_collision_skipped_launches=dynamic_collision_skipped_launches,
        self_collision_skipped_launches=self_collision_skipped_launches,
        self_candidate_count=self_candidate_count,
        self_filter_seen=self_filter_seen,
        self_filter_accepted_vv=self_filter_accepted_vv,
        self_filter_accepted_vt=self_filter_accepted_vt,
        self_filter_accepted_ee=self_filter_accepted_ee,
        self_filter_rejected_rest=self_filter_rejected_rest,
        self_filter_rejected_duplicate=self_filter_rejected_duplicate,
        self_filter_rejected_ownership=self_filter_rejected_ownership,
        self_filter_cache_hits=self_filter_cache_hits,
        self_filter_cache_misses=self_filter_cache_misses,
        self_cluster_count=self_cluster_count,
        self_cluster_owned_contacts=self_cluster_owned_contacts,
        finite=finite,
        frame_ms=perf.frame_ms if perf is not None else 0.0,
        frame_set_ms=perf.frame_set_ms if perf is not None else 0.0,
        input_refresh_ms=perf.input_refresh_ms if perf is not None else 0.0,
        pin_upload_ms=perf.pin_upload_ms if perf is not None else 0.0,
        runtime_upload_ms=perf.runtime_upload_ms if perf is not None else 0.0,
        static_upload_ms=perf.static_upload_ms if perf is not None else 0.0,
        dynamic_upload_ms=perf.dynamic_upload_ms if perf is not None else 0.0,
        cuda_step_call_ms=perf.cuda_step_call_ms if perf is not None else 0.0,
        download_ms=perf.download_ms if perf is not None else 0.0,
        writeback_ms=perf.writeback_ms if perf is not None else 0.0,
        writeback_to_local_ms=perf.writeback_to_local_ms if perf is not None else 0.0,
        writeback_foreach_set_ms=perf.writeback_foreach_set_ms if perf is not None else 0.0,
        writeback_mesh_update_ms=perf.writeback_mesh_update_ms if perf is not None else 0.0,
        writeback_performed=perf.writeback_performed if perf is not None else writeback_performed,
        diagnostics_ms=perf.diagnostics_ms if perf is not None else 0.0,
        viewport_tag_ms=perf.viewport_tag_ms if perf is not None else 0.0,
    )


def _cache_path_for_object(obj: bpy.types.Object) -> str:
    safe_name = safe_cache_stem(obj.name)
    if bpy.data.filepath:
        root = bpy.path.abspath("//")
    else:
        root = bpy.app.tempdir
    return os.path.join(root, "ssbl_cache", f"{safe_name}_xpbd.pc2")


def _write_pc2_header(handle, vertex_count: int, start_frame: int, sample_count: int) -> None:
    handle.write(
        struct.pack(
            "<12siiffi",
            b"POINTCACHE2\0",
            1,
            int(vertex_count),
            float(start_frame),
            1.0,
            int(sample_count),
        )
    )


def _write_pc2_sample(handle, world_positions: np.ndarray, matrix_world_inv: np.ndarray) -> None:
    local = to_local(np.asarray(world_positions, dtype=np.float64), matrix_world_inv)
    handle.write(np.ascontiguousarray(local, dtype="<f4").tobytes())


def _realtime_cache_temp_path(path: str) -> str:
    return f"{path}.realtime.tmp"


def _discard_realtime_cache_writer(writer: RealtimeCacheWriter) -> None:
    try:
        if not getattr(writer.handle, "closed", True):
            writer.handle.close()
    except Exception:
        pass
    try:
        if os.path.exists(writer.temp_path):
            os.remove(writer.temp_path)
    except Exception:
        pass


def _write_realtime_cache_sample(slot: ClothSlot) -> None:
    writer = slot.realtime_cache
    if writer is None:
        return
    _write_pc2_sample(writer.handle, slot.current_positions_world, slot.cloth.matrix_world_inv)
    writer.sample_count += 1


def _begin_realtime_cache_for_slot(obj: bpy.types.Object, slot: ClothSlot, start_frame: int) -> None:
    if not bool(slot.auto_cache_realtime) or slot.realtime_cache is not None:
        return
    path = _cache_path_for_object(obj)
    temp_path = _realtime_cache_temp_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    handle = open(temp_path, "w+b")
    writer = RealtimeCacheWriter(
        path=path,
        temp_path=temp_path,
        handle=handle,
        start_frame=int(start_frame),
        vertex_count=int(len(slot.current_positions_world)),
    )
    slot.realtime_cache = writer
    try:
        _write_pc2_header(handle, writer.vertex_count, writer.start_frame, 0)
        _write_realtime_cache_sample(slot)
    except Exception:
        slot.realtime_cache = None
        _discard_realtime_cache_writer(writer)
        raise


def _begin_realtime_cache_for_session(session: SceneSession) -> None:
    for slot_name in session.solve_order:
        slot = session.slots[slot_name]
        if not bool(slot.auto_cache_realtime):
            continue
        obj = bpy.data.objects.get(slot.object_name)
        if obj is None or obj.type != "MESH":
            raise ValueError(f"Missing object for realtime auto cache: {slot.object_name}")
        _begin_realtime_cache_for_slot(obj, slot, session.start_frame)


def _write_realtime_cache_samples(session: SceneSession) -> None:
    for slot_name in session.solve_order:
        slot = session.slots[slot_name]
        if slot.realtime_cache is not None:
            _write_realtime_cache_sample(slot)


def _finish_realtime_cache_for_slot(slot: ClothSlot, obj: bpy.types.Object | None, commit: bool) -> bool:
    writer = slot.realtime_cache
    slot.realtime_cache = None
    if writer is None:
        return True
    if not bool(commit) or writer.sample_count <= 0 or obj is None or obj.type != "MESH":
        _discard_realtime_cache_writer(writer)
        return True
    try:
        writer.handle.seek(0)
        _write_pc2_header(writer.handle, writer.vertex_count, writer.start_frame, writer.sample_count)
        writer.handle.flush()
        writer.handle.close()
        os.replace(writer.temp_path, writer.path)
        _bind_mesh_cache(obj, writer.path, writer.start_frame)
        obj[_CACHE_PATH_PROP] = writer.path
        return True
    except Exception:
        _discard_realtime_cache_writer(writer)
        return False


def _bind_mesh_cache(obj: bpy.types.Object, path: str, start_frame: int) -> None:
    modifier = obj.modifiers.get(_CACHE_MODIFIER_NAME)
    if modifier is None:
        modifier = obj.modifiers.new(_CACHE_MODIFIER_NAME, "MESH_CACHE")
    modifier.cache_format = "PC2"
    modifier.filepath = path
    modifier.frame_start = float(start_frame)
    modifier.frame_scale = 1.0
