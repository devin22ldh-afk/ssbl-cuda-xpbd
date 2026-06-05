from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
import re
import struct
import time
from typing import Callable, Optional

import bpy
from mathutils import Vector
import numpy as np

from .collision import clear_static_collision_cache, collect_static_triangles
from .force_fields import EMPTY_FORCE_FIELD_BATCH, ForceFieldBatch, collect_force_fields, has_force_field_sources
from .native_backend import NativeStepDiagnostics, NativeXpbdSolver, status as native_status
from .xpbd_core import (
    ClothBuildData,
    PinAttachmentBatch,
    build_cloth_data,
    clear_cloth_topology_cache,
    make_pin_attachment_batch,
    settings_to_options,
    to_local,
    world_positions_from_object,
)


_SCENE_SESSIONS: dict[str, "SceneSession"] = {}
_OBJECT_TO_SCENE_SESSION: dict[str, str] = {}
_STATUS: dict[str, str] = {}
_LAST_DIAGNOSTICS: dict[str, NativeStepDiagnostics] = {}
_CACHE_PATH_PROP = "_ssbl_xpbd_cache_path"
_CACHE_MODIFIER_NAME = "SSBL XPBD Cache"
_UNSUPPORTED_INPUT_TYPES = {"solid", "rod", "stitch", "tet"}
_OBJECT_COLLISION_LAYER_PROP = "ssbl_collision_layer"
_IDENTITY_4X4 = np.eye(4, dtype=np.float32)
_AUTO_WRITEBACK_INTERVAL = 0
_MIN_WRITEBACK_INTERVAL = 1
_MAX_AUTO_WRITEBACK_INTERVAL = 8
_WRITEBACK_EWMA_ALPHA = 0.25
STATUS_IDLE = "Idle"
STATUS_PREVIEW_RUNNING = "Preview Running"
STATUS_PREVIEW_PAUSED = "Preview Paused"
STATUS_PREVIEW_STOPPED = "Preview Stopped"
STATUS_BAKING = "Baking"
STATUS_FINISHED = "Finished"
STATUS_ERROR = "Error"


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
    force_next_writeback: bool = False


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
        self_vs_pair_build_ms=diag.self_vs_pair_build_ms,
        self_vs_pair_project_ms=diag.self_vs_pair_project_ms,
        candidate_count=diag.candidate_count,
        resolved_contacts=diag.resolved_contacts,
        min_gap=diag.min_gap,
        ccd_clamp_count=diag.ccd_clamp_count,
        recovery_passes=diag.recovery_passes,
        local_retry_count=diag.local_retry_count,
        self_active_regions=diag.self_active_regions,
        self_sleeping_regions=diag.self_sleeping_regions,
        self_skipped_sources=diag.self_skipped_sources,
        self_active_vertices=diag.self_active_vertices,
        self_active_samples=diag.self_active_samples,
        self_suspect_regions=diag.self_suspect_regions,
        self_compaction_used=diag.self_compaction_used,
        self_full_recovery_fallbacks=diag.self_full_recovery_fallbacks,
        self_vs_pair_count=diag.self_vs_pair_count,
        self_vs_pair_capacity=diag.self_vs_pair_capacity,
        self_vs_pair_overflow=diag.self_vs_pair_overflow,
        self_vs_pair_compaction_used=diag.self_vs_pair_compaction_used,
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


def has_session(obj: Optional[bpy.types.Object]) -> bool:
    return obj is not None and _session_for_object_name(obj.name) is not None


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
    _finish_session(session, STATUS_PREVIEW_STOPPED)
    return True


def reset_preview_object(obj: bpy.types.Object) -> bool:
    session = _session_for_object_name(obj.name if obj else "")
    if session is None:
        return False
    _finish_session(session, STATUS_IDLE)
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
    _finish_session(session, STATUS_IDLE)
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
    return _collection_contains_object(static_collection, candidate)


def _collision_only_object_names_for_scene(scene: bpy.types.Scene) -> set[str]:
    collision_only: set[str] = set()
    if scene is None:
        return collision_only
    scene_settings = getattr(scene, "ssbl_preview", None)
    if scene_settings is not None:
        for candidate in scene.objects:
            if candidate is not None and candidate.type == "MESH" and _settings_reference_collision_object(scene_settings, candidate):
                collision_only.add(candidate.name)
    for owner in scene.objects:
        owner_settings = _object_cloth_settings(owner)
        if owner_settings is None:
            continue
        for candidate in scene.objects:
            if candidate is None or candidate.type != "MESH":
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
        if _declared_input_type(obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        objects.append(obj)
    return sorted(objects, key=lambda item: (int(getattr(item, _OBJECT_COLLISION_LAYER_PROP, item.get(_OBJECT_COLLISION_LAYER_PROP, 1))), item.name.casefold()))


def _auto_cross_cloth_mode(slot_count: int) -> str:
    return "all_selected" if int(slot_count) > 1 else "off"


def start_preview(context: bpy.types.Context, obj: bpy.types.Object) -> SceneSession:
    try:
        if context.mode != "OBJECT":
            raise ValueError("Preview must be started in Object mode.")
        settings = _settings_for_object(context, obj, context.scene.ssbl_preview)
        auto_sphere_obj = _auto_sphere_collider_for_preview(context, obj, settings)
        cloth_objects = _preview_cloth_objects(context, obj, settings)
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
        for cloth_obj in cloth_objects:
            slot_settings = _settings_for_object(context, cloth_obj, settings)
            slot = _create_cloth_slot(context, cloth_obj, slot_settings, depsgraph, auto_sphere_obj)
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
            cross_cloth_mode=_auto_cross_cloth_mode(len(slots)),
            last_fps_time=time.perf_counter(),
            fps_sample_frames=0,
            actual_fps=0.0,
            target_fps=_target_fps_from_settings(settings),
            last_scene_frame=int(context.scene.frame_current),
        )
        _SCENE_SESSIONS[scene_key] = session
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
        return session
    except Exception:
        if obj is not None:
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
    for cloth_obj in cloth_objects:
        _ensure_supported_cloth_object(cloth_obj)
        settings = _settings_for_object(context, cloth_obj, require_enabled=True)
        if settings is None:
            continue
        slot = _create_cloth_slot(context, cloth_obj, settings, depsgraph)
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
        cross_cloth_mode=_auto_cross_cloth_mode(len(slots)),
        last_fps_time=time.perf_counter(),
        fps_sample_frames=0,
        actual_fps=0.0,
        target_fps=_target_fps_from_settings(scene.ssbl_preview),
        playback_driven=True,
        paused=False,
        last_scene_frame=int(scene.frame_current),
    )
    _SCENE_SESSIONS[scene_key] = session
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
    return session


def pause_timeline_preview(scene: bpy.types.Scene | None = None) -> None:
    scene = scene or bpy.context.scene
    session = _SCENE_SESSIONS.get(_scene_key(scene))
    if session is None or not bool(session.playback_driven):
        return
    session.paused = True
    for name in session.solve_order:
        _STATUS[name] = STATUS_PREVIEW_PAUSED


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
        perf.writeback_performed = any(writeback_by_slot.values())
        _step_session_slots(session, writeback_by_slot, perf)
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
        _finish_session(session, STATUS_PREVIEW_STOPPED)
        return True
    next_frame = session.start_frame + session.frame_index + 1
    if session.frame_index >= session.frame_count or next_frame > int(scene.frame_end):
        _finish_session(session, STATUS_FINISHED)
        return True

    step_started = time.perf_counter()
    perf = FramePerf()
    try:
        started = time.perf_counter()
        scene.frame_set(next_frame)
        perf.frame_set_ms += _elapsed_ms(started)
        _refresh_session_runtime_inputs(context, session, perf)
        writeback_by_slot = _writeback_decisions(session, next_frame, scene_end=int(scene.frame_end))
        perf.writeback_performed = any(writeback_by_slot.values())
        _step_session_slots(session, writeback_by_slot, perf)
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


def _options_from_settings(settings, runtime_mode_override: str | None = None, auto_sphere_object=None):
    if auto_sphere_object is None:
        return settings_to_options(settings, runtime_mode_override=runtime_mode_override)
    explicit_sphere = getattr(settings, "sphere_object", None)
    if bool(getattr(settings, "use_sphere", False)) and explicit_sphere is not None:
        return settings_to_options(settings, runtime_mode_override=runtime_mode_override)
    with _temporary_setting(settings, "use_sphere", True):
        with _temporary_setting(settings, "sphere_object", auto_sphere_object):
            return settings_to_options(settings, runtime_mode_override=runtime_mode_override)


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
        bool(options.self_sleep_enabled),
        int(options.self_sleep_still_frames),
        int(options.self_sleep_full_scan_interval),
        bool(options.self_compaction_enabled),
        round(float(options.self_sleep_motion_scale), 6),
        round(float(options.self_compaction_active_fraction_threshold), 6),
        bool(options.self_pair_compaction_enabled),
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
    )


def _static_collider_runtime_signature(
    collection: bpy.types.Collection | None,
    exclude_obj: bpy.types.Object | None,
    depsgraph: bpy.types.Depsgraph | None,
    use_evaluated_mesh: bool,
) -> tuple:
    if collection is None:
        return ()
    entries = []
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    for obj in sorted(collection.objects, key=lambda item: item.name):
        if obj is None or obj == exclude_obj or obj.type != "MESH":
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
        if _declared_input_type(selected_obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        if _is_collision_only_object(context.scene, selected_obj, active_obj=obj, active_settings=settings):
            continue
        selected.append(selected_obj)
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
    )
    suspended_modifiers = _suspend_preview_modifiers(obj, suspend_all=use_evaluated_mesh)
    preview_mesh = original_mesh.copy()
    preview_mesh.name = f"{original_mesh.name}_SSBL_XPBD_Preview"
    obj.data = preview_mesh
    writeback_flat_buffer = np.empty(cloth.positions_world.size, dtype=np.float32)
    writeback_local_buffer = writeback_flat_buffer.reshape(cloth.positions_world.shape)
    options = _options_from_settings(settings, runtime_mode_override="preview", auto_sphere_object=auto_sphere_object)
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
    )


def _create_native_solver(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
    use_evaluated_mesh_override: bool | None = None,
    runtime_mode_override: str | None = None,
    auto_sphere_object: bpy.types.Object | None = None,
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
        )
        static_tris, static_signature = collect_static_triangles(
            settings.static_collider_collection,
            obj,
            depsgraph=depsgraph,
            use_evaluated_mesh=use_evaluated_mesh,
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
    session.cross_cloth_mode = _auto_cross_cloth_mode(len(session.slots))
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
        slot.static_runtime_signature = (
            _static_collider_runtime_signature(
                static_collection,
                obj,
                depsgraph,
                slot.use_evaluated_mesh,
            )
            if static_collection is not None
            else ()
        )
        slot.previous_positions_world = np.array(current_positions, dtype=np.float32, copy=True)
        slot.current_positions_world = current_positions
        slot.runtime_options_signature = _runtime_options_signature(options)
        slot.solver_options_signature = solver_signature
        slot.pin_attachment_pairs = np.array(cloth.pin_attachment_pairs, dtype=np.int32, copy=True)
        slot.pin_targets_world = np.empty((0, 3), dtype=np.float32)
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
        or slot.use_evaluated_mesh
        or has_force_field_sources(context.scene, slot_settings[name])
        for name, slot in session.slots.items()
    )
    with _with_session_source_state(session):
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
            options = _options_from_settings(
                settings,
                runtime_mode_override="preview",
                auto_sphere_object=auto_sphere_object,
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
                )

            static_collection = settings.static_collider_collection
            has_static_collection = static_collection is not None
            if len(slot.cloth.pin_indices) > 0:
                pin_attachment_batch, matrix_world = _pin_attachment_batch_from_object(
                    obj,
                    slot.cloth.pin_indices,
                    slot.use_evaluated_mesh,
                    depsgraph=depsgraph,
                    expected_vertex_count=len(slot.cloth.positions_world),
                )
                pin_targets = pin_attachment_batch.targets_world
            else:
                pin_targets = np.empty((0, 3), dtype=np.float32)
                matrix_world = obj.matrix_world.copy()
            static_runtime_signature = (
                _static_collider_runtime_signature(
                    static_collection,
                    obj,
                    depsgraph,
                    slot.use_evaluated_mesh,
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
        _refresh_session_live_settings(context, session, slot_settings)
    if perf is not None:
        perf.input_refresh_ms += _elapsed_ms(refresh_started)


def _slot_should_download(download_positions, slot_name: str) -> bool:
    if isinstance(download_positions, dict):
        return bool(download_positions.get(slot_name, False))
    return bool(download_positions)


def _step_session_slots(session: SceneSession, download_positions, perf: FramePerf | None = None) -> None:
    cross_cloth_enabled = len(session.slots) > 1 and str(session.cross_cloth_mode or "off").lower() != "off"
    for slot_name in session.solve_order:
        slot = session.slots[slot_name]
        should_download = _slot_should_download(download_positions, slot_name)
        if cross_cloth_enabled:
            started = time.perf_counter()
            dynamic_triangles = _collect_cross_cloth_triangles(session, slot)
            dynamic_particles = _collect_cross_cloth_particles(session, slot)
            slot.native.update_frame_inputs(
                pin_indices=None,
                pin_positions=None,
                update_pin=False,
                options=None,
                update_runtime=False,
                static_triangles=None,
                update_static=False,
                dynamic_triangles=dynamic_triangles,
                update_dynamic=True,
                dynamic_particles=dynamic_particles,
                update_dynamic_particles=True,
                force_fields=None,
                update_force_fields=False,
            )
            if perf is not None:
                elapsed = _elapsed_ms(started)
                perf.dynamic_upload_ms += elapsed
                perf.frame_input_upload_ms += elapsed
        started = time.perf_counter()
        sample_diagnostics = bool(should_download or cross_cloth_enabled)
        slot.native.step(
            slot.substeps,
            slot.iterations,
            diagnostics=sample_diagnostics,
            synchronize=sample_diagnostics,
        )
        if perf is not None:
            perf.cuda_step_call_ms += _elapsed_ms(started)
        if should_download or cross_cloth_enabled:
            started = time.perf_counter()
            slot.previous_positions_world = np.array(slot.current_positions_world, dtype=np.float32, copy=True)
            slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
            if perf is not None:
                perf.download_ms += _elapsed_ms(started)


def _collect_cross_cloth_triangles(session: SceneSession, target: ClothSlot) -> np.ndarray:
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off":
        return np.empty((0, 3, 3), dtype=np.float32)
    target_aabb = _swept_positions_aabb(target)
    all_tris: list[np.ndarray] = []
    for source in session.slots.values():
        if not _cross_cloth_source_enabled(mode, target, source):
            continue
        positions = source.current_positions_world
        if positions is None or len(source.cloth.triangles) == 0:
            continue
        source_aabb = _swept_positions_aabb(source)
        if target_aabb is not None and source_aabb is not None:
            contact_radius = max(float(target.external_contact_distance), float(source.external_contact_distance))
            motion_padding = max(
                contact_radius * 4.0,
                _slot_max_motion(target) * 2.0,
                _slot_max_motion(source) * 2.0,
                0.08,
            )
            if _aabb_distance(target_aabb, source_aabb) > contact_radius + motion_padding:
                continue
        all_tris.append(np.asarray(positions[source.cloth.triangles], dtype=np.float32))
    if not all_tris:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(all_tris, axis=0), dtype=np.float32)


def _collect_cross_cloth_particles(session: SceneSession, target: ClothSlot) -> dict[str, np.ndarray]:
    empty = {
        "positions": np.empty((0, 3), dtype=np.float32),
        "radii": np.empty(0, dtype=np.float32),
        "inv_mass": np.empty(0, dtype=np.float32),
        "slot_ids": np.empty(0, dtype=np.int32),
        "phases": np.empty(0, dtype=np.int32),
    }
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off":
        return empty
    target_aabb = _swept_positions_aabb(target)
    positions_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    inv_mass_list: list[np.ndarray] = []
    slot_id_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    slot_ids = {name: index + 1 for index, name in enumerate(session.solve_order)}
    for source in session.slots.values():
        if not _cross_cloth_source_enabled(mode, target, source):
            continue
        positions = source.current_positions_world
        if positions is None or len(positions) == 0:
            continue
        source_aabb = _swept_positions_aabb(source)
        if target_aabb is not None and source_aabb is not None:
            contact_radius = max(float(target.external_contact_distance), float(source.external_contact_distance))
            motion_padding = max(
                contact_radius * 4.0,
                _slot_max_motion(target) * 2.0,
                _slot_max_motion(source) * 2.0,
                0.08,
            )
            if _aabb_distance(target_aabb, source_aabb) > contact_radius + motion_padding:
                continue
        source_positions = np.asarray(positions, dtype=np.float32)
        count = int(len(source_positions))
        if count <= 0:
            continue
        positions_list.append(source_positions)
        radii_list.append(np.full(count, float(source.external_contact_distance), dtype=np.float32))
        source_inv_mass = np.asarray(source.cloth.inv_mass, dtype=np.float32).reshape((-1,))
        if len(source_inv_mass) == count:
            inv_mass_list.append(source_inv_mass)
        else:
            inv_mass_list.append(np.ones(count, dtype=np.float32))
        slot_id = int(slot_ids.get(source.object_name, 0))
        slot_id_list.append(np.full(count, slot_id, dtype=np.int32))
        phase_list.append(np.full(count, int(source.collision_layer), dtype=np.int32))
    if not positions_list:
        return empty
    return {
        "positions": np.ascontiguousarray(np.concatenate(positions_list, axis=0), dtype=np.float32),
        "radii": np.ascontiguousarray(np.concatenate(radii_list, axis=0), dtype=np.float32),
        "inv_mass": np.ascontiguousarray(np.concatenate(inv_mass_list, axis=0), dtype=np.float32),
        "slot_ids": np.ascontiguousarray(np.concatenate(slot_id_list, axis=0), dtype=np.int32),
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
    if bool(getattr(settings, "use_evaluated_mesh", True)):
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


def _finish_session(session: SceneSession, status: str) -> None:
    if session.closed:
        return
    session.closed = True
    _SCENE_SESSIONS.pop(session.scene_name, None)
    for slot_name in list(session.slots.keys()):
        _OBJECT_TO_SCENE_SESSION.pop(slot_name, None)

    for slot in list(session.slots.values()):
        _LAST_DIAGNOSTICS[slot.object_name] = session.last_diagnostics
        obj = bpy.data.objects.get(slot.object_name)
        if obj is not None and obj.type == "MESH" and _rna_alive(obj):
            try:
                if _same_mesh(obj.data, slot.preview_mesh) and _rna_alive(slot.original_mesh):
                    obj.data = slot.original_mesh
            except (ReferenceError, RuntimeError, AttributeError):
                pass
            _restore_preview_modifiers(obj, slot.suspended_modifiers)
        try:
            slot.native.close()
        except Exception:
            pass
        finally:
            _safe_remove_mesh(slot.preview_mesh)
        _STATUS[slot.object_name] = status
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
    self_vs_pair_build_ms = 0.0
    self_vs_pair_project_ms = 0.0
    frame_input_upload_ms = 0.0
    candidate_count = 0
    resolved_contacts = 0
    min_gap: float | None = None
    ccd_clamp_count = 0
    recovery_passes = 0
    local_retry_count = 0
    self_active_regions = 0
    self_sleeping_regions = 0
    self_skipped_sources = 0
    self_active_vertices = 0
    self_active_samples = 0
    self_suspect_regions = 0
    self_compaction_used = 0
    self_full_recovery_fallbacks = 0
    self_vs_pair_count = 0
    self_vs_pair_capacity = 0
    self_vs_pair_overflow = 0
    self_vs_pair_compaction_used = 0
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
        self_vs_pair_build_ms += float(diag.self_vs_pair_build_ms)
        self_vs_pair_project_ms += float(diag.self_vs_pair_project_ms)
        frame_input_upload_ms += float(diag.frame_input_upload_ms)
        candidate_count += int(diag.candidate_count)
        resolved_contacts += int(diag.resolved_contacts)
        ccd_clamp_count += int(diag.ccd_clamp_count)
        recovery_passes += int(diag.recovery_passes)
        local_retry_count += int(diag.local_retry_count)
        self_active_regions += int(diag.self_active_regions)
        self_sleeping_regions += int(diag.self_sleeping_regions)
        self_skipped_sources += int(diag.self_skipped_sources)
        self_active_vertices += int(diag.self_active_vertices)
        self_active_samples += int(diag.self_active_samples)
        self_suspect_regions += int(diag.self_suspect_regions)
        self_compaction_used += int(diag.self_compaction_used)
        self_full_recovery_fallbacks += int(diag.self_full_recovery_fallbacks)
        self_vs_pair_count += int(diag.self_vs_pair_count)
        self_vs_pair_capacity += int(diag.self_vs_pair_capacity)
        self_vs_pair_overflow += int(diag.self_vs_pair_overflow)
        self_vs_pair_compaction_used += int(diag.self_vs_pair_compaction_used)
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
        self_vs_pair_build_ms=self_vs_pair_build_ms,
        self_vs_pair_project_ms=self_vs_pair_project_ms,
        frame_input_upload_ms=perf.frame_input_upload_ms if perf is not None else frame_input_upload_ms,
        candidate_count=candidate_count,
        resolved_contacts=resolved_contacts,
        min_gap=min_gap,
        ccd_clamp_count=ccd_clamp_count,
        recovery_passes=recovery_passes,
        local_retry_count=local_retry_count,
        self_active_regions=self_active_regions,
        self_sleeping_regions=self_sleeping_regions,
        self_skipped_sources=self_skipped_sources,
        self_active_vertices=self_active_vertices,
        self_active_samples=self_active_samples,
        self_suspect_regions=self_suspect_regions,
        self_compaction_used=self_compaction_used,
        self_full_recovery_fallbacks=self_full_recovery_fallbacks,
        self_vs_pair_count=self_vs_pair_count,
        self_vs_pair_capacity=self_vs_pair_capacity,
        self_vs_pair_overflow=self_vs_pair_overflow,
        self_vs_pair_compaction_used=self_vs_pair_compaction_used,
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
        dynamic_particle_count=dynamic_particle_count,
        dynamic_particle_candidate_count=dynamic_particle_candidate_count,
        dynamic_particle_contacts=dynamic_particle_contacts,
        dynamic_particle_overflow=dynamic_particle_overflow,
        dynamic_triangle_count=dynamic_triangle_count,
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
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", obj.name).strip("_") or "cloth"
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


def _bind_mesh_cache(obj: bpy.types.Object, path: str, start_frame: int) -> None:
    modifier = obj.modifiers.get(_CACHE_MODIFIER_NAME)
    if modifier is None:
        modifier = obj.modifiers.new(_CACHE_MODIFIER_NAME, "MESH_CACHE")
    modifier.cache_format = "PC2"
    modifier.filepath = path
    modifier.frame_start = float(start_frame)
    modifier.frame_scale = 1.0
