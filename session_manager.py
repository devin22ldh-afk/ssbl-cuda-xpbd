from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import re
import struct
import time
from typing import Optional

import bpy
from mathutils import Vector
import numpy as np

from .collision import clear_static_collision_cache, collect_static_triangles
from .native_backend import NativeStepDiagnostics, NativeXpbdSolver, status as native_status
from .xpbd_core import (
    ClothBuildData,
    build_cloth_data,
    clear_cloth_topology_cache,
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
_OBJECT_CROSS_COLLISION_PROP = "ssbl_enable_cross_cloth_collision"
_IDENTITY_4X4 = np.eye(4, dtype=np.float32)
STATUS_IDLE = "Idle"
STATUS_PREVIEW_RUNNING = "Preview Running"
STATUS_PREVIEW_PAUSED = "Preview Paused"
STATUS_PREVIEW_STOPPED = "Preview Stopped"
STATUS_BAKING = "Baking"
STATUS_FINISHED = "Finished"
STATUS_ERROR = "Error"


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
    pin_targets_world: np.ndarray
    runtime_options_signature: tuple
    collision_layer: int
    cross_cloth_collision: bool
    current_positions_world: np.ndarray
    writeback_local_buffer: np.ndarray
    writeback_flat_buffer: np.ndarray
    substeps: int
    iterations: int
    writeback_interval: int
    frame_count: int
    use_object_settings: bool


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
        static_collision_ms=diag.static_collision_ms,
        dynamic_collision_ms=diag.dynamic_collision_ms,
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
        finite=diag.finite,
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
    closed_mesh = _mesh_is_probably_closed(obj.data)
    self_mode = str(getattr(settings, "self_collision_mode", "off")).lower()
    if closed_mesh and self_mode != "off" and not bool(getattr(settings, "use_volume_pressure", False)):
        warnings.append("Closed meshes with self collision usually also need volume preservation enabled.")
    if len(obj.data.polygons) > 10000:
        warnings.append("Large meshes may need optimized preview settings for stable realtime playback.")
    if bool(getattr(settings, "use_ground", False)):
        bbox_min_z = min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box)
        ground_limit = float(getattr(settings, "ground_height", 0.0)) + float(getattr(settings, "collision_margin", 0.0))
        if bbox_min_z < ground_limit - 1.0e-4:
            warnings.append("The mesh starts below the ground collision plane; expect an upward correction on the first frame.")
    if bool(getattr(settings, "multi_cloth_preview", False)):
        selected_meshes = [item for item in bpy.context.selected_objects if item and item.type == "MESH"]
        if len(selected_meshes) > 1:
            warnings.append("Multi-cloth preview scales with the number and density of selected cloth meshes.")
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


def _enabled_playback_cloth_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for obj in scene.objects:
        if obj is None or obj.type != "MESH":
            continue
        settings = _object_cloth_settings(obj)
        if settings is None:
            continue
        if _declared_input_type(obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        objects.append(obj)
    return sorted(objects, key=lambda item: (int(getattr(item, _OBJECT_COLLISION_LAYER_PROP, item.get(_OBJECT_COLLISION_LAYER_PROP, 1))), item.name.casefold()))


def start_preview(context: bpy.types.Context, obj: bpy.types.Object) -> SceneSession:
    try:
        if context.mode != "OBJECT":
            raise ValueError("Preview must be started in Object mode.")
        settings = _settings_for_object(context, obj, context.scene.ssbl_preview)
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
            slot = _create_cloth_slot(context, cloth_obj, slot_settings, depsgraph)
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
            writeback_interval=max(int(getattr(settings, "preview_writeback_interval", 1)), 1),
            cross_cloth_mode=str(getattr(settings, "cross_cloth_collision", "lower_layers")),
            last_fps_time=time.perf_counter(),
            fps_sample_frames=0,
            actual_fps=0.0,
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
        writeback_interval=max(min(slot.writeback_interval for slot in slots.values()), 1),
        cross_cloth_mode=str(getattr(scene.ssbl_preview, "cross_cloth_collision", "lower_layers")),
        last_fps_time=time.perf_counter(),
        fps_sample_frames=0,
        actual_fps=0.0,
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


def _writeback_decisions(session: SceneSession, next_frame: int, scene_end: int) -> dict[str, bool]:
    cross_cloth_enabled = len(session.slots) > 1 and str(session.cross_cloth_mode or "off").lower() != "off"
    decisions: dict[str, bool] = {}
    for slot_name, slot in session.slots.items():
        decisions[slot_name] = bool(
            session.frame_index == 0
            or ((session.frame_index + 1) % max(slot.writeback_interval, 1)) == 0
            or next_frame >= scene_end
            or cross_cloth_enabled
        )
    return decisions


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
    writeback_by_slot = _writeback_decisions(session, current_frame, scene_end=int(scene.frame_end))
    perf.writeback_performed = any(writeback_by_slot.values())
    try:
        _refresh_session_runtime_inputs(context, session, perf)
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
    should_writeback = (
        session.frame_index == 0
        or ((session.frame_index + 1) % max(session.writeback_interval, 1)) == 0
        or session.frame_index + 1 >= session.frame_count
        or next_frame >= int(scene.frame_end)
        or (len(session.slots) > 1 and str(session.cross_cloth_mode or "off").lower() != "off")
    )
    perf.writeback_performed = bool(should_writeback)
    try:
        started = time.perf_counter()
        scene.frame_set(next_frame)
        perf.frame_set_ms += _elapsed_ms(started)
        _refresh_session_runtime_inputs(context, session, perf)
        _step_session_slots(session, should_writeback, perf)
        if should_writeback:
            started = time.perf_counter()
            for slot in session.slots.values():
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
    except Exception:
        _finish_session(session, STATUS_ERROR)
        raise

    session.frame_index += 1
    perf.frame_ms = _elapsed_ms(step_started)
    session.last_diagnostics = _aggregate_session_diagnostics(session, perf)
    for slot_name in session.solve_order:
        _LAST_DIAGNOSTICS[slot_name] = session.last_diagnostics
    _update_session_fps(session, step_started)
    return False


def bake_xpbd_cache(context: bpy.types.Context, obj: bpy.types.Object) -> str:
    _ensure_supported_cloth_object(obj)

    settings = _settings_for_object(context, obj, context.scene.ssbl_preview)
    start = int(settings.bake_start)
    end = int(settings.bake_end)
    if end < start:
        raise ValueError("Bake end frame must be greater than or equal to bake start frame.")

    native = None
    _STATUS[obj.name] = STATUS_BAKING
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
        sample_count = end - start + 1

        with open(path, "wb") as handle:
            _write_pc2_header(handle, len(cloth.positions_world), start, sample_count)
            _write_pc2_sample(handle, cloth.positions_world, cloth.matrix_world_inv)
            for frame in range(start + 1, end + 1):
                _refresh_bake_runtime_inputs(context, obj, cloth, native, frame, static_signature, settings)
                native.step(max(int(settings.substeps), 1), max(int(settings.iterations), 1))
                world_positions = native.download_positions()
                _write_pc2_sample(handle, world_positions, cloth.matrix_world_inv)
        _bind_mesh_cache(obj, path, start)
        obj[_CACHE_PATH_PROP] = path
        _STATUS[obj.name] = STATUS_FINISHED
        return path
    except Exception:
        _STATUS[obj.name] = STATUS_ERROR
        raise
    finally:
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
    return bool(np.array_equal(a, b))


def _matrix_signature(matrix) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for row in matrix for value in row)


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
        source = obj.evaluated_get(depsgraph) if use_evaluated_mesh else obj
        mesh = source.data
        entries.append(
            (
                obj.name,
                len(mesh.vertices),
                len(mesh.polygons),
                _matrix_signature(source.matrix_world),
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


def _pin_targets_from_mesh(mesh: bpy.types.Mesh, matrix_world, pin_indices: np.ndarray) -> np.ndarray:
    if len(pin_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    indices = np.asarray(pin_indices, dtype=np.intp)
    coords_flat = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
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
    if not bool(getattr(settings, "multi_cloth_preview", False)):
        return [obj]
    selected = []
    for selected_obj in context.selected_objects:
        if selected_obj is None or selected_obj.type != "MESH":
            continue
        if _declared_input_type(selected_obj) in _UNSUPPORTED_INPUT_TYPES:
            continue
        selected.append(selected_obj)
    if len(selected) <= 1:
        return [obj]
    return sorted(selected, key=lambda item: (int(getattr(item, _OBJECT_COLLISION_LAYER_PROP, item.get(_OBJECT_COLLISION_LAYER_PROP, 1))), item.name.casefold()))


def _create_cloth_slot(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph,
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
    )
    suspended_modifiers = _suspend_preview_modifiers(obj, suspend_all=use_evaluated_mesh)
    preview_mesh = original_mesh.copy()
    preview_mesh.name = f"{original_mesh.name}_SSBL_XPBD_Preview"
    obj.data = preview_mesh
    writeback_flat_buffer = np.empty(cloth.positions_world.size, dtype=np.float32)
    writeback_local_buffer = writeback_flat_buffer.reshape(cloth.positions_world.shape)
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
        pin_targets_world=np.array(cloth.pin_targets_world, dtype=np.float32, copy=True),
        runtime_options_signature=_runtime_options_signature(
            settings_to_options(settings, runtime_mode_override="preview")
        ),
        collision_layer=_object_collision_layer(obj),
        cross_cloth_collision=_object_cross_collision_enabled(obj),
        current_positions_world=np.array(cloth.positions_world, dtype=np.float32, copy=True),
        writeback_local_buffer=writeback_local_buffer,
        writeback_flat_buffer=writeback_flat_buffer,
        substeps=max(int(getattr(settings, "substeps", 1)), 1),
        iterations=max(int(getattr(settings, "iterations", 1)), 1),
        writeback_interval=max(int(getattr(settings, "preview_writeback_interval", 1)), 1),
        frame_count=max(int(getattr(settings, "frame_count", 1)), 1),
        use_object_settings=bool(_object_cloth_settings(obj) is not None),
    )


def _create_native_solver(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
    use_evaluated_mesh_override: bool | None = None,
    runtime_mode_override: str | None = None,
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
        options = settings_to_options(settings, runtime_mode_override=runtime_mode_override)
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


def _refresh_session_runtime_inputs(context: bpy.types.Context, session: SceneSession, perf: FramePerf | None = None) -> None:
    refresh_started = time.perf_counter()
    slot_settings = {name: _settings_for_slot(context, slot) for name, slot in session.slots.items()}
    needs_depsgraph = any(
        getattr(slot_settings[name], "static_collider_collection", None) is not None
        or (slot.use_evaluated_mesh and len(slot.cloth.pin_indices) > 0)
        for name, slot in session.slots.items()
    )
    with _with_session_source_state(session):
        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get() if needs_depsgraph else None
        for slot in session.slots.values():
            obj = bpy.data.objects.get(slot.object_name)
            if obj is None or obj.type != "MESH":
                raise ValueError(f"Missing preview object during input refresh: {slot.object_name}")
            settings = slot_settings[slot.object_name]
            options = settings_to_options(settings, runtime_mode_override="preview")
            runtime_signature = _runtime_options_signature(options)
            static_collection = settings.static_collider_collection
            has_static_collection = static_collection is not None
            if len(slot.cloth.pin_indices) > 0:
                pin_targets, matrix_world = _pin_targets_from_object(
                    obj,
                    slot.cloth.pin_indices,
                    slot.use_evaluated_mesh,
                    depsgraph=depsgraph,
                    expected_vertex_count=len(slot.cloth.positions_world),
                )
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
            _apply_runtime_inputs(
                slot,
                options,
                runtime_signature,
                pin_targets,
                matrix_world,
                static_tris,
                static_signature,
                static_runtime_signature,
                perf,
            )
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
            slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
            if perf is not None:
                perf.download_ms += _elapsed_ms(started)


def _collect_cross_cloth_triangles(session: SceneSession, target: ClothSlot) -> np.ndarray:
    mode = str(session.cross_cloth_mode or "off").lower()
    if mode == "off":
        return np.empty((0, 3, 3), dtype=np.float32)
    all_tris: list[np.ndarray] = []
    for source in session.slots.values():
        if source.object_name == target.object_name or not source.cross_cloth_collision:
            continue
        if mode == "lower_layers" and source.collision_layer >= target.collision_layer:
            continue
        positions = source.current_positions_world
        if positions is None or len(source.cloth.triangles) == 0:
            continue
        all_tris.append(np.asarray(positions[source.cloth.triangles], dtype=np.float32))
    if not all_tris:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(all_tris, axis=0), dtype=np.float32)


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
    if static_signature != expected_static_signature:
        raise ValueError("Animated static collider topology changed during bake; fixed topology is required.")
    cloth.matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    pin_targets = np.ascontiguousarray(world_positions[cloth.pin_indices], dtype=np.float32)
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
    perf: FramePerf | None = None,
) -> None:
    if static_signature != slot.static_collider_signature:
        raise ValueError("Animated static collider topology changed during preview; fixed topology is required.")
    slot.cloth.matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    pin_targets = np.ascontiguousarray(pin_targets, dtype=np.float32)
    update_pin = not _array_equal(pin_targets, slot.pin_targets_world)
    update_runtime = runtime_signature != slot.runtime_options_signature
    update_static = static_tris is not None
    if update_pin or update_runtime or update_static:
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
        )
        slot.pin_targets_world = np.array(pin_targets, dtype=np.float32, copy=True)
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


def _object_cross_collision_enabled(obj: bpy.types.Object) -> bool:
    return bool(getattr(obj, _OBJECT_CROSS_COLLISION_PROP, obj.get(_OBJECT_CROSS_COLLISION_PROP, True)))


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
    static_collision_ms = 0.0
    dynamic_collision_ms = 0.0
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
    finite = True
    writeback_performed = False
    diag_started = time.perf_counter()
    for slot in session.slots.values():
        diag = slot.native.cached_diagnostics()
        step_ms += float(diag.step_ms)
        hash_build_ms += float(diag.hash_build_ms)
        constraints_ms += float(diag.constraints_ms)
        volume_ms += float(diag.volume_ms)
        static_collision_ms += float(diag.static_collision_ms)
        dynamic_collision_ms += float(diag.dynamic_collision_ms)
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
        static_collision_ms=static_collision_ms,
        dynamic_collision_ms=dynamic_collision_ms,
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
