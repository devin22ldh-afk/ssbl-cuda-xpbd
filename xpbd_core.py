from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import os

import bmesh
import bpy
import numpy as np


_EPS = 1.0e-8
SELF_COLLISION_OFF = 0
SELF_COLLISION_FAST = 1
SELF_COLLISION_STRICT = 2
DEFAULT_HARDNESS = 0.4
_SOFT_STRETCH_COMPLIANCE = 8.0e-6
_HARD_STRETCH_COMPLIANCE = 1.0e-9
_SOFT_BEND_COMPLIANCE = 2.0e-3
_HARD_BEND_COMPLIANCE = 1.0e-9
_SOFT_TETHER_COMPLIANCE = 8.0e-4
_HARD_TETHER_COMPLIANCE = 1.0e-9
_SOFT_TETHER_SLACK = 1.20
_HARD_TETHER_SLACK = 0.95
_HARDNESS_OUTPUT_SCALE = 0.70
_TETHER_START_HARDNESS = 0.50
_LEGACY_BALANCED_STRETCH = 1.0e-6
_LEGACY_BALANCED_BEND = 1.0e-4
_STARTUP_CACHE_LIMIT = 16


@dataclass
class PinAttachmentBatch:
    pairs: np.ndarray
    targets_world: np.ndarray


@dataclass
class SolverOptions:
    dt: float
    damping: float
    gravity: np.ndarray
    stretch_compliance: float
    stretch_optimization_enabled: bool
    stretch_optimization_strength: float
    bend_compliance: float
    lra_compliance: float
    collision_margin: float
    use_ground: bool
    ground_height: float
    use_wall: bool
    wall_origin: np.ndarray
    wall_normal: np.ndarray
    use_sphere: bool
    sphere_center: np.ndarray
    sphere_radius: float
    self_collision: bool
    self_collision_mode: int
    cloth_thickness: float
    self_collision_interval: int
    max_self_collision_neighbors: int
    fast_self_collision_passes: int
    use_volume_pressure: bool
    volume_compliance: float
    pressure_strength: float
    volume_target_scale: float
    volume_solve_interval: int
    self_probe_interval: int
    self_surface_pair_interval: int
    self_sleep_enabled: bool
    self_sleep_still_frames: int
    self_sleep_full_scan_interval: int
    self_compaction_enabled: bool
    self_sleep_motion_scale: float
    self_compaction_active_fraction_threshold: float
    self_pair_compaction_enabled: bool
    jitter_stabilizer_enabled: bool
    contact_friction: float
    contact_tangent_damping: float
    contact_compliance: float


@dataclass
class ClothBuildData:
    positions_world: np.ndarray
    inv_mass: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    edge_rest_lengths: np.ndarray
    edge_color_offsets: np.ndarray
    bends: np.ndarray
    bend_rest_lengths: np.ndarray
    bend_color_offsets: np.ndarray
    lra_edges: np.ndarray
    lra_rest_lengths: np.ndarray
    lra_color_offsets: np.ndarray
    pin_indices: np.ndarray
    pin_targets_world: np.ndarray
    matrix_world_inv: np.ndarray
    rest_volume: float
    pin_attachment_pairs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    pin_attachment_targets_world: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))


@dataclass
class _TopologyCacheEntry:
    triangles: np.ndarray
    edges: np.ndarray
    edge_color_offsets: np.ndarray
    bends: np.ndarray
    bend_color_offsets: np.ndarray
    lra_edges: np.ndarray
    lra_color_offsets: np.ndarray
    pin_indices: np.ndarray


@dataclass(frozen=True)
class HardnessDerivedSettings:
    hardness: float
    stretch_compliance: float
    stretch_optimization_enabled: bool
    stretch_optimization_strength: float
    bend_compliance: float
    hidden_tether_enabled: bool
    hidden_tether_compliance: float
    hidden_tether_slack: float


_TOPOLOGY_CACHE: OrderedDict[tuple, _TopologyCacheEntry] = OrderedDict()
_TOPOLOGY_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "last_hit": False,
}


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def clear_cloth_topology_cache() -> None:
    _TOPOLOGY_CACHE.clear()
    _TOPOLOGY_CACHE_STATS["hits"] = 0
    _TOPOLOGY_CACHE_STATS["misses"] = 0
    _TOPOLOGY_CACHE_STATS["last_hit"] = False


def make_pin_attachment_batch(pin_indices: np.ndarray, targets_world: np.ndarray) -> PinAttachmentBatch:
    pin_indices_arr = np.ascontiguousarray(pin_indices, dtype=np.int32).reshape((-1,))
    targets_arr = np.ascontiguousarray(targets_world, dtype=np.float32).reshape((-1, 3))
    if len(pin_indices_arr) != len(targets_arr):
        raise ValueError("Pin attachment indices and target positions must have matching lengths.")
    if len(pin_indices_arr) == 0:
        return PinAttachmentBatch(
            pairs=np.empty((0, 2), dtype=np.int32),
            targets_world=np.empty((0, 3), dtype=np.float32),
        )
    attachment_sources = np.arange(len(pin_indices_arr), dtype=np.int32)
    pairs = np.column_stack((pin_indices_arr, attachment_sources)).astype(np.int32, copy=False)
    return PinAttachmentBatch(
        pairs=np.ascontiguousarray(pairs, dtype=np.int32),
        targets_world=targets_arr,
    )


def cloth_topology_cache_stats() -> dict[str, int | bool]:
    return {
        "hits": int(_TOPOLOGY_CACHE_STATS["hits"]),
        "misses": int(_TOPOLOGY_CACHE_STATS["misses"]),
        "last_hit": bool(_TOPOLOGY_CACHE_STATS["last_hit"]),
        "size": int(len(_TOPOLOGY_CACHE)),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _log_interpolate(soft_value: float, hard_value: float, hardness: float) -> float:
    hardness = _clamp01(hardness)
    return float(np.exp(np.log(soft_value) * (1.0 - hardness) + np.log(hard_value) * hardness))


def _inverse_log_interpolate(value: float, soft_value: float, hard_value: float) -> float:
    value = max(float(value), _EPS)
    numerator = np.log(value) - np.log(soft_value)
    denominator = np.log(hard_value) - np.log(soft_value)
    if abs(float(denominator)) <= _EPS:
        return DEFAULT_HARDNESS
    return _clamp01(float(numerator / denominator))


def derive_hardness_settings(hardness: float) -> HardnessDerivedSettings:
    hardness = _clamp01(hardness)
    material_hardness = hardness * _HARDNESS_OUTPUT_SCALE
    tether_enabled = hardness >= _TETHER_START_HARDNESS
    if tether_enabled:
        tether_hardness = ((hardness - _TETHER_START_HARDNESS) / (1.0 - _TETHER_START_HARDNESS)) * _HARDNESS_OUTPUT_SCALE
    else:
        tether_hardness = 0.0
    return HardnessDerivedSettings(
        hardness=hardness,
        stretch_compliance=_log_interpolate(_SOFT_STRETCH_COMPLIANCE, _HARD_STRETCH_COMPLIANCE, material_hardness),
        stretch_optimization_enabled=hardness > 0.0,
        stretch_optimization_strength=hardness,
        bend_compliance=_log_interpolate(_SOFT_BEND_COMPLIANCE, _HARD_BEND_COMPLIANCE, material_hardness),
        hidden_tether_enabled=tether_enabled,
        hidden_tether_compliance=_log_interpolate(_SOFT_TETHER_COMPLIANCE, _HARD_TETHER_COMPLIANCE, tether_hardness),
        hidden_tether_slack=float(np.interp(tether_hardness, [0.0, 1.0], [_SOFT_TETHER_SLACK, _HARD_TETHER_SLACK])),
    )


def infer_hardness_from_legacy_settings(settings) -> float:
    stretch = float(getattr(settings, "stretch_compliance", _LEGACY_BALANCED_STRETCH))
    bend = float(getattr(settings, "bend_compliance", _LEGACY_BALANCED_BEND))
    legacy_like_default = (
        abs(stretch - _LEGACY_BALANCED_STRETCH) <= 1.0e-12
        and abs(bend - _LEGACY_BALANCED_BEND) <= 1.0e-10
    )
    if legacy_like_default:
        return DEFAULT_HARDNESS

    stretch_hardness = _inverse_log_interpolate(stretch, _SOFT_STRETCH_COMPLIANCE, _HARD_STRETCH_COMPLIANCE)
    bend_hardness = _inverse_log_interpolate(bend, _SOFT_BEND_COMPLIANCE, _HARD_BEND_COMPLIANCE)
    return _clamp01((stretch_hardness + bend_hardness) * 0.5)


def sync_hardness_settings(settings) -> HardnessDerivedSettings:
    hardness = _clamp01(float(getattr(settings, "hardness", DEFAULT_HARDNESS)))
    if not bool(getattr(settings, "hardness_initialized", False)):
        hardness = infer_hardness_from_legacy_settings(settings)
        settings.hardness = hardness
        settings.hardness_initialized = True

    derived = derive_hardness_settings(hardness)
    settings.stretch_compliance = derived.stretch_compliance
    settings.bend_compliance = derived.bend_compliance
    settings.use_lra = derived.hidden_tether_enabled
    settings.lra_compliance = derived.hidden_tether_compliance
    settings.lra_slack = derived.hidden_tether_slack
    return derived


def preview_hardness_settings(settings) -> HardnessDerivedSettings:
    if not bool(getattr(settings, "hardness_initialized", False)):
        return derive_hardness_settings(infer_hardness_from_legacy_settings(settings))
    return derive_hardness_settings(float(getattr(settings, "hardness", DEFAULT_HARDNESS)))


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _mesh_loop_signature(mesh: bpy.types.Mesh) -> tuple[int, str]:
    loop_indices = np.empty(len(mesh.loops), dtype=np.int32)
    if len(loop_indices) > 0:
        mesh.loops.foreach_get("vertex_index", loop_indices)
    return len(loop_indices), _array_digest(loop_indices)


def _matrix_linear_signature(matrix) -> tuple[float, ...]:
    mat = np.asarray(matrix, dtype=np.float64)
    return tuple(round(float(mat[row, col]), 6) for row in range(3) for col in range(3))


def _pin_indices_signature(pin_indices: np.ndarray) -> tuple[int, str]:
    indices = np.ascontiguousarray(pin_indices, dtype=np.int32)
    return int(len(indices)), _array_digest(indices)


def _topology_cache_key(
    obj: bpy.types.Object,
    mesh: bpy.types.Mesh,
    use_evaluated_mesh: bool,
    pin_indices: np.ndarray,
    derived: HardnessDerivedSettings,
    matrix_world,
    local_positions: np.ndarray,
) -> tuple:
    lra_position_signature = (
        _array_digest(np.asarray(local_positions, dtype=np.float32))
        if bool(derived.hidden_tether_enabled)
        else ""
    )
    return (
        int(obj.as_pointer()),
        int(obj.data.as_pointer()),
        bool(use_evaluated_mesh),
        len(mesh.vertices),
        len(mesh.polygons),
        len(mesh.loops),
        _mesh_loop_signature(mesh),
        _pin_indices_signature(pin_indices),
        bool(derived.hidden_tether_enabled),
        round(float(derived.hidden_tether_slack), 6),
        _matrix_linear_signature(matrix_world),
        lra_position_signature,
    )


def _get_topology_cache_entry(key: tuple) -> _TopologyCacheEntry | None:
    entry = _TOPOLOGY_CACHE.get(key)
    if entry is None:
        _TOPOLOGY_CACHE_STATS["misses"] = int(_TOPOLOGY_CACHE_STATS["misses"]) + 1
        _TOPOLOGY_CACHE_STATS["last_hit"] = False
        return None
    _TOPOLOGY_CACHE.move_to_end(key)
    _TOPOLOGY_CACHE_STATS["hits"] = int(_TOPOLOGY_CACHE_STATS["hits"]) + 1
    _TOPOLOGY_CACHE_STATS["last_hit"] = True
    return entry


def _store_topology_cache_entry(key: tuple, entry: _TopologyCacheEntry) -> None:
    _TOPOLOGY_CACHE[key] = entry
    _TOPOLOGY_CACHE.move_to_end(key)
    while len(_TOPOLOGY_CACHE) > _STARTUP_CACHE_LIMIT:
        _TOPOLOGY_CACHE.popitem(last=False)


def _distance_rest_lengths(constraints: np.ndarray, rest_world: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if len(constraints) == 0:
        return np.empty(0, dtype=np.float32)
    delta = rest_world[constraints[:, 1]] - rest_world[constraints[:, 0]]
    rest = np.maximum(np.linalg.norm(delta, axis=1) * float(scale), _EPS)
    return np.asarray(rest, dtype=np.float32)


def _build_cloth_data_uncached(
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> ClothBuildData:
    if obj is None or obj.type != "MESH":
        raise ValueError("当前活动对象必须是网格")

    derived = sync_hardness_settings(settings)
    use_evaluated_mesh = bool(getattr(settings, "use_evaluated_mesh", True))
    local, triangles, matrix_world = mesh_input_data(
        obj,
        use_evaluated_mesh=use_evaluated_mesh,
        depsgraph=depsgraph,
        require_matching_vertex_count=True,
    )
    world, _matrix_world = to_world(local, matrix_world)
    matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    if len(triangles) == 0:
        raise ValueError("当前网格至少需要一个面")

    pin_mask = pin_mask_from_group(obj, str(settings.pin_vertex_group).strip(), len(local))
    if np.all(pin_mask):
        raise ValueError("所有顶点都被固定了，没有可模拟的部分")
    use_volume_pressure = bool(getattr(settings, "use_volume_pressure", False))
    rest_volume = signed_mesh_volume(world, triangles)
    if use_volume_pressure:
        if not is_closed_triangle_mesh(triangles):
            raise ValueError("体积压力需要闭合流形网格；暂不支持开口布片网格。")
        if abs(rest_volume) <= 1.0e-7:
            raise ValueError("体积压力需要非零的静止有符号体积；请检查网格法线和拓扑。")
        if np.count_nonzero(~pin_mask) < 4:
            raise ValueError("体积压力至少需要四个未固定顶点才能保持体积。")

    edges, edge_rest = edge_constraints(triangles, world)
    edges, edge_rest, edge_color_offsets = color_distance_constraints(edges, edge_rest, len(world))
    bends, bend_rest = bend_constraints(triangles, world)
    bends, bend_rest, bend_color_offsets = color_distance_constraints(bends, bend_rest, len(world))
    if derived.hidden_tether_enabled:
        lra_edges, lra_rest = hidden_tether_constraints(world, pin_mask, derived.hidden_tether_slack)
    else:
        lra_edges = np.empty((0, 2), dtype=np.int32)
        lra_rest = np.empty(0, dtype=np.float32)
    lra_color_offsets = np.asarray([0], dtype=np.int32)
    inv_mass = vertex_inverse_mass(world, triangles, float(settings.density), pin_mask)
    pin_indices = np.flatnonzero(pin_mask).astype(np.int32)
    pin_targets = world[pin_indices].astype(np.float32, copy=True)
    pin_attachments = make_pin_attachment_batch(pin_indices, pin_targets)

    return ClothBuildData(
        positions_world=np.ascontiguousarray(world, dtype=np.float32),
        inv_mass=np.ascontiguousarray(inv_mass, dtype=np.float32),
        triangles=np.ascontiguousarray(triangles, dtype=np.int32),
        edges=np.ascontiguousarray(edges, dtype=np.int32),
        edge_rest_lengths=np.ascontiguousarray(edge_rest, dtype=np.float32),
        edge_color_offsets=np.ascontiguousarray(edge_color_offsets, dtype=np.int32),
        bends=np.ascontiguousarray(bends, dtype=np.int32),
        bend_rest_lengths=np.ascontiguousarray(bend_rest, dtype=np.float32),
        bend_color_offsets=np.ascontiguousarray(bend_color_offsets, dtype=np.int32),
        lra_edges=np.ascontiguousarray(lra_edges, dtype=np.int32),
        lra_rest_lengths=np.ascontiguousarray(lra_rest, dtype=np.float32),
        lra_color_offsets=np.ascontiguousarray(lra_color_offsets, dtype=np.int32),
        pin_indices=pin_attachments.pairs[:, 0].copy(),
        pin_targets_world=pin_attachments.targets_world,
        matrix_world_inv=matrix_world_inv,
        rest_volume=float(rest_volume),
        pin_attachment_pairs=pin_attachments.pairs,
        pin_attachment_targets_world=pin_attachments.targets_world,
    )


def build_cloth_data(
    obj: bpy.types.Object,
    settings,
    depsgraph: bpy.types.Depsgraph | None = None,
) -> ClothBuildData:
    if obj is None or obj.type != "MESH":
        raise ValueError("A mesh object is required for cloth simulation.")

    derived = sync_hardness_settings(settings)
    use_evaluated_mesh = bool(getattr(settings, "use_evaluated_mesh", True))
    if not use_evaluated_mesh:
        return _build_cloth_data_from_mesh(
            obj,
            settings,
            derived,
            obj.data,
            mesh_local_positions(obj.data),
            obj.matrix_world.copy(),
            use_evaluated_mesh=False,
        )

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        if len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError(
                "Evaluated cloth input must keep the same vertex count as the source mesh."
            )
        return _build_cloth_data_from_mesh(
            obj,
            settings,
            derived,
            mesh,
            mesh_local_positions(mesh),
            eval_obj.matrix_world.copy(),
            use_evaluated_mesh=True,
        )
    finally:
        eval_obj.to_mesh_clear()


def _build_cloth_data_from_mesh(
    obj: bpy.types.Object,
    settings,
    derived: HardnessDerivedSettings,
    mesh: bpy.types.Mesh,
    local: np.ndarray,
    matrix_world,
    use_evaluated_mesh: bool,
) -> ClothBuildData:
    world, _matrix_world = to_world(local, matrix_world)
    matrix_world_inv = np.array(matrix_world.inverted(), dtype=np.float32)
    pin_mask = pin_mask_from_group(obj, str(settings.pin_vertex_group).strip(), len(local))
    pin_indices = np.flatnonzero(pin_mask).astype(np.int32)
    cache_key = _topology_cache_key(
        obj,
        mesh,
        use_evaluated_mesh,
        pin_indices,
        derived,
        matrix_world,
        local,
    )
    cache_entry = _get_topology_cache_entry(cache_key)
    if cache_entry is not None:
        triangles = np.array(cache_entry.triangles, dtype=np.int32, copy=True)
        edges = np.array(cache_entry.edges, dtype=np.int32, copy=True)
        edge_color_offsets = np.array(cache_entry.edge_color_offsets, dtype=np.int32, copy=True)
        bends = np.array(cache_entry.bends, dtype=np.int32, copy=True)
        bend_color_offsets = np.array(cache_entry.bend_color_offsets, dtype=np.int32, copy=True)
        lra_edges = np.array(cache_entry.lra_edges, dtype=np.int32, copy=True)
        lra_color_offsets = np.array(cache_entry.lra_color_offsets, dtype=np.int32, copy=True)
        pin_indices = np.array(cache_entry.pin_indices, dtype=np.int32, copy=True)
    else:
        triangles = triangulated_faces(mesh)
        edges, edge_rest_for_coloring = edge_constraints(triangles, world)
        edges, _edge_rest, edge_color_offsets = color_distance_constraints(edges, edge_rest_for_coloring, len(world))
        bends, bend_rest_for_coloring = bend_constraints(triangles, world)
        bends, _bend_rest, bend_color_offsets = color_distance_constraints(bends, bend_rest_for_coloring, len(world))
        if derived.hidden_tether_enabled:
            lra_edges, _lra_rest = hidden_tether_constraints(world, pin_mask, derived.hidden_tether_slack)
        else:
            lra_edges = np.empty((0, 2), dtype=np.int32)
        lra_color_offsets = np.asarray([0], dtype=np.int32)
        _store_topology_cache_entry(
            cache_key,
            _TopologyCacheEntry(
                triangles=np.ascontiguousarray(triangles, dtype=np.int32),
                edges=np.ascontiguousarray(edges, dtype=np.int32),
                edge_color_offsets=np.ascontiguousarray(edge_color_offsets, dtype=np.int32),
                bends=np.ascontiguousarray(bends, dtype=np.int32),
                bend_color_offsets=np.ascontiguousarray(bend_color_offsets, dtype=np.int32),
                lra_edges=np.ascontiguousarray(lra_edges, dtype=np.int32),
                lra_color_offsets=np.ascontiguousarray(lra_color_offsets, dtype=np.int32),
                pin_indices=np.ascontiguousarray(pin_indices, dtype=np.int32),
            ),
        )

    if len(triangles) == 0:
        raise ValueError("The cloth mesh needs at least one face.")
    if np.all(pin_mask):
        raise ValueError("All vertices are pinned; there is no simulated cloth region.")

    use_volume_pressure = bool(getattr(settings, "use_volume_pressure", False))
    rest_volume = signed_mesh_volume(world, triangles)
    if use_volume_pressure:
        if not is_closed_triangle_mesh(triangles):
            raise ValueError("Volume pressure requires a closed triangle mesh.")
        if abs(rest_volume) <= 1.0e-7:
            raise ValueError("Volume pressure requires a non-zero signed rest volume.")
        if np.count_nonzero(~pin_mask) < 4:
            raise ValueError("Volume pressure needs at least four unpinned vertices.")

    edge_rest = _distance_rest_lengths(edges, world)
    bend_rest = _distance_rest_lengths(bends, world)
    lra_rest = _distance_rest_lengths(lra_edges, world, derived.hidden_tether_slack)
    inv_mass = vertex_inverse_mass(world, triangles, float(settings.density), pin_mask)
    pin_targets = world[pin_indices].astype(np.float32, copy=True)
    pin_attachments = make_pin_attachment_batch(pin_indices, pin_targets)

    return ClothBuildData(
        positions_world=np.ascontiguousarray(world, dtype=np.float32),
        inv_mass=np.ascontiguousarray(inv_mass, dtype=np.float32),
        triangles=np.ascontiguousarray(triangles, dtype=np.int32),
        edges=np.ascontiguousarray(edges, dtype=np.int32),
        edge_rest_lengths=np.ascontiguousarray(edge_rest, dtype=np.float32),
        edge_color_offsets=np.ascontiguousarray(edge_color_offsets, dtype=np.int32),
        bends=np.ascontiguousarray(bends, dtype=np.int32),
        bend_rest_lengths=np.ascontiguousarray(bend_rest, dtype=np.float32),
        bend_color_offsets=np.ascontiguousarray(bend_color_offsets, dtype=np.int32),
        lra_edges=np.ascontiguousarray(lra_edges, dtype=np.int32),
        lra_rest_lengths=np.ascontiguousarray(lra_rest, dtype=np.float32),
        lra_color_offsets=np.ascontiguousarray(lra_color_offsets, dtype=np.int32),
        pin_indices=pin_attachments.pairs[:, 0].copy(),
        pin_targets_world=pin_attachments.targets_world,
        matrix_world_inv=matrix_world_inv,
        rest_volume=float(rest_volume),
        pin_attachment_pairs=pin_attachments.pairs,
        pin_attachment_targets_world=pin_attachments.targets_world,
    )


def mesh_local_positions(mesh_or_obj) -> np.ndarray:
    mesh = mesh_or_obj.data if hasattr(mesh_or_obj, "data") else mesh_or_obj
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    if "position" in mesh.attributes:
        mesh.attributes["position"].data.foreach_get("vector", coords)
    else:
        mesh.vertices.foreach_get("co", coords)
    return coords.reshape((-1, 3))


def mesh_input_data(
    obj: bpy.types.Object,
    use_evaluated_mesh: bool,
    depsgraph: bpy.types.Depsgraph | None = None,
    require_matching_vertex_count: bool = False,
) -> tuple[np.ndarray, np.ndarray, bpy.types.Matrix]:
    if not use_evaluated_mesh:
        return mesh_local_positions(obj.data), triangulated_faces(obj.data), obj.matrix_world.copy()

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        if require_matching_vertex_count and len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError(
                "在 v1 中，求值后的布料输入必须与源网格保持相同顶点数；"
                "暂不支持会改变拓扑的布料修改器。"
            )
        return (
            mesh_local_positions(mesh),
            triangulated_faces(mesh),
            eval_obj.matrix_world.copy(),
        )
    finally:
        eval_obj.to_mesh_clear()


def world_positions_from_object(
    obj: bpy.types.Object,
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
                raise ValueError(
                    "动画布料输入的顶点数发生了变化；求值后的布料输入必须保持固定拓扑。"
                )
            local = mesh_local_positions(mesh)
            world, _matrix_world = to_world(local, eval_obj.matrix_world.copy())
            return world, eval_obj.matrix_world.copy()
        finally:
            eval_obj.to_mesh_clear()

    local = mesh_local_positions(obj.data)
    if expected_vertex_count is not None and len(local) != expected_vertex_count:
        raise ValueError("动画布料输入的顶点数发生了变化；必须保持固定的布料拓扑。")
    world, _matrix_world = to_world(local, obj.matrix_world.copy())
    return world, obj.matrix_world.copy()


def to_world(local: np.ndarray, matrix) -> tuple[np.ndarray, np.ndarray]:
    mat = np.array(matrix, dtype=np.float64)
    local_h = np.concatenate([local, np.ones((len(local), 1), dtype=np.float64)], axis=1)
    world = (mat @ local_h.T).T[:, :3]
    return world, mat


def to_local(world: np.ndarray, matrix_inv: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float32)
    matrix_inv = np.asarray(matrix_inv, dtype=np.float32)
    return world @ matrix_inv[:3, :3].T + matrix_inv[:3, 3]


def triangulated_faces(mesh: bpy.types.Mesh) -> np.ndarray:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.faces.ensure_lookup_table()
        triangles = [[vert.index for vert in face.verts] for face in bm.faces if len(face.verts) == 3]
        return np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    finally:
        bm.free()


def is_closed_triangle_mesh(triangles: np.ndarray) -> bool:
    if len(triangles) == 0:
        return False
    edge_use: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y in ((a, b), (b, c), (c, a)):
            edge = (x, y) if x < y else (y, x)
            edge_use[edge] = edge_use.get(edge, 0) + 1
    return bool(edge_use) and all(count == 2 for count in edge_use.values())


def signed_mesh_volume(rest_world: np.ndarray, triangles: np.ndarray) -> float:
    if len(triangles) == 0:
        return 0.0
    a = rest_world[triangles[:, 0]]
    b = rest_world[triangles[:, 1]]
    c = rest_world[triangles[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def pin_mask_from_group(obj: bpy.types.Object, group_name: str, vertex_count: int) -> np.ndarray:
    mask = np.zeros(vertex_count, dtype=bool)
    if not group_name:
        return mask
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return mask
    group_index = group.index
    for vert in obj.data.vertices:
        for assignment in vert.groups:
            if assignment.group == group_index and assignment.weight > 0.0:
                mask[vert.index] = True
                break
    return mask


def edge_constraints(triangles: np.ndarray, rest_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges: set[tuple[int, int]] = set()
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    edge_array = np.asarray(sorted(edges), dtype=np.int32).reshape((-1, 2))
    if len(edge_array) == 0:
        return edge_array, np.empty(0, dtype=np.float32)
    delta = rest_world[edge_array[:, 1]] - rest_world[edge_array[:, 0]]
    rest_lengths = np.maximum(np.linalg.norm(delta, axis=1), _EPS)
    return edge_array, rest_lengths


def bend_constraints(triangles: np.ndarray, rest_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_to_opposite: dict[tuple[int, int], int] = {}
    pairs: set[tuple[int, int]] = set()
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for x, y, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edge = (min(x, y), max(x, y))
            previous = edge_to_opposite.get(edge)
            if previous is None:
                edge_to_opposite[edge] = opposite
            elif previous != opposite:
                pairs.add((min(previous, opposite), max(previous, opposite)))
    bend_array = np.asarray(sorted(pairs), dtype=np.int32).reshape((-1, 2))
    if len(bend_array) == 0:
        return bend_array, np.empty(0, dtype=np.float32)
    delta = rest_world[bend_array[:, 1]] - rest_world[bend_array[:, 0]]
    rest_lengths = np.maximum(np.linalg.norm(delta, axis=1), _EPS)
    return bend_array, rest_lengths


def hidden_tether_constraints(
    rest_world: np.ndarray,
    pin_mask: np.ndarray,
    slack: float,
) -> tuple[np.ndarray, np.ndarray]:
    pin_indices = np.flatnonzero(pin_mask).astype(np.int32)
    dynamic_indices = np.flatnonzero(~pin_mask).astype(np.int32)
    if len(pin_indices) == 0 or len(dynamic_indices) == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    if len(pin_indices) == 1:
        nearest_pin = np.full(len(dynamic_indices), int(pin_indices[0]), dtype=np.int32)
        delta = rest_world[dynamic_indices].astype(np.float64, copy=False) - rest_world[int(pin_indices[0])].astype(
            np.float64,
            copy=False,
        )
        nearest_distance = np.linalg.norm(delta, axis=1)
        pairs = np.column_stack((nearest_pin, dynamic_indices)).astype(np.int32, copy=False)
        rest = np.maximum(nearest_distance * max(float(slack), 0.5), _EPS).astype(np.float32)
        return pairs, rest

    if _env_bool("SSBL_LRA_KDTREE_ENABLED", True):
        try:
            from scipy.spatial import cKDTree
            pin_positions = rest_world[pin_indices].astype(np.float64, copy=False)
            dynamic_positions = rest_world[dynamic_indices].astype(np.float64, copy=False)
            tree = cKDTree(pin_positions)
            distances, indices = tree.query(dynamic_positions, k=1, workers=-1)
            nearest_pin = pin_indices[indices]
            pairs = np.column_stack((nearest_pin, dynamic_indices)).astype(np.int32, copy=False)
            rest = np.maximum(distances * max(float(slack), 0.5), _EPS).astype(np.float32)
            return pairs, rest
        except Exception:
            pass

        try:
            from mathutils import kdtree

            tree = kdtree.KDTree(len(pin_indices))
            pin_positions = rest_world[pin_indices].astype(np.float64, copy=False)
            for local_index, position in enumerate(pin_positions):
                tree.insert((float(position[0]), float(position[1]), float(position[2])), local_index)
            tree.balance()

            nearest_pin = np.empty(len(dynamic_indices), dtype=np.int32)
            nearest_distance = np.empty(len(dynamic_indices), dtype=np.float64)
            dynamic_positions = rest_world[dynamic_indices].astype(np.float64, copy=False)
            for local_index, position in enumerate(dynamic_positions):
                _co, pin_local_index, distance = tree.find(
                    (float(position[0]), float(position[1]), float(position[2]))
                )
                nearest_pin[local_index] = pin_indices[int(pin_local_index)]
                nearest_distance[local_index] = float(distance)

            pairs = np.column_stack((nearest_pin, dynamic_indices)).astype(np.int32, copy=False)
            rest = np.maximum(nearest_distance * max(float(slack), 0.5), _EPS).astype(np.float32)
            return pairs, rest
        except Exception:
            pass

    pin_positions = rest_world[pin_indices].astype(np.float64, copy=False)
    dynamic_positions = rest_world[dynamic_indices].astype(np.float64, copy=False)
    nearest_pin = np.empty(len(dynamic_indices), dtype=np.int32)
    nearest_distance = np.empty(len(dynamic_indices), dtype=np.float64)
    chunk_size = 2048
    for start in range(0, len(dynamic_indices), chunk_size):
        end = min(start + chunk_size, len(dynamic_indices))
        delta = dynamic_positions[start:end, None, :] - pin_positions[None, :, :]
        distances_sq = np.einsum("cpi,cpi->cp", delta, delta, optimize=True)
        local_nearest = np.argmin(distances_sq, axis=1)
        nearest_pin[start:end] = pin_indices[local_nearest]
        nearest_distance[start:end] = np.sqrt(distances_sq[np.arange(end - start), local_nearest])

    pairs = np.column_stack((nearest_pin, dynamic_indices)).astype(np.int32, copy=False)
    rest = np.maximum(nearest_distance * max(float(slack), 0.5), _EPS).astype(np.float32)
    return pairs, rest


def color_distance_constraints(
    constraints: np.ndarray,
    rest_lengths: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(constraints) == 0:
        return constraints.reshape((0, 2)), rest_lengths.astype(np.float32), np.asarray([0], dtype=np.int32)

    used_colors_by_vertex: list[set[int]] = [set() for _ in range(vertex_count)]
    colors = np.empty(len(constraints), dtype=np.int32)
    for index, row in enumerate(constraints):
        vertices = [int(v) for v in row]
        blocked: set[int] = set()
        for vertex in vertices:
            if 0 <= vertex < vertex_count:
                blocked.update(used_colors_by_vertex[vertex])
        color = 0
        while color in blocked:
            color += 1
        colors[index] = color
        for vertex in vertices:
            if 0 <= vertex < vertex_count:
                used_colors_by_vertex[vertex].add(color)

    order = np.argsort(colors, kind="stable")
    sorted_constraints = np.ascontiguousarray(constraints[order], dtype=np.int32)
    sorted_rest = np.ascontiguousarray(rest_lengths[order], dtype=np.float32)
    sorted_colors = colors[order]
    color_count = int(sorted_colors[-1]) + 1
    counts = np.bincount(sorted_colors, minlength=color_count).astype(np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return sorted_constraints, sorted_rest, offsets


def vertex_inverse_mass(
    rest_world: np.ndarray,
    triangles: np.ndarray,
    density: float,
    pin_mask: np.ndarray,
) -> np.ndarray:
    mass = np.zeros(len(rest_world), dtype=np.float64)
    p0 = rest_world[triangles[:, 0]]
    p1 = rest_world[triangles[:, 1]]
    p2 = rest_world[triangles[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    tri_mass = np.maximum(area * max(density, _EPS), _EPS)
    share = tri_mass / 3.0
    np.add.at(mass, triangles[:, 0], share)
    np.add.at(mass, triangles[:, 1], share)
    np.add.at(mass, triangles[:, 2], share)
    mass[mass <= 0.0] = 1.0
    inv_mass = 1.0 / mass
    inv_mass[pin_mask] = 0.0
    return inv_mass


def settings_to_options(settings, runtime_mode_override: str | None = None) -> SolverOptions:
    derived = sync_hardness_settings(settings)
    sphere_center = np.zeros(3, dtype=np.float32)
    sphere_radius = 0.0
    sphere_obj = getattr(settings, "sphere_object", None)
    if bool(settings.use_sphere) and sphere_obj is not None:
        sphere_center = np.array(sphere_obj.matrix_world.translation, dtype=np.float32)
        sphere_radius = max(float(max(sphere_obj.dimensions)) * 0.5, 0.0)

    wall_normal = np.array(settings.wall_normal, dtype=np.float32)
    norm = float(np.linalg.norm(wall_normal))
    if norm <= _EPS:
        wall_normal = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    else:
        wall_normal = wall_normal / norm

    mode_name = str(getattr(settings, "self_collision_mode", "fast")).lower()
    if mode_name in {"off", "quality"}:
        # Legacy scenes/scripts may still store removed mode names. Off is now
        # controlled by self_collision=false; Quality maps to the strict solver.
        mode_name = "strict" if mode_name == "quality" else "fast"
    self_collision_enabled = bool(getattr(settings, "self_collision", False))
    if not self_collision_enabled:
        mode_value = SELF_COLLISION_OFF
    elif mode_name == "strict":
        mode_value = SELF_COLLISION_STRICT
    else:
        mode_value = SELF_COLLISION_FAST

    run_mode = (
        str(runtime_mode_override).lower()
        if runtime_mode_override is not None
        else str(getattr(settings, "runtime_mode", getattr(settings, "run_mode", "preview"))).lower()
    )
    self_sleep_enabled = (
        run_mode == "preview"
        and mode_value > SELF_COLLISION_OFF
        and bool(getattr(settings, "self_sleep_enabled", False))
    )
    self_compaction_enabled = self_sleep_enabled and bool(getattr(settings, "self_compaction_enabled", True))
    self_pair_compaction_enabled = (
        self_compaction_enabled
        and bool(getattr(settings, "self_pair_compaction_enabled", True))
    )
    jitter_env = os.environ.get("SSBL_JITTER_STABILIZER_ENABLED")
    jitter_enabled = (
        run_mode == "preview"
        and mode_value == SELF_COLLISION_FAST
        and bool(getattr(settings, "jitter_stabilizer_enabled", True))
    )
    if jitter_env is not None:
        jitter_enabled = jitter_env.strip().lower() not in {"", "0", "false", "no", "off"}
        jitter_enabled = jitter_enabled and run_mode == "preview" and mode_value == SELF_COLLISION_FAST

    return SolverOptions(
        dt=float(settings.dt),
        damping=float(settings.damping),
        gravity=np.asarray(settings.gravity, dtype=np.float32),
        stretch_compliance=float(derived.stretch_compliance),
        stretch_optimization_enabled=bool(derived.stretch_optimization_enabled),
        stretch_optimization_strength=float(derived.stretch_optimization_strength),
        bend_compliance=float(derived.bend_compliance),
        lra_compliance=float(derived.hidden_tether_compliance if derived.hidden_tether_enabled else 0.0),
        collision_margin=float(settings.collision_margin),
        use_ground=bool(settings.use_ground),
        ground_height=float(settings.ground_height),
        use_wall=bool(settings.use_wall),
        wall_origin=np.asarray(settings.wall_origin, dtype=np.float32),
        wall_normal=wall_normal,
        use_sphere=bool(settings.use_sphere and sphere_obj is not None),
        sphere_center=sphere_center,
        sphere_radius=float(sphere_radius),
        self_collision=self_collision_enabled,
        self_collision_mode=mode_value,
        cloth_thickness=float(getattr(settings, "cloth_thickness", 0.02)),
        self_collision_interval=max(int(getattr(settings, "self_collision_interval", 2)), 1),
        max_self_collision_neighbors=max(int(getattr(settings, "max_self_collision_neighbors", 32)), 4),
        fast_self_collision_passes=min(max(int(getattr(settings, "fast_self_collision_passes", 4)), 1), 8),
        use_volume_pressure=bool(getattr(settings, "use_volume_pressure", False)),
        volume_compliance=float(getattr(settings, "volume_compliance", 1e-6)),
        pressure_strength=max(float(getattr(settings, "pressure_strength", 1.0)), 0.0),
        volume_target_scale=float(getattr(settings, "volume_target_scale", 1.0)),
        volume_solve_interval=max(int(getattr(settings, "volume_solve_interval", 1)), 1),
        self_probe_interval=max(int(getattr(settings, "self_probe_interval", 1)), 1),
        self_surface_pair_interval=max(int(getattr(settings, "self_surface_pair_interval", 1)), 1),
        self_sleep_enabled=self_sleep_enabled,
        self_sleep_still_frames=max(int(getattr(settings, "self_sleep_still_frames", 10)), 1),
        self_sleep_full_scan_interval=max(int(getattr(settings, "self_sleep_full_scan_interval", 30)), 1),
        self_compaction_enabled=self_compaction_enabled,
        self_sleep_motion_scale=float(getattr(settings, "self_sleep_motion_scale", 1.0 if self_sleep_enabled else 0.25)),
        self_compaction_active_fraction_threshold=float(
            getattr(settings, "self_compaction_active_fraction_threshold", 0.75)
        ),
        self_pair_compaction_enabled=self_pair_compaction_enabled,
        jitter_stabilizer_enabled=jitter_enabled,
        contact_friction=max(float(getattr(settings, "contact_friction", 0.35)), 0.0),
        contact_tangent_damping=max(float(getattr(settings, "contact_tangent_damping", 0.2)), 0.0),
        contact_compliance=max(float(getattr(settings, "contact_compliance", 0.0)), 0.0),
    )
