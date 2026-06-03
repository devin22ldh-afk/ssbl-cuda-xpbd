from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import bpy
from mathutils import Matrix, Vector


MAX_FORCE_FIELDS = 64

FORCE_FIELD_WIND = 1
FORCE_FIELD_FORCE = 2
FORCE_FIELD_VORTEX = 3
FORCE_FIELD_TURBULENCE = 4
FORCE_FIELD_CHARGE = 5
FORCE_FIELD_HARMONIC = 6
FORCE_FIELD_LENNARDJ = 7
FORCE_FIELD_MAGNET = 8
FORCE_FIELD_DRAG = 9
FORCE_FIELD_TEXTURE = 10

_SUPPORTED_TYPES = {
    "WIND": FORCE_FIELD_WIND,
    "FORCE": FORCE_FIELD_FORCE,
    "VORTEX": FORCE_FIELD_VORTEX,
    "TURBULENCE": FORCE_FIELD_TURBULENCE,
    "CHARGE": FORCE_FIELD_CHARGE,
    "HARMONIC": FORCE_FIELD_HARMONIC,
    "LENNARDJ": FORCE_FIELD_LENNARDJ,
    "MAGNET": FORCE_FIELD_MAGNET,
    "DRAG": FORCE_FIELD_DRAG,
    "TEXTURE": FORCE_FIELD_TEXTURE,
}


@dataclass(frozen=True)
class ForceFieldSample:
    field_type: int
    strength: float
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    axis: tuple[float, float, float]
    falloff_power: float
    distance_min: float
    distance_max: float
    radial_min: float
    radial_max: float
    use_min_distance: int
    use_max_distance: int
    use_radial_min: int
    use_radial_max: int
    use_2d_force: int
    noise: float
    seed: int
    linear_drag: float
    quadratic_drag: float
    harmonic_damping: float
    flow: float
    size: float
    rest_length: float
    radial_falloff: float
    texture_nabla: float


@dataclass(frozen=True)
class ForceFieldBatch:
    fields: tuple[ForceFieldSample, ...] = ()
    unsupported_count: int = 0


EMPTY_FORCE_FIELD_BATCH = ForceFieldBatch()


def _safe_normalized(vec: Vector, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if vec.length <= 1.0e-8:
        return fallback
    normalized = vec.normalized()
    return (float(normalized.x), float(normalized.y), float(normalized.z))


def _field_objects(scene: bpy.types.Scene, collection: bpy.types.Collection | None) -> Iterable[bpy.types.Object]:
    if collection is not None:
        return collection.all_objects
    return scene.objects


def has_force_field_sources(scene: bpy.types.Scene, settings) -> bool:
    if not bool(getattr(settings, "use_blender_force_fields", False)):
        return False
    collection = getattr(settings, "force_field_collection", None)
    for obj in _field_objects(scene, collection):
        field = getattr(obj, "field", None)
        if field is None:
            continue
        field_type_name = str(getattr(field, "type", "NONE")).upper()
        if field_type_name not in {"", "NONE"}:
            return True
    return False


def _evaluated_object(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph | None) -> bpy.types.Object:
    if depsgraph is None:
        return obj
    try:
        return obj.evaluated_get(depsgraph)
    except Exception:
        return obj


def _matrix_axes(matrix: Matrix) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    linear = matrix.to_3x3()
    local_z = linear @ Vector((0.0, 0.0, 1.0))
    direction = _safe_normalized(local_z, (0.0, 0.0, 1.0))
    return direction, direction


def collect_force_fields(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph | None,
    settings,
) -> ForceFieldBatch:
    if not bool(getattr(settings, "use_blender_force_fields", False)):
        return EMPTY_FORCE_FIELD_BATCH

    collection = getattr(settings, "force_field_collection", None)
    strength_scale = float(getattr(settings, "force_field_strength_scale", 1.0))
    fields: list[ForceFieldSample] = []
    unsupported_count = 0

    for source_obj in _field_objects(scene, collection):
        obj = _evaluated_object(source_obj, depsgraph)
        field = getattr(obj, "field", None)
        if field is None:
            continue
        field_type_name = str(getattr(field, "type", "NONE")).upper()
        if field_type_name in {"", "NONE"}:
            continue
        field_type = _SUPPORTED_TYPES.get(field_type_name)
        if field_type is None:
            unsupported_count += 1
            continue
        if len(fields) >= MAX_FORCE_FIELDS:
            raise ValueError(f"SSBL supports at most {MAX_FORCE_FIELDS} Blender force fields per cloth object.")

        matrix = obj.matrix_world.copy()
        direction, axis = _matrix_axes(matrix)
        origin = matrix.translation
        fields.append(
            ForceFieldSample(
                field_type=field_type,
                strength=float(getattr(field, "strength", 0.0)) * strength_scale,
                origin=(float(origin.x), float(origin.y), float(origin.z)),
                direction=direction,
                axis=axis,
                falloff_power=float(getattr(field, "falloff_power", 0.0)),
                distance_min=max(float(getattr(field, "distance_min", 0.0)), 0.0),
                distance_max=max(float(getattr(field, "distance_max", 0.0)), 0.0),
                radial_min=max(float(getattr(field, "radial_min", 0.0)), 0.0),
                radial_max=max(float(getattr(field, "radial_max", 0.0)), 0.0),
                use_min_distance=1 if bool(getattr(field, "use_min_distance", False)) else 0,
                use_max_distance=1 if bool(getattr(field, "use_max_distance", False)) else 0,
                use_radial_min=1 if bool(getattr(field, "use_radial_min", False)) else 0,
                use_radial_max=1 if bool(getattr(field, "use_radial_max", False)) else 0,
                use_2d_force=1 if bool(getattr(field, "use_2d_force", False)) else 0,
                noise=max(float(getattr(field, "noise", 0.0)), 0.0),
                seed=int(getattr(field, "seed", 0)),
                linear_drag=float(getattr(field, "linear_drag", 0.0)),
                quadratic_drag=float(getattr(field, "quadratic_drag", 0.0)),
                harmonic_damping=float(getattr(field, "harmonic_damping", 0.0)),
                flow=float(getattr(field, "flow", 0.0)),
                size=float(getattr(field, "size", 0.0)),
                rest_length=float(getattr(field, "rest_length", 0.0)),
                radial_falloff=float(getattr(field, "radial_falloff", 0.0)),
                texture_nabla=float(getattr(field, "texture_nabla", 0.0)),
            )
        )

    return ForceFieldBatch(fields=tuple(fields), unsupported_count=unsupported_count)
