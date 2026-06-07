from __future__ import annotations

from dataclasses import dataclass
import math
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

_FIELD_TYPE_TO_WEIGHT_PROPERTY = {
    "FORCE": "force_field_weight_force",
    "WIND": "force_field_weight_wind",
    "VORTEX": "force_field_weight_vortex",
    "TURBULENCE": "force_field_weight_turbulence",
    "CHARGE": "force_field_weight_charge",
    "HARMONIC": "force_field_weight_harmonic",
    "LENNARDJ": "force_field_weight_lennardjones",
    "MAGNET": "force_field_weight_magnetic",
    "DRAG": "force_field_weight_drag",
    "TEXTURE": "force_field_weight_texture",
}

_VISIBLE_WEIGHT_PROPERTY_GROUPS = (
    (
        "force_field_weight_gravity",
        "force_field_weight_all",
        "force_field_weight_force",
        "force_field_weight_vortex",
    ),
    (
        "force_field_weight_magnetic",
        "force_field_weight_harmonic",
        "force_field_weight_charge",
        "force_field_weight_lennardjones",
    ),
    (
        "force_field_weight_wind",
        "force_field_weight_texture",
    ),
    (
        "force_field_weight_turbulence",
        "force_field_weight_drag",
    ),
)


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


def visible_force_field_weight_groups() -> tuple[tuple[str, ...], ...]:
    return _VISIBLE_WEIGHT_PROPERTY_GROUPS


def visible_force_field_weight_properties() -> tuple[str, ...]:
    return tuple(prop_name for group in _VISIBLE_WEIGHT_PROPERTY_GROUPS for prop_name in group)


def _clamp_weight(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _non_negative_float(value, default: float = 0.0) -> float:
    return max(_finite_float(value, default), 0.0)


def _finite_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _setting_weight(settings, identifier: str, default: float = 1.0) -> float:
    return _clamp_weight(_finite_float(getattr(settings, identifier, default), default))


def _setting_scale(settings, identifier: str, default: float = 1.0) -> float:
    return _non_negative_float(getattr(settings, identifier, default), default)


def gravity_weight(settings) -> float:
    return _setting_weight(settings, "force_field_weight_gravity", 1.0)


def all_force_field_weight(settings) -> float:
    return _setting_weight(settings, "force_field_weight_all", 1.0)


def force_field_strength_scale(settings) -> float:
    return _setting_scale(settings, "force_field_strength_scale", 1.0)


def field_type_weight(settings, field_type_name: str) -> float:
    prop_name = _FIELD_TYPE_TO_WEIGHT_PROPERTY.get(str(field_type_name).upper())
    if prop_name is None:
        return 0.0
    return _setting_weight(settings, prop_name, 1.0)


def _safe_normalized(vec: Vector, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if vec.length <= 1.0e-8:
        return fallback
    normalized = vec.normalized()
    return (float(normalized.x), float(normalized.y), float(normalized.z))


def _field_objects(scene: bpy.types.Scene, collection: bpy.types.Collection | None) -> Iterable[bpy.types.Object]:
    if collection is not None:
        return collection.all_objects
    return scene.objects


def _iter_force_field_sources(objects: Iterable[bpy.types.Object]):
    for obj in objects:
        field = getattr(obj, "field", None)
        if field is None:
            continue
        field_type_name = str(getattr(field, "type", "NONE")).upper()
        if field_type_name in {"", "NONE"}:
            continue
        yield obj, field, field_type_name


def has_force_field_sources(scene: bpy.types.Scene, settings) -> bool:
    collection = getattr(settings, "force_field_collection", None)
    for _obj, _field, field_type_name in _iter_force_field_sources(_field_objects(scene, collection)):
        if field_type_name in _SUPPORTED_TYPES:
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
    collection = getattr(settings, "force_field_collection", None)
    all_weight = all_force_field_weight(settings) * force_field_strength_scale(settings)
    fields: list[ForceFieldSample] = []
    unsupported_count = 0

    for source_obj, _source_field, field_type_name in _iter_force_field_sources(_field_objects(scene, collection)):
        obj = _evaluated_object(source_obj, depsgraph)
        field = getattr(obj, "field", None)
        if field is None:
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
        type_weight = field_type_weight(settings, field_type_name)
        strength = _finite_float(getattr(field, "strength", 0.0), 0.0) * all_weight * type_weight
        fields.append(
            ForceFieldSample(
                field_type=field_type,
                strength=strength,
                origin=(float(origin.x), float(origin.y), float(origin.z)),
                direction=direction,
                axis=axis,
                falloff_power=_non_negative_float(getattr(field, "falloff_power", 0.0), 0.0),
                distance_min=_non_negative_float(getattr(field, "distance_min", 0.0), 0.0),
                distance_max=_non_negative_float(getattr(field, "distance_max", 0.0), 0.0),
                radial_min=_non_negative_float(getattr(field, "radial_min", 0.0), 0.0),
                radial_max=_non_negative_float(getattr(field, "radial_max", 0.0), 0.0),
                use_min_distance=1 if bool(getattr(field, "use_min_distance", False)) else 0,
                use_max_distance=1 if bool(getattr(field, "use_max_distance", False)) else 0,
                use_radial_min=1 if bool(getattr(field, "use_radial_min", False)) else 0,
                use_radial_max=1 if bool(getattr(field, "use_radial_max", False)) else 0,
                use_2d_force=1 if bool(getattr(field, "use_2d_force", False)) else 0,
                noise=_non_negative_float(getattr(field, "noise", 0.0), 0.0),
                seed=_finite_int(getattr(field, "seed", 0), 0),
                linear_drag=_non_negative_float(getattr(field, "linear_drag", 0.0), 0.0),
                quadratic_drag=_non_negative_float(getattr(field, "quadratic_drag", 0.0), 0.0),
                harmonic_damping=_non_negative_float(getattr(field, "harmonic_damping", 0.0), 0.0),
                flow=_finite_float(getattr(field, "flow", 0.0), 0.0),
                size=_non_negative_float(getattr(field, "size", 0.0), 0.0),
                rest_length=_non_negative_float(getattr(field, "rest_length", 0.0), 0.0),
                radial_falloff=_non_negative_float(getattr(field, "radial_falloff", 0.0), 0.0),
                texture_nabla=_non_negative_float(getattr(field, "texture_nabla", 0.0), 0.0),
            )
        )

    return ForceFieldBatch(fields=tuple(fields), unsupported_count=unsupported_count)
