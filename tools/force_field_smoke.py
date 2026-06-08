from __future__ import annotations

import json
import math
import os
import struct
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl.force_fields import MAX_FORCE_FIELDS, collect_force_fields, visible_force_field_weight_properties


EXPECTED_VISIBLE_WEIGHT_ORDER = (
    "force_field_weight_gravity",
    "force_field_weight_all",
    "force_field_weight_force",
    "force_field_weight_vortex",
    "force_field_weight_magnetic",
    "force_field_weight_harmonic",
    "force_field_weight_charge",
    "force_field_weight_lennardjones",
    "force_field_weight_wind",
    "force_field_weight_texture",
    "force_field_weight_turbulence",
    "force_field_weight_drag",
)


PARAMETER_EFFECT_EPSILON = 2.0e-4
NOISE_SPATIAL_EPSILON = 2.0e-5
TURBULENCE_PERF_WARNING_RATIO = 1.15


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _make_cloth(
    name: str,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=9,
        y_subdivisions=9,
        size=1.0,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = False
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.gravity = gravity
    settings.substeps = 2
    settings.iterations = 1
    settings.preview_writeback_interval = 1
    settings.bake_start = 1
    settings.bake_end = 4
    return obj


def _add_wind(name: str, strength: float, rotation=(0.0, 0.0, 0.0)) -> bpy.types.Object:
    return _add_effector("WIND", name, strength, rotation=rotation)


def _add_effector(
    field_type: str,
    name: str,
    strength: float,
    location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.object.effector_add(type=field_type, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.field.strength = strength
    return obj


def _set_field_property(field, identifier: str, value) -> None:
    if not hasattr(field, identifier):
        raise RuntimeError(f"Force field {getattr(field, 'type', 'UNKNOWN')} has no parameter {identifier!r}")
    setattr(field, identifier, value)


def _apply_field_properties(field, values: dict[str, object]) -> None:
    for identifier, value in values.items():
        _set_field_property(field, identifier, value)


def _average_axis(obj: bpy.types.Object, axis_name: str) -> float:
    return sum(float(getattr(vertex.co, axis_name)) for vertex in obj.data.vertices) / max(len(obj.data.vertices), 1)


def _shape_signature(obj: bpy.types.Object) -> tuple[float, ...]:
    vertices = list(obj.data.vertices)
    if not vertices:
        return (0.0,) * 13
    xs = [float(vertex.co.x) for vertex in vertices]
    ys = [float(vertex.co.y) for vertex in vertices]
    zs = [float(vertex.co.z) for vertex in vertices]
    count = float(len(vertices))
    return (
        sum(xs) / count,
        sum(ys) / count,
        sum(zs) / count,
        min(xs),
        max(xs),
        min(ys),
        max(ys),
        min(zs),
        max(zs),
        sum(abs(value) for value in xs) / count,
        sum(abs(value) for value in ys) / count,
        sum(abs(value) for value in zs) / count,
        math.sqrt(sum(x * x + y * y + z * z for x, y, z in zip(xs, ys, zs)) / count),
    )


def _displacement_metrics(
    obj: bpy.types.Object,
    initial_positions: list[tuple[float, float, float]],
) -> dict[str, float]:
    vertices = list(obj.data.vertices)
    if not vertices or len(vertices) != len(initial_positions):
        return {
            "center_displacement": 0.0,
            "local_displacement_rms": 0.0,
        }
    displacements = [
        (
            float(vertex.co.x) - start[0],
            float(vertex.co.y) - start[1],
            float(vertex.co.z) - start[2],
        )
        for vertex, start in zip(vertices, initial_positions)
    ]
    count = float(len(displacements))
    mean = (
        sum(delta[0] for delta in displacements) / count,
        sum(delta[1] for delta in displacements) / count,
        sum(delta[2] for delta in displacements) / count,
    )
    residual_rms = math.sqrt(
        sum(
            (delta[0] - mean[0]) ** 2
            + (delta[1] - mean[1]) ** 2
            + (delta[2] - mean[2]) ** 2
            for delta in displacements
        )
        / count
    )
    return {
        "center_displacement": math.sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]),
        "local_displacement_rms": residual_rms,
    }


def _signature_delta(left: dict[str, object], right: dict[str, object]) -> float:
    left_signature = tuple(left.get("signature", ()))
    right_signature = tuple(right.get("signature", ()))
    if len(left_signature) != len(right_signature):
        return math.inf
    return max(abs(float(a) - float(b)) for a, b in zip(left_signature, right_signature))


def _run_preview(scene: bpy.types.Scene, obj: bpy.types.Object, steps: int = 5) -> dict[str, object]:
    scene.frame_start = 1
    scene.frame_end = steps + 1
    scene.frame_current = 1
    bpy.context.view_layer.objects.active = obj
    initial_positions = [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    for frame in range(2, steps + 2):
        scene.frame_current = frame
        ssbl.solver.step_timeline_preview(bpy.context, scene)
    diag = ssbl.solver.session_diagnostics(obj)
    avg_x = _average_axis(obj, "x")
    avg_y = _average_axis(obj, "y")
    avg_z = _average_axis(obj, "z")
    signature = _shape_signature(obj)
    displacement = _displacement_metrics(obj, initial_positions)
    finite = all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )
    ssbl.solver.reset_preview_object(obj)
    return {
        "slots": len(session.slots) if session else 0,
        "avg_x": avg_x,
        "avg_y": avg_y,
        "avg_z": avg_z,
        "signature": signature,
        "center_displacement": displacement["center_displacement"],
        "local_displacement_rms": displacement["local_displacement_rms"],
        "finite": bool(finite and diag.finite),
        "force_field_count": int(diag.force_field_count),
        "unsupported_force_field_count": int(diag.unsupported_force_field_count),
        "step_ms": float(diag.step_ms),
    }


def _run_single_field_case(
    scene: bpy.types.Scene,
    name: str,
    field_type: str,
    strength: float,
    configure=None,
    cloth_location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    field_location: tuple[float, float, float] = (0.0, 0.0, 0.0),
    field_rotation=(0.0, 0.0, 0.0),
    steps: int = 5,
) -> tuple[dict[str, object], object | None]:
    _clear_scene()
    cloth = _make_cloth(f"SSBL_{name}_Cloth", location=cloth_location)
    field_obj = _add_effector(
        field_type,
        f"SSBL_{name}_{field_type}",
        strength,
        location=field_location,
        rotation=field_rotation,
    )
    if configure is not None:
        configure(field_obj.field)
    run = _run_preview(scene, cloth, steps=steps)
    batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), cloth.ssbl_cloth)
    sample = batch.fields[0] if batch.fields else None
    return run, sample


def _run_gravity_drag_case(
    scene: bpy.types.Scene,
    name: str,
    configure_drag=None,
    steps: int = 8,
) -> tuple[dict[str, object], object | None]:
    _clear_scene()
    cloth = _make_cloth(
        f"SSBL_{name}_Cloth",
        location=(0.0, 0.0, 1.0),
        gravity=(0.0, 0.0, -9.8),
    )
    if configure_drag is not None:
        field_obj = _add_effector("DRAG", f"SSBL_{name}_DRAG", 1.0)
        configure_drag(field_obj.field)
    run = _run_preview(scene, cloth, steps=steps)
    batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), cloth.ssbl_cloth)
    sample = batch.fields[0] if batch.fields else None
    return run, sample


def _has_parameter_effect(left: dict[str, object], right: dict[str, object], epsilon: float = PARAMETER_EFFECT_EPSILON) -> bool:
    return bool(left.get("finite")) and bool(right.get("finite")) and _signature_delta(left, right) > epsilon


def _sample_dict(sample) -> dict[str, object]:
    if sample is None:
        return {}
    keys = (
        "field_type",
        "strength",
        "falloff_power",
        "distance_min",
        "distance_max",
        "radial_min",
        "radial_max",
        "use_min_distance",
        "use_max_distance",
        "use_radial_min",
        "use_radial_max",
        "use_2d_force",
        "use_global_coords",
        "apply_to_location",
        "noise",
        "seed",
        "linear_drag",
        "quadratic_drag",
        "harmonic_damping",
        "flow",
        "size",
        "rest_length",
        "radial_falloff",
        "texture_nabla",
    )
    return {key: getattr(sample, key) for key in keys}


def _read_pc2_average_axis(path: str, sample_index: int, axis: int) -> float:
    with open(path, "rb") as handle:
        signature, version, vertex_count, start, sample_rate, sample_count = struct.unpack("<12siiffi", handle.read(32))
        if signature.rstrip(b"\0") != b"POINTCACHE2":
            raise RuntimeError(f"Unexpected PC2 signature: {signature!r}")
        sample_index = max(0, min(int(sample_index), int(sample_count) - 1))
        handle.seek(32 + sample_index * int(vertex_count) * 3 * 4)
        values = struct.unpack("<" + "f" * int(vertex_count) * 3, handle.read(int(vertex_count) * 3 * 4))
    samples = values[int(axis)::3]
    return sum(samples) / max(len(samples), 1)


def _vector_list(values) -> list[float]:
    return [float(value) for value in values]


BLENDER_NOISE_HASH = (
    0xA2, 0xA0, 0x19, 0x3B, 0xF8, 0xEB, 0xAA, 0xEE, 0xF3, 0x1C, 0x67, 0x28, 0x1D, 0xED, 0x00, 0xDE,
    0x95, 0x2E, 0xDC, 0x3F, 0x3A, 0x82, 0x35, 0x4D, 0x6C, 0xBA, 0x36, 0xD0, 0xF6, 0x0C, 0x79, 0x32,
    0xD1, 0x59, 0xF4, 0x08, 0x8B, 0x63, 0x89, 0x2F, 0xB8, 0xB4, 0x97, 0x83, 0xF2, 0x8F, 0x18, 0xC7,
    0x51, 0x14, 0x65, 0x87, 0x48, 0x20, 0x42, 0xA8, 0x80, 0xB5, 0x40, 0x13, 0xB2, 0x22, 0x7E, 0x57,
    0xBC, 0x7F, 0x6B, 0x9D, 0x86, 0x4C, 0xC8, 0xDB, 0x7C, 0xD5, 0x25, 0x4E, 0x5A, 0x55, 0x74, 0x50,
    0xCD, 0xB3, 0x7A, 0xBB, 0xC3, 0xCB, 0xB6, 0xE2, 0xE4, 0xEC, 0xFD, 0x98, 0x0B, 0x96, 0xD3, 0x9E,
    0x5C, 0xA1, 0x64, 0xF1, 0x81, 0x61, 0xE1, 0xC4, 0x24, 0x72, 0x49, 0x8C, 0x90, 0x4B, 0x84, 0x34,
    0x38, 0xAB, 0x78, 0xCA, 0x1F, 0x01, 0xD7, 0x93, 0x11, 0xC1, 0x58, 0xA9, 0x31, 0xF9, 0x44, 0x6D,
    0xBF, 0x33, 0x9C, 0x5F, 0x09, 0x94, 0xA3, 0x85, 0x06, 0xC6, 0x9A, 0x1E, 0x7B, 0x46, 0x15, 0x30,
    0x27, 0x2B, 0x1B, 0x71, 0x3C, 0x5B, 0xD6, 0x6F, 0x62, 0xAC, 0x4F, 0xC2, 0xC0, 0x0E, 0xB1, 0x23,
    0xA7, 0xDF, 0x47, 0xB0, 0x77, 0x69, 0x05, 0xE9, 0xE6, 0xE7, 0x76, 0x73, 0x0F, 0xFE, 0x6E, 0x9B,
    0x56, 0xEF, 0x12, 0xA5, 0x37, 0xFC, 0xAE, 0xD9, 0x03, 0x8E, 0xDD, 0x10, 0xB9, 0xCE, 0xC9, 0x8D,
    0xDA, 0x2A, 0xBD, 0x68, 0x17, 0x9F, 0xBE, 0xD4, 0x0A, 0xCC, 0xD2, 0xE8, 0x43, 0x3D, 0x70, 0xB7,
    0x02, 0x7D, 0x99, 0xD8, 0x0D, 0x60, 0x8A, 0x04, 0x2C, 0x3E, 0x92, 0xE5, 0xAF, 0x53, 0x07, 0xE0,
    0x29, 0xA6, 0xC5, 0xE3, 0xF5, 0xF7, 0x4A, 0x41, 0x26, 0x6A, 0x16, 0x5E, 0x52, 0x2D, 0x21, 0xAD,
    0xF0, 0x91, 0xFF, 0xEA, 0x54, 0xFA, 0x66, 0x1A, 0x45, 0x39, 0xCF, 0x75, 0xA4, 0x88, 0xFB, 0x5D,
) * 2


def _blender_noise_lerp(t: float, a: float, b: float) -> float:
    return a + t * (b - a)


def _blender_noise_fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _blender_noise_grad(hash_value: int, x: float, y: float, z: float) -> float:
    h = hash_value & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _blender_new_perlin(x: float, y: float, z: float) -> float:
    floor_x = math.floor(x)
    floor_y = math.floor(y)
    floor_z = math.floor(z)
    X = int(floor_x) & 255
    Y = int(floor_y) & 255
    Z = int(floor_z) & 255
    x -= floor_x
    y -= floor_y
    z -= floor_z
    u = _blender_noise_fade(x)
    v = _blender_noise_fade(y)
    w = _blender_noise_fade(z)
    A = BLENDER_NOISE_HASH[X] + Y
    AA = BLENDER_NOISE_HASH[A] + Z
    AB = BLENDER_NOISE_HASH[A + 1] + Z
    B = BLENDER_NOISE_HASH[X + 1] + Y
    BA = BLENDER_NOISE_HASH[B] + Z
    BB = BLENDER_NOISE_HASH[B + 1] + Z
    return _blender_noise_lerp(
        w,
        _blender_noise_lerp(
            v,
            _blender_noise_lerp(
                u,
                _blender_noise_grad(BLENDER_NOISE_HASH[AA], x, y, z),
                _blender_noise_grad(BLENDER_NOISE_HASH[BA], x - 1.0, y, z),
            ),
            _blender_noise_lerp(
                u,
                _blender_noise_grad(BLENDER_NOISE_HASH[AB], x, y - 1.0, z),
                _blender_noise_grad(BLENDER_NOISE_HASH[BB], x - 1.0, y - 1.0, z),
            ),
        ),
        _blender_noise_lerp(
            v,
            _blender_noise_lerp(
                u,
                _blender_noise_grad(BLENDER_NOISE_HASH[AA + 1], x, y, z - 1.0),
                _blender_noise_grad(BLENDER_NOISE_HASH[BA + 1], x - 1.0, y, z - 1.0),
            ),
            _blender_noise_lerp(
                u,
                _blender_noise_grad(BLENDER_NOISE_HASH[AB + 1], x, y - 1.0, z - 1.0),
                _blender_noise_grad(BLENDER_NOISE_HASH[BB + 1], x - 1.0, y - 1.0, z - 1.0),
            ),
        ),
    )


def _blender_generic_turbulence(noise_size: float, x: float, y: float, z: float) -> float:
    if noise_size != 0.0 and math.isfinite(noise_size):
        inv_size = 1.0 / noise_size
        x *= inv_size
        y *= inv_size
        z *= inv_size
    total = 0.0
    amp = 1.0
    fscale = 1.0
    for _ in range(3):
        total += (0.5 + 0.5 * _blender_new_perlin(fscale * x, fscale * y, fscale * z)) * amp
        amp *= 0.5
        fscale *= 2.0
    return total * (4.0 / 7.0)


def _blender_turbulence_vector(noise_size: float, co: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = co
    return (
        -1.0 + 2.0 * _blender_generic_turbulence(noise_size, x, y, z),
        -1.0 + 2.0 * _blender_generic_turbulence(noise_size, y, z, x),
        -1.0 + 2.0 * _blender_generic_turbulence(noise_size, z, x, y),
    )


def _first_field_sample(scene: bpy.types.Scene, cloth: bpy.types.Object):
    batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), cloth.ssbl_cloth)
    return batch.fields[0] if batch.fields else None


def _run_animated_transform_collect_case(scene: bpy.types.Scene) -> dict[str, object]:
    _clear_scene()
    cloth = _make_cloth("SSBL_Force_AnimatedTransform")
    wind = _add_wind("SSBL_Wind_AnimatedTransform", 2.0)
    scene.frame_set(1)
    wind.rotation_euler = (0.0, 0.0, 0.0)
    wind.keyframe_insert("rotation_euler", frame=1)
    scene.frame_set(10)
    wind.rotation_euler = (0.0, math.pi / 2.0, 0.0)
    wind.keyframe_insert("rotation_euler", frame=10)

    samples: dict[int, object] = {}
    for frame in (1, 10):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples[frame] = _first_field_sample(scene, cloth)

    frame_1 = samples[1]
    frame_10 = samples[10]
    return {
        "frame_1_direction": _vector_list(frame_1.direction) if frame_1 is not None else [],
        "frame_10_direction": _vector_list(frame_10.direction) if frame_10 is not None else [],
        "frame_1_count": 1 if frame_1 is not None else 0,
        "frame_10_count": 1 if frame_10 is not None else 0,
    }


def _run_collection_filter_edge_cases(scene: bpy.types.Scene) -> dict[str, object]:
    _clear_scene()
    empty_cloth = _make_cloth("SSBL_Force_EmptyCollection")
    empty_collection = bpy.data.collections.new("SSBL_Force_EmptyCollection_Filter")
    scene.collection.children.link(empty_collection)
    _add_wind("SSBL_Wind_OutsideEmptyCollection", 5.0)
    empty_cloth.ssbl_cloth.force_field_collection = empty_collection
    empty_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), empty_cloth.ssbl_cloth)

    _clear_scene()
    nested_cloth = _make_cloth("SSBL_Force_NestedCollection")
    parent = bpy.data.collections.new("SSBL_Force_NestedParent")
    child = bpy.data.collections.new("SSBL_Force_NestedChild")
    scene.collection.children.link(parent)
    parent.children.link(child)
    included = _add_wind("SSBL_Wind_NestedIncluded", 6.0)
    _add_wind("SSBL_Wind_NestedExcluded", 11.0)
    child.objects.link(included)
    nested_cloth.ssbl_cloth.force_field_collection = parent
    nested_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), nested_cloth.ssbl_cloth)

    return {
        "empty_count": len(empty_batch.fields),
        "empty_unsupported_count": int(empty_batch.unsupported_count),
        "nested_count": len(nested_batch.fields),
        "nested_strength": float(nested_batch.fields[0].strength) if nested_batch.fields else 0.0,
        "nested_unsupported_count": int(nested_batch.unsupported_count),
    }


def _run_collection_membership_live_case(scene: bpy.types.Scene) -> dict[str, object]:
    _clear_scene()
    scene.frame_start = 1
    scene.frame_end = 6
    scene.frame_set(1)
    collection = bpy.data.collections.new("SSBL_Force_LiveCollection_Filter")
    scene.collection.children.link(collection)
    cloth = _make_cloth("SSBL_Force_LiveCollection")
    cloth.ssbl_cloth.force_field_collection = collection
    wind = _add_wind("SSBL_Wind_LiveCollection", 30.0)
    bpy.context.view_layer.objects.active = cloth

    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("Live collection force-field case did not start a timeline session")
    try:
        scene.frame_set(2)
        ssbl.solver.step_timeline_preview(bpy.context, scene)
        before = ssbl.solver.session_diagnostics(cloth)

        collection.objects.link(wind)
        scene.frame_set(3)
        ssbl.solver.step_timeline_preview(bpy.context, scene)
        linked = ssbl.solver.session_diagnostics(cloth)

        collection.objects.unlink(wind)
        scene.frame_set(4)
        ssbl.solver.step_timeline_preview(bpy.context, scene)
        unlinked = ssbl.solver.session_diagnostics(cloth)
    finally:
        ssbl.solver.reset_preview_object(cloth)

    return {
        "before_count": int(before.force_field_count),
        "linked_count": int(linked.force_field_count),
        "unlinked_count": int(unlinked.force_field_count),
        "before_finite": bool(before.finite),
        "linked_finite": bool(linked.finite),
        "unlinked_finite": bool(unlinked.finite),
    }


def _run_nonfinite_strength_guard_case(scene: bpy.types.Scene) -> dict[str, object]:
    _clear_scene()
    cloth = _make_cloth("SSBL_Force_NonFiniteStrength")
    field_obj = _add_wind("SSBL_Wind_NonFiniteStrength", 1.0)
    assigned_nonfinite = True
    try:
        field_obj.field.strength = float("nan")
    except Exception:
        assigned_nonfinite = False
        field_obj.field.strength = 0.0
    batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), cloth.ssbl_cloth)
    strength = float(batch.fields[0].strength) if batch.fields else 0.0
    return {
        "assigned_nonfinite": bool(assigned_nonfinite),
        "count": len(batch.fields),
        "strength": strength,
        "finite_strength": math.isfinite(strength),
    }


def _run_force_field_limit_case(scene: bpy.types.Scene) -> dict[str, object]:
    _clear_scene()
    cloth = _make_cloth("SSBL_Force_MaxFields")
    for index in range(MAX_FORCE_FIELDS + 1):
        _add_wind(f"SSBL_Wind_MaxFields_{index:02d}", 1.0)
    try:
        collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), cloth.ssbl_cloth)
    except ValueError as exc:
        return {
            "raised": True,
            "limit": int(MAX_FORCE_FIELDS),
            "message": str(exc),
        }
    return {
        "raised": False,
        "limit": int(MAX_FORCE_FIELDS),
        "message": "",
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        scene = bpy.context.scene

        _clear_scene()
        off_cloth = _make_cloth("SSBL_Force_Off")
        off = _run_preview(scene, off_cloth)

        _clear_scene()
        on_cloth = _make_cloth("SSBL_Force_On")
        _add_wind("SSBL_Wind_On", 30.0)
        on = _run_preview(scene, on_cloth)

        _clear_scene()
        back_cloth = _make_cloth("SSBL_Force_Back")
        _add_wind("SSBL_Wind_Back", 30.0, rotation=(0.0, math.pi, 0.0))
        back = _run_preview(scene, back_cloth)

        _clear_scene()
        weighted_cloth = _make_cloth("SSBL_Force_Weighted")
        weighted_cloth.ssbl_cloth.force_field_weight_wind = 0.25
        _add_wind("SSBL_Wind_Weighted", 30.0)
        weighted = _run_preview(scene, weighted_cloth)
        weighted_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), weighted_cloth.ssbl_cloth)
        weighted_strength = weighted_batch.fields[0].strength if weighted_batch.fields else 0.0

        _clear_scene()
        all_zero_cloth = _make_cloth("SSBL_Force_AllZero")
        all_zero_cloth.ssbl_cloth.force_field_weight_all = 0.0
        _add_wind("SSBL_Wind_AllZero", 30.0)
        all_zero = _run_preview(scene, all_zero_cloth)
        all_zero_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), all_zero_cloth.ssbl_cloth)
        all_zero_strength = all_zero_batch.fields[0].strength if all_zero_batch.fields else 0.0

        _clear_scene()
        gravity_zero_cloth = _make_cloth(
            "SSBL_Gravity_Zero",
            location=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -9.8),
        )
        gravity_zero_cloth.ssbl_cloth.force_field_weight_gravity = 0.0
        gravity_zero = _run_preview(scene, gravity_zero_cloth)

        _clear_scene()
        gravity_on_cloth = _make_cloth(
            "SSBL_Gravity_On",
            location=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -9.8),
        )
        gravity_on = _run_preview(scene, gravity_on_cloth)

        _clear_scene()
        key_cloth = _make_cloth("SSBL_Force_Key")
        key_wind = _add_wind("SSBL_Wind_Key", 1.0)
        scene.frame_set(1)
        key_wind.field.strength = 1.0
        key_wind.field.keyframe_insert("strength", frame=1)
        scene.frame_set(10)
        key_wind.field.strength = 9.0
        key_wind.field.keyframe_insert("strength", frame=10)
        scene.frame_set(5)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        key_batch = collect_force_fields(scene, depsgraph, key_cloth.ssbl_cloth)
        key_strength = key_batch.fields[0].strength if key_batch.fields else 0.0

        _clear_scene()
        transform_cloth = _make_cloth("SSBL_Force_Transform")
        _add_wind("SSBL_Wind_Transform", 1.0, rotation=(0.0, math.pi / 2.0, 0.0))
        transform_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), transform_cloth.ssbl_cloth)
        transform_direction = transform_batch.fields[0].direction if transform_batch.fields else (0.0, 0.0, 0.0)

        _clear_scene()
        collection_cloth = _make_cloth("SSBL_Force_Collection")
        collection_cloth.ssbl_cloth.force_field_weight_all = 0.5
        collection_cloth.ssbl_cloth.force_field_weight_wind = 0.5
        included = _add_wind("SSBL_Wind_Included", 3.0)
        _add_wind("SSBL_Wind_Excluded", 5.0)
        collection = bpy.data.collections.new("SSBL_Force_Collection_Filter")
        scene.collection.children.link(collection)
        collection.objects.link(included)
        collection_cloth.ssbl_cloth.force_field_collection = collection
        collection_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), collection_cloth.ssbl_cloth)

        _clear_scene()
        supported_cloth = _make_cloth("SSBL_Force_Supported")
        supported_types = [
            "FORCE",
            "WIND",
            "VORTEX",
            "TURBULENCE",
            "CHARGE",
            "HARMONIC",
            "LENNARDJ",
            "MAGNET",
            "DRAG",
            "TEXTURE",
        ]
        unsupported_types = ["BOID", "GUIDE", "FLUID_FLOW"]
        created_supported = 0
        created_unsupported = 0
        for field_type in supported_types + unsupported_types:
            try:
                bpy.ops.object.effector_add(type=field_type)
                field_obj = bpy.context.object
                field_obj.name = f"SSBL_Field_{field_type}"
                field_obj.field.strength = 1.0
                if field_type in supported_types:
                    created_supported += 1
                else:
                    created_unsupported += 1
            except Exception:
                pass
        supported_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), supported_cloth.ssbl_cloth)
        visible_weight_order = tuple(visible_force_field_weight_properties())

        _clear_scene()
        scale_cloth = _make_cloth("SSBL_Force_StrengthScale")
        scale_cloth.ssbl_cloth.force_field_strength_scale = 0.5
        _add_wind("SSBL_Wind_StrengthScale", 4.0)
        scale_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), scale_cloth.ssbl_cloth)
        scaled_strength = scale_batch.fields[0].strength if scale_batch.fields else 0.0

        animated_transform = _run_animated_transform_collect_case(scene)
        collection_edges = _run_collection_filter_edge_cases(scene)
        collection_membership = _run_collection_membership_live_case(scene)
        nonfinite_strength_guard = _run_nonfinite_strength_guard_case(scene)
        max_field_guard = _run_force_field_limit_case(scene)

        wind_plain, wind_plain_sample = _run_single_field_case(scene, "WindPlain", "WIND", 30.0)
        wind_noisy, wind_noisy_sample = _run_single_field_case(
            scene,
            "WindNoisy",
            "WIND",
            30.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 1.1, "seed": 23}),
        )
        wind_noise_delta = _signature_delta(wind_plain, wind_noisy)

        force_plain, force_plain_sample = _run_single_field_case(
            scene,
            "ForcePlain",
            "FORCE",
            12.0,
            cloth_location=(1.0, 0.0, 0.0),
        )
        force_max_limited, force_max_limited_sample = _run_single_field_case(
            scene,
            "ForceMaxLimited",
            "FORCE",
            12.0,
            configure=lambda field: _apply_field_properties(field, {"use_max_distance": True, "distance_max": 0.35}),
            cloth_location=(1.0, 0.0, 0.0),
        )
        force_distance_delta = _signature_delta(force_plain, force_max_limited)

        force_radial_limited, force_radial_limited_sample = _run_single_field_case(
            scene,
            "ForceRadialLimited",
            "FORCE",
            12.0,
            configure=lambda field: _apply_field_properties(field, {"use_radial_max": True, "radial_max": 0.35}),
            cloth_location=(1.0, 0.0, 0.0),
        )
        force_radial_delta = _signature_delta(force_plain, force_radial_limited)

        force_3d, force_3d_sample = _run_single_field_case(
            scene,
            "Force3D",
            "FORCE",
            12.0,
            cloth_location=(0.0, 0.0, 1.0),
        )
        force_2d, force_2d_sample = _run_single_field_case(
            scene,
            "Force2D",
            "FORCE",
            12.0,
            configure=lambda field: _apply_field_properties(field, {"use_2d_force": True}),
            cloth_location=(0.0, 0.0, 1.0),
        )
        force_2d_delta = _signature_delta(force_3d, force_2d)

        turbulence_baseline, turbulence_baseline_sample = _run_single_field_case(
            scene,
            "TurbulenceBaseline",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.8, "seed": 19, "size": 0.9, "flow": 0.0}),
        )
        turbulence_tuned, turbulence_tuned_sample = _run_single_field_case(
            scene,
            "TurbulenceTuned",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 1.35, "seed": 19, "size": 0.35, "flow": 2.4}),
        )
        turbulence_parameter_delta = _signature_delta(turbulence_baseline, turbulence_tuned)

        turbulence_flow_off, turbulence_flow_off_sample = _run_single_field_case(
            scene,
            "TurbulenceFlowOff",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.9, "seed": 31, "size": 0.65, "flow": 0.0}),
        )
        turbulence_flow_on, turbulence_flow_on_sample = _run_single_field_case(
            scene,
            "TurbulenceFlowOn",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.9, "seed": 31, "size": 0.65, "flow": 2.5}),
        )
        turbulence_flow_delta = _signature_delta(turbulence_flow_off, turbulence_flow_on)

        turbulence_noise_low, turbulence_noise_low_sample = _run_single_field_case(
            scene,
            "TurbulenceNoiseLow",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.0, "seed": 41, "size": 1.25, "flow": 0.0}),
        )
        turbulence_noise_high, turbulence_noise_high_sample = _run_single_field_case(
            scene,
            "TurbulenceNoiseHigh",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 3.0, "seed": 41, "size": 1.25, "flow": 0.0}),
        )
        turbulence_noise_delta = _signature_delta(turbulence_noise_low, turbulence_noise_high)

        turbulence_local_coords, turbulence_local_coords_sample = _run_single_field_case(
            scene,
            "TurbulenceLocalCoords",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.0, "seed": 43, "size": 0.85, "flow": 0.0}),
            field_location=(0.45, -0.25, 0.15),
        )
        turbulence_global_coords, turbulence_global_coords_sample = _run_single_field_case(
            scene,
            "TurbulenceGlobalCoords",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(
                field,
                {"noise": 0.0, "seed": 43, "size": 0.85, "flow": 0.0, "use_global_coords": True},
            ),
            field_location=(0.45, -0.25, 0.15),
        )
        turbulence_coords_delta = _signature_delta(turbulence_local_coords, turbulence_global_coords)

        turbulence_apply_off, turbulence_apply_off_sample = _run_single_field_case(
            scene,
            "TurbulenceApplyOff",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(
                field,
                {"noise": 0.0, "seed": 47, "size": 0.65, "flow": 0.0, "apply_to_location": False},
            ),
        )
        turbulence_apply_off_delta = _signature_delta(off, turbulence_apply_off)

        turbulence_large_size, turbulence_large_size_sample = _run_single_field_case(
            scene,
            "TurbulenceLargeSize",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.0, "seed": 53, "size": 4.0, "flow": 0.0}),
            field_location=(0.75, -0.35, 0.20),
        )
        turbulence_small_size, turbulence_small_size_sample = _run_single_field_case(
            scene,
            "TurbulenceSmallSize",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.0, "seed": 53, "size": 0.25, "flow": 0.0}),
            field_location=(0.75, -0.35, 0.20),
        )
        turbulence_reference_vector = _blender_turbulence_vector(0.85, (0.30, -0.20, 1.10))

        perf_wind, perf_wind_sample = _run_single_field_case(
            scene,
            "PerfWind",
            "WIND",
            24.0,
            steps=10,
        )
        perf_turbulence, perf_turbulence_sample = _run_single_field_case(
            scene,
            "PerfTurbulence",
            "TURBULENCE",
            24.0,
            configure=lambda field: _apply_field_properties(field, {"noise": 0.0, "seed": 59, "size": 1.0, "flow": 0.0}),
            steps=10,
        )
        turbulence_perf_ratio = float(perf_turbulence["step_ms"]) / max(float(perf_wind["step_ms"]), 1.0e-6)
        turbulence_perf_warning = (
            turbulence_perf_ratio > TURBULENCE_PERF_WARNING_RATIO
            and float(perf_turbulence["step_ms"]) - float(perf_wind["step_ms"]) > 0.05
        )

        drag_low, drag_low_sample = _run_gravity_drag_case(
            scene,
            "DragLow",
            configure_drag=lambda field: _apply_field_properties(field, {"linear_drag": 0.02, "quadratic_drag": 0.0}),
        )
        drag_high, drag_high_sample = _run_gravity_drag_case(
            scene,
            "DragHigh",
            configure_drag=lambda field: _apply_field_properties(field, {"linear_drag": 8.0, "quadratic_drag": 2.0}),
        )
        drag_parameter_delta = _signature_delta(drag_low, drag_high)

        harmonic_short, harmonic_short_sample = _run_single_field_case(
            scene,
            "HarmonicShort",
            "HARMONIC",
            8.0,
            configure=lambda field: _apply_field_properties(field, {"rest_length": 0.0, "harmonic_damping": 0.0}),
            cloth_location=(1.0, 0.0, 0.0),
        )
        harmonic_long, harmonic_long_sample = _run_single_field_case(
            scene,
            "HarmonicLong",
            "HARMONIC",
            8.0,
            configure=lambda field: _apply_field_properties(field, {"rest_length": 2.0, "harmonic_damping": 4.0}),
            cloth_location=(1.0, 0.0, 0.0),
        )
        harmonic_parameter_delta = _signature_delta(harmonic_short, harmonic_long)

        _clear_scene()
        bake_cloth = _make_cloth("SSBL_Force_Bake")
        _add_wind("SSBL_Wind_Bake", 40.0)
        cache_path = ssbl.solver.bake_xpbd_cache(bpy.context, bake_cloth)
        first_sample_z = _read_pc2_average_axis(cache_path, 0, 2)
        last_sample_z = _read_pc2_average_axis(cache_path, 3, 2)
        cache_exists = os.path.exists(cache_path)
        ssbl.solver.clear_xpbd_cache(bake_cloth)

        result = {
            "off": off,
            "on": on,
            "preview_delta_z": on["avg_z"] - off["avg_z"],
            "back": back,
            "back_preview_delta_z": back["avg_z"] - off["avg_z"],
            "weighted": weighted,
            "weighted_strength": weighted_strength,
            "weighted_preview_delta_z": weighted["avg_z"] - off["avg_z"],
            "all_zero": all_zero,
            "all_zero_strength": all_zero_strength,
            "all_zero_preview_delta_z": all_zero["avg_z"] - off["avg_z"],
            "gravity_zero": gravity_zero,
            "gravity_on": gravity_on,
            "gravity_delta_z": gravity_zero["avg_z"] - gravity_on["avg_z"],
            "key_strength_frame_5": key_strength,
            "transform_direction": transform_direction,
            "collection_count": len(collection_batch.fields),
            "collection_strength": collection_batch.fields[0].strength if collection_batch.fields else 0.0,
            "visible_weight_order": visible_weight_order,
            "supported_field_count": len(supported_batch.fields),
            "unsupported_field_count": int(supported_batch.unsupported_count),
            "scaled_strength": scaled_strength,
            "animated_transform": animated_transform,
            "collection_edges": collection_edges,
            "collection_membership": collection_membership,
            "nonfinite_strength_guard": nonfinite_strength_guard,
            "max_field_guard": max_field_guard,
            "wind_noise_delta": wind_noise_delta,
            "wind_plain_sample": _sample_dict(wind_plain_sample),
            "wind_noisy_sample": _sample_dict(wind_noisy_sample),
            "force_distance_delta": force_distance_delta,
            "force_plain_sample": _sample_dict(force_plain_sample),
            "force_max_limited_sample": _sample_dict(force_max_limited_sample),
            "force_radial_delta": force_radial_delta,
            "force_radial_limited_sample": _sample_dict(force_radial_limited_sample),
            "force_2d_delta": force_2d_delta,
            "force_3d_sample": _sample_dict(force_3d_sample),
            "force_2d_sample": _sample_dict(force_2d_sample),
            "turbulence_parameter_delta": turbulence_parameter_delta,
            "turbulence_baseline_sample": _sample_dict(turbulence_baseline_sample),
            "turbulence_tuned_sample": _sample_dict(turbulence_tuned_sample),
            "turbulence_flow_delta": turbulence_flow_delta,
            "turbulence_flow_off_sample": _sample_dict(turbulence_flow_off_sample),
            "turbulence_flow_on_sample": _sample_dict(turbulence_flow_on_sample),
            "turbulence_noise_delta": turbulence_noise_delta,
            "turbulence_noise_low_sample": _sample_dict(turbulence_noise_low_sample),
            "turbulence_noise_high_sample": _sample_dict(turbulence_noise_high_sample),
            "turbulence_coords_delta": turbulence_coords_delta,
            "turbulence_local_coords_sample": _sample_dict(turbulence_local_coords_sample),
            "turbulence_global_coords_sample": _sample_dict(turbulence_global_coords_sample),
            "turbulence_apply_off_delta": turbulence_apply_off_delta,
            "turbulence_apply_off_sample": _sample_dict(turbulence_apply_off_sample),
            "turbulence_large_size_center_displacement": float(turbulence_large_size["center_displacement"]),
            "turbulence_large_size_local_displacement_rms": float(turbulence_large_size["local_displacement_rms"]),
            "turbulence_small_size_center_displacement": float(turbulence_small_size["center_displacement"]),
            "turbulence_small_size_local_displacement_rms": float(turbulence_small_size["local_displacement_rms"]),
            "turbulence_large_size_sample": _sample_dict(turbulence_large_size_sample),
            "turbulence_small_size_sample": _sample_dict(turbulence_small_size_sample),
            "turbulence_reference_vector": turbulence_reference_vector,
            "perf_wind_step_ms": float(perf_wind["step_ms"]),
            "perf_turbulence_step_ms": float(perf_turbulence["step_ms"]),
            "perf_turbulence_ratio": turbulence_perf_ratio,
            "perf_turbulence_warning": bool(turbulence_perf_warning),
            "perf_wind_sample": _sample_dict(perf_wind_sample),
            "perf_turbulence_sample": _sample_dict(perf_turbulence_sample),
            "drag_parameter_delta": drag_parameter_delta,
            "drag_low_sample": _sample_dict(drag_low_sample),
            "drag_high_sample": _sample_dict(drag_high_sample),
            "drag_low_avg_z": drag_low["avg_z"],
            "drag_high_avg_z": drag_high["avg_z"],
            "harmonic_parameter_delta": harmonic_parameter_delta,
            "harmonic_short_sample": _sample_dict(harmonic_short_sample),
            "harmonic_long_sample": _sample_dict(harmonic_long_sample),
            "bake_first_sample_z": first_sample_z,
            "bake_last_sample_z": last_sample_z,
            "bake_cache_exists": cache_exists,
        }
        print("SSBL_FORCE_FIELD_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            off["finite"]
            and on["finite"]
            and on["force_field_count"] == 1
            and on["unsupported_force_field_count"] == 0
            and result["preview_delta_z"] > 0.02
            and back["finite"]
            and back["force_field_count"] == 1
            and abs(result["back_preview_delta_z"]) < 1.0e-4
            and weighted["finite"]
            and weighted["force_field_count"] == 1
            and abs(result["weighted_strength"] - 7.5) < 1.0e-4
            and result["weighted_preview_delta_z"] > 0.005
            and result["weighted_preview_delta_z"] < result["preview_delta_z"]
            and all_zero["finite"]
            and all_zero["force_field_count"] == 1
            and abs(result["all_zero_strength"]) < 1.0e-6
            and abs(result["all_zero_preview_delta_z"]) < 1.0e-4
            and gravity_zero["finite"]
            and gravity_on["finite"]
            and result["gravity_delta_z"] > 0.02
            and 1.0 < key_strength < 9.0
            and transform_direction[0] > 0.85
            and result["collection_count"] == 1
            and abs(result["collection_strength"] - 0.75) < 1.0e-4
            and result["visible_weight_order"] == EXPECTED_VISIBLE_WEIGHT_ORDER
            and result["supported_field_count"] == created_supported
            and result["unsupported_field_count"] == created_unsupported
            and abs(result["scaled_strength"] - 2.0) < 1.0e-4
            and animated_transform["frame_1_count"] == 1
            and animated_transform["frame_10_count"] == 1
            and animated_transform["frame_1_direction"][2] > 0.85
            and animated_transform["frame_10_direction"][0] > 0.85
            and collection_edges["empty_count"] == 0
            and collection_edges["empty_unsupported_count"] == 0
            and collection_edges["nested_count"] == 1
            and abs(collection_edges["nested_strength"] - 6.0) < 1.0e-4
            and collection_edges["nested_unsupported_count"] == 0
            and collection_membership["before_finite"]
            and collection_membership["linked_finite"]
            and collection_membership["unlinked_finite"]
            and collection_membership["before_count"] == 0
            and collection_membership["linked_count"] == 1
            and collection_membership["unlinked_count"] == 0
            and nonfinite_strength_guard["count"] == 1
            and nonfinite_strength_guard["finite_strength"]
            and abs(nonfinite_strength_guard["strength"]) < 1.0e-6
            and max_field_guard["raised"]
            and _has_parameter_effect(wind_plain, wind_noisy)
            and wind_noisy_sample is not None
            and abs(float(wind_noisy_sample.noise) - 1.1) < 1.0e-4
            and int(wind_noisy_sample.seed) == 23
            and _has_parameter_effect(force_plain, force_max_limited)
            and force_max_limited["avg_x"] < force_plain["avg_x"] - 0.005
            and force_max_limited_sample is not None
            and int(force_max_limited_sample.use_max_distance) == 1
            and abs(float(force_max_limited_sample.distance_max) - 0.35) < 1.0e-4
            and _has_parameter_effect(force_plain, force_radial_limited)
            and force_radial_limited["avg_x"] < force_plain["avg_x"] - 0.005
            and force_radial_limited_sample is not None
            and int(force_radial_limited_sample.use_radial_max) == 1
            and abs(float(force_radial_limited_sample.radial_max) - 0.35) < 1.0e-4
            and _has_parameter_effect(force_3d, force_2d)
            and force_3d["avg_z"] > force_2d["avg_z"] + 0.005
            and force_2d_sample is not None
            and int(force_2d_sample.use_2d_force) == 1
            and _has_parameter_effect(turbulence_baseline, turbulence_tuned)
            and turbulence_tuned_sample is not None
            and abs(float(turbulence_tuned_sample.noise) - 1.35) < 1.0e-4
            and abs(float(turbulence_tuned_sample.size) - 0.35) < 1.0e-4
            and abs(float(turbulence_tuned_sample.flow) - 2.4) < 1.0e-4
            and _has_parameter_effect(turbulence_flow_off, turbulence_flow_on)
            and turbulence_flow_on_sample is not None
            and abs(float(turbulence_flow_on_sample.flow) - 2.5) < 1.0e-4
            and turbulence_noise_low["finite"]
            and turbulence_noise_high["finite"]
            and turbulence_noise_delta < NOISE_SPATIAL_EPSILON
            and turbulence_noise_high_sample is not None
            and abs(float(turbulence_noise_high_sample.noise) - 3.0) < 1.0e-4
            and _has_parameter_effect(turbulence_local_coords, turbulence_global_coords)
            and turbulence_global_coords_sample is not None
            and int(turbulence_global_coords_sample.use_global_coords) == 1
            and turbulence_apply_off["finite"]
            and turbulence_apply_off["force_field_count"] == 1
            and turbulence_apply_off_delta < NOISE_SPATIAL_EPSILON
            and turbulence_apply_off_sample is not None
            and int(turbulence_apply_off_sample.apply_to_location) == 0
            and turbulence_large_size["center_displacement"] > 1.0e-5
            and turbulence_small_size["local_displacement_rms"] > turbulence_large_size["local_displacement_rms"] + 1.0e-5
            and all(math.isfinite(float(component)) and -1.0 <= float(component) <= 1.0 for component in turbulence_reference_vector)
            and _has_parameter_effect(drag_low, drag_high)
            and drag_high["avg_z"] > drag_low["avg_z"] + 0.005
            and drag_high_sample is not None
            and abs(float(drag_high_sample.linear_drag) - 8.0) < 1.0e-4
            and abs(float(drag_high_sample.quadratic_drag) - 2.0) < 1.0e-4
            and _has_parameter_effect(harmonic_short, harmonic_long)
            and harmonic_long_sample is not None
            and abs(float(harmonic_long_sample.rest_length) - 2.0) < 1.0e-4
            and abs(float(harmonic_long_sample.harmonic_damping) - 4.0) < 1.0e-4
            and result["bake_cache_exists"]
            and result["bake_last_sample_z"] > result["bake_first_sample_z"] + 0.005
        ):
            raise RuntimeError(f"Force field smoke failed: {result}")
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
