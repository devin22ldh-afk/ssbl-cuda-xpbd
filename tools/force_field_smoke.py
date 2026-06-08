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
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    for frame in range(2, steps + 2):
        scene.frame_current = frame
        ssbl.solver.step_timeline_preview(bpy.context, scene)
    diag = ssbl.solver.session_diagnostics(obj)
    avg_x = _average_axis(obj, "x")
    avg_y = _average_axis(obj, "y")
    avg_z = _average_axis(obj, "z")
    signature = _shape_signature(obj)
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
        "finite": bool(finite and diag.finite),
        "force_field_count": int(diag.force_field_count),
        "unsupported_force_field_count": int(diag.unsupported_force_field_count),
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
