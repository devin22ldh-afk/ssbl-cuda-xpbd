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
from ssbl.force_fields import collect_force_fields


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _make_cloth(name: str) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=9, y_subdivisions=9, size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = name
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.pin_vertex_group = ""
    settings.use_evaluated_mesh = False
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.gravity = (0.0, 0.0, 0.0)
    settings.substeps = 2
    settings.iterations = 1
    settings.preview_writeback_interval = 1
    settings.bake_start = 1
    settings.bake_end = 4
    return obj


def _add_wind(name: str, strength: float, rotation=(0.0, math.pi / 2.0, 0.0)) -> bpy.types.Object:
    bpy.ops.object.effector_add(type="WIND", location=(0.0, 0.0, 0.0), rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.field.strength = strength
    return obj


def _average_x(obj: bpy.types.Object) -> float:
    return sum(float(vertex.co.x) for vertex in obj.data.vertices) / max(len(obj.data.vertices), 1)


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
    avg_x = _average_x(obj)
    finite = all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )
    ssbl.solver.reset_preview_object(obj)
    return {
        "slots": len(session.slots) if session else 0,
        "avg_x": avg_x,
        "finite": bool(finite and diag.finite),
        "force_field_count": int(diag.force_field_count),
        "unsupported_force_field_count": int(diag.unsupported_force_field_count),
    }


def _read_pc2_average_x(path: str, sample_index: int) -> float:
    with open(path, "rb") as handle:
        signature, version, vertex_count, start, sample_rate, sample_count = struct.unpack("<12siiffi", handle.read(32))
        if signature.rstrip(b"\0") != b"POINTCACHE2":
            raise RuntimeError(f"Unexpected PC2 signature: {signature!r}")
        sample_index = max(0, min(int(sample_index), int(sample_count) - 1))
        handle.seek(32 + sample_index * int(vertex_count) * 3 * 4)
        values = struct.unpack("<" + "f" * int(vertex_count) * 3, handle.read(int(vertex_count) * 3 * 4))
    xs = values[0::3]
    return sum(xs) / max(len(xs), 1)


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
        default_force_fields_enabled = bool(off_cloth.ssbl_cloth.use_blender_force_fields)
        off_cloth.ssbl_cloth.use_blender_force_fields = False
        off = _run_preview(scene, off_cloth)

        _clear_scene()
        on_cloth = _make_cloth("SSBL_Force_On")
        on_cloth.ssbl_cloth.use_blender_force_fields = True
        _add_wind("SSBL_Wind_On", 30.0)
        on = _run_preview(scene, on_cloth)

        _clear_scene()
        weighted_cloth = _make_cloth("SSBL_Force_Weighted")
        weighted_cloth.ssbl_cloth.use_blender_force_fields = True
        weighted_wind = _add_wind("SSBL_Wind_Weighted", 30.0)
        weighted_wind.ssbl_force_field_weight = 0.25
        weighted = _run_preview(scene, weighted_cloth)
        weighted_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), weighted_cloth.ssbl_cloth)
        weighted_strength = weighted_batch.fields[0].strength if weighted_batch.fields else 0.0

        _clear_scene()
        key_cloth = _make_cloth("SSBL_Force_Key")
        key_cloth.ssbl_cloth.use_blender_force_fields = True
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
        transform_cloth.ssbl_cloth.use_blender_force_fields = True
        _add_wind("SSBL_Wind_Transform", 1.0, rotation=(0.0, math.pi / 2.0, 0.0))
        transform_batch = collect_force_fields(scene, bpy.context.evaluated_depsgraph_get(), transform_cloth.ssbl_cloth)
        transform_direction = transform_batch.fields[0].direction if transform_batch.fields else (0.0, 0.0, 0.0)

        _clear_scene()
        collection_cloth = _make_cloth("SSBL_Force_Collection")
        collection_cloth.ssbl_cloth.use_blender_force_fields = True
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

        _clear_scene()
        bake_cloth = _make_cloth("SSBL_Force_Bake")
        bake_cloth.ssbl_cloth.use_blender_force_fields = True
        _add_wind("SSBL_Wind_Bake", 40.0)
        cache_path = ssbl.solver.bake_xpbd_cache(bpy.context, bake_cloth)
        first_sample_x = _read_pc2_average_x(cache_path, 0)
        last_sample_x = _read_pc2_average_x(cache_path, 3)
        cache_exists = os.path.exists(cache_path)
        ssbl.solver.clear_xpbd_cache(bake_cloth)

        result = {
            "default_force_fields_enabled": default_force_fields_enabled,
            "off": off,
            "on": on,
            "preview_delta_x": on["avg_x"] - off["avg_x"],
            "weighted": weighted,
            "weighted_strength": weighted_strength,
            "weighted_preview_delta_x": weighted["avg_x"] - off["avg_x"],
            "key_strength_frame_5": key_strength,
            "transform_direction": transform_direction,
            "collection_count": len(collection_batch.fields),
            "collection_strength": collection_batch.fields[0].strength if collection_batch.fields else 0.0,
            "supported_field_count": len(supported_batch.fields),
            "unsupported_field_count": int(supported_batch.unsupported_count),
            "bake_first_sample_x": first_sample_x,
            "bake_last_sample_x": last_sample_x,
            "bake_cache_exists": cache_exists,
        }
        print("SSBL_FORCE_FIELD_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            off["finite"]
            and result["default_force_fields_enabled"]
            and on["finite"]
            and on["force_field_count"] == 1
            and on["unsupported_force_field_count"] == 0
            and result["preview_delta_x"] > 0.02
            and weighted["finite"]
            and weighted["force_field_count"] == 1
            and abs(result["weighted_strength"] - 7.5) < 1.0e-4
            and result["weighted_preview_delta_x"] > 0.005
            and result["weighted_preview_delta_x"] < result["preview_delta_x"]
            and 1.0 < key_strength < 9.0
            and transform_direction[0] > 0.85
            and result["collection_count"] == 1
            and abs(result["collection_strength"] - 3.0) < 1.0e-4
            and result["supported_field_count"] == created_supported
            and result["unsupported_field_count"] == created_unsupported
            and result["bake_cache_exists"]
            and result["bake_last_sample_x"] > result["bake_first_sample_x"] + 0.005
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
