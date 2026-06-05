import json
import os
import struct
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _all_finite(obj):
    return all(
        float(component) == float(component) and abs(float(component)) != float("inf")
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _make_cloth():
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=15, y_subdivisions=15, size=1.2, location=(0.0, 0.0, 1.2))
    obj = bpy.context.object
    obj.name = "SSBL_Animated_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    indices = [vert.index for vert in obj.data.vertices if vert.co.y > 0.55]
    group.add(indices, 1.0, "ADD")

    obj.shape_key_add(name="Basis", from_mix=False)
    key = obj.shape_key_add(name="PinDrive", from_mix=False)
    for index, point in enumerate(key.data):
        if obj.data.vertices[index].co.y > 0.55:
            point.co.x += 0.2
            point.co.z += 0.1
    key.value = 0.0
    key.keyframe_insert(data_path="value", frame=1)
    key.value = 1.0
    key.keyframe_insert(data_path="value", frame=10)
    return obj


def _make_sphere():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.22, location=(-0.45, 0.0, 0.7))
    obj = bpy.context.object
    obj.name = "SSBL_Animated_Sphere"
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location.x = 0.45
    obj.location.z = 0.9
    obj.keyframe_insert(data_path="location", frame=10)
    return obj


def _make_static_collider():
    collection = bpy.data.collections.new("SSBL_Animated_Colliders")
    bpy.context.scene.collection.children.link(collection)
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.15))
    obj = bpy.context.object
    obj.name = "SSBL_Animated_Cube"
    obj.scale = (0.25, 0.25, 0.1)
    for parent_collection in list(obj.users_collection):
        parent_collection.objects.unlink(obj)
    collection.objects.link(obj)
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location.y = 0.3
    obj.keyframe_insert(data_path="location", frame=10)
    return collection


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    _clear_scene()
    cloth = _make_cloth()
    sphere = _make_sphere()
    collider_collection = _make_static_collider()
    bpy.context.view_layer.objects.active = cloth
    cloth.select_set(True)

    settings = bpy.context.scene.ssbl_preview
    settings.use_evaluated_mesh = True
    settings.pin_vertex_group = "ssbl_pin"
    settings.frame_count = 20
    settings.preview_target_fps = 24.0
    settings.substeps = 4
    settings.iterations = 1
    settings.use_ground = False
    settings.use_sphere = True
    settings.sphere_object = sphere
    settings.static_collider_collection = collider_collection
    settings.bake_start = 1
    settings.bake_end = 10

    source_mesh = cloth.data
    source_before = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]

    bpy.context.scene.frame_set(1)
    session = ssbl.solver.start_preview(bpy.context, cloth)
    slot = session.slots[cloth.name]
    pin_pairs = getattr(slot.cloth, "pin_attachment_pairs", [])
    pin_indices = list(slot.cloth.pin_indices)
    pin_attachment_pairs_identity = (
        len(pin_pairs) == len(pin_indices)
        and all(int(pair[0]) == int(pin_indices[index]) and int(pair[1]) == index for index, pair in enumerate(pin_pairs))
    )
    for _index in range(10):
        ssbl.solver.step_preview(bpy.context, cloth.name)
    preview_frame = int(bpy.context.scene.frame_current)
    preview_finite = _all_finite(cloth)
    ssbl.solver.request_stop(cloth)
    restored_frame = int(bpy.context.scene.frame_current)
    source_after = [component for vert in source_mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]
    original_mesh_max_abs_delta = max(
        (abs(float(after) - float(before)) for before, after in zip(source_before, source_after)),
        default=0.0,
    )

    bake_path = ssbl.solver.bake_xpbd_cache(bpy.context, cloth)
    with open(bake_path, "rb") as handle:
        signature, version, vertex_count, start, sample_rate, sample_count = struct.unpack("<12siiffi", handle.read(32))
    baked_frame = int(bpy.context.scene.frame_current)
    cache_exists_before_clear = os.path.exists(bake_path)
    cleared = ssbl.solver.clear_xpbd_cache(cloth)
    cache_exists_after_clear = os.path.exists(bake_path)

    result = {
        "session_object": session.object_name,
        "preview_frame": preview_frame,
        "restored_frame": restored_frame,
        "baked_frame": baked_frame,
        "preview_finite": preview_finite,
        "original_mesh_max_abs_delta": original_mesh_max_abs_delta,
        "pin_count": len(pin_indices),
        "pin_attachment_pair_count": len(pin_pairs),
        "pin_attachment_pairs_identity": bool(pin_attachment_pairs_identity),
        "slot_use_evaluated_mesh": bool(slot.use_evaluated_mesh),
        "pc2_signature": signature.decode("ascii", errors="replace").rstrip("\0"),
        "pc2_version": version,
        "pc2_vertex_count": vertex_count,
        "pc2_start": start,
        "pc2_sample_rate": sample_rate,
        "pc2_sample_count": sample_count,
        "cache_exists_before_clear": cache_exists_before_clear,
        "cache_cleared": cleared,
        "cache_exists_after_clear": cache_exists_after_clear,
    }
    print("SSBL_ANIMATED_INPUTS_SMOKE", json.dumps(result, ensure_ascii=False))
    if not (
        result["preview_finite"]
        and result["pin_count"] > 0
        and result["pin_attachment_pair_count"] == result["pin_count"]
        and result["pin_attachment_pairs_identity"]
        and result["slot_use_evaluated_mesh"]
        and result["pc2_signature"] == "POINTCACHE2"
        and result["cache_exists_before_clear"]
        and result["cache_cleared"]
        and not result["cache_exists_after_clear"]
        and result["original_mesh_max_abs_delta"] <= 1.0e-6
    ):
        raise RuntimeError(f"Animated input smoke failed: {result}")
    ssbl.unregister()


if __name__ == "__main__":
    main()
