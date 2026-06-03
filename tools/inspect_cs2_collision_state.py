from __future__ import annotations

import json
import sys

import bpy
from mathutils import Vector


BLEND_PATH = r"C:\Users\Administrator\Desktop\cs2.blend"
ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)


def _jsonable(value):
    if value is None:
        return None
    if hasattr(value, "name"):
        return value.name
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            return [float(item) for item in value]
        except Exception:
            return str(value)
    return value


def _bbox_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "min": [min(point[axis] for point in corners) for axis in range(3)],
        "max": [max(point[axis] for point in corners) for axis in range(3)],
    }


def _aabb_distance(a, b) -> float:
    gap = []
    for axis in range(3):
        gap.append(max(float(a["min"][axis]) - float(b["max"][axis]), float(b["min"][axis]) - float(a["max"][axis]), 0.0))
    return float(sum(value * value for value in gap) ** 0.5)


def _settings_snapshot(settings) -> dict[str, object]:
    out = {}
    for key in (
        "enabled",
        "use_evaluated_mesh",
        "pin_vertex_group",
        "preview_writeback_interval",
        "preview_target_fps",
        "frame_count",
        "substeps",
        "iterations",
        "dt",
        "damping",
        "gravity",
        "collision_margin",
        "cloth_thickness",
        "use_ground",
        "use_wall",
        "use_sphere",
        "sphere_object",
        "static_collider_collection",
        "self_collision",
        "self_collision_mode",
        "use_volume_pressure",
    ):
        if hasattr(settings, key):
            out[key] = _jsonable(getattr(settings, key))
    return out


def _vertex_group_count(obj: bpy.types.Object, group_name: str) -> int:
    if not group_name:
        return 0
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return 0
    count = 0
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group == group.index and assignment.weight > 0.0:
                count += 1
                break
    return count


def main() -> None:
    bpy.ops.wm.open_mainfile(filepath=BLEND_PATH, load_ui=False)
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    scene = bpy.context.scene
    active = bpy.context.view_layer.objects.active
    selected = list(bpy.context.selected_objects)
    mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    cloth_enabled = [
        obj for obj in mesh_objects
        if hasattr(obj, "ssbl_cloth") and bool(getattr(obj.ssbl_cloth, "enabled", False))
    ]
    objects = []
    for obj in mesh_objects:
        item = {
            "name": obj.name,
            "selected": bool(obj.select_get()),
            "active": bool(active is not None and obj.name == active.name),
            "visible": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
            "hide_viewport": bool(obj.hide_viewport),
            "verts": len(obj.data.vertices),
            "polys": len(obj.data.polygons),
            "bbox_world": _bbox_world(obj),
            "ssbl_cloth": _settings_snapshot(obj.ssbl_cloth) if hasattr(obj, "ssbl_cloth") else None,
            "ssbl_type": obj.get("ssbl_type"),
            "simulation_type": obj.get("simulation_type"),
            "collision_layer": int(getattr(obj, "ssbl_collision_layer", obj.get("ssbl_collision_layer", 1))),
            "legacy_cross_cloth_enabled": bool(getattr(obj, "ssbl_enable_cross_cloth_collision", obj.get("ssbl_enable_cross_cloth_collision", True))),
        }
        if item["ssbl_cloth"]:
            pin_group = str(item["ssbl_cloth"].get("pin_vertex_group") or "")
            item["pin_vertex_count"] = _vertex_group_count(obj, pin_group)
        objects.append(item)

    pairs = []
    for index, a in enumerate(cloth_enabled):
        for b in cloth_enabled[index + 1:]:
            pairs.append(
                {
                    "a": a.name,
                    "b": b.name,
                    "aabb_distance": _aabb_distance(_bbox_world(a), _bbox_world(b)),
                    "selected_pair": bool(a.select_get() and b.select_get()),
                    "active_in_pair": bool(active is not None and active.name in {a.name, b.name}),
                }
            )

    preview_settings = _settings_snapshot(scene.ssbl_preview)
    warnings = []
    if active is None:
        warnings.append("No active object.")
    elif active.type != "MESH":
        warnings.append("Active object is not a mesh.")
    if len(cloth_enabled) < 2:
        warnings.append("Fewer than two scene mesh objects have ssbl_cloth.enabled=True.")
    selected_cloth = [obj for obj in selected if obj in cloth_enabled]
    if active is not None and len(selected_cloth) < 2:
        warnings.append("Manual Start Preview will only create one cloth slot unless active plus another selected object both have ssbl_cloth.enabled=True.")
    sphere_name = preview_settings.get("sphere_object")
    static_collection = preview_settings.get("static_collider_collection")
    if sphere_name:
        warnings.append(f"Scene preview sphere_object is {sphere_name}; that object is collision-only when it is not the active cloth target.")
    if static_collection:
        warnings.append(f"Scene preview static_collider_collection is {static_collection}; its mesh objects are collision-only when not active.")

    out = {
        "blend_file": bpy.data.filepath,
        "scene": scene.name,
        "frame": int(scene.frame_current),
        "active": None if active is None else active.name,
        "selected": [obj.name for obj in selected],
        "preview_settings": preview_settings,
        "cloth_enabled_names": [obj.name for obj in cloth_enabled],
        "selected_cloth_names": [obj.name for obj in selected_cloth],
        "objects": objects,
        "cloth_pairs": pairs,
        "warnings": warnings,
    }
    print("SSBL_CS2_COLLISION_STATE " + json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
