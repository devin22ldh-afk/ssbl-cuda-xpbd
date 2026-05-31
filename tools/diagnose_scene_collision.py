import json
import sys
import traceback

import bpy
from mathutils import Vector


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)


def _jsonable(value):
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


def _main():
    import ssbl
    from ssbl.collision import collect_static_triangles
    from ssbl.xpbd_core import build_cloth_data

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    scene = bpy.context.scene
    settings = scene.ssbl_preview
    active = bpy.context.active_object
    selected = list(bpy.context.selected_objects)
    out = {
        "blend_file": bpy.data.filepath,
        "scene": scene.name,
        "frame": int(scene.frame_current),
        "active_object": None,
        "selected": [{"name": obj.name, "type": obj.type} for obj in selected],
        "settings": {},
        "objects": [],
        "collections": [],
        "static_collider_collection": None,
        "static_triangle_collect": None,
        "cloth_build": None,
        "preview_warnings": [],
        "likely_issues": [],
    }

    if active is not None:
        out["active_object"] = {
            "name": active.name,
            "type": active.type,
            "mode": getattr(active, "mode", None),
            "data": getattr(getattr(active, "data", None), "name", None),
        }

    for key in (
        "use_evaluated_mesh",
        "pin_vertex_group",
        "frame_count",
        "substeps",
        "iterations",
        "dt",
        "damping",
        "gravity",
        "collision_margin",
        "use_ground",
        "ground_height",
        "use_wall",
        "wall_origin",
        "wall_normal",
        "use_sphere",
        "sphere_object",
        "static_collider_collection",
        "self_collision",
        "self_collision_mode",
        "use_volume_pressure",
    ):
        value = getattr(settings, key, None)
        if key in {"sphere_object", "static_collider_collection"}:
            out["settings"][key] = None if value is None else getattr(value, "name", str(value))
        else:
            out["settings"][key] = _jsonable(value)

    for obj in scene.objects:
        item = {
            "name": obj.name,
            "type": obj.type,
            "data": getattr(getattr(obj, "data", None), "name", None),
            "selected": obj.select_get(),
            "hide_viewport": bool(obj.hide_viewport),
            "visible": bool(obj.visible_get(view_layer=bpy.context.view_layer)),
            "users_collection": [collection.name for collection in obj.users_collection],
        }
        if obj.type == "MESH":
            item.update(
                {
                    "verts": len(obj.data.vertices),
                    "polys": len(obj.data.polygons),
                    "bbox_world": _bbox_world(obj),
                }
            )
            for prop_name in ("ssbl_type", "ssbl_kind", "ppf_type", "simulation_type"):
                if prop_name in obj:
                    item[prop_name] = obj.get(prop_name)
        out["objects"].append(item)

    for collection in bpy.data.collections:
        out["collections"].append(
            {
                "name": collection.name,
                "objects": [obj.name for obj in collection.objects],
                "children": [child.name for child in collection.children],
                "all_objects_recursive": [obj.name for obj in collection.all_objects],
            }
        )

    collider_collection = getattr(settings, "static_collider_collection", None)
    if collider_collection is None:
        out["likely_issues"].append("Static Collider Collection is not assigned.")
    else:
        direct_meshes = [obj for obj in collider_collection.objects if obj.type == "MESH"]
        recursive_meshes = [obj for obj in collider_collection.all_objects if obj.type == "MESH"]
        out["static_collider_collection"] = {
            "name": collider_collection.name,
            "direct_objects": [obj.name for obj in collider_collection.objects],
            "recursive_objects": [obj.name for obj in collider_collection.all_objects],
            "child_collections": [child.name for child in collider_collection.children],
            "direct_meshes": [obj.name for obj in direct_meshes],
            "recursive_meshes": [obj.name for obj in recursive_meshes],
        }
        if not direct_meshes and recursive_meshes:
            out["likely_issues"].append(
                "Collider meshes are only in child collections. Current SSBL reads direct collection.objects, not nested all_objects."
            )
        if not direct_meshes and not recursive_meshes:
            out["likely_issues"].append("Static Collider Collection contains no mesh objects.")
        if active is not None and active in direct_meshes:
            out["likely_issues"].append(
                "The active simulated object is also inside Static Collider Collection, so it is excluded from static collision."
            )
        if active is not None:
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
                triangles, signature = collect_static_triangles(
                    collider_collection,
                    active,
                    depsgraph=depsgraph,
                    use_evaluated_mesh=bool(getattr(settings, "use_evaluated_mesh", True)),
                )
                out["static_triangle_collect"] = {
                    "triangle_count": int(len(triangles)),
                    "signature": [list(row) for row in signature],
                }
                if len(triangles) == 0:
                    out["likely_issues"].append("SSBL collected 0 static collider triangles for the active object.")
            except Exception as exc:
                out["static_triangle_collect"] = {"error": str(exc), "traceback": traceback.format_exc()}
                out["likely_issues"].append("Static collider triangle collection raised an error.")

    if active is None:
        out["likely_issues"].append("No active object selected as cloth target.")
    elif active.type != "MESH":
        out["likely_issues"].append("Active object is not a mesh cloth target.")
    else:
        try:
            out["preview_warnings"] = ssbl.solver.preview_warnings(active, settings)
        except Exception as exc:
            out["preview_warnings_error"] = str(exc)
        try:
            cloth = build_cloth_data(active, settings, depsgraph=bpy.context.evaluated_depsgraph_get())
            out["cloth_build"] = {
                "verts": int(len(cloth.positions_world)),
                "triangles": int(len(cloth.triangles)),
                "pins": int(len(cloth.pin_indices)),
                "inv_mass_zero_count": int((cloth.inv_mass == 0).sum()),
                "rest_volume": float(cloth.rest_volume),
            }
            if len(cloth.pin_indices) == len(cloth.positions_world):
                out["likely_issues"].append("All cloth vertices are pinned, so the cloth cannot move into colliders.")
        except Exception as exc:
            out["cloth_build"] = {"error": str(exc), "traceback": traceback.format_exc()}
            out["likely_issues"].append("Cloth build failed for the active object.")

    print("SSBL_SCENE_COLLISION_DIAG " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    _main()
