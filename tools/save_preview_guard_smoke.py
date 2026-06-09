from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _snapshot(obj: bpy.types.Object) -> list[tuple[float, float, float]]:
    return [(float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)) for vertex in obj.data.vertices]


def _max_source_delta(obj: bpy.types.Object, before: list[tuple[float, float, float]]) -> float:
    if len(obj.data.vertices) != len(before):
        return float("inf")
    return max(
        (
            max(
                abs(float(vertex.co.x) - old[0]),
                abs(float(vertex.co.y) - old[1]),
                abs(float(vertex.co.z) - old[2]),
            )
            for vertex, old in zip(obj.data.vertices, before)
        ),
        default=0.0,
    )


def _make_cloth() -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=17, y_subdivisions=17, size=1.2, location=(0.0, 0.0, 1.0))
    obj = bpy.context.object
    obj.name = "SSBL_Save_Guard_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.5], 1.0, "ADD")
    settings = obj.ssbl_cloth
    settings.enabled = True
    settings.auto_cache_realtime = True
    settings.pin_vertex_group = "ssbl_pin"
    settings.use_evaluated_mesh = True
    settings.use_volume_pressure = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.substeps = 2
    settings.iterations = 1
    settings.frame_count = 12
    settings.preview_writeback_interval = 1
    return obj


def _scene_has_preview_mesh_objects() -> bool:
    return any(
        getattr(obj, "type", None) == "MESH"
        and getattr(getattr(obj, "data", None), "name", "").endswith("_SSBL_XPBD_Preview")
        for obj in bpy.data.objects
    )


def _scene_cache_modifiers() -> list[str]:
    return [
        f"{obj.name}:{modifier.name}:{modifier.type}"
        for obj in bpy.data.objects
        if getattr(obj, "type", None) == "MESH"
        for modifier in obj.modifiers
        if modifier.type == "MESH_CACHE" or modifier.name == "SSBL XPBD Cache"
    ]


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="ssbl_save_guard_"))
    try:
        try:
            ssbl.unregister()
        except Exception:
            pass
        ssbl.register()
        _clear_scene()
        obj = _make_cloth()
        bpy.context.view_layer.objects.active = obj
        original_mesh = obj.data
        before = _snapshot(obj)

        ssbl.solver.start_preview(bpy.context, obj)
        ssbl.solver.step_preview(bpy.context, obj.name)
        preview_active_before_save = bool(ssbl.solver.has_session(obj) and obj.data != original_mesh)

        path = temp_dir / "save_guard.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
        stopped_by_save_guard = not ssbl.solver.has_session(obj)
        restored_after_save = obj.data == original_mesh
        restore_delta = _max_source_delta(obj, before)

        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
        saved_obj = bpy.data.objects.get("SSBL_Save_Guard_Cloth")
        saved_delta = _max_source_delta(saved_obj, before) if saved_obj is not None else float("inf")
        preview_mesh_saved = _scene_has_preview_mesh_objects()
        cache_modifiers = _scene_cache_modifiers()

        result = {
            "cache_modifiers": cache_modifiers,
            "preview_active_before_save": bool(preview_active_before_save),
            "preview_mesh_saved": bool(preview_mesh_saved),
            "restored_after_save": bool(restored_after_save),
            "restore_delta": float(restore_delta),
            "saved_delta": float(saved_delta),
            "saved_exists": bool(path.exists() and path.stat().st_size > 0),
            "stopped_by_save_guard": bool(stopped_by_save_guard),
        }
        print("SSBL_SAVE_PREVIEW_GUARD_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (
            result["preview_active_before_save"]
            and result["stopped_by_save_guard"]
            and result["restored_after_save"]
            and result["restore_delta"] == 0.0
            and result["saved_delta"] == 0.0
            and not result["preview_mesh_saved"]
            and not result["cache_modifiers"]
            and result["saved_exists"]
        ):
            raise RuntimeError(f"Save preview guard smoke failed: {result}")
    finally:
        try:
            ssbl.unregister()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
