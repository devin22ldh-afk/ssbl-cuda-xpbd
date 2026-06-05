import json
import sys

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
    obj.name = "SSBL_Preview_Restart_Cloth"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.5], 1.0, "ADD")
    return obj


def _configure() -> None:
    settings = bpy.context.scene.ssbl_preview
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


def _preview_mesh_exists(name: str) -> bool:
    return bpy.data.meshes.get(name) is not None


def _run_once(obj: bpy.types.Object, steps: int) -> tuple[bool, str]:
    session = ssbl.solver.start_preview(bpy.context, obj)
    preview_mesh_name = session.slots[session.object_name].preview_mesh.name
    for _index in range(steps):
        if ssbl.solver.step_preview(bpy.context, obj.name):
            break
    first_stop = ssbl.solver.request_stop(obj)
    second_stop = ssbl.solver.request_stop(obj)
    return first_stop and not second_stop and not _preview_mesh_exists(preview_mesh_name), preview_mesh_name


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        obj = _make_cloth()
        original_mesh = obj.data
        before = _snapshot(obj)
        _configure()

        first_ok, first_preview_mesh = _run_once(obj, 2)
        restored_after_first = obj.data == original_mesh
        second_ok, second_preview_mesh = _run_once(obj, 1)
        restored_after_second = obj.data == original_mesh
        restore_delta = _max_source_delta(obj, before)

        result = {
            "first_ok": bool(first_ok),
            "second_ok": bool(second_ok),
            "restored_after_first": bool(restored_after_first),
            "restored_after_second": bool(restored_after_second),
            "restore_delta": float(restore_delta),
            "first_preview_mesh": first_preview_mesh,
            "second_preview_mesh": second_preview_mesh,
            "status": ssbl.solver.session_status(obj),
        }
        print("SSBL_PREVIEW_RESTART_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
        if not (first_ok and second_ok and restored_after_first and restored_after_second and restore_delta == 0.0):
            raise RuntimeError(f"Preview restart smoke failed: {result}")
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
