from __future__ import annotations

import json
import os
import struct
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


CACHE_MODIFIER_NAME = "SSBL XPBD Cache"
CACHE_PATH_PROP = "_ssbl_xpbd_cache_path"


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


def _make_cloth(name: str, location: tuple[float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=11, y_subdivisions=11, size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vertex.index for vertex in obj.data.vertices if vertex.co.y > 0.42], 1.0, "ADD")
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
    settings.preview_writeback_interval = 8
    return obj


def _read_pc2_header(path: str) -> dict[str, object]:
    with open(path, "rb") as handle:
        signature, version, vertex_count, start, sample_rate, sample_count = struct.unpack("<12siiffi", handle.read(32))
    return {
        "signature": signature.decode("ascii", errors="replace").rstrip("\0"),
        "version": int(version),
        "vertex_count": int(vertex_count),
        "start": float(start),
        "sample_rate": float(sample_rate),
        "sample_count": int(sample_count),
        "size": int(os.path.getsize(path)),
    }


def _cache_result(obj: bpy.types.Object, before: list[tuple[float, float, float]], original_mesh, expected_samples: int) -> dict[str, object]:
    modifier = obj.modifiers.get(CACHE_MODIFIER_NAME)
    path = str(obj.get(CACHE_PATH_PROP, ""))
    if not path and modifier is not None:
        path = str(getattr(modifier, "filepath", ""))
    header = _read_pc2_header(path) if path and os.path.exists(path) else {}
    expected_size = 32 + int(expected_samples) * len(obj.data.vertices) * 3 * 4
    return {
        "cache_exists": bool(path and os.path.exists(path)),
        "cache_path": path,
        "header": header,
        "modifier_bound": bool(modifier is not None and modifier.type == "MESH_CACHE"),
        "restored_mesh": bool(obj.data == original_mesh),
        "restore_delta": float(_max_source_delta(obj, before)),
        "size_matches": bool(header.get("size") == expected_size),
    }


def _assert_cache_result(label: str, result: dict[str, object], expected_samples: int) -> None:
    header = result.get("header", {})
    if not (
        result["cache_exists"]
        and result["modifier_bound"]
        and result["restored_mesh"]
        and result["restore_delta"] == 0.0
        and result["size_matches"]
        and header.get("signature") == "POINTCACHE2"
        and header.get("version") == 1
        and header.get("sample_count") == expected_samples
        and header.get("sample_rate") == 1.0
    ):
        raise RuntimeError(f"{label} realtime auto cache failed: {result}")


def _run_manual(scene: bpy.types.Scene) -> dict[str, object]:
    print("SSBL_REALTIME_AUTO_CACHE_STAGE manual_start", flush=True)
    obj = _make_cloth("SSBL_AutoCache_Manual", (0.0, 0.0, 1.0))
    bpy.context.view_layer.objects.active = obj
    original_mesh = obj.data
    before = _snapshot(obj)
    scene.frame_start = 1
    scene.frame_end = 12
    scene.frame_set(1)
    ssbl.solver.start_preview(bpy.context, obj)
    for _index in range(3):
        ssbl.solver.step_preview(bpy.context, obj.name)
    ssbl.solver.request_stop(obj)
    result = _cache_result(obj, before, original_mesh, 4)
    _assert_cache_result("manual", result, 4)
    ssbl.solver.clear_xpbd_cache(obj)
    print("SSBL_REALTIME_AUTO_CACHE_STAGE manual_ok", flush=True)
    return result


def _run_timeline(scene: bpy.types.Scene) -> dict[str, object]:
    print("SSBL_REALTIME_AUTO_CACHE_STAGE timeline_start", flush=True)
    obj = _make_cloth("SSBL_AutoCache_Timeline", (0.0, 0.0, 1.0))
    bpy.context.view_layer.objects.active = obj
    original_mesh = obj.data
    before = _snapshot(obj)
    scene.frame_start = 1
    scene.frame_end = 12
    scene.frame_set(1)
    ssbl.solver.start_timeline_preview(bpy.context, scene)
    for frame in (2, 3, 4):
        scene.frame_current = frame
        ssbl.solver.step_timeline_preview(bpy.context, scene)
    stopped = ssbl.solver.stop_timeline_preview(scene)
    result = _cache_result(obj, before, original_mesh, 4)
    result["stopped"] = bool(stopped)
    _assert_cache_result("timeline", result, 4)
    if not stopped:
        raise RuntimeError(f"Timeline stop did not finalize realtime cache: {result}")
    ssbl.solver.clear_xpbd_cache(obj)
    print("SSBL_REALTIME_AUTO_CACHE_STAGE timeline_ok", flush=True)
    return result


def _run_multi(scene: bpy.types.Scene) -> dict[str, object]:
    print("SSBL_REALTIME_AUTO_CACHE_STAGE multi_start", flush=True)
    first = _make_cloth("SSBL_AutoCache_Multi_A", (-0.55, 0.0, 1.0))
    second = _make_cloth("SSBL_AutoCache_Multi_B", (0.55, 0.0, 1.0))
    first_original = first.data
    second_original = second.data
    first_before = _snapshot(first)
    second_before = _snapshot(second)
    bpy.ops.object.select_all(action="DESELECT")
    first.select_set(True)
    second.select_set(True)
    bpy.context.view_layer.objects.active = first
    scene.frame_start = 1
    scene.frame_end = 12
    scene.frame_set(1)
    session = ssbl.solver.start_preview(bpy.context, first)
    for _index in range(2):
        ssbl.solver.step_preview(bpy.context, first.name)
    ssbl.solver.request_stop(first)
    first_result = _cache_result(first, first_before, first_original, 3)
    second_result = _cache_result(second, second_before, second_original, 3)
    _assert_cache_result("multi_first", first_result, 3)
    _assert_cache_result("multi_second", second_result, 3)
    if len(session.slots) != 2:
        raise RuntimeError(f"Expected two multi-cloth slots, got {len(session.slots)}")
    ssbl.solver.clear_xpbd_cache(first)
    ssbl.solver.clear_xpbd_cache(second)
    print("SSBL_REALTIME_AUTO_CACHE_STAGE multi_ok", flush=True)
    return {
        "first": first_result,
        "second": second_result,
        "slots": len(session.slots),
    }


def main() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        _clear_scene()
        scene = bpy.context.scene
        manual = _run_manual(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        timeline = _run_timeline(scene)
        ssbl.solver.cleanup_all_sessions()
        _clear_scene()
        multi = _run_multi(scene)
        result = {
            "manual": manual,
            "timeline": timeline,
            "multi": multi,
        }
        print("SSBL_REALTIME_AUTO_CACHE_SMOKE", json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        try:
            ssbl.unregister()
        except Exception:
            pass


if __name__ == "__main__":
    main()
