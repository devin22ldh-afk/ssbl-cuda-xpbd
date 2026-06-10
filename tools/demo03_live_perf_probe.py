from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import bpy
import numpy as np


ADDONS_ROOT = os.environ.get(
    "SSBL_ADDONS_ROOT",
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons",
)
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

BLEND_NAME = "03_clothesline_multicloth_realtime.blend"
STEPS = max(int(os.environ.get("SSBL_DEMO03_STEPS", "112")), 1)
EXPECTED_SLOTS = int(os.environ.get("SSBL_DEMO03_EXPECTED_SLOTS", "7"))
EXPECTED_TRIANGLES = int(os.environ.get("SSBL_DEMO03_EXPECTED_TRIANGLES", "40156"))
FAIL_UNDER_FPS = float(os.environ.get("SSBL_DEMO03_FAIL_UNDER_FPS", "15.0"))
FAIL_P95_UNDER_FPS = float(os.environ.get("SSBL_DEMO03_FAIL_P95_UNDER_FPS", "10.0"))
REQUIRE_ABI41 = os.environ.get("SSBL_DEMO03_REQUIRE_ABI41", "1").strip().lower() not in {"0", "false", "no", "off"}
REQUIRE_GLOBAL_SCENE = os.environ.get("SSBL_DEMO03_REQUIRE_GLOBAL_SCENE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _find_blend_path() -> Path:
    override = os.environ.get("SSBL_DEMO03_BLEND", "").strip()
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"missing 03 blend from SSBL_DEMO03_BLEND: {path}")
        return path
    desktop = Path.home() / "Desktop"
    matches = sorted(desktop.rglob(BLEND_NAME), key=lambda item: str(item).casefold())
    if not matches:
        raise RuntimeError(f"could not find {BLEND_NAME} under {desktop}")
    return matches[0]


def _register_addon():
    import ssbl

    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    return ssbl


def _snapshot(obj: bpy.types.Object) -> np.ndarray:
    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    return coords


def _max_delta(before: np.ndarray, obj: bpy.types.Object) -> float:
    if len(before) != len(obj.data.vertices) * 3:
        return float("inf")
    after = _snapshot(obj)
    return float(np.max(np.abs(after - before))) if len(after) else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(len(ordered) * 0.95), 0), len(ordered) - 1)
    return float(ordered[index])


def _finite_mesh(obj: bpy.types.Object) -> bool:
    return all(
        math.isfinite(float(component))
        for vertex in obj.data.vertices
        for component in (vertex.co.x, vertex.co.y, vertex.co.z)
    )


def _enabled_cloth_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    objects = []
    for obj in scene.objects:
        if obj.type != "MESH" or not hasattr(obj, "ssbl_cloth"):
            continue
        if bool(getattr(obj.ssbl_cloth, "enabled", False)):
            objects.append(obj)
    return sorted(objects, key=lambda item: item.name.casefold())


def _simulated_triangle_count(session) -> int:
    total = 0
    for slot in getattr(session, "slots", {}).values():
        cloth = getattr(slot, "cloth", None)
        triangles = getattr(cloth, "triangles", None)
        if triangles is not None:
            total += int(len(triangles))
    return int(total)


def _session_positions_finite(session) -> bool:
    for slot in getattr(session, "slots", {}).values():
        positions = np.asarray(getattr(slot, "current_positions_world", ()), dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or not bool(np.isfinite(positions).all()):
            return False
    return True


def _configure_runtime(scene: bpy.types.Scene, cloth_objects: list[bpy.types.Object]) -> None:
    scene.ssbl_preview.cross_cloth_collision = "all_selected"
    for obj in cloth_objects:
        settings = obj.ssbl_cloth
        if hasattr(settings, "auto_cache_realtime"):
            settings.auto_cache_realtime = False
        if hasattr(settings, "preview_writeback_interval"):
            settings.preview_writeback_interval = 0
        if hasattr(settings, "preview_target_fps"):
            settings.preview_target_fps = 60.0


def main() -> None:
    blend_path = _find_blend_path()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    ssbl = _register_addon()
    from ssbl import native_backend

    scene = bpy.context.scene
    scene.frame_set(int(scene.frame_start))
    cloth_objects = _enabled_cloth_objects(scene)
    if not cloth_objects:
        raise RuntimeError("03 blend has no enabled SSBL cloth objects")
    _configure_runtime(scene, cloth_objects)
    scene.frame_end = max(int(scene.frame_end), int(scene.frame_start) + STEPS + 1)

    for obj in cloth_objects:
        obj.hide_viewport = False
        obj.hide_set(False)

    before = {obj.name: _snapshot(obj) for obj in cloth_objects}
    started_total = time.perf_counter()
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("03 timeline preview did not start")

    step_times: list[float] = []
    max_diag: dict[str, float | int] = {
        "step_ms": 0.0,
        "constraints_ms": 0.0,
        "hash_build_ms": 0.0,
        "volume_ms": 0.0,
        "self_hash_ms": 0.0,
        "self_solve_ms": 0.0,
        "self_probe_ms": 0.0,
        "self_recovery_ms": 0.0,
        "sync_ms": 0.0,
        "diagnostics_fetch_ms": 0.0,
        "abi41_direct_stretch_ms": 0.0,
        "abi41_pcg_solve_ms": 0.0,
        "abi41_pcg_system_ms": 0.0,
        "abi41_pcg_ad_ms": 0.0,
        "abi41_pcg_iterations": 0,
        "self_candidate_count": 0,
        "self_cluster_count": 0,
        "frame_ms": 0.0,
        "frame_set_ms": 0.0,
        "input_refresh_ms": 0.0,
        "cuda_step_call_ms": 0.0,
        "download_ms": 0.0,
        "writeback_ms": 0.0,
        "dynamic_collision_ms": 0.0,
        "dynamic_particle_collision_ms": 0.0,
        "dynamic_upload_ms": 0.0,
        "dynamic_collider_pack_ms": 0.0,
        "dynamic_triangle_count": 0,
        "dynamic_particle_count": 0,
        "dynamic_triangle_candidate_count": 0,
        "dynamic_particle_candidate_count": 0,
        "dynamic_particle_contacts": 0,
        "resolved_contacts": 0,
        "candidate_count": 0,
        "global_dynamic_scene_pack_ms": 0.0,
        "global_dynamic_scene_upload_ms": 0.0,
        "global_dynamic_hash_ms": 0.0,
        "global_dynamic_particle_count": 0,
        "global_dynamic_triangle_count": 0,
        "global_dynamic_hash_overflow": 0,
    }
    diag_samples: dict[str, list[float]] = {key: [] for key in max_diag}
    finite = True
    session_snapshot: dict[str, object] = {}
    try:
        for index in range(STEPS):
            frame = int(scene.frame_start) + index + 1
            step_started = time.perf_counter()
            scene.frame_set(frame)
            ssbl.solver.step_timeline_preview(bpy.context, scene)
            step_times.append((time.perf_counter() - step_started) * 1000.0)
            diag = ssbl.solver.session_diagnostics(bpy.data.objects[session.object_name])
            finite = finite and bool(getattr(diag, "finite", True))
            finite = finite and _session_positions_finite(session)
            for slot_name in session.slots:
                obj = bpy.data.objects.get(slot_name)
                finite = finite and obj is not None and _finite_mesh(obj)
            for key in list(max_diag.keys()):
                value = getattr(diag, key, 0)
                diag_samples[key].append(float(value))
                if isinstance(max_diag[key], float):
                    max_diag[key] = max(float(max_diag[key]), float(value))
                else:
                    max_diag[key] = max(int(max_diag[key]), int(value))
        session_snapshot = {
            "slots": int(len(session.slots)),
            "slot_names": list(session.slots.keys()),
            "solve_order": list(session.solve_order),
            "cross_mode": str(session.cross_cloth_mode),
            "global_scene_enabled": bool(getattr(session, "global_dynamic_scene_enabled", False)),
            "simulated_triangle_count": _simulated_triangle_count(session),
            "slot_details": {
                name: {
                    "vertices": int(len(slot.cloth.positions_world)),
                    "triangles": int(len(slot.cloth.triangles)),
                    "substeps": int(slot.substeps),
                    "iterations": int(slot.iterations),
                    "pin_count": int(len(slot.cloth.pin_indices)),
                    "pin_group": str(getattr(bpy.data.objects[name].ssbl_cloth, "pin_vertex_group", "")),
                    "use_evaluated_mesh": bool(slot.use_evaluated_mesh),
                    "collision_layer": int(getattr(slot, "collision_layer", 0)),
                    "writeback_interval": int(slot.writeback_interval),
                }
                for name, slot in session.slots.items()
            },
        }
    finally:
        stopper = bpy.data.objects.get(getattr(session, "object_name", ""))
        if stopper is not None:
            ssbl.solver.request_stop(stopper)

    elapsed = time.perf_counter() - started_total
    restore_deltas = {
        name: _max_delta(snapshot, bpy.data.objects[name])
        for name, snapshot in before.items()
        if bpy.data.objects.get(name) is not None
    }
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    p95_step_ms = _p95(step_times)
    abi_path = native_backend.dll_path()
    abi_name = Path(abi_path).name
    summary = {
        "abi": "ABI41" if "abi41" in abi_name.lower() else ("ABI40" if "abi40" in abi_name.lower() else abi_name),
        "abi_path": abi_path,
        "blend_file": bpy.data.filepath,
        "steps": int(STEPS),
        "elapsed_s": float(elapsed),
        "average_wall_step_ms": float(average_step_ms),
        "p95_wall_step_ms": float(p95_step_ms),
        "sim_fps_wall": float(1000.0 / average_step_ms) if average_step_ms > 0.0 else 0.0,
        "p95_fps_wall": float(1000.0 / p95_step_ms) if p95_step_ms > 0.0 else 0.0,
        "slots": int(session_snapshot.get("slots", 0)),
        "slot_names": list(session_snapshot.get("slot_names", [])),
        "solve_order": list(session_snapshot.get("solve_order", [])),
        "cross_mode": str(session_snapshot.get("cross_mode", "")),
        "global_scene_enabled": bool(session_snapshot.get("global_scene_enabled", False)),
        "native_supports_global_scene": bool(native_backend.supports_global_dynamic_scene()),
        "simulated_triangle_count": int(session_snapshot.get("simulated_triangle_count", 0)),
        "slot_details": dict(session_snapshot.get("slot_details", {})),
        "finite": bool(finite),
        "restore_delta_max": float(max(restore_deltas.values(), default=0.0)),
        "restore_deltas": restore_deltas,
        "max_diag": max_diag,
        "avg_diag": {
            key: float(sum(values) / len(values)) if values else 0.0
            for key, values in diag_samples.items()
        },
        "p95_diag": {
            key: _p95(values)
            for key, values in diag_samples.items()
        },
        "requirements": {
            "abi41": bool(REQUIRE_ABI41),
            "global_scene": bool(REQUIRE_GLOBAL_SCENE),
            "min_average_fps": float(FAIL_UNDER_FPS),
            "min_p95_fps": float(FAIL_P95_UNDER_FPS),
        },
    }
    summary["accepted"] = bool(
        (not REQUIRE_ABI41 or summary["abi"] == "ABI41")
        and (not REQUIRE_GLOBAL_SCENE or summary["global_scene_enabled"])
        and summary["cross_mode"] == "all_selected"
        and summary["slots"] == EXPECTED_SLOTS
        and summary["simulated_triangle_count"] == EXPECTED_TRIANGLES
        and summary["sim_fps_wall"] >= FAIL_UNDER_FPS
        and summary["p95_fps_wall"] >= FAIL_P95_UNDER_FPS
        and summary["finite"]
        and summary["restore_delta_max"] == 0.0
    )
    print("SSBL_DEMO03_LIVE_PERF_PROBE", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not summary["accepted"]:
        raise RuntimeError(f"03 live perf probe failed: {summary}")
    if os.environ.get("SSBL_DEMO03_FORCE_PROCESS_EXIT", "0").strip().lower() in {"1", "true", "yes", "on"}:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
