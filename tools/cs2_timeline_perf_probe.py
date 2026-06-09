from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import bpy


ADDONS_ROOT = os.environ.get(
    "SSBL_ADDONS_ROOT",
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons",
)
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

BLEND_PATH = Path(os.environ.get("SSBL_CS2_BLEND", r"C:\Users\Administrator\Desktop\cs2.blend"))
STEPS = max(int(os.environ.get("SSBL_CS2_STEPS", "60")), 1)
HARDNESS = max(0.0, min(1.0, float(os.environ.get("SSBL_CS2_HARDNESS", "1.0"))))


def _load_perf_module():
    path = Path(__file__).with_name("cs2_multicloth_perf_probe.py")
    spec = importlib.util.spec_from_file_location("ssbl_cs2_perf_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load CS2 perf helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not BLEND_PATH.exists():
        raise RuntimeError(f"missing cs2 blend: {BLEND_PATH}")
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH), load_ui=False)
    perf_helpers = _load_perf_module()
    ssbl = perf_helpers._register_addon()
    cs2 = perf_helpers._load_cs2_regression_module()

    active = perf_helpers._mesh_object(os.environ.get("SSBL_CS2_ACTIVE_OBJECT", "Cube"))
    other = perf_helpers._mesh_object(os.environ.get("SSBL_CS2_OTHER_OBJECT", "Suzanne"))
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(int(scene.frame_end), STEPS + 2)
    scene.frame_set(1)

    for obj in (active, other):
        obj.hide_viewport = False
        perf_helpers._configure_cloth(obj, cs2, STEPS, HARDNESS)
    perf_helpers._select_multicloth(active, other)

    before_active = perf_helpers._snapshot(active)
    before_other = perf_helpers._snapshot(other)
    started = time.perf_counter()
    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("timeline preview did not start")

    step_times: list[float] = []
    max_input_refresh_ms = 0.0
    max_cuda_step_ms = 0.0
    max_dynamic_upload_ms = 0.0
    max_dynamic_triangle_count = 0
    max_dynamic_particle_count = 0
    max_candidates = 0
    max_resolved_contacts = 0
    finite = True

    try:
        for index in range(STEPS):
            scene.frame_set(1 + index)
            step_started = time.perf_counter()
            ssbl.solver.step_timeline_preview(bpy.context, scene)
            step_times.append((time.perf_counter() - step_started) * 1000.0)
            diag = ssbl.solver.session_diagnostics(active)
            finite = finite and bool(getattr(diag, "finite", True))
            max_input_refresh_ms = max(max_input_refresh_ms, float(getattr(diag, "input_refresh_ms", 0.0)))
            max_cuda_step_ms = max(max_cuda_step_ms, float(getattr(diag, "cuda_step_call_ms", 0.0)))
            max_dynamic_upload_ms = max(max_dynamic_upload_ms, float(getattr(diag, "dynamic_upload_ms", 0.0)))
            max_dynamic_triangle_count = max(max_dynamic_triangle_count, int(getattr(diag, "dynamic_triangle_count", 0)))
            max_dynamic_particle_count = max(max_dynamic_particle_count, int(getattr(diag, "dynamic_particle_count", 0)))
            max_candidates = max(max_candidates, int(getattr(diag, "dynamic_triangle_candidate_count", 0)))
            max_resolved_contacts = max(max_resolved_contacts, int(getattr(diag, "resolved_contacts", 0)))
    finally:
        ssbl.solver.request_stop(active)

    elapsed = time.perf_counter() - started
    after_active = perf_helpers._snapshot(active)
    after_other = perf_helpers._snapshot(other)
    average_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    summary = {
        "blend_file": bpy.data.filepath,
        "steps": int(STEPS),
        "elapsed_s": float(elapsed),
        "average_wall_step_ms": float(average_step_ms),
        "sim_fps_wall": float(1000.0 / average_step_ms) if average_step_ms > 0.0 else 0.0,
        "p95_wall_step_ms": perf_helpers._p95(step_times),
        "slot_names": list(session.slots.keys()),
        "solve_order": list(session.solve_order),
        "cross_mode": str(session.cross_cloth_mode),
        "finite": bool(finite and perf_helpers._finite_mesh(active) and perf_helpers._finite_mesh(other)),
        "restore_delta_active": float(perf_helpers._max_delta(before_active, after_active)),
        "restore_delta_other": float(perf_helpers._max_delta(before_other, after_other)),
        "max_input_refresh_ms": float(max_input_refresh_ms),
        "max_cuda_step_call_ms": float(max_cuda_step_ms),
        "max_dynamic_upload_ms": float(max_dynamic_upload_ms),
        "max_dynamic_triangle_count": int(max_dynamic_triangle_count),
        "max_dynamic_particle_count": int(max_dynamic_particle_count),
        "max_dynamic_triangle_candidate_count": int(max_candidates),
        "max_resolved_contacts": int(max_resolved_contacts),
    }
    print("SSBL_CS2_TIMELINE_PERF_PROBE", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
