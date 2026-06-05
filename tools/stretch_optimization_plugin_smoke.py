import json
import math
import sys
from pathlib import Path

import bpy


ADDONS_ROOT = str(Path(__file__).resolve().parents[2])
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl.xpbd_core import settings_to_options


def _clear_scene():
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    for existing in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing, do_unlink=True)


def _make_quad(label: str):
    mesh = bpy.data.meshes.new(f"SSBL_{label}_Mesh")
    verts = [
        (-0.6, -0.6, 1.2),
        (0.6, -0.6, 1.2),
        (-0.6, 0.6, 1.2),
        (0.6, 0.6, 1.2),
    ]
    mesh.from_pydata(verts, [], [(0, 1, 3, 2)])
    mesh.update()
    obj = bpy.data.objects.new(f"SSBL_{label}", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _snapshot(obj):
    return [float(c) for vert in obj.data.vertices for c in (vert.co.x, vert.co.y, vert.co.z)]


def _max_abs_delta(before, after):
    return max((abs(a - b) for a, b in zip(before, after)), default=0.0)


def _assert_strength(options, hardness: float):
    expected_enabled = bool(hardness > 0.0)
    expected_strength = float(max(0.0, min(1.0, hardness)))
    if bool(options.stretch_optimization_enabled) != expected_enabled:
        raise RuntimeError(
            f"hardness {hardness:.3f} derived enabled={options.stretch_optimization_enabled}, "
            f"expected {expected_enabled}"
        )
    if abs(float(options.stretch_optimization_strength) - expected_strength) > 1.0e-6:
        raise RuntimeError(
            f"hardness {hardness:.3f} derived strength={options.stretch_optimization_strength:.6f}, "
            f"expected {expected_strength:.6f}"
        )


def _run_case(hardness: float):
    _clear_scene()
    obj = _make_quad(f"StretchOpt_H{int(round(hardness * 100.0))}")
    before = _snapshot(obj)

    settings = bpy.context.scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = ""
    settings.hardness = float(hardness)
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = False
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.substeps = 2
    settings.iterations = 2
    settings.frame_count = 6
    settings.damping = 0.99
    options = settings_to_options(settings, runtime_mode_override="preview")
    _assert_strength(options, hardness)

    session = ssbl.solver.start_preview(bpy.context, obj)
    for _index in range(3):
        ssbl.solver.step_preview(bpy.context, obj.name)

    during = _snapshot(obj)
    finite = all(math.isfinite(component) for component in during)
    preview_delta = _max_abs_delta(before, during)
    diagnostics = session.slots[obj.name].native.cached_diagnostics()
    ssbl.solver.request_stop(obj)
    restored = _snapshot(obj)
    restore_delta = _max_abs_delta(before, restored)

    if not finite:
        raise RuntimeError("preview produced non-finite vertex coordinates")
    if preview_delta <= 0.0:
        raise RuntimeError("preview did not write back vertex movement")
    if restore_delta > 1.0e-7:
        raise RuntimeError(f"preview did not restore source mesh: {restore_delta}")
    if options.stretch_optimization_enabled:
        if int(getattr(diagnostics, "abi41_pcg_csr_nnz", 0)) <= 0:
            raise RuntimeError("PCG stretch optimization produced an empty CSR")
        if int(getattr(diagnostics, "abi41_pcg_texture_ready", 0)) != 1:
            raise RuntimeError("PCG stretch optimization texture was not ready")
        initial = float(getattr(diagnostics, "abi41_pcg_initial_residual", 0.0))
        final = float(getattr(diagnostics, "abi41_pcg_final_residual", 0.0))
        if initial > 0.0 and int(getattr(diagnostics, "abi41_pcg_iterations", 0)) <= 0:
            raise RuntimeError("PCG stretch optimization had residual but did not iterate")
        if initial > 0.0 and not (final < initial):
            raise RuntimeError(
                f"PCG residual did not decrease: initial={initial:.8f} final={final:.8f}"
            )

    return {
        "hardness": float(hardness),
        "enabled": bool(options.stretch_optimization_enabled),
        "strength": float(options.stretch_optimization_strength),
        "finite": finite,
        "preview_delta": preview_delta,
        "restore_delta": restore_delta,
        "step_ms": float(diagnostics.step_ms),
        "pcg_iterations": int(getattr(diagnostics, "abi41_pcg_iterations", 0)),
        "pcg_csr_nnz": int(getattr(diagnostics, "abi41_pcg_csr_nnz", 0)),
        "pcg_texture_ready": int(getattr(diagnostics, "abi41_pcg_texture_ready", 0)),
        "pcg_initial_residual": float(getattr(diagnostics, "abi41_pcg_initial_residual", 0.0)),
        "pcg_final_residual": float(getattr(diagnostics, "abi41_pcg_final_residual", 0.0)),
        "pcg_max_delta": float(getattr(diagnostics, "abi41_pcg_max_delta", 0.0)),
    }


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        default_settings = bpy.context.scene.ssbl_preview
        default_options = settings_to_options(default_settings, runtime_mode_override="preview")
        _assert_strength(default_options, float(default_settings.hardness))
        result = {
            "default_hardness": float(default_settings.hardness),
            "default_enabled": bool(default_options.stretch_optimization_enabled),
            "default_strength": float(default_options.stretch_optimization_strength),
            "cases": [_run_case(0.0), _run_case(0.4), _run_case(1.0)],
        }
        print("SSBL_STRETCH_OPT_PLUGIN_SMOKE", json.dumps(result, ensure_ascii=False))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
