import json
import math
import os
import sys
from pathlib import Path

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


BLEND_PATH = Path(os.environ.get(
    "SSBL_DEMO05_BLEND",
    r"C:\Users\Administrator\Desktop\演示视频\05_tshirt_drape_realtime.blend",
))
SAMPLE_FRAMES = (60, 90, 110, 125, 145)
MAX_SETTLED_DRIFT = float(os.environ.get("SSBL_DEMO05_MAX_SETTLED_DRIFT", "3.0"))
MAX_BELOW_GROUND = float(os.environ.get("SSBL_DEMO05_MAX_BELOW_GROUND", "0.02"))


def _enabled_cloth_objects(scene):
    return [
        obj
        for obj in scene.objects
        if obj.type == "MESH" and hasattr(obj, "ssbl_cloth") and bool(obj.ssbl_cloth.enabled)
    ]


def _world_positions(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _finite_points(points):
    return all(
        math.isfinite(float(component))
        for point in points
        for component in (point.x, point.y, point.z)
    )


def _ground_limit(settings):
    ground = float(getattr(settings, "ground_height", 0.0))
    margin = max(
        float(getattr(settings, "collision_margin", 0.0)),
        float(getattr(settings, "cloth_thickness", 0.0)),
        0.01,
    )
    return ground, ground + margin * 2.5


def _low_contact_sample(objects):
    low_points = []
    min_z = float("inf")
    max_below_ground = 0.0
    finite = True
    for obj in objects:
        settings = obj.ssbl_cloth
        ground, low_limit = _ground_limit(settings)
        points = _world_positions(obj)
        finite = finite and _finite_points(points)
        for point in points:
            min_z = min(min_z, float(point.z))
            max_below_ground = max(max_below_ground, max(0.0, ground - float(point.z)))
            if float(point.z) <= low_limit:
                low_points.append(point)
    if not low_points:
        return {
            "low_count": 0,
            "low_center": None,
            "min_z": min_z,
            "max_below_ground": max_below_ground,
            "finite": finite,
        }
    inv_count = 1.0 / float(len(low_points))
    center = [
        sum(float(point.x) for point in low_points) * inv_count,
        sum(float(point.y) for point in low_points) * inv_count,
        sum(float(point.z) for point in low_points) * inv_count,
    ]
    return {
        "low_count": len(low_points),
        "low_center": center,
        "min_z": min_z,
        "max_below_ground": max_below_ground,
        "finite": finite,
    }


def _xy_distance(a, b):
    if a is None or b is None:
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def main():
    if not BLEND_PATH.exists():
        raise RuntimeError(f"Missing demo 05 blend: {BLEND_PATH}")

    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_count = int(scene.get("ssbl_demo_frame_count", int(scene.frame_end) - frame_start))
    frame_count = max(frame_count, max(SAMPLE_FRAMES))
    scene.frame_set(frame_start)

    cloth_objects = _enabled_cloth_objects(scene)
    if not cloth_objects:
        raise RuntimeError("Demo 05 has no enabled cloth objects")

    session = ssbl.solver.start_timeline_preview(bpy.context, scene)
    if session is None:
        raise RuntimeError("Demo 05 did not start a timeline preview session")

    samples = {}
    total_friction_corrections = 0
    max_friction_corrections = 0
    finite = True
    max_below_ground = 0.0
    try:
        for frame in range(0, frame_count + 1):
            if frame > 0:
                scene.frame_set(frame_start + frame)
                ssbl.solver.step_timeline_preview(bpy.context, scene)
            diag = session.last_diagnostics
            friction = int(getattr(diag, "external_friction_corrections", 0))
            total_friction_corrections += friction
            max_friction_corrections = max(max_friction_corrections, friction)
            if frame in SAMPLE_FRAMES:
                sample = _low_contact_sample(cloth_objects)
                sample["external_friction_corrections"] = friction
                samples[str(frame)] = sample
                finite = finite and bool(sample["finite"]) and bool(getattr(diag, "finite", True))
                max_below_ground = max(max_below_ground, float(sample["max_below_ground"]))
    finally:
        for obj in cloth_objects:
            ssbl.solver.request_stop(obj)
        ssbl.unregister()

    reference = None
    settled_max_drift = 0.0
    for frame in SAMPLE_FRAMES:
        if frame < 90:
            continue
        sample = samples.get(str(frame), {})
        center = sample.get("low_center")
        if center is None or int(sample.get("low_count", 0)) <= 0:
            continue
        if reference is None:
            reference = center
        drift = _xy_distance(reference, center)
        if drift is not None:
            settled_max_drift = max(settled_max_drift, drift)

    result = {
        "blend_path": str(BLEND_PATH),
        "frame_count": frame_count,
        "samples": samples,
        "total_external_friction_corrections": total_friction_corrections,
        "max_external_friction_corrections": max_friction_corrections,
        "settled_max_xy_drift_from_frame90": settled_max_drift,
        "max_below_ground": max_below_ground,
        "finite": finite,
        "thresholds": {
            "max_settled_drift": MAX_SETTLED_DRIFT,
            "max_below_ground": MAX_BELOW_GROUND,
        },
    }

    failures = []
    if not finite:
        failures.append("non-finite mesh or diagnostics")
    if max_friction_corrections <= 0:
        failures.append("no external friction corrections were reported")
    if settled_max_drift > MAX_SETTLED_DRIFT:
        failures.append(f"settled XY drift {settled_max_drift:.4f} exceeds {MAX_SETTLED_DRIFT:.4f}")
    if max_below_ground > MAX_BELOW_GROUND:
        failures.append(f"below-ground depth {max_below_ground:.4f} exceeds {MAX_BELOW_GROUND:.4f}")
    if failures:
        result["failures"] = failures
        print("SSBL_DEMO05_FRICTION_PROBE", json.dumps(result, ensure_ascii=False))
        raise RuntimeError("; ".join(failures))

    result["passed"] = True
    print("SSBL_DEMO05_FRICTION_PROBE", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
