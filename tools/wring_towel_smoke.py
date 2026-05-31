import json
import math
import sys

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl
from ssbl.xpbd_core import to_local


LENGTH = 4.0
TOWEL_RADIUS = 0.28
X_SEGMENTS = 96
RADIAL_SEGMENTS = 24
PIN_BAND = 0.55
TWIST_RADIANS = math.radians(540.0)
WRING_FRAME_COUNT = 80


def _clear_scene():
    for existing in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing, do_unlink=True)


def _rotate_yz(y, z, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return y * cos_a - z * sin_a, y * sin_a + z * cos_a


def _make_towel():
    vertices = []
    faces = []
    for ix in range(X_SEGMENTS + 1):
        x = -LENGTH * 0.5 + LENGTH * ix / X_SEGMENTS
        for ir in range(RADIAL_SEGMENTS):
            angle = math.tau * ir / RADIAL_SEGMENTS
            y = math.cos(angle) * TOWEL_RADIUS
            z = 1.25 + math.sin(angle) * TOWEL_RADIUS
            vertices.append((x, y, z))

    row = RADIAL_SEGMENTS
    for ix in range(X_SEGMENTS):
        for ir in range(RADIAL_SEGMENTS):
            a = ix * row + ir
            b = ix * row + ((ir + 1) % RADIAL_SEGMENTS)
            c = (ix + 1) * row + ((ir + 1) % RADIAL_SEGMENTS)
            d = (ix + 1) * row + ir
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new("SSBL_Wring_Towel_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("SSBL_Wring_Towel", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    pin = obj.vertex_groups.new(name="ssbl_pin")
    pin_indices = [vert.index for vert in obj.data.vertices if abs(vert.co.x) >= LENGTH * 0.5 - PIN_BAND]
    pin.add(pin_indices, 1.0, "ADD")

    return obj


def _all_finite(obj):
    return all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def _max_abs_delta(before, after):
    return max((abs(float(a) - float(b)) for a, b in zip(after, before)), default=0.0)


def _snapshot_coords(mesh):
    return [component for vert in mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]


def _wring_pin_targets(cloth, progress):
    progress = max(0.0, min(1.0, float(progress)))
    rest = np.asarray(cloth.positions_world[cloth.pin_indices], dtype=np.float32)
    targets = np.array(rest, dtype=np.float32, copy=True)
    left_limit = -LENGTH * 0.5 + PIN_BAND + 1.0e-5
    right_limit = LENGTH * 0.5 - PIN_BAND - 1.0e-5
    for index, point in enumerate(rest):
        x = float(point[0])
        if x <= left_limit:
            angle = TWIST_RADIANS * progress
        elif x >= right_limit:
            angle = -TWIST_RADIANS * progress
        else:
            continue
        y, z_delta = _rotate_yz(float(point[1]), float(point[2]) - 1.25, angle)
        targets[index, 1] = y
        targets[index, 2] = 1.25 + z_delta
    return np.ascontiguousarray(targets, dtype=np.float32)


def _apply_preview_positions(obj, world_positions, matrix_world_inv):
    local = to_local(np.asarray(world_positions, dtype=np.float32), matrix_world_inv)
    obj.data.vertices.foreach_set("co", np.asarray(local, dtype=np.float32).reshape(-1))
    obj.data.update()


def _step_wring_frame(session, obj, frame, frame_count):
    slot = session.slots[obj.name]
    progress = max(0.0, min(1.0, float(frame) / max(float(frame_count), 1.0)))
    slot.native.update_pin_targets(slot.cloth.pin_indices, _wring_pin_targets(slot.cloth, progress))
    slot.native.step(session.substeps, session.iterations)
    slot.current_positions_world = np.array(slot.native.download_positions(), dtype=np.float32, copy=True)
    _apply_preview_positions(obj, slot.current_positions_world, slot.cloth.matrix_world_inv)
    return slot.current_positions_world


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    _clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = WRING_FRAME_COUNT + 8
    scene.frame_set(1)
    towel = _make_towel()
    source_mesh = towel.data
    source_before = _snapshot_coords(source_mesh)

    settings = scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = False
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.02
    settings.substeps = 16
    settings.iterations = 4
    settings.damping = 0.995
    # Keep one spare preview frame so the 40th measured step is still visible
    # instead of triggering the preview auto-restore path.
    settings.frame_count = WRING_FRAME_COUNT + 1

    session = ssbl.solver.start_preview(bpy.context, towel)
    min_z = float("inf")
    max_radius = 0.0
    finite = True
    completed_steps = 0
    for _ in range(WRING_FRAME_COUNT):
        _step_wring_frame(session, towel, completed_steps + 1, WRING_FRAME_COUNT)
        completed_steps += 1
        finite = finite and _all_finite(towel)
        for vert in towel.data.vertices:
            min_z = min(min_z, float((towel.matrix_world @ vert.co).z))
            max_radius = max(max_radius, math.hypot(float(vert.co.y), float(vert.co.z - 1.25)))

    tethers = len(session.cloth.lra_edges)
    ssbl.solver.request_stop(towel)
    source_after = _snapshot_coords(source_mesh)
    original_mesh_max_abs_delta = _max_abs_delta(source_before, source_after)

    result = {
        "object": towel.name,
        "hardness": float(settings.hardness),
        "use_volume_pressure": bool(settings.use_volume_pressure),
        "self_collision_mode": str(settings.self_collision_mode),
        "vertices": len(towel.data.vertices),
        "polygons": len(towel.data.polygons),
        "triangles": len(session.cloth.triangles),
        "tethers": tethers,
        "completed_steps": completed_steps,
        "finite": finite,
        "min_z": min_z,
        "max_twist_radius": max_radius,
        "original_mesh_max_abs_delta": original_mesh_max_abs_delta,
    }
    print("SSBL_WRING_TOWEL_SMOKE", json.dumps(result, ensure_ascii=False))
    if not finite:
        raise RuntimeError("wring towel smoke produced non-finite vertex coordinates")
    if tethers != 0:
        raise RuntimeError(f"hardness=0 should not create hidden tether constraints, got {tethers}")
    if settings.use_volume_pressure:
        raise RuntimeError("wring towel smoke must keep volume pressure disabled")
    if original_mesh_max_abs_delta > 1.0e-7:
        raise RuntimeError(f"preview did not restore the source mesh: {original_mesh_max_abs_delta}")
    ssbl.unregister()


if __name__ == "__main__":
    main()
