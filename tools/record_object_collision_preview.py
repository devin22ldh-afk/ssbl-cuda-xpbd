import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


FRAME_COUNT = 48
MARGIN = 0.035
TOLERANCE = 0.015
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "recordings" / "object_collision_preview"
FRAMES_DIR = OUTPUT_DIR / "frames"


def _clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def _material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def _make_grid(name, center, size=0.68, segments=14, material=None):
    verts = []
    faces = []
    half = size * 0.5
    for y in range(segments + 1):
        fy = -half + size * (y / segments)
        for x in range(segments + 1):
            fx = -half + size * (x / segments)
            verts.append((fx, fy, 0.0))
    stride = segments + 1
    for y in range(segments):
        for x in range(segments):
            a = y * stride + x
            faces.append((a, a + 1, a + stride + 1, a + stride))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.show_wire = True
    if material is not None:
        obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_xy_plane(name, center, size, material=None):
    half = size * 0.5
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(
        [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    if material is not None:
        obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_xz_plane(name, center, width, height, material=None):
    hw = width * 0.5
    hh = height * 0.5
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(
        [(-hw, 0.0, -hh), (hw, 0.0, -hh), (hw, 0.0, hh), (-hw, 0.0, hh)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    if material is not None:
        obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _make_static_collection(name, collider):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    bpy.context.scene.collection.objects.unlink(collider)
    collection.objects.link(collider)
    return collection


def _make_monkey(name, size, location, material=None):
    if not hasattr(bpy.ops.mesh, "primitive_monkey_add"):
        raise RuntimeError("This Blender build does not expose primitive_monkey_add")
    bpy.ops.mesh.primitive_monkey_add(size=size, location=location)
    monkey = bpy.context.object
    monkey.name = name
    monkey.data.name = f"{name}_Mesh"
    if material is not None:
        monkey.data.materials.append(material)
    return monkey


def _object_bbox_world(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def _add_label(text, location, material, camera):
    font_curve = bpy.data.curves.new(f"{text}_Curve", "FONT")
    font_curve.body = text
    font_curve.align_x = "CENTER"
    font_curve.align_y = "CENTER"
    font_curve.size = 0.12
    obj = bpy.data.objects.new(f"{text}_Label", font_curve)
    obj.location = location
    direction = camera.location - obj.location
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _setup_render(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = 24
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT + 4
    if scene.world is not None:
        scene.world.color = (0.025, 0.03, 0.035)

    camera_data = bpy.data.cameras.new("SSBL_Object_Collision_Camera")
    camera = bpy.data.objects.new("SSBL_Object_Collision_Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (0.0, -7.0, 4.2)
    _look_at(camera, (0.0, 0.0, 0.38))
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 5.8
    scene.camera = camera

    light_data = bpy.data.lights.new("SSBL_Object_Collision_Key", "AREA")
    light = bpy.data.objects.new("SSBL_Object_Collision_Key", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (0.0, -3.0, 5.0)
    light_data.energy = 500
    light_data.size = 5.0
    return camera


def _world_positions(obj):
    return [obj.matrix_world @ vert.co for vert in obj.data.vertices]


def _flat_mesh_coords(mesh):
    return [float(component) for vert in mesh.vertices for component in (vert.co.x, vert.co.y, vert.co.z)]


def _mesh_bvh_world(obj):
    vertices = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    polygons = [tuple(poly.vertices) for poly in obj.data.polygons]
    return BVHTree.FromPolygons(vertices, polygons)


def _min_distance_to_object_surface(points, collider):
    bvh = _mesh_bvh_world(collider)
    min_distance = float("inf")
    for point in points:
        nearest = bvh.find_nearest(point)
        if nearest is None:
            continue
        _position, _normal, _index, distance = nearest
        min_distance = min(min_distance, float(distance))
    return min_distance


def _finite(obj):
    return all(
        math.isfinite(float(component))
        for vert in obj.data.vertices
        for component in (vert.co.x, vert.co.y, vert.co.z)
    )


def _configure_common(settings):
    settings.use_evaluated_mesh = True
    settings.pin_vertex_group = ""
    settings.frame_count = FRAME_COUNT + 2
    settings.preview_target_fps = 24.0
    settings.dt = 1.0 / 30.0
    settings.substeps = 8
    settings.iterations = 1
    settings.damping = 0.995
    settings.stretch_compliance = 1e-6
    settings.bend_compliance = 1e-4
    settings.self_collision = False
    settings.use_volume_pressure = False
    settings.collision_margin = MARGIN
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.sphere_object = None
    settings.static_collider_collection = None


def _configure_case(settings, case):
    _configure_common(settings)
    settings.gravity = case["gravity"]
    kind = case["kind"]
    if kind == "ground":
        settings.use_ground = True
        settings.ground_height = case["ground_height"]
    elif kind == "wall":
        settings.use_wall = True
        settings.wall_origin = case["wall_origin"]
        settings.wall_normal = case["wall_normal"]
    elif kind in {"static_monkey", "static_mesh"}:
        settings.static_collider_collection = case["collection"]


def _case_metric(case):
    positions = _world_positions(case["cloth"])
    kind = case["kind"]
    if kind == "ground":
        limit = case["ground_height"] + MARGIN
        distance = min(point.z for point in positions) - limit
    elif kind == "wall":
        origin = Vector(case["wall_origin"])
        normal = Vector(case["wall_normal"]).normalized()
        distance = min((point - origin).dot(normal) for point in positions) - MARGIN
    elif kind == "static_monkey":
        distance = _min_distance_to_object_surface(positions, case["monkey"]) - MARGIN
    else:
        limit = case["plane_z"] + MARGIN
        distance = min(point.z for point in positions) - limit
    return {
        "signed_clearance": float(distance),
        "penetration": float(max(0.0, -distance)),
        "finite": bool(_finite(case["cloth"])),
    }


def _build_scene():
    scene = bpy.context.scene
    camera = _setup_render(scene)
    cloth_mat = _material("SSBL Cloth Amber", (1.0, 0.58, 0.16, 1.0))
    ground_mat = _material("SSBL Ground Green", (0.18, 0.58, 0.34, 1.0))
    wall_mat = _material("SSBL Wall Red", (0.82, 0.22, 0.18, 1.0))
    monkey_mat = _material("SSBL Monkey Blue", (0.18, 0.38, 0.9, 1.0))
    static_mat = _material("SSBL Static Teal", (0.08, 0.68, 0.72, 1.0))
    label_mat = _material("SSBL Label White", (0.92, 0.92, 0.88, 1.0))

    cases = []

    ground_center = (-1.7, 2.1, 0.0)
    _make_xy_plane("Ground_Collider_Visual", (ground_center[0], ground_center[1], 0.0), 1.15, ground_mat)
    cases.append(
        {
            "label": "Ground",
            "kind": "ground",
            "cloth": _make_grid("Ground_Cloth", (ground_center[0], ground_center[1], 0.82), material=cloth_mat),
            "gravity": (0.0, 0.0, -9.8),
            "ground_height": 0.0,
        }
    )

    monkey_center = (1.7, 2.1, 0.38)
    monkey = _make_monkey("Static_Monkey_Collider", 0.7, monkey_center, monkey_mat)
    bpy.context.view_layer.update()
    monkey_top_z = max(point.z for point in _object_bbox_world(monkey))
    monkey_collection = _make_static_collection("SSBL_Object_Collision_Static_Monkey_Collection", monkey)
    cases.append(
        {
            "label": "Static Monkey",
            "kind": "static_monkey",
            "cloth": _make_grid(
                "Monkey_Cloth",
                (monkey_center[0], monkey_center[1], monkey_top_z + MARGIN * 0.45),
                material=cloth_mat,
            ),
            "gravity": (0.0, 0.0, -0.2),
            "monkey": monkey,
            "collection": monkey_collection,
        }
    )

    wall_y = -2.1
    _make_xz_plane("Wall_Collider_Visual", (-1.7, wall_y, 0.45), 1.1, 1.1, wall_mat)
    cases.append(
        {
            "label": "Wall",
            "kind": "wall",
            "cloth": _make_grid("Wall_Cloth", (-1.7, wall_y - 0.58, 0.45), material=cloth_mat),
            "gravity": (0.0, 9.8, 0.0),
            "wall_origin": (-1.7, wall_y, 0.0),
            "wall_normal": (0.0, -1.0, 0.0),
        }
    )

    static_center = (1.7, -2.1, 0.0)
    plane_z = 0.32
    static_plane = _make_xy_plane("Static_Mesh_Collider", (static_center[0], static_center[1], plane_z), 1.15, static_mat)
    collection = _make_static_collection("SSBL_Object_Collision_Static_Collection", static_plane)
    cases.append(
        {
            "label": "Static Plane",
            "kind": "static_mesh",
            "cloth": _make_grid("Static_Mesh_Cloth", (static_center[0], static_center[1], plane_z + MARGIN * 0.55), material=cloth_mat),
            "gravity": (0.0, 0.0, -0.2),
            "collection": collection,
            "plane_z": plane_z,
        }
    )

    for case in cases:
        loc = case["cloth"].location.copy()
        _add_label(case["label"], (loc.x, loc.y, 1.42), label_mat, camera)
    return cases


def _prepare_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for path in FRAMES_DIR.glob("*.png"):
        path.unlink()
    for path in OUTPUT_DIR.glob("*.json"):
        path.unlink()


def _render_frame(scene, index):
    scene.render.filepath = str(FRAMES_DIR / f"frame_{index:04d}.png")
    try:
        bpy.ops.render.opengl(write_still=True, view_context=False)
    except RuntimeError:
        bpy.ops.render.render(write_still=True)


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    _prepare_output()
    _clear_scene()

    scene = bpy.context.scene
    scene.frame_set(1)
    cases = _build_scene()
    settings = scene.ssbl_preview

    source_snapshots = {}
    for case in cases:
        cloth = case["cloth"]
        source_snapshots[cloth.name] = {
            "mesh": cloth.data,
            "coords": _flat_mesh_coords(cloth.data),
        }

    diagnostics = []
    _render_frame(scene, 0)

    for case in cases:
        _configure_case(settings, case)
        bpy.context.view_layer.objects.active = case["cloth"]
        case["cloth"].select_set(True)
        case["session"] = ssbl.solver.start_preview(bpy.context, case["cloth"])

    for frame in range(1, FRAME_COUNT + 1):
        frame_metrics = {"frame": frame, "cases": {}}
        for case in cases:
            _configure_case(settings, case)
            ssbl.solver.step_preview(bpy.context, case["cloth"].name)
            metric = _case_metric(case)
            frame_metrics["cases"][case["label"]] = metric
            if not metric["finite"]:
                raise RuntimeError(f"{case['label']} produced non-finite coordinates")
        diagnostics.append(frame_metrics)
        _render_frame(scene, frame)

    for case in cases:
        ssbl.solver.request_stop(case["cloth"])

    restore_metrics = {}
    for case in cases:
        cloth = case["cloth"]
        snapshot = source_snapshots[cloth.name]
        restored_coords = _flat_mesh_coords(snapshot["mesh"])
        max_delta = max(
            (abs(after - before) for before, after in zip(snapshot["coords"], restored_coords)),
            default=0.0,
        )
        restore_metrics[case["label"]] = {
            "restored_mesh": cloth.data.name,
            "source_mesh": snapshot["mesh"].name,
            "original_mesh_max_abs_delta": float(max_delta),
        }
        if max_delta > 1e-7:
            raise RuntimeError(f"{case['label']} polluted source mesh: {max_delta}")

    final_metrics = diagnostics[-1]["cases"] if diagnostics else {}
    for label, metric in final_metrics.items():
        if metric["penetration"] > TOLERANCE:
            raise RuntimeError(f"{label} final penetration {metric['penetration']:.6f} exceeds tolerance")

    summary = {
        "output_dir": str(OUTPUT_DIR),
        "frames_dir": str(FRAMES_DIR),
        "frame_count": FRAME_COUNT + 1,
        "margin": MARGIN,
        "tolerance": TOLERANCE,
        "final_metrics": final_metrics,
        "restore_metrics": restore_metrics,
        "frames": [str(path) for path in sorted(FRAMES_DIR.glob("*.png"))],
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SSBL_OBJECT_COLLISION_RECORDING", str(summary_path))
    print(json.dumps(summary, ensure_ascii=False))
    ssbl.unregister()


if __name__ == "__main__":
    main()
