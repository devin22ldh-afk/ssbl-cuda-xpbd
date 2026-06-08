from __future__ import annotations

import datetime as _dt
import math
import os
import sys
from pathlib import Path

import bpy


TOOLS_DIR = Path(__file__).resolve().parent
SSBL_ROOT = TOOLS_DIR.parent
ADDONS_ROOT = SSBL_ROOT.parent
if str(ADDONS_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDONS_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import record_realtime_demo_pack as demo
import ssbl


TARGET_DIR = Path(r"C:\Users\Administrator\Desktop\演示视频")
DESKTOP_DEMO01_BLEND = Path(r"C:\Users\Administrator\Desktop\SSBL_01_brand_flag_wind_realtime_edit.blend")
USE_DESKTOP_DEMO01_BLEND = os.environ.get("SSBL_USE_DESKTOP_DEMO01_BLEND", "").strip().lower() in {"1", "true", "yes"}


def _safe_register() -> None:
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()


def _metadata(name: str, title: str, frame_count: int, source: str) -> None:
    scene = bpy.context.scene
    scene["ssbl_demo_name"] = name
    scene["ssbl_demo_title"] = title
    scene["ssbl_demo_frame_count"] = int(frame_count)
    scene["ssbl_demo_exported_from"] = "tools/record_realtime_demo_pack.py"
    scene["ssbl_demo_source"] = source
    scene["ssbl_demo_exported_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    text = bpy.data.texts.get("SSBL_Demo_Export_Info")
    if text is None:
        text = bpy.data.texts.new("SSBL_Demo_Export_Info")
    text.clear()
    text.write(
        "\n".join(
            [
                f"name: {name}",
                f"title: {title}",
                f"frame_count: {frame_count}",
                f"source: {source}",
                "generated_from: tools/record_realtime_demo_pack.py",
                "note: Editable SSBL realtime demo scene. Re-record with Blender 5.0.1 and the SSBL addon enabled.",
            ]
        )
    )


def _assert_exportable_initial_scene(name: str) -> None:
    try:
        ssbl.solver.cleanup_all_sessions()
    except Exception as exc:
        raise RuntimeError(f"{name}: could not restore SSBL preview sessions before export: {exc}") from exc

    active_preview = [
        obj.name
        for obj in bpy.data.objects
        if getattr(obj, "type", None) == "MESH" and ssbl.solver.has_session(obj)
    ]
    preview_mesh_objects = [
        f"{obj.name}:{obj.data.name}"
        for obj in bpy.data.objects
        if getattr(obj, "type", None) == "MESH"
        and getattr(getattr(obj, "data", None), "name", "").endswith("_SSBL_XPBD_Preview")
    ]
    cache_modifiers = [
        f"{obj.name}:{modifier.name}"
        for obj in bpy.data.objects
        if getattr(obj, "type", None) == "MESH"
        for modifier in obj.modifiers
        if modifier.type == "MESH_CACHE" or modifier.name == "SSBL XPBD Cache"
    ]
    for mesh in list(bpy.data.meshes):
        if mesh.name.endswith("_SSBL_XPBD_Preview") and int(mesh.users) == 0:
            bpy.data.meshes.remove(mesh)

    problems = []
    if active_preview:
        problems.append(f"active preview sessions: {', '.join(active_preview)}")
    if preview_mesh_objects:
        problems.append(f"objects using preview meshes: {', '.join(preview_mesh_objects)}")
    if cache_modifiers:
        problems.append(f"cache modifiers: {', '.join(cache_modifiers)}")
    if problems:
        raise RuntimeError(f"{name}: refusing to export solved/baked scene state ({'; '.join(problems)})")


def _finish_scene(name: str, title: str, frame_count: int, source: str, overlay, active=None) -> Path:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    demo._update_overlay(
        overlay,
        metrics_line=demo._format_metrics_line(0, frame_count, 0.0, 0.0),
        note="",
    )
    if active is not None:
        bpy.ops.object.select_all(action="DESELECT")
        if isinstance(active, (list, tuple)):
            for obj in active:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = active[-1]
        else:
            active.select_set(True)
            bpy.context.view_layer.objects.active = active
    _assert_exportable_initial_scene(name)
    scene.frame_set(1)
    _metadata(name, title, frame_count + 1, source)
    path = TARGET_DIR / f"{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
    print(f"SSBL_DEMO_BLEND_DONE {name} {path}")
    return path


def _create_overlay(scene, camera, title):
    return demo._create_overlay(scene, camera, title)


def export_demo01() -> Path:
    name = "01_brand_flag_wind_realtime"
    title = "SSBL realtime flag - live wind control"
    if USE_DESKTOP_DEMO01_BLEND and DESKTOP_DEMO01_BLEND.exists():
        try:
            bpy.ops.wm.open_mainfile(filepath=str(DESKTOP_DEMO01_BLEND), load_ui=False)
        except Exception as exc:
            print(f"SSBL_DEMO_BLEND_EXTERNAL_SKIPPED {DESKTOP_DEMO01_BLEND}: {exc}")
        else:
            _assert_exportable_initial_scene(name)
            bpy.context.scene.frame_set(1)
            _metadata(name, title, 113, f"Desktop blend: {DESKTOP_DEMO01_BLEND}")
            path = TARGET_DIR / f"{name}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
            print(f"SSBL_DEMO_BLEND_DONE {name} {path}")
            return path

    frame_count = 112
    demo._clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = demo._configure_scene_render(
        scene,
        camera_location=(2.9, -3.7, 1.75),
        target=(0.25, 0.0, 0.82),
        ortho_scale=2.8,
    )
    overlay = _create_overlay(scene, camera, title)
    cloth = demo._make_yz_flag("SSBL_Demo_Force_Field_Flag")
    demo._configure_cloth_settings(cloth.ssbl_cloth, frame_count=frame_count, pin_group="ssbl_pin")
    cloth.ssbl_cloth.preview_writeback_interval = 1
    cloth.ssbl_cloth.substeps = 5
    cloth.ssbl_cloth.iterations = 2
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -2.4)
    cloth.ssbl_cloth.hardness = 0.25
    cloth.ssbl_cloth.use_blender_force_fields = True
    bpy.ops.object.effector_add(type="WIND", location=(-1.25, 0.0, 0.9), rotation=(0.0, math.pi / 2.0, 0.0))
    wind = bpy.context.object
    wind.name = "SSBL_Demo_Wind_Field"
    wind.field.strength = 26.0
    bpy.ops.object.effector_add(type="TURBULENCE", location=(-0.55, 0.0, 0.95))
    turbulence = bpy.context.object
    turbulence.name = "SSBL_Demo_Turbulence_Field"
    turbulence.field.strength = 18.0
    turbulence.field.size = 1.35
    turbulence.field.flow = 1.1
    turbulence.field.noise = 1.7
    turbulence.field.seed = 11
    return _finish_scene(name, title, frame_count, "Script fallback scene", overlay, cloth)


def export_demo02() -> Path:
    name = "02_wring_towel_realtime"
    title = "SSBL realtime wring towel - hook driven twist"
    frame_count = demo.WRING_FRAME_COUNT
    demo._clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    towel = demo._make_towel()

    blue = demo._material("SSBL_Demo_Towel_Blue", (0.20, 0.42, 0.95, 1.0))
    stripe = demo._material("SSBL_Demo_Towel_Stripe", (1.0, 0.54, 0.18, 1.0))
    light = demo._material("SSBL_Demo_Towel_Light", (0.75, 0.88, 1.0, 1.0))
    towel.data.materials.append(blue)
    towel.data.materials.append(stripe)
    towel.data.materials.append(light)
    for poly in towel.data.polygons:
        radial_index = int(poly.index) % demo.RADIAL_SEGMENTS
        poly.material_index = 1 if radial_index in (0, 1, 2) else (2 if radial_index % 6 == 0 else 0)
    demo._beautify_cloth(towel)

    camera = demo._configure_scene_render(
        scene,
        camera_location=(3.2, -5.2, 2.35),
        target=(0.0, 0.0, 0.8),
        ortho_scale=4.65,
    )
    overlay = _create_overlay(scene, camera, title)

    settings = scene.ssbl_preview
    settings.use_evaluated_mesh = False
    settings.pin_vertex_group = "ssbl_pin"
    settings.hardness = 0.0
    settings.hardness_initialized = True
    settings.use_volume_pressure = False
    settings.self_collision = True
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 1
    settings.max_self_collision_neighbors = 96
    settings.self_probe_interval = 2
    settings.self_surface_pair_interval = 2
    settings.use_ground = False
    settings.use_wall = False
    settings.use_sphere = False
    settings.static_collider_collection = None
    settings.collision_margin = 0.01
    settings.cloth_thickness = 0.035
    settings.substeps = 20
    settings.iterations = 5
    settings.damping = 0.995
    settings.frame_count = frame_count + 1

    return _finish_scene(name, title, frame_count, "Two-point towel wring inspiration", overlay, towel)


def export_demo03() -> Path:
    name = "03_clothesline_multicloth_realtime"
    title = "SSBL realtime inflated trio - chamber contact"
    frame_count = 112
    demo._clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = demo._configure_scene_render(
        scene,
        camera_location=(3.15, -5.35, 2.85),
        target=(0.0, 0.02, 2.18),
        ortho_scale=4.4,
    )
    overlay = _create_overlay(scene, camera, title)
    chamber_collection = demo._make_pressure_collision_chamber()
    cloths = [
        demo._make_inflated_remeshed_monkey("SSBL_Demo_Inflated_Monkey", (0.91, 0.35, 0.18, 1.0)),
        demo._make_inflated_torus("SSBL_Demo_Inflated_Torus", (0.17, 0.55, 0.95, 1.0)),
        demo._make_inflated_sphere("SSBL_Demo_Inflated_Sphere", (0.97, 0.80, 0.23, 1.0)),
    ]
    pressure_strengths = [0.15, 0.20, 0.18]
    for layer, (obj, pressure_strength) in enumerate(zip(cloths, pressure_strengths)):
        obj.ssbl_collision_layer = layer
        obj.ssbl_enable_cross_cloth_collision = True
        demo._configure_pressure_cloth(
            obj.ssbl_cloth,
            frame_count=frame_count,
            pressure_strength=pressure_strength,
            collider_collection=chamber_collection,
        )
    auto_sphere_blocker = demo._make_preview_auto_sphere_blocker()
    cloths[1].ssbl_cloth.use_sphere = True
    cloths[1].ssbl_cloth.sphere_object = auto_sphere_blocker
    return _finish_scene(
        name,
        title,
        frame_count,
        "Custom pressure chamber with remeshed Suzanne, torus, and sphere",
        overlay,
        cloths,
    )


def export_demo04() -> Path:
    name = "04_tablecloth_pull_collision_realtime"
    title = "SSBL realtime tablecloth pull - rigid edge collision"
    frame_count = 112
    demo._clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = demo._configure_scene_render(
        scene,
        camera_location=(4.9, -4.8, 3.45),
        target=(0.1, 0.0, 0.92),
        ortho_scale=4.35,
    )
    overlay = _create_overlay(scene, camera, title)
    collider_collection, _handle_a, _handle_b = demo._make_table_scene_visual()
    cloth = demo._make_tablecloth(2.42, (0.94, 0.92, 0.86, 1.0))
    demo._configure_collision_settings(cloth.ssbl_cloth, frame_count=frame_count)
    cloth.ssbl_cloth.pin_vertex_group = "ssbl_pin"
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
    cloth.ssbl_cloth.static_collider_collection = collider_collection
    cloth.ssbl_cloth.damping = 0.992
    cloth.ssbl_cloth.collision_margin = 0.028
    cloth.ssbl_cloth.cloth_thickness = 0.03
    return _finish_scene(name, title, frame_count, "Mobile scene inspiration adapted to a two-handle tablecloth pull", overlay, cloth)


def export_demo05() -> Path:
    name = "05_tshirt_drape_realtime"
    title = "SSBL realtime spiral cloth drop - floor pile"
    frame_count = 144
    demo._clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count + 4
    scene.frame_set(1)
    camera = demo._configure_scene_render(
        scene,
        camera_location=(4.20, -5.35, 2.85),
        target=(0.0, -0.04, 1.18),
        ortho_scale=5.05,
    )
    overlay = _create_overlay(scene, camera, title)
    demo._make_spiral_floor_visual()
    cloth = demo._make_spiral_drop_cloth("SSBL_Demo_Tshirt", (0.86, 0.48, 0.62, 1.0))
    demo._configure_collision_settings(cloth.ssbl_cloth, frame_count=frame_count)
    cloth.ssbl_cloth.gravity = (0.0, 0.0, -9.8)
    cloth.ssbl_cloth.pin_vertex_group = "ssbl_pin"
    cloth.ssbl_cloth.use_ground = True
    cloth.ssbl_cloth.ground_height = 0.0
    cloth.ssbl_cloth.self_collision = True
    cloth.ssbl_cloth.self_collision_mode = "fast"
    cloth.ssbl_cloth.self_collision_interval = 1
    cloth.ssbl_cloth.max_self_collision_neighbors = 96
    cloth.ssbl_cloth.fast_self_collision_passes = 6
    cloth.ssbl_cloth.static_collider_collection = None
    cloth.ssbl_cloth.damping = 0.992
    cloth.ssbl_cloth.hardness = 0.34
    cloth.ssbl_cloth.hardness_initialized = True
    cloth.ssbl_cloth.collision_margin = 0.032
    cloth.ssbl_cloth.cloth_thickness = 0.032
    cloth.ssbl_cloth.substeps = 16
    cloth.ssbl_cloth.iterations = 4
    cloth.ssbl_cloth.contact_friction = 0.94
    cloth.ssbl_cloth.contact_tangent_damping = 0.72
    return _finish_scene(name, title, frame_count, "Custom spiral-guided cloth drop onto an SSBL ground plane", overlay, cloth)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    _safe_register()
    try:
        paths = [
            export_demo01(),
            export_demo02(),
            export_demo03(),
            export_demo04(),
            export_demo05(),
        ]
        print("SSBL_DEMO_BLEND_PACK_DONE " + " ".join(str(path) for path in paths))
    finally:
        try:
            ssbl.solver.cleanup_all_sessions()
        except Exception:
            pass
        ssbl.unregister()


if __name__ == "__main__":
    main()
