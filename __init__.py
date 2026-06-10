bl_info = {
    "name": "SSBL CUDA XPBD",
    "author": "OpenAI",
    "version": (0, 4, 3),
    "blender": (5, 0, 0),
    "location": "3D View > Sidebar > SSBL",
    "description": "Native CUDA XPBD cloth preview and PC2 baking add-on for Blender",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from . import operators, solver, translation, ui
from .xpbd_core import DEFAULT_HARDNESS, sync_hardness_settings


def _self_collision_mode_name(settings) -> str:
    mode = str(getattr(settings, "self_collision_mode", "fast")).lower()
    return "strict" if mode in {"strict", "quality"} else "fast"


def _apply_fast_self_collision_defaults(settings) -> None:
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.fast_self_collision_passes = 4
    settings.self_probe_interval = 4
    settings.self_surface_pair_interval = 4


def _apply_strict_self_collision_defaults(settings) -> None:
    settings.self_collision_interval = 1
    settings.max_self_collision_neighbors = 64
    settings.self_probe_interval = 1
    settings.self_surface_pair_interval = 1


def _apply_self_collision_mode(settings, _context):
    if not bool(getattr(settings, "self_collision", False)):
        return
    if _self_collision_mode_name(settings) == "strict":
        _apply_strict_self_collision_defaults(settings)
    else:
        _apply_fast_self_collision_defaults(settings)


def _sync_self_collision_runtime_settings(settings) -> None:
    enabled = bool(getattr(settings, "self_collision", False))
    if not enabled:
        return
    settings.self_collision = True
    if _self_collision_mode_name(settings) == "strict":
        _apply_strict_self_collision_defaults(settings)
    else:
        _apply_fast_self_collision_defaults(settings)


def _apply_self_collision_toggle(settings, _context):
    enabled = bool(getattr(settings, "self_collision", False))
    if enabled:
        _apply_self_collision_mode(settings, _context)


def _apply_hardness(settings, _context):
    settings.hardness_initialized = True
    sync_hardness_settings(settings)


def _scene_for_settings(settings, context=None):
    if context is not None:
        scene = getattr(context, "scene", None)
        if scene is not None:
            return scene

    owner = getattr(settings, "id_data", None)
    if owner is not None and isinstance(owner, bpy.types.Scene):
        return owner

    context_scene = getattr(bpy.context, "scene", None)
    if owner is not None and isinstance(owner, bpy.types.Object):
        if context_scene is not None:
            try:
                if context_scene.objects.get(owner.name) is owner:
                    return context_scene
            except Exception:
                pass
        for scene in getattr(bpy.data, "scenes", ()):
            try:
                if scene.objects.get(owner.name) is owner:
                    return scene
            except Exception:
                pass
    return context_scene


def _get_scene_gravity(settings):
    scene = _scene_for_settings(settings)
    if scene is None:
        return (0.0, 0.0, -9.8)
    if hasattr(scene, "use_gravity") and not bool(scene.use_gravity):
        return (0.0, 0.0, 0.0)
    gravity = getattr(scene, "gravity", None)
    if gravity is None:
        return (0.0, 0.0, -9.8)
    return tuple(float(component) for component in gravity)


def _set_scene_gravity(settings, value):
    scene = _scene_for_settings(settings)
    if scene is None or not hasattr(scene, "gravity"):
        return
    scene.gravity = tuple(float(component) for component in value)
    if hasattr(scene, "use_gravity"):
        has_non_zero_component = any(abs(float(component)) > 1.0e-8 for component in value)
        if has_non_zero_component and not bool(scene.use_gravity):
            scene.use_gravity = True


_OBJECT_SETTING_COPY_SKIP = {
    "rna_type",
    "enabled",
    "object_settings_initialized",
    "bake_in_progress",
    "bake_progress_percent",
    "bake_progress_current",
    "bake_progress_total",
}


def _copy_scene_defaults_to_object_settings(settings, context) -> None:
    if context is None or not hasattr(context.scene, "ssbl_preview"):
        return
    source = context.scene.ssbl_preview
    if source is settings:
        return
    for prop in source.bl_rna.properties:
        identifier = prop.identifier
        if identifier in _OBJECT_SETTING_COPY_SKIP or prop.is_readonly:
            continue
        try:
            setattr(settings, identifier, getattr(source, identifier))
        except Exception:
            pass


def _apply_enabled(settings, context):
    owner = getattr(settings, "id_data", None)
    if owner is None or not isinstance(owner, bpy.types.Object):
        return
    if bool(settings.enabled):
        if not bool(getattr(settings, "object_settings_initialized", False)):
            _copy_scene_defaults_to_object_settings(settings, context)
            settings.object_settings_initialized = True
            sync_hardness_settings(settings)
        return
    try:
        solver.reset_preview_object(owner)
    except Exception:
        pass


@persistent
def _initialize_hardness_for_scenes(_dummy=None):
    if not hasattr(bpy.types.Scene, "ssbl_preview"):
        return None
    if not hasattr(bpy.data, "scenes"):
        return None
    for scene in bpy.data.scenes:
        try:
            sync_hardness_settings(scene.ssbl_preview)
            _sync_self_collision_runtime_settings(scene.ssbl_preview)
        except Exception:
            pass
    if hasattr(bpy.types.Object, "ssbl_cloth"):
        for obj in bpy.data.objects:
            try:
                sync_hardness_settings(obj.ssbl_cloth)
                _sync_self_collision_runtime_settings(obj.ssbl_cloth)
            except Exception:
                pass
    return None


@persistent
def _restore_preview_source_before_save(_dummy=None):
    try:
        solver.cleanup_all_sessions()
    except Exception:
        pass


class SSBL_PreviewSettings(PropertyGroup):
    enabled: BoolProperty(
        name="Enable SSBL Cloth Simulation",
        default=False,
        update=_apply_enabled,
        description="Enable SSBL CUDA XPBD cloth preview. Press Space to play the timeline and run the preview.",
    )
    object_settings_initialized: BoolProperty(
        name="SSBL Object Settings Initialized",
        default=False,
        options={"HIDDEN"},
        description="Internal flag: copy scene defaults the first time this object is enabled.",
    )
    runtime_mode: EnumProperty(
        name="Run Mode",
        items=[
            ("preview", "Preview", "Run realtime viewport preview"),
            ("bake", "Bake", "Bake to a PC2 cache"),
        ],
        default="preview",
        options={"HIDDEN"},
        description="Choose realtime preview or PC2 cache baking mode.",
    )
    hardness: FloatProperty(
        name="Bend Stiffness",
        default=DEFAULT_HARDNESS,
        min=0.0,
        max=1.0,
        precision=3,
        update=_apply_hardness,
        description="Overall cloth material hardness. 0 behaves like soft silk, 1 behaves like stiff leather.",
    )
    show_advanced_settings: BoolProperty(
        name="Show Advanced Settings",
        default=False,
        description="Expand to show diagnostics and advanced solver settings.",
    )
    hardness_initialized: BoolProperty(
        name="Hardness Initialized",
        default=False,
        options={"HIDDEN"},
        description="Internal migration flag used to infer hardness from legacy scene settings.",
    )
    frame_count: IntProperty(
        name="Preview Frame Count",
        default=120,
        min=1,
        soft_max=1000,
        options={"HIDDEN"},
        description="Frame count simulated by the preview timer.",
    )
    preview_target_fps: FloatProperty(
        name="Target Preview FPS",
        default=30.0,
        min=1.0,
        soft_max=120.0,
        description="Target playback rate for viewport preview.",
    )
    preview_writeback_interval: IntProperty(
        name="Preview writeback interval",
        default=0,
        min=0,
        soft_max=8,
        options={"HIDDEN"},
        description="0 uses adaptive preview mesh writeback; values >= 1 force a fixed interval. Baking still writes every frame",
    )
    auto_cache_realtime: BoolProperty(
        name="Auto-Cache Real-Time Simulation",
        default=False,
        description="Write a PC2 cache while realtime preview or timeline playback is running",
    )
    use_evaluated_mesh: BoolProperty(
        name="Use Evaluated Mesh",
        default=False,
        description="Read the mesh after Blender modifiers and bindings. Vertex count and topology must stay unchanged.",
    )
    multi_cloth_preview: BoolProperty(
        options={"HIDDEN"},
        name="Multi-Cloth Preview",
        default=False,
        description="Solve multiple selected cloth meshes together when preview starts.",
    )
    cross_cloth_collision: EnumProperty(
        options={"HIDDEN"},
        name="Cross-Cloth Collision",
        items=[
            ("off", "Off", "Do not use other cloth objects as dynamic colliders"),
            ("lower_layers", "Lower Layers", "Only accept cloth objects on lower collision layers as dynamic colliders"),
            ("all_selected", "All Selected", "Accept all other selected cloth objects as dynamic colliders"),
        ],
        default="off",
        description="Choose how dynamic cloth colliders are collected during multi-cloth preview.",
    )
    bake_start: IntProperty(
        name="Start Frame",
        default=1,
        description="First frame written to the PC2 cache.",
    )
    bake_end: IntProperty(
        name="End Frame",
        default=120,
        min=1,
        description="Last frame written to the PC2 cache.",
    )
    bake_in_progress: BoolProperty(
        name="Bake in progress",
        default=False,
        options={"HIDDEN"},
        description="Internal runtime flag used to show SSBL bake progress",
    )
    bake_progress_percent: FloatProperty(
        name="Bake progress",
        default=0.0,
        min=0.0,
        max=100.0,
        precision=0,
        subtype="PERCENTAGE",
        options={"HIDDEN"},
        description="Internal SSBL bake progress percentage",
    )
    bake_progress_current: IntProperty(
        name="Bake progress current",
        default=0,
        min=0,
        options={"HIDDEN"},
        description="Internal SSBL bake progress current sample",
    )
    bake_progress_total: IntProperty(
        name="Bake progress total",
        default=0,
        min=0,
        options={"HIDDEN"},
        description="Internal SSBL bake progress total samples",
    )
    dt: FloatProperty(
        name="Time Step",
        default=0.02,
        min=1e-4,
        soft_max=0.1,
        subtype="TIME",
        description="Simulation time represented by each output frame.",
    )
    substeps: IntProperty(
        name="Substeps",
        default=14,
        min=1,
        soft_max=32,
        description="Number of XPBD substeps per frame.",
    )
    iterations: IntProperty(
        name="Constraint Iterations",
        default=2,
        min=1,
        soft_max=128,
        description="Constraint projection iterations per substep.",
    )
    damping: FloatProperty(
        name="Damping",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Velocity damping applied each substep.",
    )
    density: FloatProperty(
        name="Density",
        default=1.0,
        min=1e-4,
        soft_max=100.0,
        description="Surface density used for cloth vertex mass.",
    )
    stretch_compliance: FloatProperty(
        name="Stretch Compliance",
        default=1e-6,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        description="XPBD stretch compliance. Lower values are stiffer.",
    )
    bend_compliance: FloatProperty(
        name="Bend Compliance",
        default=1e-4,
        min=0.0,
        soft_max=1e-2,
        precision=6,
        description="XPBD bend compliance. Lower values are stiffer.",
    )
    use_lra: BoolProperty(
        name="Legacy Compatibility Toggle",
        default=False,
        options={"HIDDEN"},
        description="Hidden legacy field. Hardness now controls anti-stretch tethers automatically.",
    )
    lra_compliance: FloatProperty(
        name="Legacy Compatibility Compliance",
        default=0.0,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        options={"HIDDEN"},
        description="Hidden legacy field. Hardness now controls anti-stretch tethers automatically.",
    )
    lra_slack: FloatProperty(
        name="Legacy Compatibility Slack",
        default=1.0,
        min=1.0,
        soft_max=2.0,
        precision=3,
        options={"HIDDEN"},
        description="Hidden legacy field. Hardness now controls anti-stretch tethers automatically.",
    )
    use_volume_pressure: BoolProperty(
        name="Inflation / Overpressure",
        default=False,
        description="Apply SSBL-style outward overpressure along cloth surface normals",
    )
    volume_compliance: FloatProperty(
        name="Legacy volume compliance",
        default=1e-6,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        options={"HIDDEN"},
        description="Legacy field kept for old .blend compatibility; ignored by ABI38 overpressure",
    )
    pressure_strength: FloatProperty(
        name="Overpressure strength",
        default=0.02,
        min=0.0,
        soft_max=0.2,
        precision=3,
        description="Density- and mesh-adaptive outward pressure applied along cloth surface normals",
    )
    volume_target_scale: FloatProperty(
        name="Legacy volume target scale",
        default=1.0,
        min=0.05,
        soft_max=2.0,
        precision=3,
        options={"HIDDEN"},
        description="Legacy field kept for old .blend compatibility; ignored by ABI38 overpressure",
    )
    volume_solve_interval: IntProperty(
        name="Legacy volume solve interval",
        default=1,
        min=1,
        soft_max=8,
        options={"HIDDEN"},
        description="Legacy field kept for old .blend compatibility; ignored by ABI38 overpressure",
    )
    gravity: FloatVectorProperty(
        name="Gravity",
        default=(0.0, 0.0, -9.8),
        size=3,
        subtype="ACCELERATION",
        get=_get_scene_gravity,
        set=_set_scene_gravity,
        options={"HIDDEN"},
        description="Legacy proxy to Blender scene gravity; SSBL now reads scene gravity automatically",
    )
    use_blender_force_fields: BoolProperty(
        name="Use Blender Force Fields",
        default=True,
        description="Sample Blender force fields from the scene and apply them as external forces to SSBL cloth.",
    )
    force_field_collection: PointerProperty(
        name="Effector Collection",
        type=bpy.types.Collection,
        description="Read all force fields from the current scene when empty, or only read force fields from this collection.",
    )
    force_field_strength_scale: FloatProperty(
        name="Force Field Strength Scale",
        default=1.0,
        min=0.0,
        soft_max=10.0,
        precision=3,
        description="Global strength multiplier applied when SSBL samples Blender force fields.",
    )
    force_field_weight_gravity: FloatProperty(
        name="Gravity",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale scene gravity for SSBL cloth",
    )
    force_field_weight_all: FloatProperty(
        name="All",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale all non-gravity Blender force fields for SSBL cloth",
    )
    force_field_weight_force: FloatProperty(
        name="Force",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Force effector influence for SSBL cloth",
    )
    force_field_weight_vortex: FloatProperty(
        name="Vortex",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Vortex effector influence for SSBL cloth",
    )
    force_field_weight_magnetic: FloatProperty(
        name="Magnetic",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Magnetic effector influence for SSBL cloth",
    )
    force_field_weight_harmonic: FloatProperty(
        name="Harmonic",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Harmonic effector influence for SSBL cloth",
    )
    force_field_weight_charge: FloatProperty(
        name="Charge",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Charge effector influence for SSBL cloth",
    )
    force_field_weight_lennardjones: FloatProperty(
        name="Lennard-Jones",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Lennard-Jones effector influence for SSBL cloth",
    )
    force_field_weight_wind: FloatProperty(
        name="Wind",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Wind effector influence for SSBL cloth",
    )
    force_field_weight_texture: FloatProperty(
        name="Texture",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Texture effector influence for SSBL cloth",
    )
    force_field_weight_turbulence: FloatProperty(
        name="Turbulence",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Turbulence effector influence for SSBL cloth",
    )
    force_field_weight_drag: FloatProperty(
        name="Drag",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Drag effector influence for SSBL cloth",
    )
    pin_vertex_group: StringProperty(
        name="Pin Vertex Group",
        default="ssbl_pin",
        description="Vertices in this vertex group stay fixed in place.",
    )
    pin_hardness: FloatProperty(
        name="Pin Stiffness",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Global multiplier for pin vertex-group weights. 0 disables pins, 1 keeps the original weights.",
    )
    collision_margin: FloatProperty(
        name="Collision Margin",
        default=0.005,
        min=0.0,
        soft_max=0.2,
        precision=4,
        description="Minimum separation distance preserved during collision projection.",
    )
    self_collision: BoolProperty(
        update=_apply_self_collision_toggle,
        name="Self Collision",
        default=True,
        description="Legacy-compatible toggle. When enabled, it maps to the fast self-collision mode.",
    )
    self_collision_mode: EnumProperty(
        name="Mode",
        items=[
            ("fast", "Fast", "Use the preview-first self-collision fast path"),
            ("strict", "Strict", "Use the full probe/recovery self-collision path with zero-intersection priority"),
        ],
        default="fast",
        update=_apply_self_collision_mode,
        description="Self-collision mode tuned for large cloth meshes.",
    )
    cloth_thickness: FloatProperty(
        name="Cloth Thickness",
        default=0.02,
        min=0.001,
        soft_max=0.1,
        precision=4,
        description="Minimum separation distance between cloth surfaces during self collision. Larger values make cloth appear thicker and reduce overlap.",
    )
    self_collision_distance: FloatProperty(
        name="Self-Collision Distance",
        default=0.0,
        min=0.0,
        soft_max=0.1,
        precision=4,
        description="0 means automatic; positive values override the self-collision contact distance",
    )
    self_collision_interval: IntProperty(
        name="Self-Collision Interval",
        default=1,
        min=1,
        soft_max=8,
        description="Run self collision every N substeps. Fast mode usually works well at 2.",
    )
    max_self_collision_neighbors: IntProperty(
        name="Max Neighbor Checks",
        default=64,
        min=4,
        soft_max=256,
        description="Maximum candidate neighbor count handled per vertex or edge during self collision.",
    )
    fast_self_collision_passes: IntProperty(
        name="Fast Passes",
        default=4,
        min=1,
        max=8,
        soft_max=8,
        description="Native self-collision pass count used only by fast mode. Strict mode is unaffected.",
    )
    self_probe_interval: IntProperty(
        name="Self probe interval",
        default=1,
        min=1,
        soft_max=8,
        description="Run expensive self-collision probe/recovery every N self-collision passes; 1 preserves the original behavior",
    )
    self_surface_pair_interval: IntProperty(
        name="Surface pair interval",
        default=1,
        min=1,
        soft_max=8,
        description="Run sample-sample self-collision every N self-collision passes; 1 preserves the original behavior",
    )
    jitter_stabilizer_enabled: BoolProperty(
        name="Jitter stabilizer",
        default=True,
        options={"HIDDEN"},
        description="Internal fast-preview filter for high-frequency self-collision surface jitter",
    )
    contact_friction: FloatProperty(
        name="Contact friction",
        default=0.35,
        min=0.0,
        max=4.0,
        precision=3,
        options={"HIDDEN"},
        description="Internal static/dynamic triangle contact friction coefficient",
    )
    contact_tangent_damping: FloatProperty(
        name="Contact tangent damping",
        default=0.2,
        min=0.0,
        max=1.0,
        precision=3,
        options={"HIDDEN"},
        description="Internal static/dynamic triangle tangential contact damping",
    )
    contact_compliance: FloatProperty(
        name="Contact compliance",
        default=0.0,
        min=0.0,
        soft_max=1.0e-4,
        precision=6,
        options={"HIDDEN"},
        description="Internal XPBD compliance for static/dynamic triangle contacts",
    )
    static_sdf_voxel_size: FloatProperty(
        name="Static SDF voxel size",
        default=0.0,
        min=0.0,
        soft_max=0.05,
        precision=4,
        description="Static mesh SDF voxel size; 0 uses max(collision margin * 0.5, 0.002)",
    )
    static_sdf_band_voxels: IntProperty(
        name="Static SDF band voxels",
        default=4,
        min=1,
        max=32,
        soft_max=16,
        description="Voxel padding band around static mesh SDF collision surfaces",
    )
    static_sdf_max_resolution: IntProperty(
        name="Static SDF max resolution",
        default=160,
        min=16,
        max=512,
        soft_max=256,
        description="Maximum grid resolution per axis for CUDA static mesh SDF rebuilds",
    )
    use_ground: BoolProperty(
        name="Infinite Ground Plane",
        default=True,
        description="Enable a simple world-space ground plane collider.",
    )
    ground_height: FloatProperty(
        name="Ground Height",
        default=0.0,
        description="World-space Z height of the ground plane.",
    )
    use_wall: BoolProperty(
        name="Wall Plane",
        default=False,
        description="Enable collision against an infinite plane.",
    )
    wall_origin: FloatVectorProperty(
        name="Wall Origin",
        default=(0.0, 0.0, 0.0),
        size=3,
        subtype="XYZ",
        description="A point on the wall plane in world space.",
    )
    wall_normal: FloatVectorProperty(
        name="Wall Normal",
        default=(0.0, 0.0, 1.0),
        size=3,
        subtype="DIRECTION",
        description="Normal direction of the wall plane in world space.",
    )
    use_sphere: BoolProperty(
        name="Sphere Collider",
        default=False,
        description="Enable collision with a sphere object.",
    )
    sphere_object: PointerProperty(
        name="Sphere Object",
        type=bpy.types.Object,
        description="Object whose world position and scale define the sphere collider.",
    )
    static_collider_collection: PointerProperty(
        name="Static Collider Collection",
        type=bpy.types.Collection,
        description="Mesh objects in this collection are used as static triangle colliders.",
    )
    dynamic_collider_collection: PointerProperty(
        name="Dynamic Collider Collection",
        type=bpy.types.Collection,
        description="Collection of animated mesh objects used as one-way evaluated dynamic collision sources for this cloth",
    )


CLASSES = (
    SSBL_PreviewSettings,
    operators.SSBL_OT_interactive_pin_monitor,
    operators.SSBL_OT_start_preview,
    operators.SSBL_OT_stop_preview,
    operators.SSBL_OT_reset_preview,
    operators.SSBL_OT_bake_xpbd_cache,
    operators.SSBL_OT_clear_xpbd_cache,
) + ui.CLASSES


def register():
    translation.register(__name__)
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ssbl_preview = PointerProperty(type=SSBL_PreviewSettings)
    bpy.types.Object.ssbl_cloth = PointerProperty(type=SSBL_PreviewSettings)
    _initialize_hardness_for_scenes()
    if _initialize_hardness_for_scenes not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_initialize_hardness_for_scenes)
    if _restore_preview_source_before_save not in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.append(_restore_preview_source_before_save)
    operators.register_playback_handlers()
    bpy.types.Object.ssbl_force_field_weight = FloatProperty(
        name="SSBL Weight",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="SSBL multiplies this object force field by the given weight before uploading it to the cloth solver",
    )
    bpy.types.Object.ssbl_collision_layer = IntProperty(
        options={"HIDDEN"},
        name="Collision Layer",
        default=1,
        min=0,
        soft_max=4,
        description="Layer index used by layered multi-cloth collision. Lower values are more internal and solve first.",
    )
    bpy.types.Object.ssbl_enable_cross_cloth_collision = BoolProperty(
        options={"HIDDEN"},
        name="Cross-Cloth Collider",
        default=True,
        description="When enabled, this object can act as a dynamic collider for other cloth objects during multi-cloth preview.",
    )


def unregister():
    try:
        translation.unregister(__name__)
    except Exception:
        pass
    try:
        operators.cleanup_fps_overlays()
    except Exception:
        pass
    try:
        operators.cleanup_interactive_pin_monitor()
    except Exception:
        pass
    try:
        solver.cleanup_all_sessions()
    except Exception:
        pass
    if hasattr(bpy.types.Scene, "ssbl_preview"):
        del bpy.types.Scene.ssbl_preview
    if hasattr(bpy.types.Object, "ssbl_cloth"):
        del bpy.types.Object.ssbl_cloth
    if hasattr(bpy.types.Object, "ssbl_force_field_weight"):
        del bpy.types.Object.ssbl_force_field_weight
    if hasattr(bpy.types.Object, "ssbl_collision_layer"):
        del bpy.types.Object.ssbl_collision_layer
    if hasattr(bpy.types.Object, "ssbl_enable_cross_cloth_collision"):
        del bpy.types.Object.ssbl_enable_cross_cloth_collision
    if _initialize_hardness_for_scenes in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_initialize_hardness_for_scenes)
    if _restore_preview_source_before_save in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(_restore_preview_source_before_save)
    operators.unregister_playback_handlers()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
