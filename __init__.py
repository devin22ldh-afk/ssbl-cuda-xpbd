bl_info = {
    "name": "SSBL CUDA XPBD",
    "author": "OpenAI",
    "version": (0, 4, 2),
    "blender": (5, 0, 0),
    "location": "3D 视图 > 侧边栏 > SSBL",
    "description": "Blender 本地 CUDA XPBD 布料预览与 PC2 烘焙插件",
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

from . import operators, solver, ui
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


class SSBL_PreviewSettings(PropertyGroup):
    enabled: BoolProperty(
        name="启用 SSBL 布料",
        default=False,
        update=_apply_enabled,
        description="启用后，按空格播放时间轴即可运行 SSBL CUDA XPBD 布料预览",
    )
    object_settings_initialized: BoolProperty(
        name="SSBL 对象设置已初始化",
        default=False,
        options={"HIDDEN"},
        description="内部标记：对象首次启用时从场景默认设置复制一次",
    )
    runtime_mode: EnumProperty(
        name="运行模式",
        items=[
            ("preview", "预览", "运行视口实时预览"),
            ("bake", "烘焙", "烘焙为 PC2 缓存"),
        ],
        default="preview",
        options={"HIDDEN"},
        description="选择实时预览或 PC2 缓存烘焙模式",
    )
    hardness: FloatProperty(
        name="硬度",
        default=DEFAULT_HARDNESS,
        min=0.0,
        max=1.0,
        precision=3,
        update=_apply_hardness,
        description="布料材料硬度；0 为最柔软接近丝绸，1 为最硬接近皮革",
    )
    show_advanced_settings: BoolProperty(
        name="显示高级设置",
        default=False,
        description="展开后显示诊断和高级解算参数",
    )
    hardness_initialized: BoolProperty(
        name="硬度已初始化",
        default=False,
        options={"HIDDEN"},
        description="内部迁移标记，用于从旧场景参数推导硬度",
    )
    frame_count: IntProperty(
        name="预览帧数",
        default=120,
        min=1,
        soft_max=1000,
        options={"HIDDEN"},
        description="预览计时器要模拟的帧数",
    )
    preview_target_fps: FloatProperty(
        name="预览目标 FPS",
        default=30.0,
        min=1.0,
        soft_max=120.0,
        description="预览时视口播放的目标频率",
    )
    preview_writeback_interval: IntProperty(
        name="Preview writeback interval",
        default=0,
        min=0,
        soft_max=8,
        options={"HIDDEN"},
        description="0 uses adaptive preview mesh writeback; values >= 1 force a fixed interval. Baking still writes every frame",
    )
    use_evaluated_mesh: BoolProperty(
        name="动态网格",
        default=True,
        description="读取 Blender 修改器和绑定后的动态网格输入；顶点数和拓扑必须保持不变",
    )
    multi_cloth_preview: BoolProperty(
        options={"HIDDEN"},
        name="多布料预览",
        default=False,
        description="启用后，开始预览会同时解算当前选中的多个布料网格",
    )
    cross_cloth_collision: EnumProperty(
        options={"HIDDEN"},
        name="跨布料碰撞",
        items=[
            ("off", "关闭", "不使用其他布料作为动态碰撞体"),
            ("lower_layers", "较低层级", "只接收碰撞层级更低的布料作为动态碰撞体"),
            ("all_selected", "所有已选", "接收所有其他已选布料作为动态碰撞体"),
        ],
        default="off",
        description="多布料预览时如何收集动态布料碰撞体",
    )
    bake_start: IntProperty(
        name="烘焙开始帧",
        default=1,
        description="写入 PC2 缓存的第一帧",
    )
    bake_end: IntProperty(
        name="烘焙结束帧",
        default=120,
        min=1,
        description="写入 PC2 缓存的最后一帧",
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
        name="时间步长",
        default=0.02,
        min=1e-4,
        soft_max=0.1,
        subtype="TIME",
        description="每个输出帧对应的模拟时间步长",
    )
    substeps: IntProperty(
        name="子步数",
        default=14,
        min=1,
        soft_max=32,
        description="每帧的 XPBD 子步数量",
    )
    iterations: IntProperty(
        name="迭代次数",
        default=2,
        min=1,
        soft_max=128,
        description="每个子步的约束投影迭代次数",
    )
    damping: FloatProperty(
        name="速度阻尼",
        default=0.9999,
        min=0.0,
        max=1.0,
        description="每个子步施加的速度阻尼",
    )
    density: FloatProperty(
        name="密度",
        default=1.0,
        min=1e-4,
        soft_max=100.0,
        description="布料顶点使用的面密度",
    )
    stretch_compliance: FloatProperty(
        name="拉伸柔顺度",
        default=1e-6,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        description="XPBD 拉伸柔顺度；值越小越硬",
    )
    bend_compliance: FloatProperty(
        name="弯曲柔顺度",
        default=1e-4,
        min=0.0,
        soft_max=1e-2,
        precision=6,
        description="XPBD 弯曲柔顺度；值越小越硬",
    )
    use_lra: BoolProperty(
        name="旧版兼容开关",
        default=False,
        options={"HIDDEN"},
        description="内部隐藏字段；硬度会自动控制抗拉长 tether",
    )
    lra_compliance: FloatProperty(
        name="旧版兼容柔顺度",
        default=0.0,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        options={"HIDDEN"},
        description="内部隐藏字段；硬度会自动控制抗拉长 tether",
    )
    lra_slack: FloatProperty(
        name="旧版兼容松弛系数",
        default=1.0,
        min=1.0,
        soft_max=2.0,
        precision=3,
        options={"HIDDEN"},
        description="内部隐藏字段；硬度会自动控制抗拉长 tether",
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
        soft_max=2.0,
        precision=3,
        description="Outward force strength applied along the cloth surface normal",
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
        name="使用 Blender 力场",
        default=True,
        description="读取场景中的 Blender Force Field，并作为 SSBL 布料外力参与解算",
    )
    force_field_collection: PointerProperty(
        name="力场集合",
        type=bpy.types.Collection,
        description="为空时读取当前场景全部 Force Field；指定后只读取该集合内的 Force Field",
    )
    force_field_strength_scale: FloatProperty(
        name="力场强度倍率",
        default=1.0,
        min=0.0,
        soft_max=10.0,
        precision=3,
        description="SSBL 接收 Blender 力场时使用的整体强度倍率",
    )
    force_field_weight_gravity: FloatProperty(
        name="重力",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale scene gravity for SSBL cloth",
    )
    force_field_weight_all: FloatProperty(
        name="全部",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale all non-gravity Blender force fields for SSBL cloth",
    )
    force_field_weight_force: FloatProperty(
        name="常力",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Force effector influence for SSBL cloth",
    )
    force_field_weight_vortex: FloatProperty(
        name="涡流",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Vortex effector influence for SSBL cloth",
    )
    force_field_weight_magnetic: FloatProperty(
        name="磁力",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Magnetic effector influence for SSBL cloth",
    )
    force_field_weight_harmonic: FloatProperty(
        name="谐振",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Harmonic effector influence for SSBL cloth",
    )
    force_field_weight_charge: FloatProperty(
        name="电荷",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Charge effector influence for SSBL cloth",
    )
    force_field_weight_lennardjones: FloatProperty(
        name="兰纳琼斯分子力",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Lennard-Jones effector influence for SSBL cloth",
    )
    force_field_weight_wind: FloatProperty(
        name="风力",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Wind effector influence for SSBL cloth",
    )
    force_field_weight_texture: FloatProperty(
        name="纹理",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Texture effector influence for SSBL cloth",
    )
    force_field_weight_turbulence: FloatProperty(
        name="紊流",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Turbulence effector influence for SSBL cloth",
    )
    force_field_weight_drag: FloatProperty(
        name="拖拽",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Scale Drag effector influence for SSBL cloth",
    )
    pin_vertex_group: StringProperty(
        name="固定点组",
        default="ssbl_pin",
        description="该顶点组中的顶点会保持在静止位置",
    )
    collision_margin: FloatProperty(
        name="碰撞边距",
        default=0.005,
        min=0.0,
        soft_max=0.2,
        precision=4,
        description="碰撞投影时保持的最小分离距离",
    )
    self_collision: BoolProperty(
        update=_apply_self_collision_toggle,
        name="自碰撞",
        default=False,
        description="兼容旧选项；启用后会映射到“快速”自碰撞",
    )
    self_collision_mode: EnumProperty(
        name="自碰撞模式",
        items=[
            ("fast", "快速", "使用预览优先的自碰撞快速路径"),
            ("strict", "严格", "使用完整 probe/recovery 的零相交优先自碰撞路径"),
        ],
        default="fast",
        update=_apply_self_collision_mode,
        description="面向大规模布料网格调优的自碰撞模式",
    )
    cloth_thickness: FloatProperty(
        name="布料厚度",
        default=0.02,
        min=0.001,
        soft_max=0.1,
        precision=4,
        description="自碰撞时布料表面之间保持的最小分离距离；值越大布料看起来越厚、不会重叠",
    )
    self_collision_interval: IntProperty(
        name="自碰撞间隔",
        default=1,
        min=1,
        soft_max=8,
        description="每 N 个子步运行一次自碰撞；快速模式通常用 2 即可",
    )
    max_self_collision_neighbors: IntProperty(
        name="最大自碰撞邻居数",
        default=64,
        min=4,
        soft_max=256,
        description="每个顶点或边在自碰撞中允许处理的最大邻居候选数",
    )
    fast_self_collision_passes: IntProperty(
        name="快速自碰撞步数",
        default=4,
        min=1,
        max=8,
        soft_max=8,
        description="仅用于 fast 自碰撞模式的 native 自碰撞 pass 数；strict 模式不受影响",
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
        name="地面平面",
        default=True,
        description="启用简单的世界空间地面平面碰撞",
    )
    ground_height: FloatProperty(
        name="地面 Z",
        default=0.0,
        description="地面平面在世界空间中的 Z 高度",
    )
    use_wall: BoolProperty(
        name="墙面平面",
        default=False,
        description="启用与无限平面的碰撞",
    )
    wall_origin: FloatVectorProperty(
        name="墙面原点",
        default=(0.0, 0.0, 0.0),
        size=3,
        subtype="XYZ",
        description="墙面平面在世界空间中的一个点",
    )
    wall_normal: FloatVectorProperty(
        name="墙面法线",
        default=(0.0, 0.0, 1.0),
        size=3,
        subtype="DIRECTION",
        description="墙面平面在世界空间中的法线方向",
    )
    use_sphere: BoolProperty(
        name="球体碰撞器",
        default=False,
        description="启用与球体对象的碰撞",
    )
    sphere_object: PointerProperty(
        name="球体对象",
        type=bpy.types.Object,
        description="其世界位置和尺寸用于定义球体碰撞器的对象",
    )
    static_collider_collection: PointerProperty(
        name="静态碰撞集合",
        type=bpy.types.Collection,
        description="该集合中的网格对象会被用作静态三角形碰撞体",
    )


CLASSES = (
    SSBL_PreviewSettings,
    operators.SSBL_OT_start_preview,
    operators.SSBL_OT_stop_preview,
    operators.SSBL_OT_reset_preview,
    operators.SSBL_OT_bake_xpbd_cache,
    operators.SSBL_OT_clear_xpbd_cache,
) + ui.CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ssbl_preview = PointerProperty(type=SSBL_PreviewSettings)
    bpy.types.Object.ssbl_cloth = PointerProperty(type=SSBL_PreviewSettings)
    _initialize_hardness_for_scenes()
    if _initialize_hardness_for_scenes not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_initialize_hardness_for_scenes)
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
        name="碰撞层级",
        default=1,
        min=0,
        soft_max=4,
        description="多布料分层碰撞中的层级；数值越小越靠内并先解算",
    )
    bpy.types.Object.ssbl_enable_cross_cloth_collision = BoolProperty(
        options={"HIDDEN"},
        name="跨布料碰撞体",
        default=True,
        description="启用后，该对象会在多布料预览中作为其他布料的动态碰撞体",
    )


def unregister():
    try:
        operators.cleanup_fps_overlays()
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
    operators.unregister_playback_handlers()
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
