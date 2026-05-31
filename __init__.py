bl_info = {
    "name": "SSBL CUDA XPBD",
    "author": "OpenAI",
    "version": (0, 4, 1),
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


def _apply_self_collision_mode(settings, _context):
    mode = settings.self_collision_mode
    if mode == "fast":
        settings.self_collision_interval = 2
        settings.max_self_collision_neighbors = 32


def _apply_solver_preset(settings, _context):
    preset = settings.solver_preset
    if preset == "fast":
        settings.substeps = 8
        settings.iterations = 1
    elif preset == "stable":
        settings.substeps = 16
        settings.iterations = 2
    else:
        settings.substeps = 12
        settings.iterations = 1


def _apply_hardness(settings, _context):
    settings.hardness_initialized = True
    sync_hardness_settings(settings)


@persistent
def _initialize_hardness_for_scenes(_dummy=None):
    if not hasattr(bpy.types.Scene, "ssbl_preview"):
        return
    for scene in bpy.data.scenes:
        try:
            sync_hardness_settings(scene.ssbl_preview)
        except Exception:
            pass


class SSBL_PreviewSettings(PropertyGroup):
    runtime_mode: EnumProperty(
        name="运行模式",
        items=[
            ("preview", "预览", "运行视口实时预览"),
            ("bake", "烘焙", "烘焙为 PC2 缓存"),
        ],
        default="preview",
        description="选择实时预览或 PC2 缓存烘焙模式",
    )
    solver_preset: EnumProperty(
        name="解算质量预设",
        items=[
            ("fast", "快速预览", "为较轻网格优化的小步预览配置"),
            ("balanced", "平衡", "默认的小步 XPBD 布料设置"),
            ("stable", "稳定布料", "为更复杂布料和碰撞场景提供更多子步"),
        ],
        default="balanced",
        update=_apply_solver_preset,
        description="应用推荐的解算质量配置，只调整稳定性和性能相关参数",
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
        description="预览计时器要模拟的帧数",
    )
    preview_target_fps: FloatProperty(
        name="预览目标 FPS",
        default=30.0,
        min=1.0,
        soft_max=120.0,
        description="预览时视口播放的目标频率",
    )
    use_evaluated_mesh: BoolProperty(
        name="使用求值网格",
        default=True,
        description="在拓扑不变时，从 Blender 求值后的依赖图读取布料和碰撞体输入",
    )
    multi_cloth_preview: BoolProperty(
        name="多布料预览",
        default=False,
        description="启用后，开始预览会同时解算当前选中的多个布料网格",
    )
    cross_cloth_collision: EnumProperty(
        name="跨布料碰撞",
        items=[
            ("off", "关闭", "不使用其他布料作为动态碰撞体"),
            ("lower_layers", "较低层级", "只接收碰撞层级更低的布料作为动态碰撞体"),
            ("all_selected", "所有已选", "接收所有其他已选布料作为动态碰撞体"),
        ],
        default="lower_layers",
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
    dt: FloatProperty(
        name="时间步长",
        default=1.0 / 60.0,
        min=1e-4,
        soft_max=0.1,
        subtype="TIME",
        description="每个输出帧对应的模拟时间步长",
    )
    substeps: IntProperty(
        name="子步数",
        default=12,
        min=1,
        soft_max=32,
        description="每帧的 XPBD 子步数量",
    )
    iterations: IntProperty(
        name="迭代次数",
        default=1,
        min=1,
        soft_max=128,
        description="每个子步的约束投影迭代次数",
    )
    damping: FloatProperty(
        name="速度阻尼",
        default=0.995,
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
        name="充气 / 体积保持",
        default=False,
        description="为 Suzanne 这类闭合软壳保持有符号体积；常规开口布料默认关闭",
    )
    volume_compliance: FloatProperty(
        name="体积柔顺度",
        default=1e-6,
        min=0.0,
        soft_max=1e-3,
        precision=6,
        description="全局体积保持的 XPBD 柔顺度；值越小体积保持越强",
    )
    pressure_strength: FloatProperty(
        name="充气强度",
        default=1.0,
        min=0.0,
        soft_max=2.0,
        precision=3,
        description="缩放每次投影施加的体积修正强度",
    )
    volume_target_scale: FloatProperty(
        name="目标体积比例",
        default=1.0,
        min=0.05,
        soft_max=2.0,
        precision=3,
        description="目标体积相对于静止有符号体积的倍数",
    )
    gravity: FloatVectorProperty(
        name="重力",
        default=(0.0, 0.0, -9.8),
        size=3,
        subtype="ACCELERATION",
        description="世界空间中的重力向量",
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
        name="自碰撞",
        default=False,
        description="兼容旧选项；启用后会映射到“快速”自碰撞",
    )
    self_collision_mode: EnumProperty(
        name="自碰撞模式",
        items=[
            ("off", "关闭", "关闭布料自碰撞"),
            ("fast", "快速", "使用受限顶点空间哈希自碰撞"),
        ],
        default="off",
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
        default=128,
        min=4,
        soft_max=256,
        description="每个顶点或边在自碰撞中允许处理的最大邻居候选数",
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
    ui.SSBL_PT_preview_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ssbl_preview = PointerProperty(type=SSBL_PreviewSettings)
    _initialize_hardness_for_scenes()
    if _initialize_hardness_for_scenes not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_initialize_hardness_for_scenes)
    bpy.types.Object.ssbl_collision_layer = IntProperty(
        name="碰撞层级",
        default=1,
        min=0,
        soft_max=4,
        description="多布料分层碰撞中的层级；数值越小越靠内并先解算",
    )
    bpy.types.Object.ssbl_enable_cross_cloth_collision = BoolProperty(
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
    if hasattr(bpy.types.Object, "ssbl_collision_layer"):
        del bpy.types.Object.ssbl_collision_layer
    if hasattr(bpy.types.Object, "ssbl_enable_cross_cloth_collision"):
        del bpy.types.Object.ssbl_enable_cross_cloth_collision
    if _initialize_hardness_for_scenes in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_initialize_hardness_for_scenes)
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
