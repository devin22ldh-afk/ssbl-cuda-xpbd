from __future__ import annotations

import bpy

from . import solver
from .force_fields import visible_force_field_weight_groups


def _active_cloth_settings(context: bpy.types.Context):
    obj = context.active_object
    if obj is not None and hasattr(obj, "ssbl_cloth"):
        return obj.ssbl_cloth
    return context.scene.ssbl_preview



class SSBL_PT_physics_panel(bpy.types.Panel):
    bl_label = "SSBL CUDA XPBD"
    bl_idname = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        obj = context.active_object
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Main toggle - Awesome Design: Large, prominent CTA
        row = layout.row(align=True)
        row.scale_y = 1.5
        is_enabled = settings.enabled
        
        if is_enabled:
            row.prop(settings, "enabled", text="SSBL 布料解算中", toggle=True, icon="PLAY")
        else:
            row.prop(settings, "enabled", text="启用 SSBL 布料模拟", toggle=True, icon="PHYSICS")

        if settings.enabled:
            layout.separator()
            
            # Status Dashboard - Awesome Design: Boxed logical grouping
            col = layout.column(align=True)
            col.label(text="运行状态", icon="STATUSBAR")
            status_box = col.box()
            box_col = status_box.column(align=True)
            
            # Status display
            status_str = f"状态: {solver.session_status(obj)}"
            fps_str = f" | FPS: {solver.session_fps(obj):.1f}" if solver.has_session(obj) else ""
            
            row_status = box_col.row()
            row_status.label(text=status_str + fps_str, icon="INFO")
            if solver.has_session(obj):
                diag = solver.session_diagnostics(obj)
                row_perf = box_col.row()
                row_perf.label(
                    text=(
                        "Frame %.1fms | CUDA %.1f | Input %.1f | DynUp %.1f | DynCol %.2f"
                        % (
                            float(getattr(diag, "frame_ms", 0.0)),
                            float(getattr(diag, "cuda_step_call_ms", 0.0)),
                            float(getattr(diag, "input_refresh_ms", 0.0)),
                            float(getattr(diag, "dynamic_upload_ms", 0.0)),
                            float(getattr(diag, "dynamic_collision_ms", 0.0)),
                        )
                    ),
                    icon="TIME",
                )
                row_perf2 = box_col.row()
                row_perf2.label(
                    text=(
                        "Download %.1f | Writeback %.1f | Viewport %.2f"
                        % (
                            float(getattr(diag, "download_ms", 0.0)),
                            float(getattr(diag, "writeback_ms", 0.0)),
                            float(getattr(diag, "viewport_tag_ms", 0.0)),
                        )
                    ),
                    icon="GRAPH",
                )
            
            row_backend = box_col.row()
            row_backend.label(text=solver.backend_status_text(), icon="CONSOLE")

            box_col.separator(factor=0.5)
            row_reset = box_col.row()
            row_reset.operator("ssbl.reset_preview", text="重置引擎状态", icon="LOOP_BACK")


class SSBL_PT_material(bpy.types.Panel):
    bl_label = "物理与材质 (Physics & Material)"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Base Physics
        col = layout.column(align=True)
        col.label(text="基础属性", icon="MOD_CLOTH")
        box_base = col.box()
        box_base.prop(settings, "hardness", text="弯曲硬度", slider=True)
        box_base.prop(settings, "cloth_thickness", text="表面厚度")
        box_base.prop(settings, "density", text="材质密度")
        
        layout.separator()

        # Constraints
        col = layout.column(align=True)
        col.label(text="空间约束", icon="PINNED")
        box_pin = col.box()
        box_pin.prop(settings, "pin_vertex_group", text="钉固顶点组")
        box_pin.prop(settings, "pin_hardness", text="钉固硬度", slider=True)
        
        layout.separator()

        # Inflation
        col = layout.column(align=True)
        col.label(text="充气与体积", icon="OUTLINER_OB_FORCE_FIELD")
        box_pressure = col.box()
        box_pressure.prop(settings, "use_volume_pressure", text="启用体积气压", toggle=True)
        if settings.use_volume_pressure:
            sub = box_pressure.column(align=True)
            sub.separator()
            sub.prop(settings, "pressure_strength", text="内压强度")
            if settings.pressure_strength >= 0.2:
                sub.label(text="High pressure: use cautiously on dense meshes", icon="ERROR")


class SSBL_PT_collision(bpy.types.Panel):
    bl_label = "环境与碰撞 (Environment & Collision)"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        # World Interaction
        col = layout.column(align=True)
        col.label(text="场景交互", icon="WORLD")
        box_world = col.box()
        box_world.prop(settings, "collision_margin", text="碰撞容差 (Margin)")
        
        box_world.separator()
        box_world.prop(settings, "use_ground", text="无限平面底座", toggle=True)
        if settings.use_ground:
            sub = box_world.column(align=True)
            sub.prop(settings, "ground_height", text="平面高度")
            
        box_world.separator()
        box_world.prop(settings, "static_collider_collection", text="静态碰撞体集合")
        
        box_world.prop(settings, "dynamic_collider_collection", text="动画碰撞体集合")

        # Static SDF Settings
        if settings.static_collider_collection:
            sdf_col = box_world.column(align=True)
            sdf_col.separator()
            sdf_col.label(text="Voxel SDF 精度:", icon="MESH_GRID")
            sdf_col.prop(settings, "static_sdf_voxel_size", text="体素尺寸")
            sdf_col.prop(settings, "static_sdf_band_voxels", text="拓展带 (Band)")
            sdf_col.prop(settings, "static_sdf_max_resolution", text="最大分辨率")

        layout.separator()

        # Self Collision
        col_self = layout.column(align=True)
        col_self.label(text="自碰撞解算", icon="OUTLINER_OB_MESH")
        box_self = col_self.box()
        box_self.prop(settings, "self_collision", text="启用自碰撞", toggle=True)
        if settings.self_collision:
            sub_self = box_self.column(align=True)
            sub_self.separator()
            sub_self.prop(settings, "self_collision_mode", text="算法模式")
            sub_self.prop(settings, "self_collision_distance", text="自碰撞距离")
            if str(getattr(settings, "self_collision_mode", "fast")).lower() == "fast":
                sub_self.prop(settings, "fast_self_collision_passes", text="快速遍数 (Passes)")
            sub_self.prop(settings, "max_self_collision_neighbors", text="最大邻居探测数")


class SSBL_PT_force_fields(bpy.types.Panel):
    bl_label = "外力与场 (Force Fields)"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        col = layout.column(align=True)
        col.label(text="效果器", icon="FORCE_FORCE")
        box = col.box()
        box.prop(settings, "force_field_collection", text="效果器集合")
        
        box.separator()
        flow = box.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=True)
        for group in visible_force_field_weight_groups():
            flow_col = flow.column()
            for prop_name in group:
                flow_col.prop(settings, prop_name, slider=True)


class SSBL_PT_advanced(bpy.types.Panel):
    bl_label = "解算器调优 (Solver Tuning)"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        # Precision & Quality
        col = layout.column(align=True)
        col.label(text="引擎核心 (Core)", icon="MOD_BUILD")
        box_solver = col.box()
        box_solver.prop(settings, "dt", text="步长 (Time Step)")
        box_solver.prop(settings, "substeps", text="子步数 (Substeps)")
        box_solver.prop(settings, "iterations", text="约束迭代 (Iterations)")
        box_solver.prop(settings, "damping", text="全局阻尼 (Damping)")

        layout.separator()

        # Contact Dynamics
        col_contact = layout.column(align=True)
        col_contact.label(text="接触动力学", icon="GRAPH")
        box_contact = col_contact.box()
        box_contact.prop(settings, "contact_friction", text="表面摩擦力")
        box_contact.prop(settings, "contact_tangent_damping", text="切向滑动阻尼")
        box_contact.prop(settings, "contact_compliance", text="接触顺应性 (Compliance)")

        if settings.self_collision:
            layout.separator()
            # Advanced Self Collision
            col_adv = layout.column(align=True)
            col_adv.label(text="高级自碰撞设定", icon="TOOL_SETTINGS")
            box_adv_self = col_adv.box()
            box_adv_self.prop(settings, "self_collision_interval", text="求交判定间隔")
            box_adv_self.prop(settings, "self_probe_interval", text="宽相 Probe 间隔")
            box_adv_self.prop(settings, "self_surface_pair_interval", text="表面配对刷新间隔")


class SSBL_PT_cache(bpy.types.Panel):
    bl_label = "缓存与烘焙 (Cache & Bake)"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        obj = context.active_object
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        is_baking = bool(getattr(settings, "bake_in_progress", False))
        
        # Playback Settings
        col = layout.column(align=True)
        col.label(text="预览设定", icon="PLAY")
        box_preview = col.box()
        box_preview.prop(settings, "preview_target_fps", text="目标预览帧率")
        box_preview.prop(settings, "auto_cache_realtime", text="实时模拟自动缓存")
        box_preview.prop(settings, "use_evaluated_mesh", text="使用修改器形态 (Evaluated)")
        
        layout.separator()

        # Bake Workflow
        col_bake = layout.column(align=True)
        col_bake.label(text="烘焙工作流", icon="RENDER_ANIMATION")
        box_bake = col_bake.box()
        
        if is_baking:
            current = int(getattr(settings, "bake_progress_current", 0))
            total = int(getattr(settings, "bake_progress_total", 0))
            percent = float(getattr(settings, "bake_progress_percent", 0.0))
            row_progress = box_bake.row()
            row_progress.scale_y = 1.2
            row_progress.label(text=f"烘焙中: {current}/{total} ({percent:.0f}%)", icon="TIME")

        sub_bake = box_bake.column(align=True)
        sub_bake.enabled = not is_baking
        
        row_frames = sub_bake.row(align=True)
        row_frames.prop(settings, "bake_start", text="起始帧")
        row_frames.prop(settings, "bake_end", text="结束帧")
        
        sub_bake.separator()
        row_actions = sub_bake.row(align=True)
        row_actions.scale_y = 1.5
        row_actions.operator("ssbl.bake_xpbd_cache", text="开始烘焙", icon="REC")
        row_actions.operator("ssbl.clear_xpbd_cache", text="清除缓存", icon="TRASH")


CLASSES = (
    SSBL_PT_physics_panel,
    SSBL_PT_material,
    SSBL_PT_collision,
    SSBL_PT_force_fields,
    SSBL_PT_advanced,
    SSBL_PT_cache,
)
