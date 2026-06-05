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

        # Main toggle
        row = layout.row(align=True)
        row.scale_y = 1.2
        row.prop(settings, "enabled", text="启用布料模拟" if not settings.enabled else "布料模拟已启用", toggle=True, icon="PHYSICS")

        if settings.enabled:
            # Status display
            col = layout.column(align=True)
            status_str = f"状态: {solver.session_status(obj)}"
            fps_str = f"  |  FPS: {solver.session_fps(obj):.1f}" if solver.has_session(obj) else ""
            col.label(text=status_str + fps_str, icon="INFO")
            col.label(text=solver.backend_status_text(), icon="CONSOLE")

            layout.separator()
            layout.operator("ssbl.reset_preview", text="重置预览状态", icon="LOOP_BACK")


class SSBL_PT_material(bpy.types.Panel):
    bl_label = "Material / Inflation"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(settings, "hardness", slider=True)
        layout.prop(settings, "pin_vertex_group")
        
        layout.separator()
        layout.prop(settings, "use_volume_pressure")
        if settings.use_volume_pressure:
            col = layout.column(align=True)
            col.prop(settings, "pressure_strength")


class SSBL_PT_collision(bpy.types.Panel):
    bl_label = "碰撞"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(settings, "self_collision")
        if settings.self_collision:
            layout.prop(settings, "self_collision_mode")

        layout.separator()
        layout.prop(settings, "collision_margin")
        layout.prop(settings, "cloth_thickness")

        layout.separator()
        layout.prop(settings, "use_ground")
        if settings.use_ground:
            layout.prop(settings, "ground_height")
        layout.prop(settings, "static_collider_collection")
        sdf_box = layout.box()
        sdf_col = sdf_box.column(align=True)
        sdf_col.prop(settings, "static_sdf_voxel_size")
        sdf_col.prop(settings, "static_sdf_band_voxels")
        sdf_col.prop(settings, "static_sdf_max_resolution")


class SSBL_PT_cache(bpy.types.Panel):
    bl_label = "预览与烘焙"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        obj = context.active_object
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        is_baking = bool(getattr(settings, "bake_in_progress", False))
        
        col = layout.column(align=True)
        col.prop(settings, "preview_target_fps")
        col.prop(settings, "use_evaluated_mesh")
        
        layout.separator()
        col = layout.column(align=True)
        col.prop(settings, "multi_cloth_preview")
        if settings.multi_cloth_preview:
            col.prop(settings, "cross_cloth_collision")
            if obj is not None:
                col.prop(obj, "ssbl_collision_layer")
                col.prop(obj, "ssbl_enable_cross_cloth_collision")

        layout.separator()
        if is_baking:
            current = int(getattr(settings, "bake_progress_current", 0))
            total = int(getattr(settings, "bake_progress_total", 0))
            percent = float(getattr(settings, "bake_progress_percent", 0.0))
            layout.label(text=f"烘焙中: {current}/{total} ({percent:.0f}%)", icon="TIME")

        col = layout.column(align=True)
        col.enabled = not is_baking
        
        row = col.row(align=True)
        row.prop(settings, "bake_start")
        row.prop(settings, "bake_end")
        
        row = col.row(align=True)
        row.operator("ssbl.bake_xpbd_cache", text="开始烘焙", icon="REC")
        row.operator("ssbl.clear_xpbd_cache", text="清除", icon="TRASH")


class SSBL_PT_force_fields(bpy.types.Panel):
    bl_label = "外力与场"
    bl_parent_id = "SSBL_PT_physics_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = _active_cloth_settings(context)
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(settings, "force_field_collection", text="效果器集合")
        
        layout.separator()
        flow = layout.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=True)
        for group in visible_force_field_weight_groups():
            col = flow.column()
            for prop_name in group:
                col.prop(settings, prop_name, slider=True)


class SSBL_PT_advanced(bpy.types.Panel):
    bl_label = "高级解算"
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
        col.prop(settings, "dt")
        col.prop(settings, "substeps")
        col.prop(settings, "iterations")
        col.prop(settings, "damping")
        col.prop(settings, "density")

        layout.separator()
        col = layout.column(align=True)
        col.label(text="接触:")
        col.prop(settings, "contact_friction")
        col.prop(settings, "contact_tangent_damping")
        col.prop(settings, "contact_compliance")

        if settings.self_collision:
            layout.separator()
            col = layout.column(align=True)
            col.label(text="自碰撞:")
            col.prop(settings, "self_collision_interval")
            col.prop(settings, "max_self_collision_neighbors")
            if str(getattr(settings, "self_collision_mode", "fast")).lower() == "fast":
                col.prop(settings, "fast_self_collision_passes")
            col.prop(settings, "self_probe_interval")
            col.prop(settings, "self_surface_pair_interval")
            
            col.prop(settings, "self_sleep_enabled")
            if settings.self_sleep_enabled:
                col.prop(settings, "self_sleep_still_frames")
                col.prop(settings, "self_sleep_full_scan_interval")


CLASSES = (
    SSBL_PT_physics_panel,
    SSBL_PT_material,
    SSBL_PT_collision,
    SSBL_PT_cache,
    SSBL_PT_force_fields,
    SSBL_PT_advanced,
)
