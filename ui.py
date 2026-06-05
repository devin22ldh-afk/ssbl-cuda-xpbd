from __future__ import annotations

import bpy

from . import solver
from .xpbd_core import preview_hardness_settings


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
        derived = preview_hardness_settings(settings)

        status_box = layout.box()
        status_box.prop(settings, "enabled", toggle=True, icon="MOD_CLOTH")
        status_box.label(text=f"状态: {solver.session_status(obj)}")
        if solver.has_session(obj):
            status_box.label(text=f"预览 FPS: {solver.session_fps(obj):.1f}")
        else:
            status_box.label(text="预览 FPS: -")
        if obj is not None:
            status_box.label(text=f"对象: {obj.name}")
        status_box.label(text=solver.backend_status_text())
        status_box.operator("ssbl.reset_preview", icon="LOOP_BACK")

        preview_box = layout.box()
        preview_box.label(text="预览")
        preview_box.prop(settings, "use_evaluated_mesh")

        sim_box = layout.box()
        sim_box.label(text="材料")
        sim_box.prop(settings, "hardness", slider=True)
        sim_box.label(text="0 = 丝绸    1 = 皮革")
        sim_box.prop(settings, "use_volume_pressure")
        if settings.use_volume_pressure:
            sim_box.prop(settings, "pressure_strength")
            sim_box.prop(settings, "volume_target_scale")
        sim_box.label(text="Gravity: uses Scene setting", icon="SCENE_DATA")
        sim_box.prop(settings, "use_blender_force_fields")
        if settings.use_blender_force_fields:
            sim_box.prop(settings, "force_field_collection")
            sim_box.prop(settings, "force_field_strength_scale")
        sim_box.prop(settings, "pin_vertex_group")

        collision_box = layout.box()
        collision_box.label(text="碰撞")
        collision_box.prop(settings, "collision_margin")
        collision_box.prop(settings, "self_collision")
        if settings.self_collision:
            collision_box.prop(settings, "self_collision_mode")
        collision_box.prop(settings, "cloth_thickness")
        collision_box.prop(settings, "use_ground")
        if settings.use_ground:
            collision_box.prop(settings, "ground_height")
        collision_box.prop(settings, "static_collider_collection")

        bake_box = layout.box()
        bake_box.label(text="烘焙缓存")
        if getattr(settings, "bake_in_progress", False):
            current = int(getattr(settings, "bake_progress_current", 0))
            total = int(getattr(settings, "bake_progress_total", 0))
            bake_box.label(text=f"烘焙中: {current}/{total}")
            progress_row = bake_box.row()
            progress_row.enabled = False
            progress_row.prop(settings, "bake_progress_percent", slider=True, text="进度")
        frame_row = bake_box.row(align=True)
        frame_row.enabled = not bool(getattr(settings, "bake_in_progress", False))
        frame_row.prop(settings, "bake_start")
        frame_row.prop(settings, "bake_end")
        action_row = bake_box.row(align=True)
        action_row.enabled = not bool(getattr(settings, "bake_in_progress", False))
        action_row.operator("ssbl.bake_xpbd_cache", icon="REC")
        action_row.operator("ssbl.clear_xpbd_cache", icon="TRASH")

        advanced_box = layout.box()
        icon = "TRIA_DOWN" if settings.show_advanced_settings else "TRIA_RIGHT"
        advanced_box.prop(settings, "show_advanced_settings", text="高级设置", emboss=False, icon=icon)
        if settings.show_advanced_settings:
            derived_box = advanced_box.box()
            derived_box.label(text="硬度派生项")
            derived_box.label(text=f"拉伸柔顺度: {derived.stretch_compliance:.3g}")
            derived_box.label(text=f"弯曲柔顺度: {derived.bend_compliance:.3g}")
            derived_box.label(text=f"当前硬度: {derived.hardness:.3f}")

            tuning_box = advanced_box.box()
            tuning_box.label(text="高级解算")
            tuning_box.prop(settings, "dt")
            tuning_box.prop(settings, "substeps")
            tuning_box.prop(settings, "iterations")
            tuning_box.prop(settings, "damping")
            tuning_box.prop(settings, "density")
            if settings.self_collision:
                tuning_box.prop(settings, "self_collision_interval")
                tuning_box.prop(settings, "max_self_collision_neighbors")
                if str(getattr(settings, "self_collision_mode", "fast")).lower() == "fast":
                    tuning_box.prop(settings, "fast_self_collision_passes")
                tuning_box.prop(settings, "self_probe_interval")
                tuning_box.prop(settings, "self_surface_pair_interval")
                tuning_box.prop(settings, "self_sleep_enabled")
                if settings.self_sleep_enabled:
                    tuning_box.prop(settings, "self_sleep_still_frames")
                    tuning_box.prop(settings, "self_sleep_full_scan_interval")
            if settings.use_volume_pressure:
                tuning_box.prop(settings, "volume_compliance")
                tuning_box.prop(settings, "volume_solve_interval")


class SSBL_PT_force_field_panel(bpy.types.Panel):
    bl_label = "SSBL Force Field"
    bl_idname = "SSBL_PT_force_field_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        obj = context.active_object
        field = getattr(obj, "field", None) if obj is not None else None
        return obj is not None and field is not None and str(getattr(field, "type", "NONE")).upper() not in {"", "NONE"}

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        obj = context.active_object
        field = obj.field
        weight = float(getattr(obj, "ssbl_force_field_weight", 1.0))
        strength = float(getattr(field, "strength", 0.0))
        box = layout.box()
        box.prop(obj, "ssbl_force_field_weight", slider=True)
        box.label(text=f"Effective strength: {strength * weight:.3f}")
