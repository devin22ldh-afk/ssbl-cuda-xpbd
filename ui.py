from __future__ import annotations

import bpy

from . import solver


def _active_cloth_settings(context: bpy.types.Context):
    obj = context.active_object
    if obj is not None and hasattr(obj, "ssbl_cloth"):
        return obj.ssbl_cloth
    return context.scene.ssbl_preview


def _panel_header(layout: bpy.types.UILayout, text: str, icon: str):
    row = layout.row(align=True)
    row.label(text=text, icon=icon)
    return row


def _draw_status_box(layout: bpy.types.UILayout, obj: bpy.types.Object, settings) -> None:
    box = layout.box()
    _panel_header(box, "状态与操作", "MOD_CLOTH")
    box.prop(settings, "enabled", text="启用当前布料", toggle=True, icon="CHECKMARK")

    grid = box.grid_flow(columns=1, even_columns=False, even_rows=False, align=True)
    grid.label(text=f"对象: {obj.name if obj is not None else '-'}", icon="OBJECT_DATA")
    grid.label(text=f"状态: {solver.session_status(obj)}", icon="INFO")
    fps_text = f"{solver.session_fps(obj):.1f}" if solver.has_session(obj) else "-"
    grid.label(text=f"预览 FPS: {fps_text}", icon="TIME")
    grid.label(text=solver.backend_status_text(), icon="CONSOLE")

    action_row = box.row(align=True)
    action_row.operator("ssbl.reset_preview", text="重置预览", icon="LOOP_BACK")


def _draw_preview_box(layout: bpy.types.UILayout, obj: bpy.types.Object, settings) -> None:
    box = layout.box()
    _panel_header(box, "预览输入", "PLAY")
    box.prop(settings, "use_evaluated_mesh", text="使用修改器后的动态网格")
    box.prop(settings, "preview_target_fps")

    multi_box = box.box()
    _panel_header(multi_box, "多布料", "OUTLINER_OB_GROUP_INSTANCE")
    multi_box.prop(settings, "multi_cloth_preview", text="启用多布料预览")
    if settings.multi_cloth_preview:
        multi_box.prop(settings, "cross_cloth_collision")
        if obj is not None:
            multi_box.prop(obj, "ssbl_collision_layer")
            multi_box.prop(obj, "ssbl_enable_cross_cloth_collision")


def _draw_material_force_box(layout: bpy.types.UILayout, settings) -> None:
    box = layout.box()
    _panel_header(box, "材料与外力", "MATERIAL")

    box.prop(settings, "hardness", slider=True)
    box.label(text="0 = 丝绸    1 = 皮革", icon="IPO_EASE_IN_OUT")
    box.prop(settings, "pin_vertex_group")

    volume_box = box.box()
    _panel_header(volume_box, "充气 / 体积保持", "META_BALL")
    volume_box.prop(settings, "use_volume_pressure")
    if settings.use_volume_pressure:
        volume_box.prop(settings, "pressure_strength")
        volume_box.prop(settings, "volume_target_scale")

    force_box = box.box()
    _panel_header(force_box, "Blender 力场", "FORCE_FORCE")
    force_box.label(text="重力使用当前场景设置", icon="SCENE_DATA")
    force_box.prop(settings, "use_blender_force_fields")
    if settings.use_blender_force_fields:
        force_box.prop(settings, "force_field_collection")
        force_box.prop(settings, "force_field_strength_scale")


def _draw_collision_box(layout: bpy.types.UILayout, settings) -> None:
    box = layout.box()
    _panel_header(box, "碰撞", "MOD_PHYSICS")
    box.prop(settings, "collision_margin")
    box.prop(settings, "cloth_thickness")
    box.prop(settings, "self_collision")
    if settings.self_collision:
        box.prop(settings, "self_collision_mode")

    ground_row = box.row(align=True)
    ground_row.prop(settings, "use_ground")
    if settings.use_ground:
        ground_row.prop(settings, "ground_height")
    box.prop(settings, "static_collider_collection")


def _draw_bake_box(layout: bpy.types.UILayout, settings) -> None:
    box = layout.box()
    _panel_header(box, "烘焙", "REC")

    is_baking = bool(getattr(settings, "bake_in_progress", False))
    if is_baking:
        current = int(getattr(settings, "bake_progress_current", 0))
        total = int(getattr(settings, "bake_progress_total", 0))
        percent = float(getattr(settings, "bake_progress_percent", 0.0))
        box.label(text=f"烘焙中: {current}/{total} ({percent:.0f}%)", icon="TIME")

    frame_row = box.row(align=True)
    frame_row.enabled = not is_baking
    frame_row.prop(settings, "bake_start")
    frame_row.prop(settings, "bake_end")

    action_row = box.row(align=True)
    action_row.enabled = not is_baking
    action_row.operator("ssbl.bake_xpbd_cache", text="烘焙 XPBD", icon="REC")
    action_row.operator("ssbl.clear_xpbd_cache", text="清除缓存", icon="TRASH")


def _draw_advanced_box(layout: bpy.types.UILayout, settings) -> None:
    box = layout.box()
    icon = "TRIA_DOWN" if settings.show_advanced_settings else "TRIA_RIGHT"
    box.prop(settings, "show_advanced_settings", text="高级调参", emboss=False, icon=icon)
    if not settings.show_advanced_settings:
        return

    solve_box = box.box()
    _panel_header(solve_box, "解算", "PREFERENCES")
    solve_box.prop(settings, "dt")
    solve_box.prop(settings, "substeps")
    solve_box.prop(settings, "iterations")
    solve_box.prop(settings, "damping")
    solve_box.prop(settings, "density")

    if settings.use_volume_pressure:
        volume_box = box.box()
        _panel_header(volume_box, "体积", "META_BALL")
        volume_box.prop(settings, "volume_compliance")
        volume_box.prop(settings, "volume_solve_interval")

    contact_box = box.box()
    _panel_header(contact_box, "接触", "SNAP_FACE")
    contact_box.prop(settings, "contact_friction")
    contact_box.prop(settings, "contact_tangent_damping")
    contact_box.prop(settings, "contact_compliance")

    if settings.self_collision:
        self_box = box.box()
        _panel_header(self_box, "自碰撞", "MOD_CLOTH")
        self_box.prop(settings, "self_collision_interval")
        self_box.prop(settings, "max_self_collision_neighbors")
        if str(getattr(settings, "self_collision_mode", "fast")).lower() == "fast":
            self_box.prop(settings, "fast_self_collision_passes")
        self_box.prop(settings, "self_probe_interval")
        self_box.prop(settings, "self_surface_pair_interval")
        self_box.prop(settings, "self_sleep_enabled")
        if settings.self_sleep_enabled:
            self_box.prop(settings, "self_sleep_still_frames")
            self_box.prop(settings, "self_sleep_full_scan_interval")


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

        _draw_status_box(layout, obj, settings)
        _draw_preview_box(layout, obj, settings)
        _draw_material_force_box(layout, settings)
        _draw_collision_box(layout, settings)
        _draw_bake_box(layout, settings)
        _draw_advanced_box(layout, settings)


class SSBL_PT_force_field_panel(bpy.types.Panel):
    bl_label = "SSBL 力场权重"
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
        layout.use_property_split = True
        layout.use_property_decorate = False
        box = layout.box()
        _panel_header(box, "当前力场", "FORCE_FORCE")
        box.prop(obj, "ssbl_force_field_weight", text="SSBL 权重", slider=True)
        box.label(text=f"有效强度: {strength * weight:.3f}", icon="DRIVER")
