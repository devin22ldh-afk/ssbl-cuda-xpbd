from __future__ import annotations

import bpy

from . import solver
from .xpbd_core import preview_hardness_settings


class SSBL_PT_preview_panel(bpy.types.Panel):
    bl_label = "SSBL CUDA XPBD"
    bl_idname = "SSBL_PT_preview_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SSBL"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        settings = context.scene.ssbl_preview
        obj = context.active_object
        derived = preview_hardness_settings(settings)
        diagnostics = solver.session_diagnostics(obj)

        status_box = layout.box()
        status_box.label(text=f"状态：{solver.session_status(obj)}")
        if solver.has_session(obj):
            status_box.label(text=f"预览 FPS：{solver.session_fps(obj):.1f}")
        else:
            status_box.label(text="预览 FPS：-")
        if obj is not None:
            status_box.label(text=f"对象：{obj.name}")
        status_box.label(text=solver.backend_status_text())
        status_box.label(text="适用范围：仅支持 cloth MESH，暂不支持 rod/solid/stitch/tet")

        runtime_box = layout.box()
        runtime_box.label(text="运行")
        runtime_box.prop(settings, "runtime_mode")
        runtime_box.prop(settings, "solver_preset")
        runtime_box.prop(settings, "preview_target_fps")
        runtime_box.prop(settings, "use_evaluated_mesh")
        runtime_box.prop(settings, "multi_cloth_preview")
        if settings.multi_cloth_preview:
            runtime_box.prop(settings, "cross_cloth_collision")
        runtime_box.prop(settings, "frame_count")
        runtime_box.prop(settings, "bake_start")
        runtime_box.prop(settings, "bake_end")

        if obj is not None and obj.type == "MESH":
            object_box = layout.box()
            object_box.label(text="活动对象")
            object_box.prop(obj, "ssbl_collision_layer")
            object_box.prop(obj, "ssbl_enable_cross_cloth_collision")

        sim_box = layout.box()
        sim_box.label(text="材料")
        sim_box.prop(settings, "hardness", slider=True)
        sim_box.label(text="0 = 丝绸    1 = 皮革")
        sim_box.prop(settings, "use_volume_pressure")
        if settings.use_volume_pressure:
            sim_box.prop(settings, "pressure_strength")
            sim_box.prop(settings, "volume_target_scale")
        sim_box.prop(settings, "gravity")
        sim_box.prop(settings, "pin_vertex_group")

        collision_box = layout.box()
        collision_box.label(text="碰撞")
        collision_box.prop(settings, "collision_margin")
        collision_box.prop(settings, "self_collision_mode")
        collision_box.prop(settings, "cloth_thickness")
        collision_box.prop(settings, "use_ground")
        if settings.use_ground:
            collision_box.prop(settings, "ground_height")
        collision_box.prop(settings, "use_wall")
        if settings.use_wall:
            collision_box.prop(settings, "wall_origin")
            collision_box.prop(settings, "wall_normal")
        collision_box.prop(settings, "use_sphere")
        if settings.use_sphere:
            collision_box.prop(settings, "sphere_object")
        collision_box.prop(settings, "static_collider_collection")

        preview_row = layout.row(align=True)
        preview_row.operator("ssbl.start_preview", icon="PLAY")
        preview_row.operator("ssbl.stop_preview", icon="PAUSE")
        layout.operator("ssbl.reset_preview", icon="LOOP_BACK")

        bake_row = layout.row(align=True)
        bake_row.operator("ssbl.bake_xpbd_cache", icon="REC")
        bake_row.operator("ssbl.clear_xpbd_cache", icon="TRASH")

        advanced_box = layout.box()
        icon = "TRIA_DOWN" if settings.show_advanced_settings else "TRIA_RIGHT"
        advanced_box.prop(settings, "show_advanced_settings", text="高级设置", emboss=False, icon=icon)
        if settings.show_advanced_settings:
            derived_box = advanced_box.box()
            derived_box.label(text="刚度派生项（由硬度自动计算）")
            derived_box.label(text=f"拉伸柔顺度：{derived.stretch_compliance:.3g}")
            derived_box.label(text=f"弯曲柔顺度：{derived.bend_compliance:.3g}")
            derived_box.label(text=f"当前硬度：{derived.hardness:.3f}")

            diag_box = advanced_box.box()
            diag_box.label(text="接触诊断")
            diag_box.label(text=f"step_ms：{diagnostics.step_ms:.2f}")
            diag_box.label(text=f"hash_build_ms：{diagnostics.hash_build_ms:.2f}")
            diag_box.label(text=f"candidate_count：{diagnostics.candidate_count}")
            diag_box.label(text=f"resolved_contacts：{diagnostics.resolved_contacts}")
            diag_box.label(text="min_gap：-" if diagnostics.min_gap is None else f"min_gap：{diagnostics.min_gap:.5f}")
            diag_box.label(text=f"penetration_depth：{diagnostics.penetration_depth:.5f}")
            diag_box.label(text=f"ccd_clamp_count：{diagnostics.ccd_clamp_count}")
            diag_box.label(text=f"recovery_passes：{diagnostics.recovery_passes}")
            diag_box.label(text=f"local_retry_count：{diagnostics.local_retry_count}")
            diag_box.label(text=f"finite_flag：{1 if diagnostics.finite else 0}")

            tuning_box = advanced_box.box()
            tuning_box.label(text="高级解算")
            tuning_box.prop(settings, "dt")
            tuning_box.prop(settings, "substeps")
            tuning_box.prop(settings, "iterations")
            tuning_box.prop(settings, "damping")
            tuning_box.prop(settings, "density")
            if settings.self_collision_mode != "off":
                tuning_box.prop(settings, "self_collision_interval")
                tuning_box.prop(settings, "max_self_collision_neighbors")
            if settings.use_volume_pressure:
                tuning_box.prop(settings, "volume_compliance")
