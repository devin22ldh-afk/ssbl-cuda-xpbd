from __future__ import annotations

import bpy
from bpy.app.translations import pgettext_iface as iface_

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
            row.prop(settings, "enabled", text="SSBL Cloth Simulation On", toggle=True, icon="PLAY")
        else:
            row.prop(settings, "enabled", text="Enable SSBL Cloth Simulation", toggle=True, icon="PHYSICS")

        if settings.enabled:
            layout.separator()
            
            # Status Dashboard - Awesome Design: Boxed logical grouping
            col = layout.column(align=True)
            col.label(text="Status", icon="STATUSBAR")
            status_box = col.box()
            box_col = status_box.column(align=True)
            
            # Status display
            status_str = iface_("Status: {status}").format(status=iface_(solver.session_status(obj)))
            fps_str = iface_(" | FPS: {fps}").format(fps=f"{solver.session_fps(obj):.1f}") if solver.has_session(obj) else ""
            
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
            row_backend.label(text=iface_(solver.backend_status_text()), icon="CONSOLE")

            box_col.separator(factor=0.5)
            row_reset = box_col.row()
            row_reset.operator("ssbl.reset_preview", text="Reset Engine State", icon="LOOP_BACK")


class SSBL_PT_material(bpy.types.Panel):
    bl_label = "Physics & Material"
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
        col.label(text="Base Properties", icon="MOD_CLOTH")
        box_base = col.box()
        box_base.prop(settings, "hardness", text="Bend Stiffness", slider=True)
        box_base.prop(settings, "cloth_thickness", text="Surface Thickness")
        box_base.prop(settings, "density", text="Material Density")
        
        layout.separator()

        # Constraints
        col = layout.column(align=True)
        col.label(text="Spatial Constraints", icon="PINNED")
        box_pin = col.box()
        box_pin.prop(settings, "pin_vertex_group", text="Pin Vertex Group")
        box_pin.prop(settings, "pin_hardness", text="Pin Stiffness", slider=True)
        
        layout.separator()

        # Inflation
        col = layout.column(align=True)
        col.label(text="Pressure & Volume", icon="OUTLINER_OB_FORCE_FIELD")
        box_pressure = col.box()
        box_pressure.prop(settings, "use_volume_pressure", text="Enable Volume Pressure", toggle=True)
        if settings.use_volume_pressure:
            sub = box_pressure.column(align=True)
            sub.separator()
            sub.prop(settings, "pressure_strength", text="Internal Pressure")
            if settings.pressure_strength >= 0.2:
                sub.label(text="High pressure: use cautiously on dense meshes", icon="ERROR")


class SSBL_PT_collision(bpy.types.Panel):
    bl_label = "Environment & Collision"
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
        col.label(text="Scene Interaction", icon="WORLD")
        box_world = col.box()
        box_world.prop(settings, "collision_margin", text="Collision Margin")
        
        box_world.separator()
        box_world.prop(settings, "use_ground", text="Infinite Ground Plane", toggle=True)
        if settings.use_ground:
            sub = box_world.column(align=True)
            sub.prop(settings, "ground_height", text="Ground Height")
            
        box_world.separator()
        box_world.prop(settings, "static_collider_collection", text="Static Collider Collection")
        
        box_world.prop(settings, "dynamic_collider_collection", text="Dynamic Collider Collection")

        # Static SDF Settings
        if settings.static_collider_collection:
            sdf_col = box_world.column(align=True)
            sdf_col.separator()
            sdf_col.label(text="Voxel SDF Settings:", icon="MESH_GRID")
            sdf_col.prop(settings, "static_sdf_voxel_size", text="Voxel Size")
            sdf_col.prop(settings, "static_sdf_band_voxels", text="Band Width")
            sdf_col.prop(settings, "static_sdf_max_resolution", text="Max Resolution")

        layout.separator()

        # Self Collision
        col_self = layout.column(align=True)
        col_self.label(text="Self-Collision", icon="OUTLINER_OB_MESH")
        box_self = col_self.box()
        box_self.prop(settings, "self_collision", text="Enable Self-Collision", toggle=True)
        if settings.self_collision:
            sub_self = box_self.column(align=True)
            sub_self.separator()
            sub_self.prop(settings, "self_collision_mode", text="Mode")
            sub_self.prop(settings, "self_collision_distance", text="Self-Collision Distance")
            if str(getattr(settings, "self_collision_mode", "fast")).lower() == "fast":
                sub_self.prop(settings, "fast_self_collision_passes", text="Fast Passes")
            sub_self.prop(settings, "max_self_collision_neighbors", text="Max Neighbor Checks")


class SSBL_PT_force_fields(bpy.types.Panel):
    bl_label = "Force Fields"
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
        col.label(text="Effectors", icon="FORCE_FORCE")
        box = col.box()
        box.prop(settings, "force_field_collection", text="Effector Collection")
        
        box.separator()
        flow = box.grid_flow(row_major=True, columns=0, even_columns=True, even_rows=False, align=True)
        for group in visible_force_field_weight_groups():
            flow_col = flow.column()
            for prop_name in group:
                flow_col.prop(settings, prop_name, slider=True)


class SSBL_PT_advanced(bpy.types.Panel):
    bl_label = "Solver Tuning"
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
        col.label(text="Solver Core", icon="MOD_BUILD")
        box_solver = col.box()
        box_solver.prop(settings, "dt", text="Time Step")
        box_solver.prop(settings, "substeps", text="Substeps")
        box_solver.prop(settings, "iterations", text="Constraint Iterations")
        box_solver.prop(settings, "damping", text="Global Damping")

        layout.separator()

        # Contact Dynamics
        col_contact = layout.column(align=True)
        col_contact.label(text="Contact Dynamics", icon="GRAPH")
        box_contact = col_contact.box()
        box_contact.prop(settings, "contact_friction", text="Surface Friction")
        box_contact.prop(settings, "contact_tangent_damping", text="Tangential Damping")
        box_contact.prop(settings, "contact_compliance", text="Contact Compliance")

        if settings.self_collision:
            layout.separator()
            # Advanced Self Collision
            col_adv = layout.column(align=True)
            col_adv.label(text="Advanced Self-Collision", icon="TOOL_SETTINGS")
            box_adv_self = col_adv.box()
            box_adv_self.prop(settings, "self_collision_interval", text="Intersection Check Interval")
            box_adv_self.prop(settings, "self_probe_interval", text="Broadphase Probe Interval")
            box_adv_self.prop(settings, "self_surface_pair_interval", text="Surface Pair Refresh Interval")


class SSBL_PT_cache(bpy.types.Panel):
    bl_label = "Cache & Bake"
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
        col.label(text="Preview Settings", icon="PLAY")
        box_preview = col.box()
        box_preview.prop(settings, "preview_target_fps", text="Target Preview FPS")
        box_preview.prop(settings, "auto_cache_realtime", text="Auto-Cache Real-Time Simulation")
        box_preview.prop(settings, "use_evaluated_mesh", text="Use Evaluated Mesh")
        
        layout.separator()

        # Bake Workflow
        col_bake = layout.column(align=True)
        col_bake.label(text="Bake Workflow", icon="RENDER_ANIMATION")
        box_bake = col_bake.box()
        
        if is_baking:
            current = int(getattr(settings, "bake_progress_current", 0))
            total = int(getattr(settings, "bake_progress_total", 0))
            percent = float(getattr(settings, "bake_progress_percent", 0.0))
            row_progress = box_bake.row()
            row_progress.scale_y = 1.2
            row_progress.label(
                text=iface_("Baking: {current}/{total} ({percent:.0f}%)").format(
                    current=current,
                    total=total,
                    percent=percent,
                ),
                icon="TIME",
            )

        sub_bake = box_bake.column(align=True)
        sub_bake.enabled = not is_baking
        
        row_frames = sub_bake.row(align=True)
        row_frames.prop(settings, "bake_start", text="Start Frame")
        row_frames.prop(settings, "bake_end", text="End Frame")
        
        sub_bake.separator()
        row_actions = sub_bake.row(align=True)
        row_actions.scale_y = 1.5
        row_actions.operator("ssbl.bake_xpbd_cache", text="Start Bake", icon="REC")
        row_actions.operator("ssbl.clear_xpbd_cache", text="Clear Cache", icon="TRASH")


CLASSES = (
    SSBL_PT_physics_panel,
    SSBL_PT_material,
    SSBL_PT_collision,
    SSBL_PT_force_fields,
    SSBL_PT_advanced,
    SSBL_PT_cache,
)
