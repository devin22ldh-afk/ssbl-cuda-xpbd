from __future__ import annotations

import bpy
import blf

from . import solver


_FPS_OVERLAY_HANDLERS: dict[str, object] = {}


def _draw_preview_fps(object_name: str) -> None:
    obj = bpy.data.objects.get(object_name)
    if obj is None or not solver.has_session(obj):
        return

    fps = solver.session_fps(obj)
    fps_text = "采样中" if fps <= 0.0 else f"{fps:.1f}"
    font_id = 0
    blf.position(font_id, 16, 52, 0)
    blf.size(font_id, 15)
    blf.color(font_id, 0.92, 0.97, 1.0, 1.0)
    blf.draw(font_id, f"SSBL 预览 FPS：{fps_text}")


def _add_fps_overlay(object_name: str) -> None:
    if object_name in _FPS_OVERLAY_HANDLERS:
        return
    _FPS_OVERLAY_HANDLERS[object_name] = bpy.types.SpaceView3D.draw_handler_add(
        _draw_preview_fps,
        (object_name,),
        "WINDOW",
        "POST_PIXEL",
    )


def _remove_fps_overlay(object_name: str) -> None:
    handler = _FPS_OVERLAY_HANDLERS.pop(object_name, None)
    if handler is None:
        return
    bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")


def cleanup_fps_overlays() -> None:
    for object_name in list(_FPS_OVERLAY_HANDLERS.keys()):
        _remove_fps_overlay(object_name)


def _tag_viewports(context: bpy.types.Context) -> None:
    window_manager = context.window_manager
    for window in window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class SSBL_OT_start_preview(bpy.types.Operator):
    bl_idname = "ssbl.start_preview"
    bl_label = "开始预览"
    bl_description = "对当前活动的布料网格运行本地 CUDA XPBD 预览"
    bl_options = {"REGISTER"}

    _timer = None
    _object_name = ""

    def invoke(self, context: bpy.types.Context, event):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "请先选择一个网格对象")
            return {"CANCELLED"}

        try:
            warnings = solver.preview_warnings(obj, context.scene.ssbl_preview)
            session = solver.start_preview(context, obj)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        wm = context.window_manager
        interval = 1.0 / max(float(context.scene.ssbl_preview.preview_target_fps), 1.0)
        self._timer = wm.event_timer_add(interval, window=context.window)
        self._object_name = session.object_name
        _add_fps_overlay(self._object_name)
        wm.modal_handler_add(self)
        for warning in warnings[:3]:
            self.report({"WARNING"}, warning)
        self.report({"INFO"}, f"已开始 {obj.name} 的 CUDA XPBD 预览")
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event):
        if event.type == "ESC":
            obj = bpy.data.objects.get(self._object_name)
            if obj is not None:
                solver.request_stop(obj)

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            finished = solver.step_preview(context, self._object_name)
        except Exception as exc:
            finished = True
            self.report({"ERROR"}, str(exc))

        _tag_viewports(context)
        if not finished:
            return {"RUNNING_MODAL"}

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _remove_fps_overlay(self._object_name)
        obj = bpy.data.objects.get(self._object_name)
        self.report({"INFO"}, solver.session_status(obj))
        return {"FINISHED"}


class SSBL_OT_stop_preview(bpy.types.Operator):
    bl_idname = "ssbl.stop_preview"
    bl_label = "停止预览"
    bl_description = "停止本地 CUDA XPBD 预览并恢复源网格"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None or not solver.request_stop(obj):
            self.report({"WARNING"}, "当前活动对象没有正在运行的预览")
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, f"已停止 {obj.name} 的预览")
        return {"FINISHED"}


class SSBL_OT_reset_preview(bpy.types.Operator):
    bl_idname = "ssbl.reset_preview"
    bl_label = "重置预览"
    bl_description = "将活动对象从当前预览会话恢复到原始状态"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"WARNING"}, "当前没有活动对象")
            return {"CANCELLED"}
        if not solver.reset_preview_object(obj):
            self.report({"WARNING"}, "当前活动对象没有缓存的预览状态")
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, f"已重置 {obj.name} 的预览")
        return {"FINISHED"}


class SSBL_OT_bake_xpbd_cache(bpy.types.Operator):
    bl_idname = "ssbl.bake_xpbd_cache"
    bl_label = "烘焙 XPBD 缓存"
    bl_description = "使用 CUDA XPBD 将当前活动布料网格烘焙到本地 PC2 缓存"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, "请先选择一个网格对象")
            return {"CANCELLED"}
        try:
            path = solver.bake_xpbd_cache(context, obj)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, f"已烘焙 XPBD 缓存：{path}")
        return {"FINISHED"}


class SSBL_OT_clear_xpbd_cache(bpy.types.Operator):
    bl_idname = "ssbl.clear_xpbd_cache"
    bl_label = "清除 XPBD 缓存"
    bl_description = "移除活动对象上的 SSBL XPBD PC2 缓存绑定"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"WARNING"}, "当前没有活动对象")
            return {"CANCELLED"}
        if not solver.clear_xpbd_cache(obj):
            self.report({"WARNING"}, "当前活动对象没有 SSBL XPBD 缓存")
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, f"已清除 {obj.name} 上的 XPBD 缓存")
        return {"FINISHED"}
