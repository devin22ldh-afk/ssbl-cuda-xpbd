from __future__ import annotations

import bpy
import blf
import time
from bpy.app.translations import pgettext_iface as iface_
from bpy_extras import view3d_utils
from mathutils import Vector
from bpy.app.handlers import persistent

from . import solver


_FPS_OVERLAY_HANDLERS: dict[str, object] = {}
_LEGACY_PREVIEW_TIMER_INTERVAL = 1.0 / 60.0
_INTERACTIVE_MONITOR_TIMER_INTERVAL = 0.5
_INTERACTIVE_MONITOR_STALE_SECONDS = 2.0
_ALT_EVENT_TYPES = {"LEFT_ALT", "RIGHT_ALT"}
_INTERACTIVE_PIN_MOUSE_EVENTS = {"LEFTMOUSE", "MIDDLEMOUSE"}


def _scene_from_handler_args(args) -> bpy.types.Scene | None:
    if args and isinstance(args[0], bpy.types.Scene):
        return args[0]
    return bpy.context.scene


def _is_animation_playing() -> bool:
    screen = getattr(bpy.context, "screen", None)
    return bool(screen is not None and getattr(screen, "is_animation_playing", False))


@persistent
def _ssbl_animation_playback_pre(*args):
    scene = _scene_from_handler_args(args)
    if scene is None:
        return
    try:
        solver.start_timeline_preview(bpy.context, scene)
        _ensure_interactive_pin_monitor(bpy.context)
        _tag_viewports(bpy.context)
    except Exception:
        pass


@persistent
def _ssbl_frame_change_post(*args):
    is_playing = _is_animation_playing()
    scene = _scene_from_handler_args(args)
    if scene is None:
        return
    try:
        if is_playing:
            solver.step_timeline_preview(bpy.context, scene)
            _tag_viewports(bpy.context)
        elif solver.reset_timeline_preview_if_endpoint(scene):
            _tag_viewports(bpy.context)
    except Exception:
        pass


@persistent
def _ssbl_animation_playback_post(*args):
    scene = _scene_from_handler_args(args)
    if scene is None:
        return
    try:
        if not solver.reset_timeline_preview_if_endpoint(scene):
            if not solver.stop_timeline_preview(scene):
                solver.pause_timeline_preview(scene)
        _tag_viewports(bpy.context)
    except Exception:
        pass


def register_playback_handlers() -> None:
    if _ssbl_animation_playback_pre not in bpy.app.handlers.animation_playback_pre:
        bpy.app.handlers.animation_playback_pre.append(_ssbl_animation_playback_pre)
    if _ssbl_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_ssbl_frame_change_post)
    if _ssbl_animation_playback_post not in bpy.app.handlers.animation_playback_post:
        bpy.app.handlers.animation_playback_post.append(_ssbl_animation_playback_post)
    register_interactive_pin_monitor()


def unregister_playback_handlers() -> None:
    unregister_interactive_pin_monitor()
    for handlers, handler in (
        (bpy.app.handlers.animation_playback_pre, _ssbl_animation_playback_pre),
        (bpy.app.handlers.frame_change_post, _ssbl_frame_change_post),
        (bpy.app.handlers.animation_playback_post, _ssbl_animation_playback_post),
    ):
        if handler in handlers:
            handlers.remove(handler)


def _draw_preview_fps(object_name: str) -> None:
    obj = bpy.data.objects.get(object_name)
    if obj is None or not solver.has_session(obj):
        return

    fps = solver.session_fps(obj)
    fps_text = iface_("Sampling") if fps <= 0.0 else f"{fps:.1f}"
    diag = solver.session_diagnostics(obj)
    font_id = 0
    blf.position(font_id, 16, 52, 0)
    blf.size(font_id, 15)
    blf.color(font_id, 0.92, 0.97, 1.0, 1.0)
    blf.draw(font_id, iface_("SSBL Preview FPS: {fps}").format(fps=fps_text))
    blf.position(font_id, 16, 32, 0)
    blf.size(font_id, 12)
    blf.color(font_id, 0.82, 0.90, 1.0, 1.0)
    blf.draw(
        font_id,
        "Frame %.1fms | CUDA %.1f | Input %.1f | DynUp %.1f | DynCol %.2f | VP %.2f"
        % (
            float(getattr(diag, "frame_ms", 0.0)),
            float(getattr(diag, "cuda_step_call_ms", 0.0)),
            float(getattr(diag, "input_refresh_ms", 0.0)),
            float(getattr(diag, "dynamic_upload_ms", 0.0)),
            float(getattr(diag, "dynamic_collision_ms", 0.0)),
            float(getattr(diag, "viewport_tag_ms", 0.0)),
        ),
    )


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


def cleanup_interactive_pin_monitor() -> None:
    solver.cleanup_interactive_pins()
    SSBL_OT_interactive_pin_monitor._shutdown_requested = True
    SSBL_OT_interactive_pin_monitor._active = False
    SSBL_OT_interactive_pin_monitor._last_event_time = 0.0


def register_interactive_pin_monitor() -> None:
    SSBL_OT_interactive_pin_monitor._shutdown_requested = False
    try:
        if not bpy.app.timers.is_registered(_interactive_pin_monitor_timer):
            bpy.app.timers.register(_interactive_pin_monitor_timer, first_interval=_INTERACTIVE_MONITOR_TIMER_INTERVAL, persistent=True)
    except Exception:
        pass


def unregister_interactive_pin_monitor() -> None:
    cleanup_interactive_pin_monitor()
    try:
        if bpy.app.timers.is_registered(_interactive_pin_monitor_timer):
            bpy.app.timers.unregister(_interactive_pin_monitor_timer)
    except Exception:
        pass


def _interactive_pin_monitor_timer():
    if SSBL_OT_interactive_pin_monitor._shutdown_requested:
        return None
    if (
        not SSBL_OT_interactive_pin_monitor._active
        or (time.perf_counter() - SSBL_OT_interactive_pin_monitor._last_event_time) > _INTERACTIVE_MONITOR_STALE_SECONDS
    ):
        _ensure_interactive_pin_monitor(bpy.context)
    return _INTERACTIVE_MONITOR_TIMER_INTERVAL


def _tag_viewports(context: bpy.types.Context) -> None:
    _tag_areas(context, {"VIEW_3D"})


def _tag_areas(context: bpy.types.Context, area_types: set[str] | None = None) -> None:
    window_manager = context.window_manager
    for window in window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area_types is None or area.type in area_types:
                area.tag_redraw()


def _has_scene_settings(context: bpy.types.Context) -> bool:
    scene = getattr(context, "scene", None)
    return scene is not None and hasattr(scene, "ssbl_preview")


def _active_mesh_object(context: bpy.types.Context) -> bpy.types.Object | None:
    obj = getattr(context, "active_object", None)
    if obj is None or getattr(obj, "type", None) != "MESH":
        return None
    return obj


def _has_active_object(context: bpy.types.Context) -> bool:
    return getattr(context, "active_object", None) is not None


def _event_region_xy(region: bpy.types.Region, event):
    mouse_region_x = getattr(event, "mouse_region_x", None)
    mouse_region_y = getattr(event, "mouse_region_y", None)
    if mouse_region_x is None or mouse_region_y is None:
        return None
    if 0 <= mouse_region_x < region.width and 0 <= mouse_region_y < region.height:
        return float(mouse_region_x), float(mouse_region_y)
    return None


def _space_region_3d(area: bpy.types.Area):
    space = next((item for item in area.spaces if item.type == "VIEW_3D"), None)
    return getattr(space, "region_3d", None)


def _ensure_interactive_pin_monitor(context: bpy.types.Context) -> None:
    if SSBL_OT_interactive_pin_monitor._active and (
        time.perf_counter() - SSBL_OT_interactive_pin_monitor._last_event_time
    ) <= _INTERACTIVE_MONITOR_STALE_SECONDS:
        return
    if SSBL_OT_interactive_pin_monitor._active:
        SSBL_OT_interactive_pin_monitor._active = False
    for window in context.window_manager.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((item for item in area.regions if item.type == "WINDOW"), None)
            if region is None:
                continue
            try:
                with context.temp_override(window=window, screen=screen, area=area, region=region):
                    bpy.ops.ssbl.interactive_pin_monitor("INVOKE_DEFAULT")
            except Exception:
                pass
            if SSBL_OT_interactive_pin_monitor._active:
                return


def _view3d_under_mouse(context: bpy.types.Context, event):
    window = getattr(context, "window", None)
    screen = getattr(window, "screen", None)
    if screen is None:
        return None
    context_area = getattr(context, "area", None)
    context_region = getattr(context, "region", None)
    if (
        context_area is not None
        and context_region is not None
        and context_area.type == "VIEW_3D"
        and context_region.type == "WINDOW"
    ):
        region_xy = _event_region_xy(context_region, event)
        rv3d = _space_region_3d(context_area)
        if region_xy is not None and rv3d is not None:
            return context_area, context_region, rv3d, region_xy
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            region_xy = _event_region_xy(region, event)
            if region_xy is None:
                if not (region.x <= event.mouse_x < region.x + region.width):
                    continue
                if not (region.y <= event.mouse_y < region.y + region.height):
                    continue
                region_xy = (event.mouse_x - region.x, event.mouse_y - region.y)
            rv3d = _space_region_3d(area)
            if rv3d is None:
                return None
            return area, region, rv3d, region_xy
    return None


def _raycast_session_vertex(context: bpy.types.Context, event):
    view = _view3d_under_mouse(context, event)
    if view is None:
        return None
    _area, region, rv3d, region_xy = view
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, region_xy)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, region_xy).normalized()
    best = None
    for object_name in solver.session_object_names(context.scene):
        obj = bpy.data.objects.get(object_name)
        if obj is None or obj.type != "MESH" or len(obj.data.polygons) == 0:
            continue
        try:
            inv = obj.matrix_world.inverted()
            origin_local = inv @ origin
            direction_local = (inv.to_3x3() @ direction).normalized()
            hit, location_local, _normal, face_index = obj.ray_cast(origin_local, direction_local, distance=1000000.0)
        except Exception:
            continue
        if not hit or face_index < 0 or face_index >= len(obj.data.polygons):
            continue
        hit_world = obj.matrix_world @ location_local
        distance = (hit_world - origin).length
        polygon = obj.data.polygons[face_index]
        vertex_index = min(
            (int(index) for index in polygon.vertices),
            key=lambda index: (obj.matrix_world @ obj.data.vertices[index].co - hit_world).length_squared,
        )
        vertex_world = obj.matrix_world @ obj.data.vertices[vertex_index].co
        if best is None or distance < best["distance"]:
            best = {
                "object_name": obj.name,
                "vertex_index": vertex_index,
                "vertex_world": vertex_world,
                "distance": distance,
            }
    return best


def _mouse_depth_location(context: bpy.types.Context, event, depth_world: Vector) -> Vector | None:
    view = _view3d_under_mouse(context, event)
    if view is None:
        return None
    _area, region, rv3d, region_xy = view
    return view3d_utils.region_2d_to_location_3d(region, rv3d, region_xy, depth_world)


class SSBL_OT_interactive_pin_monitor(bpy.types.Operator):
    bl_idname = "ssbl.interactive_pin_monitor"
    bl_label = "SSBL Interactive Pin Monitor"

    _active = False
    _shutdown_requested = False
    _last_event_time = 0.0

    _drag_object_name = ""
    _drag_depth_world = None
    _alt_down = False
    _timer = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_scene_settings(context) and getattr(context, "window", None) is not None

    def invoke(self, context: bpy.types.Context, event):
        if type(self)._active:
            return {"CANCELLED"}
        type(self)._active = True
        type(self)._shutdown_requested = False
        self._drag_object_name = ""
        self._drag_depth_world = None
        self._alt_down = False
        self._timer = context.window_manager.event_timer_add(_INTERACTIVE_MONITOR_TIMER_INTERVAL, window=context.window)
        type(self)._last_event_time = time.perf_counter()
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event):
        type(self)._last_event_time = time.perf_counter()
        if type(self)._shutdown_requested or not _has_scene_settings(context):
            self._cleanup_drag()
            self._remove_timer(context)
            type(self)._active = False
            return {"FINISHED"}

        if event.type in _ALT_EVENT_TYPES:
            self._alt_down = event.value != "RELEASE"
        alt_held = bool(getattr(event, "alt", False)) or bool(self._alt_down)

        if self._drag_object_name:
            alt_released = event.type in _ALT_EVENT_TYPES and event.value == "RELEASE"
            if event.type == "ESC" or alt_released or not alt_held:
                self._cleanup_drag()
                _tag_viewports(context)
                return {"RUNNING_MODAL"}
            if event.type == "MOUSEMOVE":
                location = _mouse_depth_location(context, event, self._drag_depth_world)
                if location is not None:
                    solver.move_interactive_pin(self._drag_object_name, location)
                    _tag_viewports(context)
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type in _INTERACTIVE_PIN_MOUSE_EVENTS and event.value == "PRESS" and alt_held:
            hit = _raycast_session_vertex(context, event)
            if hit is None:
                return {"PASS_THROUGH"}
            handle = solver.begin_interactive_pin(
                context,
                hit["object_name"],
                hit["vertex_index"],
                hit["vertex_world"],
            )
            if handle is None:
                return {"PASS_THROUGH"}
            self._drag_object_name = hit["object_name"]
            self._drag_depth_world = Vector(hit["vertex_world"])
            _tag_viewports(context)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    def _cleanup_drag(self) -> None:
        if self._drag_object_name:
            solver.end_interactive_pin(self._drag_object_name)
        self._drag_object_name = ""
        self._drag_depth_world = None
        self._alt_down = False

    def _remove_timer(self, context: bpy.types.Context) -> None:
        timer = getattr(self, "_timer", None)
        if timer is None:
            return
        try:
            context.window_manager.event_timer_remove(timer)
        except Exception:
            pass
        self._timer = None


class SSBL_OT_start_preview(bpy.types.Operator):
    bl_idname = "ssbl.start_preview"
    bl_label = "Start Preview"
    bl_description = "Run a local CUDA XPBD preview on the active cloth mesh"
    bl_options = {"REGISTER"}

    _timer = None
    _object_name = ""

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_scene_settings(context) and _active_mesh_object(context) is not None

    def invoke(self, context: bpy.types.Context, event):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, iface_("Select a mesh object first."))
            return {"CANCELLED"}

        try:
            warnings = solver.preview_warnings(obj, context.scene.ssbl_preview)
            session = solver.start_preview(context, obj)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        wm = context.window_manager
        self._timer = wm.event_timer_add(_LEGACY_PREVIEW_TIMER_INTERVAL, window=context.window)
        self._object_name = session.object_name
        wm.modal_handler_add(self)
        _ensure_interactive_pin_monitor(context)
        for warning in warnings[:3]:
            self.report({"WARNING"}, iface_(warning))
        self.report({"INFO"}, iface_("Started CUDA XPBD preview for {object_name}.").format(object_name=obj.name))
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

        viewport_started = time.perf_counter()
        _tag_viewports(context)
        obj = bpy.data.objects.get(self._object_name)
        if solver.has_session(obj):
            solver.record_viewport_tag_ms(self._object_name, (time.perf_counter() - viewport_started) * 1000.0)
        if not finished:
            return {"RUNNING_MODAL"}

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self.report({"INFO"}, iface_(solver.session_status(obj)))
        return {"FINISHED"}


class SSBL_OT_stop_preview(bpy.types.Operator):
    bl_idname = "ssbl.stop_preview"
    bl_label = "Stop Preview"
    bl_description = "Stop the local CUDA XPBD preview and restore the source mesh"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_active_object(context)

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None or not solver.request_stop(obj):
            self.report({"WARNING"}, iface_("The active object does not have a running preview."))
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, iface_("Stopped preview for {object_name}.").format(object_name=obj.name))
        return {"FINISHED"}


class SSBL_OT_reset_preview(bpy.types.Operator):
    bl_idname = "ssbl.reset_preview"
    bl_label = "Reset Preview"
    bl_description = "Restore the active object to its original state from the current preview session"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_active_object(context)

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"WARNING"}, iface_("No active object."))
            return {"CANCELLED"}
        if not solver.reset_preview_object(obj):
            self.report({"WARNING"}, iface_("The active object has no cached preview state."))
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, iface_("Reset preview for {object_name}.").format(object_name=obj.name))
        return {"FINISHED"}


class SSBL_OT_bake_xpbd_cache(bpy.types.Operator):
    bl_idname = "ssbl.bake_xpbd_cache"
    bl_label = "Bake XPBD Cache"
    bl_description = "Bake the active cloth mesh to a local PC2 cache using CUDA XPBD"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_scene_settings(context) and _active_mesh_object(context) is not None

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"ERROR"}, iface_("Select a mesh object first."))
            return {"CANCELLED"}
        wm = context.window_manager
        workspace = getattr(context, "workspace", None)
        progress_started = False

        def _update_progress(current: int, total: int) -> None:
            nonlocal progress_started
            total = max(int(total), 1)
            current = max(0, min(int(current), total))
            if not progress_started:
                wm.progress_begin(0, total)
                progress_started = True
            wm.progress_update(current)
            if workspace is not None:
                try:
                    workspace.status_text_set(
                        iface_("SSBL Baking {object_name}: {current}/{total} ({percent:.0f}%)").format(
                            object_name=obj.name,
                            current=current,
                            total=total,
                            percent=(float(current) / float(total)) * 100.0,
                        )
                    )
                except Exception:
                    pass
            _tag_areas(context)
        try:
            path = solver.bake_xpbd_cache(context, obj, progress_callback=_update_progress)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            if progress_started:
                wm.progress_end()
            if workspace is not None:
                try:
                    workspace.status_text_set(None)
                except Exception:
                    pass
            _tag_areas(context)
        _tag_viewports(context)
        self.report({"INFO"}, iface_("Baked XPBD cache: {path}").format(path=path))
        return {"FINISHED"}


class SSBL_OT_clear_xpbd_cache(bpy.types.Operator):
    bl_idname = "ssbl.clear_xpbd_cache"
    bl_label = "Clear XPBD Cache"
    bl_description = "Remove the SSBL XPBD PC2 cache binding from the active object"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return _has_active_object(context)

    def execute(self, context: bpy.types.Context):
        obj = context.active_object
        if obj is None:
            self.report({"WARNING"}, iface_("No active object."))
            return {"CANCELLED"}
        if not solver.clear_xpbd_cache(obj):
            self.report({"WARNING"}, iface_("The active object has no SSBL XPBD cache."))
            return {"CANCELLED"}
        _tag_viewports(context)
        self.report({"INFO"}, iface_("Cleared XPBD cache from {object_name}.").format(object_name=obj.name))
        return {"FINISHED"}
