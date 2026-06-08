from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


CATEGORY_PERFORMANCE = "performance"
CATEGORY_QUALITY = "quality"
CATEGORY_RESTORE_FINITE = "restore/finite"
CATEGORY_DIAGNOSTICS_PRESSURE = "diagnostics-pressure"
CATEGORY_VISUAL_STABILITY = "visual-stability"

DEFAULT_ENV_PREFIXES = ("SSBL_CS2_GATE",)
LOG_JSON_MARKERS = (
    "SSBL_CS2_SELF_INTERSECTION_REGRESSION",
    "SSBL_CS2_PERF_BENCH",
    "SSBL_SCENE_MONKEY_SUMMARY_JSON",
)

FPS_FIELDS = (
    "sim_fps_wall",
    "fps",
    "average_simulation_fps",
)
RESTORE_FIELDS = {
    "restore_delta",
    "original_mesh_max_abs_delta",
    "cloth_restore_delta",
    "sphere_restore_delta",
    "monkey_restore_delta",
}
FINITE_FIELDS = (
    "finite",
    "preview_finite",
    "all_vertex_coords_finite",
)
CONTACT_PRESSURE_FIELDS = (
    "fast_exact_vt_candidates",
    "native_fast_exact_vt_candidates",
    "fast_exact_vt_projected",
    "native_fast_exact_vt_projected",
    "fast_hard_projection_count",
    "native_fast_hard_projection_count",
    "fast_manifold_contacts",
    "native_fast_manifold_contacts",
    "fast_barrier_projected",
    "native_fast_barrier_projected",
    "fast_barrier_smoothed_vertices",
    "native_fast_barrier_smoothed_vertices",
    "fast_edge_edge_candidates",
    "native_fast_edge_edge_candidates",
    "fast_edge_edge_contacts",
    "native_fast_edge_edge_contacts",
    "fast_triangle_pair_candidates",
    "native_fast_triangle_pair_candidates",
    "fast_triangle_pair_contacts",
    "native_fast_triangle_pair_contacts",
    "fast_region_cluster_candidates",
    "native_fast_region_cluster_candidates",
    "fast_region_cluster_contacts",
    "native_fast_region_cluster_contacts",
    "fast_overlap_island_candidates",
    "native_fast_overlap_island_candidates",
    "fast_overlap_island_vertex_refs",
    "native_fast_overlap_island_vertex_refs",
    "fast_overlap_island_applied_vertices",
    "native_fast_overlap_island_applied_vertices",
    "fast_cc_overlap_seed_triangles",
    "native_fast_cc_overlap_seed_triangles",
    "fast_cc_overlap_owned_vertices",
    "native_fast_cc_overlap_owned_vertices",
    "fast_cc_overlap_applied_vertices",
    "native_fast_cc_overlap_applied_vertices",
)

SOFT_ABI41_PRESSURE_FIELDS = (
    "candidate_count",
    "native_candidate_count",
    "self_candidate_count",
    "dynamic_particle_candidate_count",
    "dynamic_particle_contacts",
    "resolved",
    "resolved_contacts",
    "native_resolved_contacts",
    "fast_soft_repulsion_candidates",
    "native_fast_soft_repulsion_candidates",
    "fast_soft_repulsion_applied",
    "native_fast_soft_repulsion_applied",
)


@dataclass(frozen=True)
class GateThresholds:
    min_fps: float = 15.0
    solver_intersection_lt: int = 300
    restore_tolerance: float = 1.0e-7
    max_resolved_contacts: int = 120_000
    max_ccd_clamp_count: int = 2_500
    max_contact_pressure: int = 200_000
    max_soft_contact_pressure: int = 3_010_000
    min_preview_frames: int = 3
    require_preview_frames: bool = False
    check_existing_frame_files: bool = True
    raw_intersection_lt: int | None = None


def _bool_value(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value
    return None


def _env_names(prefixes: tuple[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{suffix}" for prefix in prefixes)


def _float_env(prefixes: tuple[str, ...], suffix: str, default: float) -> float:
    value = _env_value(_env_names(prefixes, suffix))
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _int_env(prefixes: tuple[str, ...], suffix: str, default: int) -> int:
    value = _env_value(_env_names(prefixes, suffix))
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _optional_int_env(prefixes: tuple[str, ...], suffix: str, default: int | None) -> int | None:
    value = _env_value(_env_names(prefixes, suffix))
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_env(prefixes: tuple[str, ...], suffix: str, default: bool) -> bool:
    value = _env_value(_env_names(prefixes, suffix))
    if value is None:
        return bool(default)
    return _bool_value(value, bool(default))


def thresholds_from_env(prefixes: tuple[str, ...] = DEFAULT_ENV_PREFIXES) -> GateThresholds:
    defaults = GateThresholds()
    return GateThresholds(
        min_fps=_float_env(prefixes, "MIN_FPS", defaults.min_fps),
        solver_intersection_lt=_int_env(prefixes, "SOLVER_INTERSECTION_LT", defaults.solver_intersection_lt),
        restore_tolerance=_float_env(prefixes, "RESTORE_TOLERANCE", defaults.restore_tolerance),
        max_resolved_contacts=_int_env(prefixes, "MAX_RESOLVED_CONTACTS", defaults.max_resolved_contacts),
        max_ccd_clamp_count=_int_env(prefixes, "MAX_CCD_CLAMP_COUNT", defaults.max_ccd_clamp_count),
        max_contact_pressure=_int_env(prefixes, "MAX_CONTACT_PRESSURE", defaults.max_contact_pressure),
        max_soft_contact_pressure=_int_env(prefixes, "MAX_SOFT_CONTACT_PRESSURE", defaults.max_soft_contact_pressure),
        min_preview_frames=_int_env(prefixes, "MIN_PREVIEW_FRAMES", defaults.min_preview_frames),
        require_preview_frames=_bool_env(prefixes, "REQUIRE_PREVIEW_FRAMES", defaults.require_preview_frames),
        check_existing_frame_files=_bool_env(prefixes, "CHECK_FRAME_FILES", defaults.check_existing_frame_files),
        raw_intersection_lt=_optional_int_env(prefixes, "RAW_INTERSECTION_LT", defaults.raw_intersection_lt),
    )


def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-fps", type=float, default=None, help="Performance gate. Default/env: 15 FPS.")
    parser.add_argument(
        "--solver-intersection-lt",
        type=int,
        default=None,
        help="Quality gate; solver_relevant_intersection_count_capped must be strictly below this value.",
    )
    parser.add_argument("--restore-tolerance", type=float, default=None)
    parser.add_argument("--max-resolved-contacts", type=int, default=None)
    parser.add_argument("--max-ccd-clamp-count", type=int, default=None)
    parser.add_argument("--max-contact-pressure", type=int, default=None)
    parser.add_argument("--max-soft-contact-pressure", type=int, default=None)
    parser.add_argument("--min-preview-frames", type=int, default=None)
    parser.add_argument("--require-preview-frames", action="store_true", default=None)
    parser.add_argument("--skip-frame-file-check", action="store_true", default=False)
    parser.add_argument(
        "--raw-intersection-lt",
        type=int,
        default=None,
        help="Optional raw intersection cap. Disabled by default; solver-relevant intersections are always checked.",
    )


def thresholds_from_args(
    args: argparse.Namespace,
    base: GateThresholds | None = None,
) -> GateThresholds:
    values = asdict(base or thresholds_from_env())
    for attr in (
        "min_fps",
        "solver_intersection_lt",
        "restore_tolerance",
        "max_resolved_contacts",
        "max_ccd_clamp_count",
        "max_contact_pressure",
        "max_soft_contact_pressure",
        "min_preview_frames",
        "require_preview_frames",
        "raw_intersection_lt",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            values[attr] = value
    if getattr(args, "skip_frame_file_check", False):
        values["check_existing_frame_files"] = False
    return GateThresholds(**values)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _as_float(value: Any) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    if not _is_number(value):
        return None
    return int(value)


def _failure(
    category: str,
    field: str,
    observed: Any,
    threshold: Any,
    scope: str,
    message: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "field": field,
        "observed": observed,
        "threshold": threshold,
        "scope": scope,
        "message": message,
    }


def _update_peak(observed: dict[str, Any], key: str, value: float | int | None) -> None:
    if value is None:
        return
    previous = observed.get(key)
    if previous is None or value > previous:
        observed[key] = value


def _iter_payloads(payload: dict[str, Any], scope: str) -> list[tuple[dict[str, Any], str]]:
    if isinstance(payload.get("cases"), list):
        result = []
        for index, case in enumerate(payload["cases"]):
            if not isinstance(case, dict):
                continue
            label = f"{scope}.cases[{index}]"
            if "hardness" in case:
                label += f"(hardness={case['hardness']})"
            result.extend(_iter_payloads(case, label))
        return result
    return [(payload, scope)]


def _iter_metric_sources(payload: dict[str, Any], scope: str) -> list[tuple[dict[str, Any], str]]:
    sources = [(payload, scope)]
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, list):
        for index, item in enumerate(diagnostics):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or index)
            sources.append((item, f"{scope}.diagnostics[{label}]"))
    return sources


def _restore_field_names(source: dict[str, Any]) -> list[str]:
    names = []
    for key in source:
        if key in RESTORE_FIELDS or key.startswith("restore_delta_") or key.endswith("_restore_delta"):
            names.append(key)
    return names


def _contact_pressure(source: dict[str, Any], fields: tuple[str, ...] = CONTACT_PRESSURE_FIELDS) -> tuple[int, str | None]:
    peak_value = 0
    peak_field = None
    for field in fields:
        value = _as_int(source.get(field))
        if value is None:
            continue
        value = max(0, value)
        if value > peak_value:
            peak_value = value
            peak_field = field
    return peak_value, peak_field


def _check_finite_and_restore(
    source: dict[str, Any],
    scope: str,
    thresholds: GateThresholds,
    failures: list[dict[str, Any]],
    observed: dict[str, Any],
) -> None:
    for field in FINITE_FIELDS:
        if field in source and not bool(source[field]):
            failures.append(
                _failure(
                    CATEGORY_RESTORE_FINITE,
                    field,
                    bool(source[field]),
                    True,
                    scope,
                    f"{field} is false",
                )
            )
    finite_flag = _as_int(source.get("finite_flag"))
    if finite_flag is not None and finite_flag <= 0:
        failures.append(
            _failure(
                CATEGORY_RESTORE_FINITE,
                "finite_flag",
                finite_flag,
                "> 0",
                scope,
                "finite_flag reports non-finite state",
            )
        )

    for field in _restore_field_names(source):
        value = _as_float(source.get(field))
        if value is None:
            continue
        _update_peak(observed, "max_restore_delta", value)
        if value > thresholds.restore_tolerance:
            failures.append(
                _failure(
                    CATEGORY_RESTORE_FINITE,
                    field,
                    value,
                    f"<= {thresholds.restore_tolerance}",
                    scope,
                    f"{field} exceeds restore tolerance",
                )
            )


def _check_performance(
    source: dict[str, Any],
    scope: str,
    thresholds: GateThresholds,
    failures: list[dict[str, Any]],
    observed: dict[str, Any],
) -> None:
    fps_values = []
    for field in FPS_FIELDS:
        value = _as_float(source.get(field))
        if value is not None:
            fps_values.append((field, value))
    average_wall_step_ms = _as_float(source.get("average_wall_step_ms"))
    if average_wall_step_ms is not None and average_wall_step_ms > 0.0:
        fps_values.append(("average_wall_step_ms", 1000.0 / average_wall_step_ms))

    for field, fps in fps_values:
        if "min_fps" not in observed or fps < observed["min_fps"]:
            observed["min_fps"] = fps
        if fps < thresholds.min_fps:
            failures.append(
                _failure(
                    CATEGORY_PERFORMANCE,
                    field,
                    fps,
                    f">= {thresholds.min_fps}",
                    scope,
                    f"{field} is below the FPS gate",
                )
            )


def _check_quality(
    source: dict[str, Any],
    scope: str,
    thresholds: GateThresholds,
    failures: list[dict[str, Any]],
    observed: dict[str, Any],
) -> None:
    solver_intersections = _as_int(source.get("solver_relevant_intersection_count_capped"))
    if solver_intersections is not None:
        _update_peak(observed, "max_solver_relevant_intersection_count_capped", solver_intersections)
        if solver_intersections >= thresholds.solver_intersection_lt:
            failures.append(
                _failure(
                    CATEGORY_QUALITY,
                    "solver_relevant_intersection_count_capped",
                    solver_intersections,
                    f"< {thresholds.solver_intersection_lt}",
                    scope,
                    "solver-relevant intersections exceeded the quality gate",
                )
            )

    raw_limit = thresholds.raw_intersection_lt
    raw_intersections = _as_int(source.get("intersection_count_capped"))
    if raw_limit is not None and raw_intersections is not None:
        _update_peak(observed, "max_intersection_count_capped", raw_intersections)
        if raw_intersections >= raw_limit:
            failures.append(
                _failure(
                    CATEGORY_QUALITY,
                    "intersection_count_capped",
                    raw_intersections,
                    f"< {raw_limit}",
                    scope,
                    "raw intersections exceeded the optional raw-intersection gate",
                )
            )


def _check_diagnostics_pressure(
    source: dict[str, Any],
    scope: str,
    thresholds: GateThresholds,
    failures: list[dict[str, Any]],
    observed: dict[str, Any],
) -> None:
    ccd = _as_int(source.get("native_ccd_clamp_count"))
    if ccd is None:
        ccd = _as_int(source.get("ccd_clamp_count"))
    if ccd is not None:
        ccd = max(0, ccd)
        _update_peak(observed, "max_ccd_clamp_count", ccd)
        if ccd > thresholds.max_ccd_clamp_count:
            failures.append(
                _failure(
                    CATEGORY_DIAGNOSTICS_PRESSURE,
                    "ccd_clamp_count",
                    ccd,
                    f"<= {thresholds.max_ccd_clamp_count}",
                    scope,
                    "CCD clamp count exceeded the diagnostics-pressure gate",
                )
            )

    pressure, pressure_field = _contact_pressure(source)
    if pressure_field is not None:
        _update_peak(observed, "max_contact_pressure", pressure)
        if pressure > thresholds.max_contact_pressure:
            failures.append(
                _failure(
                    CATEGORY_DIAGNOSTICS_PRESSURE,
                    pressure_field,
                    pressure,
                    f"<= {thresholds.max_contact_pressure}",
                    scope,
                    "hard/exact contact pressure exceeded the diagnostics-pressure gate",
                )
            )

    soft_pressure, soft_pressure_field = _contact_pressure(source, SOFT_ABI41_PRESSURE_FIELDS)
    if soft_pressure_field is not None:
        _update_peak(observed, "max_soft_contact_pressure", soft_pressure)
        if soft_pressure > thresholds.max_soft_contact_pressure:
            failures.append(
                _failure(
                    CATEGORY_DIAGNOSTICS_PRESSURE,
                    soft_pressure_field,
                    soft_pressure,
                    f"<= {thresholds.max_soft_contact_pressure}",
                    scope,
                    "soft self-collision candidate pressure exceeded the diagnostics-pressure gate",
                )
            )

    resolved = _as_int(source.get("native_resolved_contacts"))
    if resolved is None:
        resolved = _as_int(source.get("resolved_contacts"))
    if resolved is None:
        resolved = _as_int(source.get("resolved"))
    if resolved is not None:
        resolved = max(0, resolved)
        _update_peak(observed, "max_resolved_contacts", resolved)
        hard_pressure_active = pressure_field is not None and pressure > 0
        ccd_pressure_active = ccd is not None and ccd > 0
        if resolved > thresholds.max_resolved_contacts and (hard_pressure_active or ccd_pressure_active):
            failures.append(
                _failure(
                    CATEGORY_DIAGNOSTICS_PRESSURE,
                    "resolved_contacts",
                    resolved,
                    f"<= {thresholds.max_resolved_contacts}",
                    scope,
                    "resolved hard/exact contact count exceeded the diagnostics-pressure gate",
                )
            )


def _image_paths_from_summary(preview_summary: dict[str, Any]) -> list[Path]:
    frame_paths = preview_summary.get("frame_paths")
    if isinstance(frame_paths, list):
        return [Path(str(path)) for path in frame_paths if str(path).strip()]
    frames_dir = preview_summary.get("frames_dir")
    if not frames_dir:
        return []
    directory = Path(str(frames_dir))
    if not directory.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in directory.glob("*.*") if path.suffix.lower() in suffixes)


def _check_visual_stability(
    preview_summary: dict[str, Any],
    scope: str,
    thresholds: GateThresholds,
    failures: list[dict[str, Any]],
    observed: dict[str, Any],
) -> None:
    has_preview_metadata = any(key in preview_summary for key in ("frame_paths", "frames_dir", "diagnostics"))
    if not has_preview_metadata:
        if thresholds.require_preview_frames:
            failures.append(
                _failure(
                    CATEGORY_VISUAL_STABILITY,
                    "frame_paths",
                    0,
                    f">= {thresholds.min_preview_frames}",
                    scope,
                    "preview frame metadata is required but missing",
                )
            )
        return

    image_paths = _image_paths_from_summary(preview_summary)
    observed["preview_frame_count"] = max(int(observed.get("preview_frame_count", 0)), len(image_paths))
    if len(image_paths) < thresholds.min_preview_frames:
        failures.append(
            _failure(
                CATEGORY_VISUAL_STABILITY,
                "frame_paths",
                len(image_paths),
                f">= {thresholds.min_preview_frames}",
                scope,
                "rendered preview frame metadata has too few frames",
            )
        )

    if thresholds.check_existing_frame_files and image_paths:
        missing = [str(path) for path in image_paths if not path.exists() or path.stat().st_size <= 0]
        if missing:
            failures.append(
                _failure(
                    CATEGORY_VISUAL_STABILITY,
                    "frame_paths",
                    missing[:10],
                    "existing non-empty image files",
                    scope,
                    "rendered preview metadata points at missing or empty frame files",
                )
            )

    diagnostics = preview_summary.get("diagnostics")
    if isinstance(diagnostics, list):
        for index, item in enumerate(diagnostics):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or index)
            diag_scope = f"{scope}.diagnostics[{label}]"
            if item.get("visible_get") is False or item.get("hide_viewport") is True or item.get("hide_render") is True:
                failures.append(
                    _failure(
                        CATEGORY_VISUAL_STABILITY,
                        "visibility",
                        {
                            "visible_get": item.get("visible_get"),
                            "hide_viewport": item.get("hide_viewport"),
                            "hide_render": item.get("hide_render"),
                        },
                        "visible renderable object",
                        diag_scope,
                        "preview diagnostics report a hidden or non-visible object",
                    )
                )
            for field in ("bbox_min_z", "bbox_max_z", "data_min_z", "data_max_z"):
                if field in item and _as_float(item[field]) is None:
                    failures.append(
                        _failure(
                            CATEGORY_VISUAL_STABILITY,
                            field,
                            item[field],
                            "finite numeric value",
                            diag_scope,
                            f"{field} is not finite in preview diagnostics",
                        )
                    )


def evaluate_summary(
    summary: dict[str, Any],
    preview_summary: dict[str, Any] | None = None,
    thresholds: GateThresholds | None = None,
    scope: str = "summary",
) -> dict[str, Any]:
    thresholds = thresholds or thresholds_from_env()
    failures: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}

    for payload, payload_scope in _iter_payloads(summary, scope):
        for source, source_scope in _iter_metric_sources(payload, payload_scope):
            _check_finite_and_restore(source, source_scope, thresholds, failures, observed)
            _check_performance(source, source_scope, thresholds, failures, observed)
            _check_quality(source, source_scope, thresholds, failures, observed)
            _check_diagnostics_pressure(source, source_scope, thresholds, failures, observed)
        if any(key in payload for key in ("frame_paths", "frames_dir", "diagnostics")):
            _check_visual_stability(payload, payload_scope, thresholds, failures, observed)

    if preview_summary is not None and preview_summary is not summary:
        for payload, payload_scope in _iter_payloads(preview_summary, f"{scope}.preview"):
            for source, source_scope in _iter_metric_sources(payload, payload_scope):
                _check_finite_and_restore(source, source_scope, thresholds, failures, observed)
                _check_diagnostics_pressure(source, source_scope, thresholds, failures, observed)
            _check_visual_stability(payload, payload_scope, thresholds, failures, observed)

    categories = sorted({str(failure["category"]) for failure in failures})
    return {
        "passed": not failures,
        "failure_categories": categories,
        "failures": failures,
        "thresholds": asdict(thresholds),
        "observed": observed,
    }


def evaluate_summaries(
    summaries: list[dict[str, Any]],
    preview_summaries: list[dict[str, Any]],
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or thresholds_from_env()
    aggregate_failures: list[dict[str, Any]] = []
    aggregate_observed: dict[str, Any] = {}

    if summaries:
        for index, summary in enumerate(summaries):
            preview = preview_summaries[index] if index < len(preview_summaries) else None
            result = evaluate_summary(summary, preview, thresholds, scope=f"summary[{index}]")
            aggregate_failures.extend(result["failures"])
            for key, value in result["observed"].items():
                if key == "min_fps":
                    previous = aggregate_observed.get(key)
                    if previous is None or value < previous:
                        aggregate_observed[key] = value
                else:
                    _update_peak(aggregate_observed, key, value)
    else:
        for index, preview in enumerate(preview_summaries):
            result = evaluate_summary({}, preview, thresholds, scope=f"preview[{index}]")
            aggregate_failures.extend(result["failures"])
            for key, value in result["observed"].items():
                if key == "min_fps":
                    previous = aggregate_observed.get(key)
                    if previous is None or value < previous:
                        aggregate_observed[key] = value
                else:
                    _update_peak(aggregate_observed, key, value)

    categories = sorted({str(failure["category"]) for failure in aggregate_failures})
    return {
        "passed": not aggregate_failures,
        "failure_categories": categories,
        "failures": aggregate_failures,
        "thresholds": asdict(thresholds),
        "observed": aggregate_observed,
    }


def _parse_json_from_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            return json.loads(line)
        if line.startswith("SSBL_SCENE_MONKEY_SUMMARY "):
            summary_path = Path(line.split(" ", 1)[1].strip())
            return json.loads(summary_path.read_text(encoding="utf-8"))
        for marker in LOG_JSON_MARKERS:
            prefix = marker + " "
            if line.startswith(prefix):
                return json.loads(line[len(prefix) :].strip())
    raise ValueError("Could not find a JSON summary or supported SSBL marker in input")


def load_summary(path: Path) -> dict[str, Any]:
    return _parse_json_from_text(path.read_text(encoding="utf-8"))


def _run_self_test() -> None:
    thresholds = replace(GateThresholds(), check_existing_frame_files=False)
    passing = {
        "sim_fps_wall": 16.0,
        "finite": True,
        "restore_delta": 0.0,
        "solver_relevant_intersection_count_capped": 42,
        "native_resolved_contacts": 5,
        "native_ccd_clamp_count": 0,
        "native_candidate_count": 99,
    }
    passing_preview = {
        "frame_paths": ["before.png", "mid.png", "after.png"],
        "diagnostics": [{"label": "mid", "visible_get": True, "hide_viewport": False, "hide_render": False}],
    }
    pass_result = evaluate_summary(passing, passing_preview, thresholds)
    if not pass_result["passed"]:
        raise AssertionError(f"expected passing gate, got {pass_result}")

    failing = {
        "sim_fps_wall": 10.0,
        "finite": False,
        "restore_delta": 1.0e-4,
        "solver_relevant_intersection_count_capped": 300,
        "native_resolved_contacts": thresholds.max_resolved_contacts + 1,
        "native_ccd_clamp_count": thresholds.max_ccd_clamp_count + 1,
        "native_candidate_count": thresholds.max_contact_pressure + 1,
    }
    failing_preview = {
        "frame_paths": [],
        "diagnostics": [{"label": "hidden", "visible_get": False, "hide_viewport": False, "hide_render": False}],
    }
    fail_result = evaluate_summary(failing, failing_preview, thresholds)
    expected_categories = {
        CATEGORY_PERFORMANCE,
        CATEGORY_QUALITY,
        CATEGORY_RESTORE_FINITE,
        CATEGORY_DIAGNOSTICS_PRESSURE,
        CATEGORY_VISUAL_STABILITY,
    }
    actual_categories = set(fail_result["failure_categories"])
    if actual_categories != expected_categories:
        raise AssertionError(f"expected {expected_categories}, got {actual_categories}: {fail_result}")
    print("SSBL_CS2_FAST_QUALITY_GATE_SELF_TEST_OK")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify SSBL CS2 diagnostics into a fast quality gate.")
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--preview-summary", action="append", type=Path, default=[])
    parser.add_argument("--stdin", action="store_true", help="Read one JSON summary or supported marker log from stdin.")
    parser.add_argument("--self-test", action="store_true", help="Run a pure-Python gate smoke test.")
    add_threshold_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _run_self_test()
        return 0

    base_thresholds = thresholds_from_env()
    thresholds = thresholds_from_args(args, base_thresholds)
    summaries = [load_summary(path) for path in args.summary]
    preview_summaries = [load_summary(path) for path in args.preview_summary]
    if args.stdin:
        summaries.append(_parse_json_from_text(sys.stdin.read()))
    if not summaries and not preview_summaries:
        raise SystemExit("Provide --summary, --preview-summary, --stdin, or --self-test.")

    result = evaluate_summaries(summaries, preview_summaries, thresholds)
    print("SSBL_CS2_FAST_QUALITY_GATE " + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
