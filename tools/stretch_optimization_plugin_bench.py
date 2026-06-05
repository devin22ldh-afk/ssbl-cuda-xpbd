import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import bpy


ADDONS_ROOT = str(Path(__file__).resolve().parents[2])
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def _load_smoke_module():
    smoke_path = Path(__file__).with_name("stretch_optimization_plugin_smoke.py")
    spec = importlib.util.spec_from_file_location("ssbl_stretch_optimization_plugin_smoke", smoke_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load smoke module from {smoke_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Run in-process stretch optimization preview benchmark.")
    parser.add_argument("--rounds", type=int, default=10)
    return parser.parse_args(argv)


def _summarize(rows, hardness):
    values = [row["step_ms"] for row in rows if abs(row["hardness"] - hardness) < 1.0e-6]
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "values": values,
    }


def main():
    args = _parse_args()
    rounds = max(1, int(args.rounds))
    smoke = _load_smoke_module()
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()
    try:
        rows = []
        for round_index in range(rounds):
            for hardness in (0.0, 0.4, 1.0):
                row = smoke._run_case(hardness)
                row["round"] = round_index
                rows.append(row)
        result = {
            "rounds": rounds,
            "cases": {
                "0.0": _summarize(rows, 0.0),
                "0.4": _summarize(rows, 0.4),
                "1.0": _summarize(rows, 1.0),
            },
        }
        print("SSBL_STRETCH_OPT_PLUGIN_BENCH", json.dumps(result, ensure_ascii=False))
    finally:
        ssbl.unregister()


if __name__ == "__main__":
    main()
