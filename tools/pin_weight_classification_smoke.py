from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

from ssbl.xpbd_core import PIN_HARD_WEIGHT_THRESHOLD, build_cloth_data


def _make_settings(pin_hardness: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        use_evaluated_mesh=False,
        pin_vertex_group="ssbl_pin",
        pin_hardness=float(pin_hardness),
        density=1.0,
        hardness=0.0,
        hardness_initialized=True,
        stretch_compliance=1.0e-6,
        bend_compliance=1.0e-4,
    )


def main() -> None:
    mesh = bpy.data.meshes.new("SSBL_PinWeightSmokeMesh")
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2), (1, 3, 2)],
    )
    mesh.update()
    obj = bpy.data.objects.new("SSBL_PinWeightSmoke", mesh)
    bpy.context.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([1], 0.375, "ADD")
    group.add([2], PIN_HARD_WEIGHT_THRESHOLD, "ADD")
    group.add([3], 1.0, "ADD")

    default_cloth = build_cloth_data(obj, _make_settings())
    default_pin_indices = [int(value) for value in default_cloth.pin_indices.tolist()]
    default_pin_weights = [round(float(value), 6) for value in default_cloth.pin_weights.tolist()]
    default_inv_mass = np.asarray(default_cloth.inv_mass, dtype=np.float32)

    expected_indices = [1, 2, 3]
    expected_weights = [0.375, 0.75, 1.0]
    if default_pin_indices != expected_indices:
        raise RuntimeError(f"Default pin indices mismatch: {default_pin_indices}")
    if default_pin_weights != expected_weights:
        raise RuntimeError(f"Default pin weights mismatch: {default_pin_weights}")
    if not (float(default_inv_mass[0]) > 0.0 and float(default_inv_mass[1]) > 0.0):
        raise RuntimeError(f"Unpinned/soft vertices should keep mass: {default_inv_mass.tolist()}")
    if not (float(default_inv_mass[2]) == 0.0 and float(default_inv_mass[3]) == 0.0):
        raise RuntimeError(f"Hard-pinned vertices should have zero inverse mass: {default_inv_mass.tolist()}")

    soft_cloth = build_cloth_data(obj, _make_settings(pin_hardness=0.5))
    soft_pin_indices = [int(value) for value in soft_cloth.pin_indices.tolist()]
    soft_pin_weights = [round(float(value), 6) for value in soft_cloth.pin_weights.tolist()]
    soft_inv_mass = np.asarray(soft_cloth.inv_mass, dtype=np.float32)
    if soft_pin_indices != expected_indices:
        raise RuntimeError(f"Soft pin indices mismatch: {soft_pin_indices}")
    if soft_pin_weights != [0.1875, 0.375, 0.5]:
        raise RuntimeError(f"Soft pin weights mismatch: {soft_pin_weights}")
    if not all(float(value) > 0.0 for value in soft_inv_mass.tolist()):
        raise RuntimeError(f"Softened pin hardness should keep all vertices simulatable: {soft_inv_mass.tolist()}")

    disabled_cloth = build_cloth_data(obj, _make_settings(pin_hardness=0.0))
    disabled_pin_indices = [int(value) for value in disabled_cloth.pin_indices.tolist()]
    disabled_pin_weights = [round(float(value), 6) for value in disabled_cloth.pin_weights.tolist()]
    disabled_inv_mass = np.asarray(disabled_cloth.inv_mass, dtype=np.float32)
    if disabled_pin_indices:
        raise RuntimeError(f"Disabled pin hardness should remove pin indices: {disabled_pin_indices}")
    if disabled_pin_weights:
        raise RuntimeError(f"Disabled pin hardness should remove pin weights: {disabled_pin_weights}")
    if not all(float(value) > 0.0 for value in disabled_inv_mass.tolist()):
        raise RuntimeError(f"Disabled pin hardness should keep all vertices simulatable: {disabled_inv_mass.tolist()}")

    print(
        "PIN_WEIGHT_CLASSIFICATION_OK "
        + json.dumps(
            {
                "default_pin_indices": default_pin_indices,
                "default_pin_weights": default_pin_weights,
                "default_inv_mass": [round(float(value), 6) for value in default_inv_mass.tolist()],
                "soft_pin_indices": soft_pin_indices,
                "soft_pin_weights": soft_pin_weights,
                "soft_inv_mass": [round(float(value), 6) for value in soft_inv_mass.tolist()],
                "disabled_pin_indices": disabled_pin_indices,
                "disabled_pin_weights": disabled_pin_weights,
                "disabled_inv_mass": [round(float(value), 6) for value in disabled_inv_mass.tolist()],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
