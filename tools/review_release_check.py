from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib.util
import sys
import tomllib
import zipfile
from pathlib import Path


ADDON_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ADDON_ROOT / "blender_manifest.toml"
INIT_FILE = ADDON_ROOT / "__init__.py"
FORBIDDEN_PARTS = {".git", "__MACOSX", "__pycache__", "recordings", "dist", "ssbl_cache"}
FORBIDDEN_NAMES = {".gitignore", "findings.md", "progress.md", "task_plan.md"}
FORBIDDEN_SUFFIXES = {
    ".blend",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp4",
    ".png",
    ".webp",
}


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)


def _manifest_exclude_patterns(manifest: dict) -> list[str]:
    build = manifest.get("build", {})
    patterns = build.get("paths_exclude_pattern", [])
    if not isinstance(patterns, list):
        raise AssertionError("[build].paths_exclude_pattern must be a list")
    return [str(pattern).replace("\\", "/") for pattern in patterns]


def _directory_probes(relative_path: str, is_dir: bool) -> list[str]:
    parts = relative_path.split("/")
    max_depth = len(parts) if is_dir else max(len(parts) - 1, 0)
    return ["/".join(parts[:depth]) + "/" for depth in range(1, max_depth + 1)]


def _excluded_by_manifest(relative_path: str, is_dir: bool, patterns: list[str]) -> bool:
    probe = relative_path.replace("\\", "/")
    probes = [probe]
    dir_probes = _directory_probes(probe, is_dir)
    if is_dir and not probe.endswith("/"):
        probes.append(probe + "/")
    for pattern in patterns:
        if pattern.endswith("/") and any(fnmatch.fnmatchcase(item, pattern) for item in dir_probes):
            return True
        if any(fnmatch.fnmatchcase(item, pattern) for item in probes):
            return True
    return False


def _load_bl_info() -> dict:
    source = INIT_FILE.read_text(encoding="utf-8-sig")
    module = ast.parse(source, filename="__init__.py")
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "bl_info" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, dict):
            break
        return value
    raise AssertionError("Could not parse bl_info from __init__.py")


def _iter_package_paths(manifest: dict) -> list[str]:
    patterns = _manifest_exclude_patterns(manifest)
    paths: list[str] = []
    for path in ADDON_ROOT.rglob("*"):
        rel = path.relative_to(ADDON_ROOT).as_posix()
        if _excluded_by_manifest(rel, path.is_dir(), patterns):
            continue
        if path.is_dir():
            continue
        paths.append(rel)
    return sorted(paths)


def _assert_no_forbidden_paths(paths: list[str]) -> None:
    offenders = []
    for rel in paths:
        path = Path(rel)
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            offenders.append(rel)
        elif path.name in FORBIDDEN_NAMES:
            offenders.append(rel)
        elif path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(rel)
        elif path.parts[:2] == ("native", "bin") and rel != "native/bin/ssbl_xpbd_cuda_abi41.dll":
            offenders.append(rel)
    if offenders:
        formatted = "\n".join(f"  - {item}" for item in offenders[:40])
        raise AssertionError(f"Forbidden release paths would be included:\n{formatted}")


def check_manifest() -> dict:
    if not MANIFEST.exists():
        raise AssertionError("blender_manifest.toml is missing next to __init__.py")
    if (ADDON_ROOT / "blender_manifest.toml.txt").exists():
        raise AssertionError("Found blender_manifest.toml.txt; manifest filename must not have .txt")
    manifest = _load_manifest()
    if manifest.get("website", "").startswith("TODO"):
        print("WARNING: website still contains TODO; replace it with the GitHub Issues URL before Ready for Review.")
    permissions = manifest.get("permissions", {})
    if "files" not in permissions:
        raise AssertionError("manifest must declare [permissions].files for PC2 cache writes")
    if "network" in permissions:
        raise AssertionError("manifest must not declare network permission unless runtime network access is added")
    return manifest


def check_version_alignment(manifest: dict) -> None:
    manifest_version = str(manifest.get("version", "")).strip()
    if not manifest_version:
        raise AssertionError("manifest version must be a non-empty string")

    version_parts = manifest_version.split(".")
    try:
        manifest_tuple = tuple(int(part) for part in version_parts)
    except ValueError as exc:
        raise AssertionError(f"manifest version must be numeric dotted semver, got {manifest_version!r}") from exc

    bl_info = _load_bl_info()
    bl_info_version = bl_info.get("version")
    if not isinstance(bl_info_version, tuple) or not all(isinstance(item, int) for item in bl_info_version):
        raise AssertionError("bl_info['version'] must be an integer tuple")
    if tuple(bl_info_version) != manifest_tuple:
        raise AssertionError(
            f"Version mismatch: blender_manifest.toml has {manifest_version}, "
            f"but __init__.py bl_info has {bl_info_version}"
        )


def check_operator_polls() -> None:
    source = (ADDON_ROOT / "operators.py").read_text(encoding="utf-8-sig")
    module = ast.parse(source, filename="operators.py")
    missing = []
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(base, ast.Attribute) and base.attr == "Operator"
            for base in node.bases
        ):
            continue
        if not any(isinstance(item, ast.FunctionDef) and item.name == "poll" for item in node.body):
            missing.append(node.name)
    if missing:
        raise AssertionError(f"Operator classes missing poll(): {', '.join(missing)}")


def check_safe_cache_names() -> None:
    module_path = ADDON_ROOT / "cache_names.py"
    spec = importlib.util.spec_from_file_location("ssbl_cache_names", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    safe_cache_stem = module.safe_cache_stem

    cases = {
        "": "cloth",
        "???": "cloth",
        "'quoted cloth'": "quoted_cloth",
        "__cloth__": "_cloth_",
        "cloth:name?.mesh": "cloth_name_.mesh",
        "布料 01": "01",
    }
    for raw, expected in cases.items():
        actual = safe_cache_stem(raw)
        if actual != expected:
            raise AssertionError(f"safe_cache_stem({raw!r}) -> {actual!r}, expected {expected!r}")


def check_package_paths() -> list[str]:
    manifest = check_manifest()
    check_version_alignment(manifest)
    paths = _iter_package_paths(manifest)
    _assert_no_forbidden_paths(paths)
    required = {
        "LICENSE",
        "__init__.py",
        "blender_manifest.toml",
        "cache_names.py",
        "native/bin/ssbl_xpbd_cuda_abi41.dll",
        "native/include/ssbl_xpbd_cuda.h",
        "native/src/ssbl_xpbd_cuda_abi41.cu",
        "native/README.md",
        "translation/__init__.py",
        "translation/zh_CN.py",
    }
    missing = sorted(required.difference(paths))
    if missing:
        raise AssertionError(f"Required release paths are missing: {', '.join(missing)}")
    return paths


def build_zip(output: Path) -> None:
    paths = check_package_paths()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in paths:
            archive.write(ADDON_ROOT / rel, rel)
    print(f"Wrote {output} with {len(paths)} files")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-zip", type=Path)
    args = parser.parse_args()

    manifest = check_manifest()
    check_version_alignment(manifest)
    check_operator_polls()
    check_safe_cache_names()
    paths = check_package_paths()
    print(f"SSBL_REVIEW_RELEASE_CHECK_OK package_files={len(paths)}")
    if args.build_zip:
        build_zip(args.build_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
