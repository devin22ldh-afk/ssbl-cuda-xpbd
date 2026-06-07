from __future__ import annotations

import re


_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_cache_stem(name: str) -> str:
    stem = _SAFE_FILENAME_PATTERN.sub("_", str(name))
    stem = stem.removeprefix("_").removesuffix("_")
    return stem or "cloth"
