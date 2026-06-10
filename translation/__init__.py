"""Blender translation helpers for the SSBL add-on."""

from __future__ import annotations

from typing import Mapping

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - allows local imports outside Blender
    bpy = None

from .zh_CN import SOURCE_TRANSLATIONS

_CONTEXTS = ("*", "Operator")
_LOCALES = ("zh_CN", "zh_HANS")
_registered_owner: str | None = None


def _default_owner() -> str:
    package_name = __package__ or __name__
    return package_name.rsplit(".", 1)[0]


def _build_context_table(source_table: Mapping[str, str]) -> dict[tuple[str, str], str]:
    context_table: dict[tuple[str, str], str] = {}
    for source_text, translated_text in source_table.items():
        for context in _CONTEXTS:
            context_table[(context, source_text)] = translated_text
    return context_table


def build_translation_dict() -> dict[str, dict[tuple[str, str], str]]:
    context_table = _build_context_table(SOURCE_TRANSLATIONS)
    return {locale: dict(context_table) for locale in _LOCALES}


def register(owner: str | None = None) -> None:
    global _registered_owner

    if bpy is None:
        return

    resolved_owner = owner or _default_owner()
    unregister(resolved_owner)
    bpy.app.translations.register(resolved_owner, build_translation_dict())
    _registered_owner = resolved_owner


def unregister(owner: str | None = None) -> None:
    global _registered_owner

    if bpy is None:
        return

    resolved_owner = owner or _registered_owner or _default_owner()
    try:
        bpy.app.translations.unregister(resolved_owner)
    except (RuntimeError, ValueError):
        pass
    if _registered_owner == resolved_owner:
        _registered_owner = None


__all__ = ["build_translation_dict", "register", "unregister"]
