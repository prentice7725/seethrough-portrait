"""Validation helpers for the producer-owned Portrait Bundle ``derived`` block."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


DERIVED_STATUSES = frozenset({"computed", "not_computed"})


def _relative_path(value: Any, *, root: Path | None = None) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"derived path must be a non-empty POSIX-relative string: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.drive:
        raise ValueError(f"derived path must be bundle-relative: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"derived path contains invalid traversal segments: {value!r}")
    if root is None:
        return candidate
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"derived path escapes bundle root: {value!r}") from error
    return resolved


def _validate_artifact(path_value: Any, *, root: Path | None, canonical_paths: set[Path]) -> str:
    resolved = _relative_path(path_value, root=root)
    normalized = str(resolved if root is not None else resolved).replace(os.sep, "/")
    comparison_path = resolved.resolve() if root is not None else resolved
    if comparison_path in canonical_paths:
        raise ValueError(f"derived artifact aliases a canonical layer: {path_value!r}")
    if root is not None and not resolved.is_file():
        raise ValueError(f"derived artifact is missing: {path_value!r}")
    return str(path_value)


def _validate_stage(stage: Any) -> str:
    if stage != "production_repaired":
        raise ValueError(
            "derived.source_stage must be 'production_repaired'"
        )
    return stage


def validate_derived_contract(
    derived: Mapping[str, Any] | None,
    *,
    root: str | os.PathLike[str] | None = None,
    canonical_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate and return a detached, normalized derived metadata block.

    ``root`` is optional so producer unit tests can validate a manifest shape
    without materializing PNGs.  Readers pass it to enforce artifact
    existence as part of the production contract.
    """
    if derived is None:
        return {}
    if not isinstance(derived, Mapping):
        raise ValueError("derived must be an object")

    root_path = Path(root).resolve() if root is not None else None
    canonical_resolved: set[Path] = set()
    for value in (canonical_paths or {}).values():
        if isinstance(value, str):
            canonical_resolved.add(_relative_path(value, root=root_path).resolve())

    result: dict[str, Any] = {
        "source_stage": _validate_stage(derived.get("source_stage")),
    }
    for name in ("left_right", "depth"):
        stage = derived.get(name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"derived.{name} must be an object")
        status = stage.get("status")
        if status not in DERIVED_STATUSES:
            raise ValueError(f"derived.{name}.status must be computed or not_computed")
        paths = stage.get("paths", {})
        if not isinstance(paths, Mapping):
            raise ValueError(f"derived.{name}.paths must be an object")
        if status == "not_computed" and paths:
            raise ValueError(f"derived.{name}.paths must be empty when not_computed")

        checked_paths: dict[str, Any] = {}
        if name == "left_right":
            for semantic, members in paths.items():
                if not isinstance(semantic, str) or not isinstance(members, Mapping):
                    raise ValueError("derived.left_right.paths must map semantic tags to objects")
                if set(members) != {"left", "right"}:
                    raise ValueError(
                        f"derived.left_right.paths[{semantic!r}] must contain left and right"
                    )
                checked_paths[semantic] = {
                    side: _validate_artifact(
                        members[side], root=root_path, canonical_paths=canonical_resolved
                    )
                    for side in ("left", "right")
                }
        else:
            for semantic, artifact in paths.items():
                if not isinstance(semantic, str):
                    raise ValueError("derived.depth.paths keys must be semantic tags")
                checked_paths[semantic] = _validate_artifact(
                    artifact, root=root_path, canonical_paths=canonical_resolved
                )
        if status == "computed" and not checked_paths:
            raise ValueError(f"derived.{name} is computed but has no artifacts")
        result[name] = {"status": status, "paths": checked_paths}
    return result


__all__ = ["DERIVED_STATUSES", "validate_derived_contract"]
