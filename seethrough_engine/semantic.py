"""Canonical semantic portrait tag policy.

This module belongs to static portrait production.  Exporters and repair use
the same back-to-front order, while rig-specific subdivisions are deliberately
absent from it.
"""

from __future__ import annotations

SEMANTIC_Z_ORDER: tuple[str, ...] = (
    "body_remainder",
    "wings",
    "hair", "back hair", "hairb",
    "tail",
    "objects",
    "footwear",
    "legwear",
    "bottomwear",
    "neck",
    "topwear",
    "neckwear",
    "handwear", "handwearl", "handwearr",
    "head",
    "ears", "earl", "earr",
    "earwear",
    "face",
    "eyebrow", "eyebrowl", "eyebrowr", "browl", "browr",
    "eyewhite", "eyewhitel", "eyewhiter",
    "irides", "iridesl", "iridesr",
    "eyelash", "eyelashl", "eyelashr",
    "eyes", "eyel", "eyer",
    "nose",
    "mouth",
    "eyewear",
    "front hair", "hairf",
    "headwear",
)

_RANK = {tag: index for index, tag in enumerate(SEMANTIC_Z_ORDER)}


def semantic_rank(tag: str) -> int:
    """Back-to-front rank; unknown tags stay behind known facial layers."""
    return _RANK.get(tag, -1)


def ordered_tags(tags) -> list[str]:
    """Return tags in canonical back-to-front order."""
    return sorted(tags, key=semantic_rank)
