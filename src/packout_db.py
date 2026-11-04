"""Milwaukee Packout case specifications and footprint helpers.

Interior sizes are Milwaukee's published figures (W x D x H, inches) and
exclude dividers/bins.  The Deep Organizer's interior W x D is taken from the
Packout dimension catalog (18" x 12") and its 5.5" depth from Milwaukee's
press release; it is flagged ``estimated``.  Measure your own case and use
Custom if it differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .assets import packout_images

IN = 25.4
GRID = 42.0
CUSTOM_KEY = "custom"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    part_no: str
    name: str
    interior_in: Tuple[float, float, float]      # (W, D, H) inches, as Milwaukee lists them
    estimated: bool = False

    @property
    def interior_mm(self) -> Tuple[float, float, float]:
        """(x = width, y = depth, z = height) in mm -- the engine's axes."""
        return tuple(v * IN for v in self.interior_in)  # type: ignore[return-value]

    @property
    def label(self) -> str:
        return f"{self.name} #{self.part_no}"

    @property
    def dims_ldh_in(self) -> Tuple[float, float, float]:
        """Length (long side) x width (short side) x height, inches."""
        w, d, h = self.interior_in
        return (max(w, d), min(w, d), h)

    @property
    def dims_ldh_mm(self) -> Tuple[float, float, float]:
        return tuple(v * IN for v in self.dims_ldh_in)  # type: ignore[return-value]


_CASE_LIST = [
    CaseSpec("8435", "48-22-8435", "Compact Organizer", (8.0, 12.0, 3.9)),
    CaseSpec("8436", "48-22-8436", "Compact Low-Profile Organizer", (8.0, 12.0, 1.8)),
    CaseSpec("8430", "48-22-8430", "Organizer", (18.0, 12.0, 3.9)),
    CaseSpec("8431", "48-22-8431", "Low-Profile Organizer", (18.0, 12.0, 1.8)),
    CaseSpec("8432", "48-22-8432", "Deep Organizer", (18.0, 12.0, 5.5), estimated=True),
]
CASES: Dict[str, CaseSpec] = {c.key: c for c in _CASE_LIST}


def case_labels() -> List[str]:
    return [c.label for c in CASES.values()]


def get_case(ident: str) -> CaseSpec:
    """Find a case by key ('8435'), part number ('48-22-8435') or label."""
    needle = ident.strip().lower()
    for c in CASES.values():
        if needle in (c.key, c.part_no.lower(), c.label.lower()):
            return c
    raise KeyError(f"unknown Packout case {ident!r}; choose one of {case_labels()}")


def footprint(interior_mm: Tuple[float, float, float], pitch: float = GRID) -> Dict[str, float]:
    """Grid cells that fit in an interior and how much of the floor they use."""
    cx, cy = int(interior_mm[0] // pitch), int(interior_mm[1] // pitch)
    used = (cx * pitch, cy * pitch)
    total = interior_mm[0] * interior_mm[1]
    return {"cells_x": cx, "cells_y": cy, "used_x": used[0], "used_y": used[1],
            "leftover_x": interior_mm[0] - used[0], "leftover_y": interior_mm[1] - used[1],
            "usable_pct": 100.0 * used[0] * used[1] / total if total else 0.0}


def thumbnail_svg(case: Optional[CaseSpec], selected: bool = False) -> str:
    """Offline SVG thumbnail for a case (``None`` = the Custom card)."""
    if case is None:
        return packout_images.custom_svg(selected)
    w, d, h = case.interior_mm
    return packout_images.organizer_svg(w, d, h, selected)
