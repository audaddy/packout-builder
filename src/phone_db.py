"""Phone model database (physical size + recommended slot clearance).

Dimensions are millimetres in (length, width, thickness) order.  The built-in
table can be extended at runtime with :func:`register_phone` or from a JSON
file (see :func:`load_phones_json`); ``assets/phones_custom.json`` is loaded
automatically when present.

JSON format::

    [{"name": "Google Pixel 8", "brand": "Google",
      "length": 150.5, "width": 70.8, "thickness": 8.9, "clearance": 0.6}]
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_CLEARANCE = 0.6
CUSTOM_PHONES_JSON = Path(__file__).resolve().parent.parent / "assets" / "phones_custom.json"


@dataclass(frozen=True)
class PhoneSpec:
    key: str
    name: str
    brand: str
    length: float
    width: float
    thickness: float
    clearance: float = DEFAULT_CLEARANCE

    @property
    def dims(self) -> Tuple[float, float, float]:
        """(length, width, thickness) in mm."""
        return (self.length, self.width, self.thickness)

    def __post_init__(self):
        if min(self.length, self.width, self.thickness) <= 0:
            raise ValueError(f"{self.name}: dimensions must be positive")
        if not 0 <= self.clearance <= 5:
            raise ValueError(f"{self.name}: clearance must be 0-5 mm")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


PHONES: Dict[str, PhoneSpec] = {}


def register_phone(name: str, length: float, width: float, thickness: float,
                   clearance: float = DEFAULT_CLEARANCE, brand: str = "Custom",
                   key: Optional[str] = None) -> PhoneSpec:
    """Add (or replace) a phone model and return its spec."""
    spec = PhoneSpec(key or _slug(name), name, brand, float(length), float(width),
                     float(thickness), float(clearance))
    PHONES[spec.key] = spec
    return spec


def load_phones_json(path) -> int:
    """Register every phone in a JSON file; returns how many were added."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError("phone JSON must be a list of objects")
    for r in rows:
        register_phone(r["name"], r["length"], r["width"], r["thickness"],
                       r.get("clearance", DEFAULT_CLEARANCE), r.get("brand", "Custom"), r.get("key"))
    return len(rows)


def list_phones() -> List[str]:
    """Display names in registration order."""
    return [p.name for p in PHONES.values()]


def get_phone(name: str) -> PhoneSpec:
    """Look a phone up by display name or key (case-insensitive)."""
    needle = name.strip().lower()
    for p in PHONES.values():
        if needle in (p.key, p.name.lower()):
            return p
    close = difflib.get_close_matches(needle, [p.name.lower() for p in PHONES.values()], n=3)
    raise KeyError(f"unknown phone {name!r}" + (f"; did you mean {close}?" if close else ""))


# Samsung Galaxy S23 family (L x W x T, mm)
register_phone("Samsung Galaxy S23", 146.3, 70.9, 7.6, brand="Samsung", key="s23")
register_phone("Samsung Galaxy S23+", 157.8, 76.2, 7.6, brand="Samsung", key="s23_plus")
register_phone("Samsung Galaxy S23 Ultra", 163.4, 78.1, 8.9, brand="Samsung", key="s23_ultra")

if CUSTOM_PHONES_JSON.exists():
    try:
        load_phones_json(CUSTOM_PHONES_JSON)
    except Exception as exc:  # a bad user file must not break the app
        log.warning("ignoring %s: %s", CUSTOM_PHONES_JSON, exc)
