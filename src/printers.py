"""3D-printer presets (build volumes from the manufacturers' published specs)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PrinterSpec:
    key: str
    name: str
    x: float                     # mm; the diameter for a round bed
    y: float
    z: float
    round_bed: bool = False

    @property
    def bed(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def label(self) -> str:
        if self.round_bed:
            return f"{self.name} (Ø{self.x:.0f} × {self.z:.0f} mm, round bed)"
        return f"{self.name} ({self.x:.0f} × {self.y:.0f} × {self.z:.0f} mm)"


PRINTERS: Dict[str, PrinterSpec] = {p.key: p for p in (
    PrinterSpec("bambu_p2s", "Bambu Lab P2S", 256, 256, 256),
    PrinterSpec("prusa_xl", "Prusa XL", 360, 360, 360),
    PrinterSpec("prusa_core_one_l", "Prusa CORE One L", 300, 300, 330),
    PrinterSpec("prusa_ht90", "Prusa HT90", 300, 300, 400, round_bed=True),   # round plate: diameter 300
)}

DEFAULT_PRINTER = "bambu_p2s"
CUSTOM_LABEL = "Custom"


def get_printer(key: str) -> PrinterSpec:
    try:
        return PRINTERS[key]
    except KeyError:
        raise KeyError(f"unknown printer {key!r}; choose one of {list(PRINTERS)}") from None


def printer_labels() -> List[str]:
    return [p.label for p in PRINTERS.values()]


def printer_by_label(label: str):
    """PrinterSpec for a selectbox label, or None for 'Custom'."""
    for p in PRINTERS.values():
        if p.label == label:
            return p
    return None
