"""PackoutOrganizer: plans baseplate tiling for a Milwaukee Packout organizer,
builds insert blocks and exports everything as STL/STEP.

All dimensions in millimetres.
"""
from __future__ import annotations

import io
import logging
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cadquery as cq
import numpy as np
import trimesh

from src import geometry as geo
from src.geometry import GRID
from src.packout_db import CASES
from src.phone_db import PhoneSpec, get_phone

log = logging.getLogger(__name__)

IN = 25.4

# Interior dimensions come from src/packout_db.py (Milwaukee's published specs;
# dividers/bins not accounted for).  Kept as a label -> dict map for callers.
PACKOUT_MODELS: Dict[str, Dict] = {c.label: {"interior": c.interior_mm} for c in CASES.values()}


@dataclass
class Tile:
    """A baseplate tile: cell origin (cx, cy) and size (nx, ny) in cells."""
    cx: int
    cy: int
    nx: int
    ny: int
    ix: int = 0
    iy: int = 0

    @property
    def name(self) -> str:
        return f"baseplate_r{self.iy + 1}c{self.ix + 1}_{self.nx}x{self.ny}"


@dataclass
class Item:
    """A placed (or placeable) insert block."""
    name: str
    kind: str                     # phone | stl | hub
    solid: cq.Workplane
    nx: int
    ny: int
    height: float
    pos: Optional[Tuple[int, int]] = None   # cell origin, None until placed
    manual: bool = False
    notes: List[str] = field(default_factory=list)   # shown as layout warnings
    profile: Optional[object] = None                 # shapely outline (scanned items) for SVG/DXF export


def split_cells(total: int, max_per_tile: int) -> List[int]:
    """Split ``total`` cells into the fewest tiles of <= max_per_tile, balanced."""
    if max_per_tile < 1:
        raise ValueError("printer bed is smaller than one 42 mm cell")
    n = math.ceil(total / max_per_tile)
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


class PackoutOrganizer:
    def __init__(
        self,
        model: Optional[str] = next(iter(PACKOUT_MODELS)),
        interior: Optional[Sequence[float]] = None,
        bed: Sequence[float] = (200.0, 200.0, 200.0),
        bed_round: bool = False,
        interlock: bool = False,
    ):
        if interior is not None:
            self.interior = tuple(float(v) for v in interior)
            self.model = "Custom"
        else:
            if model not in PACKOUT_MODELS:
                raise ValueError(f"unknown model {model!r}; choose one of {list(PACKOUT_MODELS)} or pass interior")
            self.interior = PACKOUT_MODELS[model]["interior"]
            self.model = model
        self.bed = tuple(float(v) for v in bed)
        self.bed_round = bool(bed_round)
        # interlock: no baseplates; every block gets a flat underside with half dovetail
        # sockets on all its cell edges, and neighbouring blocks are joined by loose clips
        self.interlock = bool(interlock)
        if self.bed_round:                       # round plate: bed[0] is the diameter
            self.bed = (self.bed[0], self.bed[0], self.bed[2])
        if min(self.interior) <= 0 or min(self.bed) <= 0:
            raise ValueError("dimensions must be positive")
        self.cells_x = int(self.interior[0] // GRID)
        self.cells_y = int(self.interior[1] // GRID)
        if self.cells_x < 1 or self.cells_y < 1:
            raise ValueError("interior is smaller than one 42 mm cell")
        self.items: List[Item] = []

    # ------------------------------------------------------------------ plan
    @property
    def leftover(self) -> Tuple[float, float]:
        return (self.interior[0] - self.cells_x * GRID, self.interior[1] - self.cells_y * GRID)

    def footprint_fits(self, w: float, h: float) -> bool:
        """Does a w x h mm part fit on the plate (rotating it 90 deg if that helps)?"""
        if self.bed_round:
            return math.hypot(w, h) <= self.bed[0] + 1e-6
        return (w <= self.bed[0] + 1e-6 and h <= self.bed[1] + 1e-6) or \
               (h <= self.bed[0] + 1e-6 and w <= self.bed[1] + 1e-6)

    @property
    def max_cells_per_tile(self) -> Tuple[int, int]:
        if not self.bed_round:
            return (int(self.bed[0] // GRID), int(self.bed[1] // GRID))
        # round plate: a tile's diagonal must fit in the circle, so pick the (x, y) limits
        # that give the fewest tiles, then the fewest dovetail seams
        top = int(self.bed[0] // GRID)
        best = None
        for mx in range(1, top + 1):
            for my in range(1, top + 1):
                if math.hypot(mx * GRID, my * GRID) > self.bed[0] + 1e-6:
                    continue
                cols, rows = len(split_cells(self.cells_x, mx)), len(split_cells(self.cells_y, my))
                score = (cols * rows, (cols - 1) * self.cells_y + (rows - 1) * self.cells_x, -(mx * my))
                if best is None or score < best[0]:
                    best = (score, (mx, my))
        if best is None:
            raise ValueError("printer bed is smaller than one 42 mm cell")
        return best[1]

    def plan_tiles(self) -> List[Tile]:
        mx, my = self.max_cells_per_tile
        cols, rows = split_cells(self.cells_x, mx), split_cells(self.cells_y, my)
        tiles, cy = [], 0
        for iy, ny in enumerate(rows):
            cx = 0
            for ix, nx in enumerate(cols):
                tiles.append(Tile(cx, cy, nx, ny, ix, iy))
                cx += nx
            cy += ny
        return tiles

    def tile_keys(self, tile: Tile) -> List[Tuple[float, float, bool]]:
        """Dovetail socket positions (tile-local) on every seam the tile touches.
        One key per cell along the seam, centred on the cell."""
        keys = []
        if tile.cx > 0:                                   # west seam
            keys += [(0.0, (j + 0.5) * GRID, False) for j in range(tile.ny)]
        if tile.cx + tile.nx < self.cells_x:              # east seam
            keys += [(tile.nx * GRID, (j + 0.5) * GRID, False) for j in range(tile.ny)]
        if tile.cy > 0:                                   # south seam
            keys += [((i + 0.5) * GRID, 0.0, True) for i in range(tile.nx)]
        if tile.cy + tile.ny < self.cells_y:              # north seam
            keys += [((i + 0.5) * GRID, tile.ny * GRID, True) for i in range(tile.nx)]
        return keys

    def seam_key_count(self) -> int:
        """Number of loose dovetail clips needed (each seam cell has one)."""
        if self.interlock:
            return self.interlock_key_count()
        cols = len(split_cells(self.cells_x, self.max_cells_per_tile[0]))
        rows = len(split_cells(self.cells_y, self.max_cells_per_tile[1]))
        return (cols - 1) * self.cells_y + (rows - 1) * self.cells_x

    def interlock_key_count(self) -> int:
        """Clips needed to join every pair of touching blocks (one per shared cell edge)
        in the current layout."""
        if any(it.pos is None for it in self.items):
            self.layout()
        owner = -np.ones((self.cells_x, self.cells_y), dtype=int)
        for k, it in enumerate(self.items):
            if it.pos is not None:
                owner[it.pos[0]:it.pos[0] + it.nx, it.pos[1]:it.pos[1] + it.ny] = k
        n = 0
        for a, b in ((owner[:-1, :], owner[1:, :]), (owner[:, :-1], owner[:, 1:])):
            n += int(((a >= 0) & (b >= 0) & (a != b)).sum())
        return n

    def build_baseplates(self) -> Dict[str, cq.Workplane]:
        if self.interlock:
            return {}
        cache: Dict[tuple, cq.Workplane] = {}
        out = {}
        for t in self.plan_tiles():
            keys = tuple(self.tile_keys(t))
            sig = (t.nx, t.ny, keys)
            if sig not in cache:
                cache[sig] = geo.create_baseplate_tile(t.nx, t.ny, keys)
            out[t.name] = cache[sig]
        return out

    # ----------------------------------------------------------------- items
    def _register(self, name: str, kind: str, built, height: float, count: int) -> List[Item]:
        solid, nx, ny = built
        if self.interlock:
            solid = geo.add_interlock_base(solid, nx, ny)
        if not self.footprint_fits(nx * GRID, ny * GRID):
            raise ValueError(f"{name}: {nx}x{ny} cells ({nx*GRID:.0f}x{ny*GRID:.0f} mm) exceeds the printer bed")
        if height > self.bed[2] + 1e-6:
            raise ValueError(f"{name}: height {height:.1f} mm exceeds bed Z {self.bed[2]:.0f} mm")
        added = []
        for i in range(count):
            label = name if count == 1 else f"{name}_{i + 1}"
            it = Item(label, kind, solid, nx, ny, height)
            self.items.append(it)
            added.append(it)
        return added

    def add_phone_slots(self, count: int = 1, phone=None, **kw) -> List[Item]:
        """``phone``: model name / PhoneSpec from the phone DB (sets dims and
        default clearance); otherwise pass ``phone_dims=`` yourself."""
        height = kw.get("height_mm", 70.0)
        name = "phone_slot"
        if phone is not None:
            spec = get_phone(phone) if isinstance(phone, str) else phone
            name = f"{spec.key}_slot"
        items = self._register(name, "phone", geo.create_phone_slot_block(phone=phone, **kw), height, count)
        spec = (get_phone(phone) if isinstance(phone, str) else phone) if phone is not None else None
        dims = spec.dims if spec else kw.get("phone_dims", geo.S23_DIMS)
        clr = kw["clearance"] if kw.get("clearance") is not None else (spec.clearance if spec else 0.6)
        top = geo.phone_rise(dims, clr, kw.get("angle", 15.0), kw.get("floor_clearance", 1.5))
        if top > self.interior[2] + 1e-6:
            for it in items:
                it.notes.append(f"a seated phone reaches {top:.0f} mm but the interior is {self.interior[2]:.0f} mm "
                                f"- the lid will not close; increase the tilt angle")
        return items

    def add_stl_insert(self, stl_path: str, name: str = "watch_charger", grid_units=None,
                       clearance_mm: float = 0.4, cable_channel: bool = False, **kw) -> List[Item]:
        """Subtract any STL from a grid block (see
        :func:`src.geometry.create_custom_stl_subtraction_block`)."""
        gx, gy = grid_units if grid_units else (None, None)
        block, nx, ny, info = geo.create_custom_stl_subtraction_block(
            stl_path, gx, gy, clearance_mm, cable_channel, return_info=True, **kw)
        items = self._register(name, "stl", (block, nx, ny), self._height(block), 1)
        if info["mode"] == "convex_hull":
            items[0].notes.append("exact STL shape could not be used; the pocket is its convex hull")
        return items

    def add_stl_on_base(self, stl_path: str, name: str = "stl_object", grid_units=None, **kw) -> List[Item]:
        """The STL itself on an integrated grid base (see
        :func:`src.geometry.create_stl_on_base`)."""
        block, nx, ny, info = geo.create_stl_on_base(stl_path, grid_units, return_info=True, **kw)
        items = self._register(name, "stl", (block, nx, ny), info["height_mm"], 1)
        if info["mode"] == "convex_hull":
            items[0].notes.append("exact STL shape could not be used; the object is its convex hull")
        return items

    def add_scanned_block(self, polygon, name: str = "scanned_tool", grid_units=None,
                          depth_mm: float = 20.0, **kw) -> List[Item]:
        """Block cut from a scanned (clearance-expanded) shapely outline, see
        :func:`src.geometry.create_scanned_contour_block`.  The centred outline is
        kept on the item so :meth:`export_zip` can write it as SVG/DXF."""
        gx, gy = grid_units if grid_units else (None, None)
        block, nx, ny, info = geo.create_scanned_contour_block(polygon, gx, gy, depth_mm, return_info=True, **kw)
        items = self._register(name, "scan", (block, nx, ny), info["height_mm"], 1)
        items[0].profile = info["profile"]
        return items

    def add_hub(self, **kw) -> List[Item]:
        built = geo.create_hub_cutout_block(**kw)
        return self._register("usbc_hub", "hub", built, self._height(built[0]), 1)

    @staticmethod
    def _height(solid: cq.Workplane) -> float:
        return geo.exact_bbox(solid)[5]

    # ---------------------------------------------------------------- layout
    def layout(self) -> List[str]:
        """Place items on the cell grid (manual positions are kept, the rest
        are first-fit packed, largest first).  Returns warning strings for
        anything that does not fit or exceeds the interior height."""
        occ = np.zeros((self.cells_x, self.cells_y), dtype=bool)
        warnings: List[str] = []

        def free(x, y, w, h):
            return (x >= 0 and y >= 0 and x + w <= self.cells_x and y + h <= self.cells_y
                    and not occ[x:x + w, y:y + h].any())

        for it in self.items:
            if it.manual and it.pos is not None:
                x, y = it.pos
                if free(x, y, it.nx, it.ny):
                    occ[x:x + it.nx, y:y + it.ny] = True
                else:
                    warnings.append(f"{it.name}: manual position {it.pos} is outside the grid or overlaps; auto-placing")
                    it.manual = False
        for it in sorted((i for i in self.items if not (i.manual and i.pos is not None)),
                         key=lambda i: -i.nx * i.ny):
            it.pos = None
            for y in range(self.cells_y):
                for x in range(self.cells_x):
                    if free(x, y, it.nx, it.ny):
                        it.pos = (x, y)
                        occ[x:x + it.nx, y:y + it.ny] = True
                        break
                if it.pos:
                    break
            if it.pos is None:
                warnings.append(f"{it.name}: {it.nx}x{it.ny} cells does not fit in the remaining {self.cells_x}x{self.cells_y} grid")
        for it in self.items:
            warnings += [f"{it.name}: {n}" for n in it.notes]
            if it.height > self.interior[2] + 1e-6:
                warnings.append(f"{it.name}: height {it.height:.0f} mm exceeds interior height {self.interior[2]:.0f} mm (lid may not close)")
        return warnings

    # ----------------------------------------------------------------- parts
    def build_parts(self, include_baseplates: bool = True, include_keys: bool = True
                    ) -> Dict[str, cq.Workplane]:
        parts: Dict[str, cq.Workplane] = {}
        if include_baseplates:
            parts.update(self.build_baseplates())
        if include_keys:
            n = self.seam_key_count()
            if n:
                key = geo.create_dovetail_key()
                for i in range(n):
                    parts[f"dovetail_key_{i + 1:02d}"] = key
        for it in self.items:
            parts[it.name] = it.solid
        return parts

    # ---------------------------------------------------------------- export
    @staticmethod
    def check_part(wp: cq.Workplane) -> Dict:
        """Validity / watertightness / size report for one part."""
        v = wp.val()
        x0, y0, z0, x1, y1, z1 = geo.exact_bbox(wp)
        mesh = PackoutOrganizer.to_mesh(wp)
        return {"valid": bool(v.isValid()), "watertight": bool(mesh.is_watertight),
                "volume_cm3": round(v.Volume() / 1000, 2),
                "size": (round(x1 - x0, 1), round(y1 - y0, 1), round(z1 - z0, 1))}

    @staticmethod
    def to_mesh(wp: cq.Workplane, tolerance: float = 0.02, angular: float = 0.2) -> trimesh.Trimesh:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.stl")
            cq.exporters.export(wp, p, tolerance=tolerance, angularTolerance=angular)
            return trimesh.load(p, force="mesh")

    def fits_bed(self, wp: cq.Workplane) -> bool:
        x0, y0, z0, x1, y1, z1 = geo.exact_bbox(wp)
        return self.footprint_fits(x1 - x0, y1 - y0) and (z1 - z0) <= self.bed[2] + 1e-6

    def export_zip(self, parts: Dict[str, cq.Workplane], fmt: str = "stl") -> bytes:
        """Zip of printable files. fmt: 'stl', 'step' or 'both'.  Scanned items
        also get their flat outline as ``profiles/<name>.svg`` and ``.dxf``."""
        fmts = ["stl", "step"] if fmt == "both" else [fmt.lower()]
        for f in fmts:
            if f not in ("stl", "step"):
                raise ValueError("fmt must be 'stl', 'step' or 'both'")
        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, wp in parts.items():
                for f in fmts:
                    p = os.path.join(d, f"{name}.{f}")
                    if f == "stl":
                        cq.exporters.export(wp, p, tolerance=0.01, angularTolerance=0.1)
                    else:
                        cq.exporters.export(wp, p)
                    z.write(p, f"{f}/{name}.{f}")
            from src import scanner
            for it in self.items:
                if it.profile is not None and it.name in parts:
                    z.writestr(f"profiles/{it.name}.svg", scanner.polygon_to_svg(it.profile))
                    z.writestr(f"profiles/{it.name}.dxf", scanner.polygon_to_dxf(it.profile))
            z.writestr("README.txt", self.summary())
        return buf.getvalue()

    def summary(self) -> str:
        lx, ly = self.leftover
        lines = [
            f"Packout insert plan -- {self.model}",
            f"Interior {self.interior[0]:.0f} x {self.interior[1]:.0f} x {self.interior[2]:.0f} mm",
            f"Grid {self.cells_x} x {self.cells_y} cells of {GRID:.0f} mm (unused: {lx:.1f} x {ly:.1f} mm)",
            (f"Print bed \u00D8{self.bed[0]:.0f} x {self.bed[2]:.0f} mm (round)" if self.bed_round
             else f"Print bed {self.bed[0]:.0f} x {self.bed[1]:.0f} x {self.bed[2]:.0f} mm"),
            (f"Baseplate-free interlocking blocks: no baseplates, {self.seam_key_count()} dovetail clip(s) join the blocks "
             f"(clips go in the underside sockets where two blocks touch)" if self.interlock
             else f"Baseplate tiles: {len(self.plan_tiles())}, dovetail keys: {self.seam_key_count()}"),
            "", "Placement (cell x, y from the front-left interior corner):",
        ]
        for it in self.items:
            lines.append(f"  {it.name}: {it.nx}x{it.ny} cells at {it.pos}, height {it.height:.1f} mm")
        return "\n".join(lines) + "\n"
