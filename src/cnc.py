"""G-code for a Shapeoko 4 XL: mill a scanned-tool pocket (and optionally cut the block out of stock).

The job is 2.5D: the flat-bottom pocket is the scanned cut-out profile (islands such as the loops of
scissors stay standing), cleared in Z levels with concentric offset passes and a ramped entry; the
optional outside cut is a through-cut of the block's rectangle with holding tabs.

Work coordinates (set these in Carbide Motion before running):
  * X0 Y0 = lower-left corner of the stock
  * Z0    = top surface of the stock
Output is plain G0/G1 (no arcs, no canned cycles), mm, absolute, feed per minute.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

INCH = 25.4


@dataclass(frozen=True)
class CncMachine:
    key: str
    name: str
    x: float          # travel, mm
    y: float
    z: float
    rapid: float = 5000.0     # mm/min, used for the time estimate only

    @property
    def label(self) -> str:
        return f"{self.name} ({self.x / INCH:.0f} x {self.y / INCH:.1f} x {self.z / INCH:.0f} in cutting area)"


# Carbide 3D: Shapeoko 4 XL cutting area 33" (X) x 17.5" (Y) x 4" (Z)
MACHINES: Dict[str, CncMachine] = {m.key: m for m in (
    CncMachine("shapeoko_4_xl", "Shapeoko 4 XL", 33 * INCH, 17.5 * INCH, 4 * INCH),
)}
DEFAULT_MACHINE = "shapeoko_4_xl"

# Starting points for a 1/4" (6.35 mm) flat endmill; scaled for other diameters.  Always test on scrap.
MATERIALS: Dict[str, Dict[str, float]] = {
    "MDF / plywood": dict(feed=1500.0, plunge=500.0, stepdown=3.0, rpm=18000),
    "HDPE / plastics": dict(feed=1500.0, plunge=500.0, stepdown=2.5, rpm=16000),
    "Hardwood": dict(feed=1000.0, plunge=400.0, stepdown=2.0, rpm=18000),
    "Foam (EVA / XPS)": dict(feed=2500.0, plunge=1000.0, stepdown=10.0, rpm=16000),
}
DEFAULT_MATERIAL = "MDF / plywood"
TOOL_SIZES = {"1/8 in (3.175 mm)": 3.175, "3 mm": 3.0, "1/4 in (6.35 mm)": 6.35, "6 mm": 6.0}


class CncError(ValueError):
    pass


def material_defaults(material: str, tool_diameter: float) -> Dict[str, float]:
    """Feed / plunge / stepdown / rpm for ``material``, scaled from the 6.35 mm baseline."""
    if material not in MATERIALS:
        raise CncError(f"unknown material {material!r}; choose one of {list(MATERIALS)}")
    base = MATERIALS[material]
    k = tool_diameter / 6.35
    return dict(feed=round(base["feed"] * min(max(k, 0.5), 1.5), -1),
                plunge=round(base["plunge"] * min(max(k, 0.5), 1.5), -1),
                stepdown=round(base["stepdown"] * min(max(k, 0.4), 1.5), 1),
                rpm=base["rpm"])


@dataclass
class CncJob:
    name: str = "scanned_tool"
    tool_diameter: float = 6.35
    stock_thickness: float = 25.4
    pocket_depth: float = 20.0
    material: str = DEFAULT_MATERIAL
    feed: Optional[float] = None          # mm/min; None = material default
    plunge: Optional[float] = None        # ramp / plunge feed, mm/min
    stepdown: Optional[float] = None      # mm per Z level
    rpm: Optional[float] = None
    stepover_pct: float = 40.0            # of the tool diameter
    cut_out: bool = True                  # through-cut the block's outline with tabs
    margin: float = 15.0                  # stock border around the block, mm
    tabs: int = 4
    tab_width: float = 8.0
    tab_height: float = 2.5
    through_extra: float = 0.3            # how far the cut-out goes into the spoilboard
    safe_z: float = 5.0                   # rapid height above the stock
    ramp_angle: float = 3.0               # degrees
    flute_length: float = 19.0            # of the bit, for the warning
    tool_change: bool = True              # emit M6 T1 (Carbide Motion pauses to confirm the bit)


@dataclass
class CncResult:
    gcode: str
    stock: Tuple[float, float, float]                  # W x H x T of the stock to cut, mm
    block_origin: Tuple[float, float]                  # block lower-left on the stock, mm
    minutes: float
    cut_length_mm: float
    layers: int
    uncut_area_mm2: float
    uncut_pct: float
    warnings: List[str] = field(default_factory=list)
    pocket_paths: List[np.ndarray] = field(default_factory=list)     # (n, 2) preview polylines
    cutout_paths: List[np.ndarray] = field(default_factory=list)
    settings: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers
def _n(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _loop_points(coords, max_seg: float = 1.5, tol: float = 0.01) -> np.ndarray:
    """Closed ring -> simplified, densified (n, 2) array whose last point equals the first."""
    ls = LineString(coords).simplify(tol)
    pts = np.asarray(ls.coords, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(d / max_seg)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * i / n)
    out = np.array(out)
    out[-1] = out[0]
    return out


class _Loop:
    """A closed toolpath with arc-length lookup (wraps around)."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        seg = np.hypot(*np.diff(pts, axis=0).T)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def walk(self, s0: float, s1: float) -> np.ndarray:
        """Points along the loop from arc length s0 to s1 (s1 may exceed the length: wraps)."""
        laps = int(math.ceil(s1 / self.length)) + 1
        ext_s = np.concatenate([self.cum] + [self.cum[1:] + k * self.length for k in range(1, laps + 1)])
        ext_x = np.concatenate([self.pts[:, 0]] + [self.pts[1:, 0]] * laps)
        ext_y = np.concatenate([self.pts[:, 1]] + [self.pts[1:, 1]] * laps)
        inner = (ext_s > s0 + 1e-9) & (ext_s < s1 - 1e-9)
        ss = np.concatenate([[s0], ext_s[inner], [s1]])
        return np.column_stack([np.interp(ss, ext_s, ext_x), np.interp(ss, ext_s, ext_y)])

    def point(self, s: float) -> np.ndarray:
        return self.walk(s, s + 1e-6)[0]


class _Program:
    def __init__(self, machine: CncMachine):
        self.lines: List[str] = []
        self.machine = machine
        self.pos = [None, None, None]
        self._f = None
        self.minutes = 0.0
        self.cut_len = 0.0

    def comment(self, text: str) -> None:
        self.lines.append("(" + text.replace("(", "[").replace(")", "]") + ")")

    def raw(self, text: str) -> None:
        self.lines.append(text)

    def rapid(self, x=None, y=None, z=None) -> None:
        words = ["G0"] + [f"{a}{_n(v)}" for a, v in (("X", x), ("Y", y), ("Z", z)) if v is not None]
        self._account(x, y, z, None)
        self.lines.append(" ".join(words))

    def cut(self, x=None, y=None, z=None, f: float = 1000.0) -> None:
        words = ["G1"] + [f"{a}{_n(v)}" for a, v in (("X", x), ("Y", y), ("Z", z)) if v is not None]
        if self._f != f:
            words.append(f"F{_n(f)}")
            self._f = f
        self._account(x, y, z, f)
        self.lines.append(" ".join(words))

    def _account(self, x, y, z, f) -> None:
        new = [x if x is not None else self.pos[0], y if y is not None else self.pos[1], z if z is not None else self.pos[2]]
        if None not in self.pos and None not in new:
            d = math.dist(self.pos, new)
            if f is None:
                self.minutes += d / self.machine.rapid
            else:
                self.minutes += d / f
                self.cut_len += d
        self.pos = new


def _levels(depth: float, stepdown: float) -> List[float]:
    n = max(1, int(math.ceil(depth / stepdown - 1e-9)))
    return [-depth * (i + 1) / n for i in range(n)]


def _cut_layer(prog: _Program, loop: _Loop, z_prev: float, z_cur: float, o: dict,
               tab_zones: Sequence[Tuple[float, float]] = (), tab_top: Optional[float] = None) -> None:
    """One Z level of one closed loop: ramp down along the loop, then a full lap at ``z_cur``."""
    P = loop.length
    dz = abs(z_prev - z_cur)
    ramp_len = min(max(dz / math.tan(math.radians(o["ramp_angle"])), 1.0), 4.0 * P)
    x0, y0 = loop.point(0.0)
    safe = o["safe_z"]
    prog.rapid(z=safe)
    prog.rapid(x=x0, y=y0)
    prog.rapid(z=z_prev + 1.0)
    prog.cut(z=z_prev, f=o["plunge"])
    ramp = loop.walk(0.0, ramp_len)
    s_ramp = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(ramp, axis=0).T))])
    for (x, y), s in zip(ramp[1:], s_ramp[1:]):
        prog.cut(x=x, y=y, z=z_prev + (z_cur - z_prev) * s / s_ramp[-1], f=o["plunge"])

    zones = [z for z in tab_zones] if (tab_top is not None and z_cur < tab_top) else []
    marks = [ramp_len] + [v for a, b in zones for v in (a, b)] + [P, P + ramp_len]
    raised = False
    for i in range(len(marks) - 1):
        seg_raised = bool(zones) and i % 2 == 1 and i <= 2 * len(zones)      # intervals (a_k, b_k) are the odd ones
        z_seg = tab_top if seg_raised else z_cur
        if seg_raised != raised:
            prog.cut(z=z_seg, f=o["plunge"])
            raised = seg_raised
        pts = loop.walk(marks[i], marks[i + 1])
        for x, y in pts[1:]:
            prog.cut(x=x, y=y, z=z_seg, f=o["feed"])
    if raised:
        prog.cut(z=z_cur, f=o["plunge"])
    prog.rapid(z=safe)


def _tab_zones(P: float, ramp_len: float, count: int, width: float) -> List[Tuple[float, float]]:
    """Tab intervals along the loop, kept clear of the ramp zone at the start."""
    lo, hi = ramp_len + width, P - width
    if count <= 0 or hi - lo < count * width * 1.5:
        return []
    step = (hi - lo) / count
    return [(lo + (i + 0.5) * step - width / 2, lo + (i + 0.5) * step + width / 2) for i in range(count)]


def _tool_rings(pocket: Polygon, r: float, step: float):
    """Tool-centre loops, outside-in (ring 1, 2, ...); the boundary ring 0 comes last as the finishing pass."""
    rings: List[List[np.ndarray]] = []
    k = 0
    while True:
        g = pocket.buffer(-(r + k * step), quad_segs=16)
        if g.is_empty:
            break
        parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        loops = []
        for part in parts:
            if part.area < 0.01:
                continue
            part = orient(part, 1.0)            # outer ring CCW, islands CW: climb milling with a clockwise spindle
            for ring in [part.exterior] + list(part.interiors):
                if ring.length >= 1.0:
                    loops.append(_loop_points(ring.coords))
        if not loops:
            break
        rings.append(loops)
        k += 1
        if k > 400:
            break
    if not rings:
        raise CncError("the bit is wider than the cut-out; use a smaller bit")
    return rings[1:] + rings[:1]


# --------------------------------------------------------------------------- main entry
def generate_gcode(profile: Polygon, block_size: Tuple[float, float], job: Optional[CncJob] = None,
                   machine: str = DEFAULT_MACHINE) -> CncResult:
    """G-code for ``profile`` (block coordinates, mm, y up; as returned by
    ``create_scanned_contour_block(...)[3]['profile']``) in a block of ``block_size`` = (W, H) mm."""
    job = job or CncJob()
    if machine not in MACHINES:
        raise CncError(f"unknown machine {machine!r}")
    m = MACHINES[machine]
    if not isinstance(profile, Polygon) or profile.is_empty:
        raise CncError("profile must be a single polygon")
    if job.tool_diameter <= 0:
        raise CncError("tool diameter must be > 0")
    d = material_defaults(job.material, job.tool_diameter)
    o = dict(feed=job.feed or d["feed"], plunge=job.plunge or d["plunge"], stepdown=job.stepdown or d["stepdown"],
             rpm=job.rpm or d["rpm"], safe_z=job.safe_z, ramp_angle=job.ramp_angle)
    if min(o["feed"], o["plunge"], o["stepdown"], o["rpm"]) <= 0:
        raise CncError("feeds, stepdown and rpm must be > 0")
    if not 5 <= job.stepover_pct <= 100:
        raise CncError("stepover must be between 5% and 100% of the tool diameter")
    if not 0.5 <= job.ramp_angle <= 20:
        raise CncError("ramp angle must be between 0.5 and 20 degrees")
    if job.pocket_depth <= 0:
        raise CncError("pocket depth must be > 0")
    if job.stock_thickness - job.pocket_depth < 1.0 - 1e-9:
        raise CncError(f"a {job.stock_thickness:g} mm stock leaves only {job.stock_thickness - job.pocket_depth:g} mm of floor "
                       f"under a {job.pocket_depth:g} mm pocket; use thicker stock or a shallower pocket (min 1 mm floor)")
    if job.stock_thickness > m.z + 1e-6:
        raise CncError(f"stock is {job.stock_thickness:g} mm thick but the {m.name} has only {m.z:.0f} mm of Z travel")

    W, H = float(block_size[0]), float(block_size[1])
    r = job.tool_diameter / 2.0
    stock = (W + 2 * job.margin, H + 2 * job.margin, job.stock_thickness)
    warnings: List[str] = []
    if stock[0] > m.x + 1e-6 or stock[1] > m.y + 1e-6:
        if stock[1] <= m.x + 1e-6 and stock[0] <= m.y + 1e-6:
            warnings.append(f"the {stock[0]:.0f} x {stock[1]:.0f} mm stock only fits the {m.name} turned 90 degrees "
                            f"(its table is {m.x:.0f} x {m.y:.0f} mm)")
        else:
            raise CncError(f"the {stock[0]:.0f} x {stock[1]:.0f} mm stock exceeds the {m.name} cutting area "
                           f"({m.x:.0f} x {m.y:.0f} mm)")
    if job.pocket_depth > job.flute_length + 1e-6:
        warnings.append(f"pocket depth {job.pocket_depth:g} mm is more than the {job.flute_length:g} mm flute length; "
                        f"use a longer bit or a shallower pocket")
    if job.cut_out and job.margin < job.tool_diameter + 2:
        raise CncError(f"margin {job.margin:g} mm is too small to cut the block out with a {job.tool_diameter:g} mm bit; "
                       f"use at least {job.tool_diameter + 2:g} mm")
    if job.cut_out and job.tool_diameter < 1.0:
        raise CncError("tool diameter too small for the cut-out")

    # place the block (and its pocket) on the stock
    from shapely import affinity
    pocket = affinity.translate(profile if profile.is_valid else profile.buffer(0), job.margin, job.margin)
    step = job.tool_diameter * job.stepover_pct / 100.0
    rings = _tool_rings(pocket, r, step)

    prog = _Program(m)
    prog.comment(f"Packout Builder - {job.name} - {m.name}")
    prog.comment(f"Stock {stock[0]:.1f} x {stock[1]:.1f} x {stock[2]:.1f} mm ({job.material})")
    prog.comment("Work zero: X0 Y0 = lower-left corner of the stock, Z0 = top of the stock")
    prog.comment(f"Block {W:.1f} x {H:.1f} mm placed at X{job.margin:g} Y{job.margin:g}")
    prog.comment(f"Bit: {job.tool_diameter:g} mm flat endmill, {o['rpm']:g} rpm, feed {o['feed']:g}, plunge {o['plunge']:g} mm/min, "
                 f"{o['stepdown']:g} mm per level")
    prog.comment("Carbide Compact Router: set the speed dial by hand; other spindles use the S word")
    prog.comment("Run a test on scrap first; feeds are starting points")
    prog.raw("G21")
    prog.raw("G90")
    prog.raw("G17")
    prog.raw("G94")
    if job.tool_change:
        prog.raw("M6 T1")
    prog.raw(f"M3 S{_n(o['rpm'])}")
    prog.rapid(z=o["safe_z"])

    levels = _levels(job.pocket_depth, o["stepdown"])
    prog.comment(f"Pocket: {len(levels)} levels to Z-{job.pocket_depth:g}")
    pocket_paths: List[np.ndarray] = []
    for li, z in enumerate(levels):
        z_prev = levels[li - 1] if li else 0.0
        prog.comment(f"Pocket level {li + 1} of {len(levels)}: Z{_n(z)}")
        for loops in rings:
            for pts in loops:
                _cut_layer(prog, _Loop(pts), z_prev, z, o)
                if li == len(levels) - 1:
                    pocket_paths.append(pts)

    cutout_paths: List[np.ndarray] = []
    if job.cut_out:
        z_bot = -(job.stock_thickness + job.through_extra)
        tab_top = z_bot + job.tab_height if job.tabs > 0 else None
        outline = orient(Polygon(box(job.margin, job.margin, job.margin + W, job.margin + H).buffer(r, quad_segs=16).exterior), -1.0)
        loop = _Loop(_loop_points(outline.exterior.coords))
        ramp_len = min(max(abs(z_bot) / len(_levels(-z_bot, o["stepdown"])) / math.tan(math.radians(o["ramp_angle"])), 1.0), 4 * loop.length)
        zones = _tab_zones(loop.length, ramp_len, job.tabs, job.tab_width)
        if job.tabs > 0 and not zones:
            warnings.append("the block outline is too short for the requested tabs; none were added")
        elif zones and job.tab_height >= job.stock_thickness:
            raise CncError("tab height must be less than the stock thickness")
        clev = _levels(-z_bot, o["stepdown"])
        prog.comment(f"Cut-out: through-cut with {len(zones)} tab(s), {len(clev)} levels to Z{_n(z_bot)}")
        for li, z in enumerate(clev):
            z_prev = clev[li - 1] if li else 0.0
            prog.comment(f"Cut-out level {li + 1} of {len(clev)}: Z{_n(z)}")
            _cut_layer(prog, loop, z_prev, z, o, zones, tab_top)
        cutout_paths.append(loop.pts)
        if zones:
            warnings.append(f"{len(zones)} tabs of {job.tab_width:g} mm hold the block in the stock; cut or snap them off and sand flush")

    prog.raw("M5")
    prog.rapid(z=o["safe_z"])
    prog.rapid(x=0, y=0)
    prog.raw("M30")

    # what the bit cannot reach (narrow gaps, sharp inside corners)
    swept = unary_union([LineString(p).buffer(r) for loops in rings for p in loops])
    uncut = pocket.difference(swept)
    uncut_area = float(uncut.area)
    pct = 100.0 * uncut_area / pocket.area
    if pct > 3.0:
        warnings.append(f"{uncut_area:.0f} mm² ({pct:.0f}%) of the outline is too narrow for a {job.tool_diameter:g} mm bit "
                        f"and stays uncut; use a smaller bit or accept that detail")

    return CncResult(
        gcode="\n".join(prog.lines) + "\n", stock=stock, block_origin=(job.margin, job.margin),
        minutes=prog.minutes, cut_length_mm=prog.cut_len, layers=len(levels), uncut_area_mm2=uncut_area, uncut_pct=pct,
        warnings=warnings, pocket_paths=pocket_paths, cutout_paths=cutout_paths,
        settings=dict(feed=o["feed"], plunge=o["plunge"], stepdown=o["stepdown"], rpm=o["rpm"]))
