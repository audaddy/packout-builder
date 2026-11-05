"""Parametric CadQuery geometry for Milwaukee Packout inserts.

Conventions
-----------
* All units are millimetres.
* Every block is built with its footprint corner at the origin and extends to
  (nx * GRID, ny * GRID) so layouts are simple integer cell offsets.
* Blocks use a (simplified) Gridfinity foot profile so they drop into the
  baseplate tiles produced by :func:`create_baseplate_tile`.
* Pocket floors always sit above the foot (``FOOT_H`` + floor clearance) so a
  cut never exposes the separate feet underneath.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cadquery as cq
import numpy as np
import trimesh

from .phone_db import DEFAULT_CLEARANCE, PhoneSpec, get_phone

log = logging.getLogger(__name__)

GRID = 42.0            # Gridfinity pitch
CELL_CLEAR = 0.25      # per-side clearance of the foot top inside a cell
FOOT_H = 4.75          # 0.8 + 1.8 + 2.15 profile height
SLAB_H = 3.0           # baseplate slab under the pockets
KEY_DEPTH = 2.4        # dovetail socket depth (from baseplate bottom)

S23_DIMS = (146.3, 70.9, 7.6)  # Samsung Galaxy S23: length, width, thickness


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def cell_foot(grow: float = 0.0) -> cq.Workplane:
    """One Gridfinity-style foot, centred on XY, z = 0 .. FOOT_H.

    ``grow`` widens every section (used for baseplate pocket clearance).
    """
    top = GRID - 2 * CELL_CLEAR + 2 * grow
    mid = top - 2 * 2.15
    bot = mid - 2 * 0.8
    return (
        cq.Workplane("XY")
        .rect(bot, bot)
        .workplane(offset=0.8).rect(mid, mid)
        .workplane(offset=1.8).rect(mid, mid)
        .workplane(offset=2.15).rect(top, top)
        .loft(ruled=True, combine=True)
    )


def create_base_block(nx: int, ny: int, height_mm: float) -> cq.Workplane:
    """Solid nx x ny block with Gridfinity feet, total height ``height_mm``."""
    if height_mm <= FOOT_H + 1:
        raise ValueError(f"height_mm must exceed {FOOT_H + 1:.2f} (foot height + 1)")
    w = nx * GRID - 2 * CELL_CLEAR
    d = ny * GRID - 2 * CELL_CLEAR
    body = (
        cq.Workplane("XY")
        .workplane(offset=FOOT_H)
        .center(nx * GRID / 2, ny * GRID / 2)
        .rect(w, d)
        .extrude(height_mm - FOOT_H)
    )
    foot = cell_foot().val()
    for i in range(nx):
        for j in range(ny):
            body = body.union(
                cq.Workplane("XY").add(foot.translate(
                    cq.Vector((i + 0.5) * GRID, (j + 0.5) * GRID, 0))),
                clean=False,
            )
    return body.clean()


def exact_bbox(wp) -> Tuple[float, float, float, float, float, float]:
    """(xmin, ymin, zmin, xmax, ymax, zmax) from the BREP only.

    ``Shape.BoundingBox()`` silently switches to the tessellation once a shape
    has been meshed (e.g. exported), which inflates curved parts by ~2 mm.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    shape = wp.val() if isinstance(wp, cq.Workplane) else wp
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax


def _safe_fillet(wp: cq.Workplane, selector: str, r: float) -> cq.Workplane:
    try:
        return wp.edges(selector).fillet(r)
    except Exception:  # pragma: no cover - geometry dependent
        log.warning("fillet %s r=%.2f failed; skipping", selector, r)
        return wp


def _cells_for(span: float, wall: float) -> int:
    return max(1, math.ceil((span + 2 * wall) / GRID))


def _y_cylinder(x: float, z: float, radius: float, y0: float, length: float):
    """Cylinder along +Y starting at y0."""
    return cq.Workplane("XY").add(
        cq.Solid.makeCylinder(radius, length, cq.Vector(x, y0, z), cq.Vector(0, 1, 0)))


def _x_cylinder(y: float, z: float, radius: float, x0: float, length: float):
    return cq.Workplane("XY").add(
        cq.Solid.makeCylinder(radius, length, cq.Vector(x0, y, z), cq.Vector(1, 0, 0)))


# --------------------------------------------------------------------------- #
# Phone slot
# --------------------------------------------------------------------------- #
def create_phone_slot_block(
    phone_dims: Sequence[float] = S23_DIMS,
    clearance: Optional[float] = None,
    angle: float = 15.0,
    height_mm: float = 70.0,
    floor_clearance: float = 1.5,
    wall_min: float = 2.5,
    finger_scoops: bool = True,
    scoop_radius: float = 14.0,
    grid_size: Optional[Tuple[int, int]] = None,
    phone: Union[str, PhoneSpec, None] = None,
) -> Tuple[cq.Workplane, int, int]:
    """Block with one phone-shaped slot tilted ``angle`` degrees from vertical
    (0 = upright, 90 = lying flat), leaning towards +Y.

    The cutter is the real phone length, extended past its top so the slot
    always opens through the top face.  Side finger scoops are half-cylinders
    cut across the block at both slot edges.

    ``phone`` is a preset (a :class:`~src.phone_db.PhoneSpec` or a model name
    such as ``"Samsung Galaxy S23 Ultra"``); it overrides ``phone_dims`` and
    supplies the recommended clearance unless ``clearance`` is given.

    Returns ``(solid, nx, ny)``.
    """
    if phone is not None:
        spec = get_phone(phone) if isinstance(phone, str) else phone
        phone_dims = spec.dims
        if clearance is None:
            clearance = spec.clearance
    if clearance is None:
        clearance = DEFAULT_CLEARANCE
    length, width, thick = phone_dims
    if not 0 <= angle < 90:
        raise ValueError("angle must be in [0, 90)")
    W = width + 2 * clearance
    T = thick + 2 * clearance
    L = length + 2 * clearance
    floor_z = FOOT_H + floor_clearance

    cutter = cq.Workplane("XY").box(W, T, L + 200, centered=(True, True, False))
    # Only the long (vertical-axis) edges are rounded.  Filleting the slot's
    # bottom edges as well produced STLs with open edges after the tilt/cut.
    cutter = _safe_fillet(cutter, "|Z", min(2.0, T / 2 - 0.05))
    cutter = cutter.rotate((0, 0, 0), (1, 0, 0), -angle)
    bb = cutter.val().BoundingBox()
    cutter = cutter.translate((0, 0, floor_z - bb.zmin))

    # Y extent of the cutter actually inside the block's vertical range.
    clip = cq.Workplane("XY").box(10 * W, 10 * L, height_mm - floor_z, centered=(True, True, False)) \
        .translate((0, 0, floor_z))
    inside = cutter.intersect(clip).val().BoundingBox()
    span_y = inside.ymax - inside.ymin

    nx, ny = grid_size if grid_size else (_cells_for(W, wall_min), _cells_for(span_y, wall_min))
    if (nx * GRID - W) / 2 < wall_min - 1e-6 or (ny * GRID - span_y) / 2 < wall_min - 1e-6:
        raise ValueError(f"grid_size {nx}x{ny} leaves walls < {wall_min} mm for this phone/angle")

    cutter = cutter.translate((nx * GRID / 2, ny * GRID / 2 - (inside.ymin + inside.ymax) / 2, 0))
    block = create_base_block(nx, ny, height_mm).cut(cutter)

    if finger_scoops:
        for sx in (-1, 1):
            block = block.cut(_y_cylinder(nx * GRID / 2 + sx * W / 2, height_mm,
                                          scoop_radius, -1, ny * GRID + 2))
    return block.clean(), nx, ny


def phone_rise(phone_dims: Sequence[float], clearance: float, angle: float,
               floor_clearance: float = 1.5) -> float:
    """Height above z=0 reached by the top of a phone seated in its slot."""
    length, _, thick = phone_dims
    a = math.radians(angle)
    return FOOT_H + floor_clearance + (length + 2 * clearance) * math.cos(a) + (thick + 2 * clearance) * math.sin(a)


def min_tilt_for_height(phone_dims: Sequence[float], clearance: float, max_height: float,
                        floor_clearance: float = 1.5) -> Optional[int]:
    """Smallest tilt (whole degrees from vertical, 0-85) at which the seated
    phone stays below ``max_height``; ``None`` if even 85 degrees is too tall."""
    for angle in range(0, 86):
        if phone_rise(phone_dims, clearance, angle, floor_clearance) <= max_height:
            return angle
    return None


# --------------------------------------------------------------------------- #
# STL -> BREP
# --------------------------------------------------------------------------- #
def offset_mesh(mesh: trimesh.Trimesh, offset: float, max_scale: float = 2.5) -> trimesh.Trimesh:
    """Grow a mesh outward by ``offset`` mm.

    Vertices move along their smoothed normal, scaled by 1/min(n_v . n_face) so
    every adjacent face plane moves by at least ``offset`` (a miter offset);
    otherwise 90 degree edges would only gain ~0.7 * offset.
    """
    if offset == 0:
        return mesh.copy()
    out = mesh.copy()
    vn = out.vertex_normals
    fn = out.face_normals
    vf = out.vertex_faces                       # (V, K) padded with -1
    valid = vf >= 0
    dots = np.einsum("vkj,vj->vk", fn[np.where(valid, vf, 0)], vn)
    dots = np.where(valid, dots, np.inf).min(axis=1)
    dots = np.clip(np.where(np.isfinite(dots), dots, 1.0), 1.0 / max_scale, 1.0)
    out.vertices = out.vertices + vn * (offset / dots)[:, None]
    return out


def mesh_to_solid(mesh: trimesh.Trimesh) -> cq.Solid:
    """Sew a triangle mesh into a closed CadQuery solid (coplanar faces merged)."""
    faces = []
    for tri in mesh.triangles:
        if trimesh.triangles.area(tri[None])[0] < 1e-9:
            continue
        wire = cq.Wire.makePolygon([cq.Vector(*map(float, p)) for p in tri], close=True)
        faces.append(cq.Face.makeFromWires(wire))
    solid = cq.Solid.makeSolid(cq.Shell.makeShell(faces)).fix()
    if solid.Volume() < 0:
        solid = cq.Solid(solid.wrapped.Reversed())
    solid = solid.clean()
    if not solid.isValid() or solid.Volume() < 1e-3:
        raise ValueError("mesh could not be converted into a valid solid")
    return solid


def _check_rotation(rotation_deg) -> Tuple[int, int, int]:
    rot = tuple(int(r) for r in rotation_deg)
    if len(rot) != 3 or any(r % 90 for r in rot):
        raise ValueError("rotation_deg must be three multiples of 90 (degrees about X, Y, Z)")
    return rot  # type: ignore[return-value]


def _load_oriented_mesh(stl_path: str, rotation_deg=(0, 0, 0)) -> trimesh.Trimesh:
    """Load an STL, clean it up and apply X, Y, Z rotations (multiples of 90)."""
    rx, ry, rz = _check_rotation(rotation_deg)
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{stl_path} is not a usable triangle mesh")
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    for angle, axis in ((rx, [1, 0, 0]), (ry, [0, 1, 0]), (rz, [0, 0, 1])):
        if angle % 360:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle), axis))
    return mesh


def analyze_stl(stl_path: str, rotation_deg=(0, 0, 0)) -> Dict:
    """Size, watertightness and sanity warnings for an uploaded STL.

    ``size`` is the oriented bounding box (x, y, z) in mm *after* rotation.
    """
    mesh = _load_oriented_mesh(stl_path, rotation_deg)
    lo, hi = mesh.bounds
    size = tuple(float(v) for v in (hi - lo))
    watertight = bool(mesh.is_watertight)
    warnings: List[str] = []
    if not watertight:
        repaired = mesh.copy()
        trimesh.repair.fix_normals(repaired)
        trimesh.repair.fill_holes(repaired)
        if repaired.is_watertight:
            warnings.append("Mesh has holes; they will be auto-filled.")
        else:
            warnings.append("Mesh is not watertight and could not be repaired; the convex hull will be used instead of the exact shape.")
    if max(size) < 3:
        warnings.append("Model is under 3 mm - was it exported in inches or metres?")
    if max(size) > 500:
        warnings.append("Model is over 500 mm - was it exported in micrometres?")
    return {"size": size, "watertight": watertight, "faces": int(len(mesh.faces)),
            "volume_cm3": round(abs(float(mesh.volume)) / 1000, 2) if watertight else None,
            "winding_consistent": bool(mesh.is_winding_consistent), "warnings": warnings}


def suggest_grid_units(size_xy: Sequence[float], clearance_mm: float = 0.4,
                       wall_min: float = 2.5) -> Tuple[int, int]:
    """Smallest grid (nx, ny) that leaves ``wall_min`` walls around the part."""
    return (_cells_for(size_xy[0] + 2 * clearance_mm, wall_min),
            _cells_for(size_xy[1] + 2 * clearance_mm, wall_min))


def load_stl_as_solid(stl_path: str, offset: float = 0.4, max_faces: int = 2500,
                      rotation_deg=(0, 0, 0)) -> Tuple[cq.Solid, Dict[str, float]]:
    """Load an STL, centre it on XY with its bottom at z=0, offset it by
    ``offset`` mm and return a CadQuery solid + info dict.

    Falls back to the convex hull (and says so in ``info['mode']``) if the
    exact mesh cannot be converted to a valid solid.
    """
    mesh = _load_oriented_mesh(stl_path, rotation_deg)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)

    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception as exc:
            log.warning("decimation unavailable (%s); using convex hull", exc)
            mesh = mesh.convex_hull
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    grown = offset_mesh(mesh, offset)
    grown.apply_translation([0, 0, -offset])       # keep the bottom of the pocket at z=0 - offset

    mode = "mesh"
    try:
        if not grown.is_watertight:
            raise ValueError("offset mesh not watertight")
        solid = mesh_to_solid(grown)
    except Exception as exc:
        log.warning("exact mesh conversion failed (%s); falling back to convex hull", exc)
        solid = mesh_to_solid(grown.convex_hull)
        mode = "convex_hull"
    size = np.array(hi) - np.array(lo)
    return solid, {"mode": mode, "size_x": float(size[0]), "size_y": float(size[1]),
                   "size_z": float(size[2]), "faces": int(len(grown.faces))}


def stl_silhouette(stl_path: str, offset: float = 0.4, rotation_deg=(0, 0, 0)):
    """Top-down outline of an STL (union of all projected triangles), grown by
    ``offset`` mm, as ``(polygon, size_xyz)``.  The polygon is centred on the
    origin.  Extruding it straight down gives a pocket the part can always be
    dropped into, whatever overhangs or hooks it has.
    """
    import shapely
    from shapely import affinity
    from shapely.geometry import MultiPolygon

    mesh = _load_oriented_mesh(stl_path, rotation_deg)
    lo, hi = mesh.bounds
    tris = mesh.triangles[:, :, :2]
    keep = np.abs(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])) > 1e-9
    union = shapely.union_all(shapely.polygons(tris[keep]))
    if isinstance(union, MultiPolygon):
        union = union.convex_hull if len(union.geoms) > 8 else union.buffer(0.05).buffer(-0.05)
    poly = union.buffer(offset, join_style="round", quad_segs=8).simplify(0.05)
    if isinstance(poly, MultiPolygon):                  # separate islands: bridge them
        poly = poly.buffer(0.5).buffer(-0.5)
        if isinstance(poly, MultiPolygon):
            poly = poly.convex_hull
    poly = shapely.Polygon(poly.exterior)               # fill through-holes: simpler pocket, part still drops in
    minx, miny, maxx, maxy = poly.bounds
    poly = affinity.translate(poly, -(minx + maxx) / 2, -(miny + maxy) / 2)
    return poly, tuple(float(v) for v in (hi - lo))


def create_stl_insert_block(
    stl_path: str,
    block_grid_size: Optional[Tuple[int, int]] = None,
    offset: float = 0.4,
    height_mm: Optional[float] = None,
    floor_clearance: float = 1.5,
    wall_min: float = 2.5,
    cable_channel: bool = True,
    cable_width: float = 6.0,
    cable_height: float = 5.0,
    finger_scoop: bool = True,
    scoop_radius: float = 12.0,
    rotation_deg=(0, 0, 0),
    cable_floor_groove: bool = False,
    groove_depth: float = 3.0,
    pocket: str = "silhouette",
    return_info: bool = False,
):
    """Block with the STL (e.g. Galaxy Watch Ultra charger) subtracted.

    The STL is rotated by ``rotation_deg`` (multiples of 90 about X, Y, Z),
    centred, and sits with its top flush with the block top (block height
    defaults to the part height; a smaller ``height_mm`` leaves the part
    protruding).  Optional cable features, both exiting towards -Y:
    ``cable_channel`` is a side tunnel at floor level and
    ``cable_floor_groove`` a groove sunk below the pocket floor.  A finger
    scoop is cut towards +Y.

    ``pocket`` chooses the cavity: ``"silhouette"`` (default) cuts the part's
    top-down outline straight down to the floor - the part drops in from above
    and the block keeps its material (stands, hooks and other overhanging parts
    would otherwise be encased in an almost solid block); ``"exact"`` cuts the
    true 3D negative of the part (a mould; only sensible for parts without
    overhangs).

    Returns ``(block, nx, ny)``, or ``(block, nx, ny, info)`` with
    ``return_info=True`` (``info['mode']`` is ``'silhouette'``, ``'mesh'`` or
    ``'convex_hull'`` if the exact mesh could not be converted).
    """
    if pocket not in ("silhouette", "exact"):
        raise ValueError("pocket must be 'silhouette' or 'exact'")
    if pocket == "silhouette":
        outline, size = stl_silhouette(stl_path, offset, rotation_deg)
        solid = None
        info = {"mode": "silhouette", "size_x": size[0], "size_y": size[1], "size_z": size[2], "faces": 0}
        sx, sy = outline.bounds[2] - outline.bounds[0], outline.bounds[3] - outline.bounds[1]
    else:
        solid, info = load_stl_as_solid(stl_path, offset, rotation_deg=rotation_deg)
        sx, sy = info["size_x"] + 2 * offset, info["size_y"] + 2 * offset
    part_h = info["size_z"]
    floor_z = FOOT_H + floor_clearance + (groove_depth if cable_floor_groove else 0.0)
    H = height_mm if height_mm else floor_z + part_h + (offset if solid is not None else 0.0)
    nx, ny = block_grid_size if block_grid_size else (_cells_for(sx, wall_min), _cells_for(sy, wall_min))
    if (nx * GRID - sx) / 2 < wall_min - 1e-6 or (ny * GRID - sy) / 2 < wall_min - 1e-6:
        raise ValueError(f"grid_size {nx}x{ny} leaves walls < {wall_min} mm around the STL")

    cx, cy = nx * GRID / 2, ny * GRID / 2
    if solid is None:
        cutter = cq.Workplane("XY").workplane(offset=floor_z)
        for ring in [outline.exterior] + list(outline.interiors):
            cutter = cutter.polyline([(x + cx, y + cy) for x, y in _ring_points(ring.coords)]).close()
        cutter = cutter.extrude(max(H - floor_z, 1.0) + 1.0)       # +1 mm: opens cleanly through the top
    else:
        cutter = cq.Workplane("XY").add(solid).translate((cx, cy, floor_z + offset))
        # Open the top: union with a copy lifted past the block top.
        lift = max(H - floor_z, 1.0) + 1.0
        cutter = cutter.union(cutter.translate((0, 0, lift)), clean=True)

    block = create_base_block(nx, ny, H).cut(cutter)
    if cable_channel:
        ch = cq.Workplane("XY").box(cable_width, cy + 1, cable_height, centered=(True, False, False)) \
            .translate((cx, -1, floor_z))
        block = block.cut(ch)
    if cable_floor_groove:
        gr = cq.Workplane("XY").box(cable_width, cy + 1, groove_depth + 0.01, centered=(True, False, False)) \
            .translate((cx, -1, floor_z - offset - groove_depth))
        block = block.cut(gr)
    if finger_scoop:
        block = block.cut(_x_cylinder(cy + sy / 2, H, scoop_radius, cx - scoop_radius, 2 * scoop_radius))
    block = block.clean()
    return (block, nx, ny, info) if return_info else (block, nx, ny)


def create_custom_stl_subtraction_block(
    stl_file_path: str,
    grid_units_x: Optional[int] = None,
    grid_units_y: Optional[int] = None,
    clearance_mm: float = 0.4,
    cable_channel_bool: bool = False,
    rotation_deg=(0, 0, 0),
    cable_floor_groove: bool = False,
    height_mm: Optional[float] = None,
    finger_scoop: bool = True,
    pocket: str = "silhouette",
    return_info: bool = False,
):
    """Universal STL holder: a ``grid_units_x`` x ``grid_units_y`` block with
    any STL subtracted (CSG cut) at its top.

    ``clearance_mm``: 0.4 = press fit, 0.6 = loose fit.  Grid units default to
    the smallest grid that fits (see :func:`suggest_grid_units`).  Raises
    ``ValueError`` if the part does not fit the requested grid.
    """
    if (grid_units_x is None) != (grid_units_y is None):
        raise ValueError("give both grid_units_x and grid_units_y, or neither")
    if grid_units_x is not None and (grid_units_x < 1 or grid_units_y < 1):
        raise ValueError("grid units must be >= 1")
    if clearance_mm < 0:
        raise ValueError("clearance_mm must be >= 0")
    grid = (int(grid_units_x), int(grid_units_y)) if grid_units_x is not None else None
    return create_stl_insert_block(
        stl_file_path, block_grid_size=grid, offset=clearance_mm, height_mm=height_mm,
        cable_channel=cable_channel_bool, finger_scoop=finger_scoop, rotation_deg=rotation_deg,
        cable_floor_groove=cable_floor_groove, pocket=pocket, return_info=return_info)


def create_stl_on_base(
    stl_path: str,
    block_grid_size: Optional[Tuple[int, int]] = None,
    base_thickness: float = 3.0,
    margin: float = 2.0,
    rotation_deg=(0, 0, 0),
    max_faces: int = 20000,
    return_info: bool = False,
):
    """The STL itself, standing on an integrated base plate that fits the grid.

    The part is rotated by ``rotation_deg``, centred on an ``nx`` x ``ny`` cell
    base (Gridfinity feet underneath, ``base_thickness`` mm slab on top of the
    feet) and fused to it - print it as one piece and it sits in the case like
    any other block.  Its bottom is sunk 0.3 mm into the slab so the two fuse
    into one solid.  Grid size defaults to the smallest that leaves ``margin``
    mm of base around the part.

    Returns ``(block, nx, ny)`` or, with ``return_info``, ``(block, nx, ny, info)``.
    """
    if base_thickness < 1.0:
        raise ValueError("base_thickness must be >= 1 mm")
    solid, info = load_stl_as_solid(stl_path, 0.0, max_faces=max_faces, rotation_deg=rotation_deg)
    sx, sy, part_h = info["size_x"], info["size_y"], info["size_z"]
    nx, ny = block_grid_size if block_grid_size else (_cells_for(sx, margin), _cells_for(sy, margin))
    if nx < 1 or ny < 1:
        raise ValueError("grid size must be >= 1")
    if sx > nx * GRID - 2 * CELL_CLEAR + 1e-6 or sy > ny * GRID - 2 * CELL_CLEAR + 1e-6:
        raise ValueError(f"the {sx:.1f}x{sy:.1f} mm part does not fit a {nx}x{ny} cell base; use at least "
                         f"{_cells_for(sx, 0.0)}x{_cells_for(sy, 0.0)}")
    sink = 0.3
    top = FOOT_H + base_thickness
    base = create_base_block(nx, ny, top)
    part = cq.Workplane("XY").add(solid.translate(cq.Vector(nx * GRID / 2, ny * GRID / 2, top - sink)))
    block = base.union(part, clean=True)
    if not block.val().isValid():
        raise ValueError("the part could not be fused to its base (invalid solid)")
    info = dict(info, base_top=top, height_mm=top - sink + part_h)
    return (block, nx, ny, info) if return_info else (block, nx, ny)


# --------------------------------------------------------------------------- #
# Scanned contour (photo scanner) -> block
# --------------------------------------------------------------------------- #
def _ring_points(coords, tol: float = 1e-6) -> List[Tuple[float, float]]:
    """Closed ring -> open point list without repeated/zero-length edges."""
    pts: List[Tuple[float, float]] = []
    for x, y in coords:
        if not pts or abs(x - pts[-1][0]) > tol or abs(y - pts[-1][1]) > tol:
            pts.append((float(x), float(y)))
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= tol and abs(pts[0][1] - pts[-1][1]) <= tol:
        pts.pop()
    return pts


def create_scanned_contour_block(
    shapely_polygon,
    block_grid_x: Optional[int] = None,
    block_grid_y: Optional[int] = None,
    depth_mm: float = 20.0,
    height_mm: Optional[float] = None,
    floor_clearance: float = 1.5,
    wall_min: float = 2.5,
    return_info: bool = False,
):
    """Block with a scanned 2D outline cut ``depth_mm`` deep from its top face.

    ``shapely_polygon`` is the (already clearance-expanded) cutter profile in mm,
    y up; it is centred in the block.  Interior rings (islands, e.g. the loops
    of scissors) are kept as solid posts.  Grid size defaults to the smallest
    that leaves ``wall_min`` walls.  The block height defaults to
    ``FOOT_H + floor_clearance + depth_mm`` (pocket floor just above the feet);
    a smaller ``height_mm`` is rejected if the floor would reach the feet.

    Returns ``(block, nx, ny)`` or, with ``return_info``,
    ``(block, nx, ny, info)`` where ``info['profile']`` is the centred polygon
    (block coordinates) and ``info['area_mm2']`` its area.
    """
    from shapely import affinity
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(shapely_polygon, MultiPolygon):
        raise ValueError("profile must be a single polygon, got a MultiPolygon")
    if not isinstance(shapely_polygon, Polygon):
        raise ValueError("shapely_polygon must be a shapely Polygon")
    poly = shapely_polygon if shapely_polygon.is_valid else shapely_polygon.buffer(0)
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area < 1.0:
        raise ValueError("profile polygon is empty or smaller than 1 mm^2")
    if depth_mm <= 0:
        raise ValueError("depth_mm must be > 0")
    if (block_grid_x is None) != (block_grid_y is None):
        raise ValueError("give both block_grid_x and block_grid_y, or neither")

    minx, miny, maxx, maxy = poly.bounds
    sx, sy = maxx - minx, maxy - miny
    nx, ny = (int(block_grid_x), int(block_grid_y)) if block_grid_x is not None else (
        _cells_for(sx, wall_min), _cells_for(sy, wall_min))
    if nx < 1 or ny < 1:
        raise ValueError("grid size must be >= 1")
    if (nx * GRID - sx) / 2 < wall_min - 1e-6 or (ny * GRID - sy) / 2 < wall_min - 1e-6:
        raise ValueError(f"grid {nx}x{ny} ({nx * GRID:.0f}x{ny * GRID:.0f} mm) leaves walls < {wall_min} mm around "
                         f"the {sx:.1f}x{sy:.1f} mm outline; use at least "
                         f"{_cells_for(sx, wall_min)}x{_cells_for(sy, wall_min)}")

    H = height_mm if height_mm else FOOT_H + floor_clearance + depth_mm
    floor_z = H - depth_mm
    if floor_z < FOOT_H + 1.0 - 1e-6:
        raise ValueError(f"depth_mm {depth_mm:.1f} leaves a pocket floor at z={floor_z:.1f}, below the "
                         f"{FOOT_H + 1.0:.2f} mm needed above the feet; reduce depth or raise height_mm")

    centred = affinity.translate(poly, nx * GRID / 2 - (minx + maxx) / 2, ny * GRID / 2 - (miny + maxy) / 2)
    cutter = cq.Workplane("XY").workplane(offset=floor_z)
    for ring in [centred.exterior] + list(centred.interiors):
        pts = _ring_points(ring.coords)
        if len(pts) < 3:
            raise ValueError("profile has a degenerate ring")
        cutter = cutter.polyline(pts).close()
    cutter = cutter.extrude(depth_mm + 1.0)       # +1 mm so the cut opens cleanly through the top face

    block = create_base_block(nx, ny, H).cut(cutter).clean()
    if not block.val().isValid():
        raise ValueError("the scanned outline produced an invalid solid (self-intersecting profile?)")
    if not return_info:
        return block, nx, ny
    return block, nx, ny, {"profile": centred, "area_mm2": float(centred.area), "depth_mm": float(depth_mm),
                           "height_mm": float(H)}


# --------------------------------------------------------------------------- #
# USB-C hub
# --------------------------------------------------------------------------- #
def create_hub_cutout_block(
    hub_dims: Sequence[float] = (105.0, 60.0, 25.0),
    cable_pass_through: bool = True,
    clearance: float = 0.5,
    floor_clearance: float = 4.0,
    wall_min: float = 2.5,
    channel_width: float = 12.0,
    channel_depth: float = 3.0,
    side_channel_height: float = 8.0,
    corner_radius: float = 4.0,
    grid_size: Optional[Tuple[int, int]] = None,
) -> Tuple[cq.Workplane, int, int]:
    """Rectangular hub pocket (open top, flush with block top).

    With ``cable_pass_through``:
      * a groove ``channel_depth`` deep runs under the hub along X and exits
        both short ends (cables dive down and leave sideways);
      * tunnels of ``side_channel_height`` exit both long sides at floor level.
    Bottom-face channels are deliberately not used: the baseplate sits there.
    """
    hl, hw, hh = hub_dims
    L, Wd = hl + 2 * clearance, hw + 2 * clearance
    floor_z = FOOT_H + floor_clearance + (channel_depth if cable_pass_through else 0.0)
    H = floor_z + hh
    nx, ny = grid_size if grid_size else (_cells_for(L, wall_min), _cells_for(Wd, wall_min))
    if (nx * GRID - L) / 2 < wall_min - 1e-6 or (ny * GRID - Wd) / 2 < wall_min - 1e-6:
        raise ValueError(f"grid_size {nx}x{ny} leaves walls < {wall_min} mm around the hub")
    cx, cy = nx * GRID / 2, ny * GRID / 2

    pocket = cq.Workplane("XY").box(L, Wd, hh + 2, centered=(True, True, False))
    pocket = _safe_fillet(pocket, "|Z", min(corner_radius, Wd / 2 - 0.1))
    pocket = pocket.translate((cx, cy, floor_z))
    block = create_base_block(nx, ny, H).cut(pocket)

    if cable_pass_through:
        groove = cq.Workplane("XY").box(nx * GRID + 2, channel_width, channel_depth + 0.01,
                                        centered=(False, True, False)).translate((-1, cy, floor_z - channel_depth))
        tunnel = cq.Workplane("XY").box(channel_width, ny * GRID + 2, side_channel_height,
                                        centered=(True, False, False)).translate((cx, -1, floor_z))
        block = block.cut(groove).cut(tunnel)
    return block.clean(), nx, ny


# --------------------------------------------------------------------------- #
# Baseplate-free interlocking blocks
# --------------------------------------------------------------------------- #
def _key_socket(clearance: float = 0.15) -> cq.Workplane:
    """Dovetail socket (bow-tie, long axis along X) opening on the underside."""
    return (cq.Workplane("XY").polyline(_bowtie_pts(8.0, 8.0, 14.0)).close()
            .offset2D(clearance).extrude(KEY_DEPTH))


def interlock_socket_positions(nx: int, ny: int) -> List[Tuple[float, float, bool]]:
    """``(x, y, rotated)`` for a half-socket on every perimeter cell edge, centred on
    the cell boundary (so two blocks that touch share one full socket)."""
    pos = []
    for j in range(ny):
        pos += [(0.0, (j + 0.5) * GRID, False), (nx * GRID, (j + 0.5) * GRID, False)]
    for i in range(nx):
        pos += [((i + 0.5) * GRID, 0.0, True), ((i + 0.5) * GRID, ny * GRID, True)]
    return pos


def add_interlock_base(block: cq.Workplane, nx: int, ny: int, key_clearance: float = 0.15) -> cq.Workplane:
    """Make a block baseplate-free.

    The Gridfinity feet are replaced by a flat (chamfered) underside, and a
    half dovetail socket is cut into the bottom at the middle of every cell edge
    around the perimeter.  Any two blocks placed next to each other on the grid
    then share a full socket that a loose :func:`create_dovetail_key` clip locks,
    so the layout holds together without printing baseplates.
    """
    w, d = nx * GRID - 2 * CELL_CLEAR, ny * GRID - 2 * CELL_CLEAR
    prism = (cq.Workplane("XY").center(nx * GRID / 2, ny * GRID / 2).rect(w, d).extrude(FOOT_H)
             .edges("<Z").chamfer(0.5))
    block = block.union(prism, clean=True)
    proto = _key_socket(key_clearance)
    sockets = []
    for (x, y, rotated) in interlock_socket_positions(nx, ny):
        s = proto.rotate((0, 0, 0), (0, 0, 1), 90) if rotated else proto
        sockets.append(s.translate((x, y, 0)).val())
    return block.cut(cq.Workplane("XY").add(cq.Compound.makeCompound(sockets))).clean()


# --------------------------------------------------------------------------- #
# Baseplate tiles + dovetail keys
# --------------------------------------------------------------------------- #
def _bowtie_pts(half_len: float, neck: float, wide: float):
    a, n, w = half_len, neck / 2, wide / 2
    return [(-a, -w), (0, -n), (a, -w), (a, w), (0, n), (-a, w)]


def create_dovetail_key(half_len: float = 8.0, neck: float = 8.0, wide: float = 14.0,
                        thickness: float = KEY_DEPTH - 0.2) -> cq.Workplane:
    """Double-dovetail (bow-tie) clip that joins two baseplate tiles."""
    return cq.Workplane("XY").polyline(_bowtie_pts(half_len, neck, wide)).close().extrude(thickness)


def create_baseplate_tile(
    nx: int,
    ny: int,
    keys: Sequence[Tuple[float, float, bool]] = (),
    tolerance: float = 0.075,
    key_clearance: float = 0.15,
) -> cq.Workplane:
    """nx x ny baseplate tile with foot-shaped pockets.

    ``keys`` is a list of ``(x, y, rotated)`` seam positions in tile-local
    coordinates; a dovetail socket is cut from the underside there
    (``rotated`` turns the bow-tie 90 degrees for seams parallel to X).
    """
    total = SLAB_H + FOOT_H
    plate = cq.Workplane("XY").box(nx * GRID, ny * GRID, total, centered=(False, False, False))
    foot = cell_foot(grow=tolerance).val()
    top_cap = (cq.Workplane("XY").rect(GRID - 2 * CELL_CLEAR + 2 * tolerance,
                                       GRID - 2 * CELL_CLEAR + 2 * tolerance)
               .extrude(1.0).translate((0, 0, FOOT_H)).val())
    for i in range(nx):
        for j in range(ny):
            pocket = cq.Workplane("XY").add(foot.fuse(top_cap).translate(
                cq.Vector((i + 0.5) * GRID, (j + 0.5) * GRID, SLAB_H)))
            plate = plate.cut(pocket)
    socket_proto = (cq.Workplane("XY").polyline(_bowtie_pts(8.0, 8.0, 14.0)).close()
                    .offset2D(key_clearance).extrude(KEY_DEPTH))
    for (x, y, rotated) in keys:
        s = socket_proto.rotate((0, 0, 0), (0, 0, 1), 90) if rotated else socket_proto
        plate = plate.cut(s.translate((x, y, 0)))
    return plate.clean()
