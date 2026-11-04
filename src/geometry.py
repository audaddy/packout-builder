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

    Returns ``(block, nx, ny)``, or ``(block, nx, ny, info)`` with
    ``return_info=True`` (``info['mode']`` is ``'convex_hull'`` if the exact
    mesh could not be converted).
    """
    solid, info = load_stl_as_solid(stl_path, offset, rotation_deg=rotation_deg)
    part_h = info["size_z"]
    sx, sy = info["size_x"] + 2 * offset, info["size_y"] + 2 * offset
    floor_z = FOOT_H + floor_clearance + (groove_depth if cable_floor_groove else 0.0)
    H = height_mm if height_mm else floor_z + part_h + offset
    nx, ny = block_grid_size if block_grid_size else (_cells_for(sx, wall_min), _cells_for(sy, wall_min))
    if (nx * GRID - sx) / 2 < wall_min - 1e-6 or (ny * GRID - sy) / 2 < wall_min - 1e-6:
        raise ValueError(f"grid_size {nx}x{ny} leaves walls < {wall_min} mm around the STL")

    cx, cy = nx * GRID / 2, ny * GRID / 2
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
        cable_floor_groove=cable_floor_groove, return_info=return_info)


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
