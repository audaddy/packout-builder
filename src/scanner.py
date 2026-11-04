"""Optical tool scanner: calibration sheet -> photo -> millimetre-accurate outline.

Workflow
--------
1. :func:`calibration_sheet_pdf` makes a printable sheet: a 200 x 200 mm
   workspace with four ArUco markers in the corners and centre crosshairs.
2. The user places an item on the sheet and photographs it from any angle.
3. :func:`scan_photo` finds the markers, fits a homography, un-tilts the photo
   to a top-down view at a known pixel/mm scale and extracts the silhouette.
4. :func:`prepare_cutout` smooths the outline, adds clearance and optional
   finger scoops (shapely); ``geometry.create_scanned_contour_block`` extrudes
   it into a CadQuery block, and :func:`polygon_to_svg` / :func:`polygon_to_dxf`
   export the flat profile for laser/CNC users.

Coordinates
-----------
Polygons are in millimetres with **y pointing up** (CAD orientation), origin at
the bottom-left of the 200 mm workspace, so a block cut from the polygon looks
like the photo when seen from above.  Sheet/image coordinates (y down) only
appear inside the vision code.

Limitations: the item must contrast with the white paper, lighting should be
even (hard shadows are detected as part of the item), and the printed sheet
must be 100% scale - measure the border and pass ``print_scale`` if it is not.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Sheet layout (all values in mm, sheet coordinates: origin top-left, y down)
# --------------------------------------------------------------------------- #
WORKSPACE_MM = 200.0
MARKER_MM = 25.0                 # outer side of each marker's black border
MARKER_INSET_MM = 5.0            # marker distance from the workspace edge
MARKER_GRID = 6                  # 4x4 payload bits + 1-bit border each side
MARKER_IDS = (0, 1, 2, 3)        # top-left, top-right, bottom-right, bottom-left
CROSS_ARM_MM = 30.0              # half-length of the centre crosshair
BORDER_MASK_MM = 3.5             # ignore this band around the workspace edge
MARKER_MASK_PAD_MM = 3.0         # extra ignore margin around markers

PAPER_SIZES = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}


class ScanError(ValueError):
    """The photo could not be turned into a usable outline."""


def _aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_layout() -> Dict[int, Tuple[float, float]]:
    """Top-left corner (x, y mm, y down) of each marker on the sheet."""
    lo = MARKER_INSET_MM
    hi = WORKSPACE_MM - MARKER_INSET_MM - MARKER_MM
    return {0: (lo, lo), 1: (hi, lo), 2: (hi, hi), 3: (lo, hi)}


def marker_corners_mm(marker_id: int) -> np.ndarray:
    """The marker's 4 corners in ArUco order (TL, TR, BR, BL) in sheet mm."""
    x, y = marker_layout()[marker_id]
    m = MARKER_MM
    return np.array([(x, y), (x + m, y), (x + m, y + m), (x, y + m)], dtype=np.float64)


def _marker_bits(marker_id: int) -> np.ndarray:
    """6x6 boolean array, True = black cell (border + payload)."""
    img = cv2.aruco.generateImageMarker(_aruco_dictionary(), marker_id, MARKER_GRID, borderBits=1)
    return img == 0


# --------------------------------------------------------------------------- #
# Calibration sheet: PDF (vector) and raster (tests / demo)
# --------------------------------------------------------------------------- #
def calibration_sheet_pdf(paper: str = "A4") -> bytes:
    """Printable calibration sheet as PDF bytes (``paper``: 'A4' or 'Letter')."""
    if paper not in PAPER_SIZES:
        raise ValueError(f"paper must be one of {list(PAPER_SIZES)}")
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    pw, ph = (v * mm for v in PAPER_SIZES[paper])
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(pw, ph))
    c.setTitle("Packout tool scanner calibration sheet")
    ox, oy = (pw - WORKSPACE_MM * mm) / 2, (ph - WORKSPACE_MM * mm) / 2

    def X(x_mm: float) -> float:        # sheet mm -> PDF points
        return ox + x_mm * mm

    def Y(y_mm: float) -> float:        # sheet mm (y down) -> PDF points (y up)
        return oy + (WORKSPACE_MM - y_mm) * mm

    # outer 200 x 200 mm border
    c.setLineWidth(0.6)
    c.setStrokeGray(0.0)
    c.rect(ox, oy, WORKSPACE_MM * mm, WORKSPACE_MM * mm)
    # markers, drawn as vector cells
    cell = MARKER_MM / MARKER_GRID
    c.setFillGray(0.0)
    eps = 0.04     # overlap neighbouring black cells so PDF viewers/printers show no anti-aliasing seams
    for mid, (x0, y0) in marker_layout().items():
        bits = _marker_bits(mid)
        for r in range(MARKER_GRID):
            for col in range(MARKER_GRID):
                if bits[r, col]:
                    c.rect(X(x0 + col * cell - eps), Y(y0 + (r + 1) * cell + eps),
                           (cell + 2 * eps) * mm, (cell + 2 * eps) * mm, stroke=0, fill=1)
    # centre crosshairs (light grey so they never read as part of the item)
    cx = cy = WORKSPACE_MM / 2
    c.setStrokeGray(0.72)
    c.setLineWidth(0.4)
    c.line(X(cx - CROSS_ARM_MM), Y(cy), X(cx + CROSS_ARM_MM), Y(cy))
    c.line(X(cx), Y(cy - CROSS_ARM_MM), X(cx), Y(cy + CROSS_ARM_MM))
    c.circle(X(cx), Y(cy), 3 * mm, stroke=1, fill=0)

    # instructions above, scale-check bar below (both outside the workspace)
    top = oy + WORKSPACE_MM * mm
    c.setFillGray(0.0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(ox, top + 22 * mm, "Packout tool scanner - calibration sheet")
    c.setFont("Helvetica", 8.5)
    lines = [
        "1. Print at 100% / 'Actual size' (NOT 'fit to page'). The square border must measure 200 mm.",
        "2. Put the item flat on the sheet, centred on the crosshair, with nothing touching the markers.",
        "3. Photograph from above (any tilt is fine) with all 4 corner markers visible and even lighting.",
        "4. Use a plain item colour that contrasts with white paper; avoid hard shadows.",
    ]
    for i, text in enumerate(lines):
        c.drawString(ox, top + (16 - i * 4.2) * mm, text)
    bar_y = oy - 12 * mm
    c.setLineWidth(0.8)
    c.line(ox, bar_y, ox + 100 * mm, bar_y)
    for t in range(0, 101, 10):
        c.line(ox + t * mm, bar_y - 1.2 * mm, ox + t * mm, bar_y + 1.2 * mm)
    c.drawString(ox, bar_y - 7 * mm, "Scale check: this bar must measure 100 mm. If not, note the border size and enter it in the app.")
    c.setFont("Helvetica", 7)
    c.drawString(ox, bar_y - 11.5 * mm, "ArUco DICT_4X4_50, ids 0-3 (clockwise from top-left). Sheet v1.")
    c.showPage()
    c.save()
    return buf.getvalue()


def render_calibration_image(px_per_mm: float = 4.0, margin_mm: float = 15.0) -> np.ndarray:
    """Raster (BGR) of the sheet: white paper with ``margin_mm`` around the
    workspace.  Sheet mm (x, y) maps to pixel ((margin + x) * px_per_mm, ...)."""
    size = int(round((WORKSPACE_MM + 2 * margin_mm) * px_per_mm))
    img = np.full((size, size, 3), 255, np.uint8)

    def px(v: float) -> int:
        return int(round((margin_mm + v) * px_per_mm))

    thick = max(1, int(round(0.6 * px_per_mm)))
    cv2.rectangle(img, (px(0), px(0)), (px(WORKSPACE_MM), px(WORKSPACE_MM)), (0, 0, 0), thick)
    cell = MARKER_MM / MARKER_GRID
    for mid, (x0, y0) in marker_layout().items():
        bits = _marker_bits(mid)
        for r in range(MARKER_GRID):
            for col in range(MARKER_GRID):
                if bits[r, col]:
                    img[px(y0 + r * cell):px(y0 + (r + 1) * cell), px(x0 + col * cell):px(x0 + (col + 1) * cell)] = 0
    cx = cy = WORKSPACE_MM / 2
    arm = (184, 184, 184)
    t = max(1, int(round(0.4 * px_per_mm)))
    cv2.line(img, (px(cx - CROSS_ARM_MM), px(cy)), (px(cx + CROSS_ARM_MM), px(cy)), arm, t)
    cv2.line(img, (px(cx), px(cy - CROSS_ARM_MM)), (px(cx), px(cy + CROSS_ARM_MM)), arm, t)
    return img


def calibration_sheet_png(paper: str = "A4", dpi: int = 300) -> bytes:
    """The same sheet as a full-page PNG with its print size (``dpi``) embedded.

    Fallback for devices/browsers that cannot open the PDF: any image viewer or
    print dialog can print it at real size (200 mm border) because the file
    carries a physical resolution.  Use ``Actual size`` / ``100%`` when printing."""
    if paper not in PAPER_SIZES:
        raise ValueError(f"paper must be one of {list(PAPER_SIZES)}")
    from PIL import Image

    ppm = dpi / 25.4
    pw, ph = (int(round(v * ppm)) for v in PAPER_SIZES[paper])
    page = np.full((ph, pw, 3), 255, np.uint8)
    ws = render_calibration_image(px_per_mm=ppm, margin_mm=0.0)
    n = ws.shape[0]
    ox, oy = (pw - n) // 2, (ph - n) // 2
    page[oy:oy + n, ox:ox + n] = ws

    def mm_px(v: float) -> int:
        return int(round(v * ppm))

    font, k = cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0
    black = (0, 0, 0)
    cv2.putText(page, "Packout tool scanner - calibration sheet", (ox, oy - mm_px(18)), font, 1.5 * k, black, max(1, int(3 * k)), cv2.LINE_AA)
    cv2.putText(page, "Print at 100% / Actual size (NOT fit to page). The square border must measure 200 mm.",
                (ox, oy - mm_px(8)), font, 0.8 * k, black, max(1, int(2 * k)), cv2.LINE_AA)
    bar_y = oy + n + mm_px(12)
    thick = max(2, int(0.8 * ppm))
    cv2.line(page, (ox, bar_y), (ox + mm_px(100), bar_y), black, thick)
    for t in range(0, 101, 10):
        cv2.line(page, (ox + mm_px(t), bar_y - mm_px(1.2)), (ox + mm_px(t), bar_y + mm_px(1.2)), black, max(1, thick // 2))
    cv2.putText(page, "Scale check: this bar must measure 100 mm. If not, enter the measured border size in the app.",
                (ox, bar_y + mm_px(8)), font, 0.7 * k, black, max(1, int(2 * k)), cv2.LINE_AA)
    buf = io.BytesIO()
    Image.fromarray(page[:, :, ::-1]).save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Vision pipeline
# --------------------------------------------------------------------------- #
@dataclass
class ScanResult:
    """Everything the UI needs from one photo."""
    warped: np.ndarray                 # top-down BGR image, WORKSPACE_MM square
    mask: np.ndarray                   # uint8 0/255 object mask on ``warped``
    polygon: Polygon                   # outline, mm, y up, origin bottom-left of the workspace
    px_per_mm: float                   # resolution of ``warped`` (1 px = 1/px_per_mm mm)
    source_mm_per_px: float            # resolution of the original photo at the sheet
    marker_ids: List[int]
    reproj_rms_mm: float               # marker fit error
    scale_error_pct: Optional[float]   # re-measured marker size vs 25 mm, in the warped image
    threshold: float
    homography: np.ndarray             # photo px -> warped px
    warnings: List[str] = field(default_factory=list)

    @property
    def mm_per_px(self) -> float:
        """Millimetres per pixel of ``warped``."""
        return 1.0 / self.px_per_mm


def detect_markers(image_bgr: np.ndarray) -> Dict[int, np.ndarray]:
    """Detect the sheet's ArUco markers -> {id: (4, 2) corner array in pixels}."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 63
    detector = cv2.aruco.ArucoDetector(_aruco_dictionary(), params)

    # ArUco's adaptive threshold only copes with markers up to ~100 px wide, so
    # large photos (markers of several hundred px) are searched at smaller scales.
    long_side = max(gray.shape[:2])
    first = min(1.0, 1600.0 / long_side)
    scales = [first]
    for s in (first * 0.5, first * 0.25, 1.0):
        if s * long_side >= 300 and s not in scales:
            scales.append(s)
    best: Dict[int, np.ndarray] = {}
    for s in scales:
        small = gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        corners, ids, _ = detector.detectMarkers(small)
        found: Dict[int, np.ndarray] = {}
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in MARKER_IDS and int(i) not in found:
                    found[int(i)] = c.reshape(4, 2).astype(np.float64) / s
        if len(found) > len(best):
            best = found
        if len(best) == len(MARKER_IDS):
            break
    return best


def compute_homography(markers: Dict[int, np.ndarray], px_per_mm: float) -> Tuple[np.ndarray, float, int]:
    """Photo px -> warped px homography from all marker corners.

    Returns ``(H, rms_error_mm, n_inlier_points)``.
    """
    src, dst = [], []
    for mid, corners in markers.items():
        src.extend(corners)
        dst.extend(marker_corners_mm(mid) * px_per_mm)
    src, dst = np.array(src, np.float64), np.array(dst, np.float64)
    if len(src) < 8:
        raise ScanError("need at least 2 markers for a homography")
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 8:
        raise ScanError("markers are inconsistent (could not fit a stable homography)")
    keep = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst[keep]) ** 2, axis=1)))) / px_per_mm
    return H, rms, int(keep.sum())


def _source_px_per_mm(H: np.ndarray, px_per_mm: float) -> float:
    """Pixels of the original photo per sheet mm, measured at the sheet centre."""
    Hinv = np.linalg.inv(H)
    c = WORKSPACE_MM / 2
    pts = np.array([(c, c), (c + 10, c), (c, c + 10)], np.float64) * px_per_mm
    src = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    return float((np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[0])) / 2 / 10.0)


def verify_scale(warped: np.ndarray, px_per_mm: float) -> Optional[float]:
    """Re-detect the markers in the flattened image and compare their side
    length with the known size.  Returns the error in percent (None if the
    markers cannot be re-detected)."""
    found = detect_markers(warped)
    if not found:
        return None
    sides = []
    for corners in found.values():
        sides.extend(np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4))
    return (float(np.mean(sides)) / px_per_mm / MARKER_MM - 1.0) * 100.0


def _printed_masks(size: int, px_per_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """(valid, cross): valid = 255 where an item may be (markers/border
    excluded); cross = 255 on the printed crosshair lines."""
    valid = np.full((size, size), 255, np.uint8)
    b = int(round(BORDER_MASK_MM * px_per_mm))
    valid[:b, :] = valid[-b:, :] = 0
    valid[:, :b] = valid[:, -b:] = 0
    pad = MARKER_MASK_PAD_MM
    for x0, y0 in marker_layout().values():
        a, z = int(round((x0 - pad) * px_per_mm)), int(round((x0 + MARKER_MM + pad) * px_per_mm))
        p, q = int(round((y0 - pad) * px_per_mm)), int(round((y0 + MARKER_MM + pad) * px_per_mm))
        valid[max(p, 0):q, max(a, 0):z] = 0
    cross = np.zeros((size, size), np.uint8)
    c = int(round(WORKSPACE_MM / 2 * px_per_mm))
    arm = int(round((CROSS_ARM_MM + 4) * px_per_mm))
    t = max(2, int(round(1.8 * px_per_mm)))
    cv2.line(cross, (c - arm, c), (c + arm, c), 255, t)
    cv2.line(cross, (c, c - arm), (c, c + arm), 255, t)
    cv2.circle(cross, (c, c), int(round(4 * px_per_mm)), 255, t)
    return valid, cross


def _paper_surface(gray: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth model of the paper brightness (handles uneven lighting).

    Returns (surface float32 same size as gray, boolean map of paper pixels).
    """
    h, w = gray.shape
    f = 8
    small = cv2.resize(gray, (w // f, h // f), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid, (w // f, h // f), interpolation=cv2.INTER_NEAREST) > 0
    vals = small[vs]
    if vals.size < 100:
        return np.full(gray.shape, float(np.median(gray)), np.float32), np.ones(gray.shape, bool)
    t0, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_small = vs & (small > t0)
    if paper_small.mean() < 0.15:                      # item fills most of the sheet: assume flat lighting
        return np.full(gray.shape, float(np.median(vals)), np.float32), np.ones(gray.shape, bool)
    ys, xs = np.nonzero(paper_small)
    u, v = xs / small.shape[1], ys / small.shape[0]
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)
    coef, *_ = np.linalg.lstsq(A, small[ys, xs].astype(np.float64), rcond=None)
    gu, gv = np.meshgrid(np.linspace(0, 1, small.shape[1]), np.linspace(0, 1, small.shape[0]))
    surf_small = (coef[0] + coef[1] * gu + coef[2] * gv + coef[3] * gu * gu + coef[4] * gu * gv + coef[5] * gv * gv)
    surf = cv2.resize(surf_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    paper_full = cv2.resize(paper_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return np.maximum(surf, 30.0), paper_full


def _object_score(warped: np.ndarray, valid: np.ndarray, cross: np.ndarray) -> np.ndarray:
    """uint8 'how un-paper-like is this pixel' map (0 = paper)."""
    work = cv2.inpaint(warped, cross, 3, cv2.INPAINT_TELEA)       # erase the printed crosshair
    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    surf, paper = _paper_surface(gray, valid)
    flat = np.clip(gray.astype(np.float32) / surf * 255.0, 0, 255)
    dark = 255.0 - flat
    sat = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[:, :, 1], (5, 5), 1.2).astype(np.float32)
    sat0 = float(np.median(sat[paper & (valid > 0)])) if (paper & (valid > 0)).any() else 0.0
    colour = np.clip(sat - sat0 - 8.0, 0, 255)
    return np.clip(np.maximum(dark, colour), 0, 255).astype(np.uint8)


def segment_item(warped: np.ndarray, px_per_mm: float, method: str = "otsu",
                 threshold: Optional[float] = None, min_area_mm2: float = 25.0,
                 min_hole_area_mm2: float = 40.0, keep_holes: bool = True
                 ) -> Tuple[np.ndarray, Polygon, float, List[str]]:
    """Extract the item silhouette from a flattened top-down image.

    ``method``: ``'otsu'`` (global threshold on the lighting-corrected score)
    or ``'adaptive'`` (local threshold, better for uneven lighting; interior
    holes are always filled).  Returns ``(mask, polygon_mm, threshold, warnings)``;
    the polygon uses y-up CAD coordinates.
    """
    if method not in ("otsu", "adaptive"):
        raise ValueError("method must be 'otsu' or 'adaptive'")
    size = warped.shape[0]
    valid, cross = _printed_masks(size, px_per_mm)
    score = _object_score(warped, valid, cross)
    warnings: List[str] = []

    floor = 30.0
    if method == "otsu":
        t, _ = cv2.threshold(score[valid > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(threshold) if threshold is not None else max(float(t), floor)
        mask = (score > t).astype(np.uint8) * 255
    else:
        block = int(round(25 * px_per_mm)) | 1
        mask = cv2.adaptiveThreshold(score, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -12)
        t = float(threshold) if threshold is not None else floor
        mask[score <= t] = 0
        keep_holes = False
    mask[valid == 0] = 0

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(0.8 * px_per_mm)) | 1),) * 2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(1.6 * px_per_mm)) | 1),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask[valid == 0] = 0

    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_px = min_area_mm2 * px_per_mm ** 2
    outer = [i for i, h in enumerate(hier[0]) if h[3] < 0 and cv2.contourArea(contours[i]) >= min_px] if hier is not None else []
    if not outer:
        raise ScanError("No object found on the sheet. Check lighting/contrast, or lower the threshold.")
    best = max(outer, key=lambda i: cv2.contourArea(contours[i]))

    shell = contours[best][:, 0, :].astype(np.float64)
    holes = []
    if keep_holes:
        for i, h in enumerate(hier[0]):
            if h[3] == best and cv2.contourArea(contours[i]) >= min_hole_area_mm2 * px_per_mm ** 2:
                holes.append(contours[i][:, 0, :].astype(np.float64))

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contours[best]], -1, 255, -1)
    if (cv2.dilate(clean, np.ones((5, 5), np.uint8)) > 0)[valid == 0].any():
        warnings.append("The item touches the markers or the sheet border - move it toward the centre so the whole outline is captured.")
    n_comp = len(outer)
    if n_comp > 1:
        warnings.append(f"{n_comp} separate objects detected; using the largest.")

    def to_mm(pts: np.ndarray) -> List[Tuple[float, float]]:
        # warped pixel centre (u, v) == u / px_per_mm mm; flip y for CAD orientation
        return [(float(u) / px_per_mm, WORKSPACE_MM - float(v) / px_per_mm) for u, v in pts]

    poly = Polygon(to_mm(shell), [to_mm(h) for h in holes if len(h) >= 4])
    poly = _clean_polygon(poly)
    # contour tracing follows boundary pixel centres, i.e. half a pixel inside the true edge
    poly = _largest_polygon(poly.buffer(0.5 / px_per_mm, quad_segs=2))
    return mask, poly, t, warnings


def scan_photo(image_bgr: np.ndarray, px_per_mm: float = 5.0, method: str = "otsu",
               threshold: Optional[float] = None, min_area_mm2: float = 25.0) -> ScanResult:
    """Photo of an item on the calibration sheet -> :class:`ScanResult`.

    Raises :class:`ScanError` (a ``ValueError``) when the markers or the item
    cannot be found.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ScanError("expected a colour (BGR) image")
    if not 2.0 <= px_per_mm <= 12.0:
        raise ValueError("px_per_mm must be between 2 and 12")
    warnings: List[str] = []
    markers = detect_markers(image_bgr)
    if len(markers) < 3:
        raise ScanError(
            f"Found {len(markers)} of 4 calibration markers (ids {sorted(markers) or 'none'}). "
            "Keep all four corner markers fully visible, sharp and unobstructed.")
    if len(markers) == 3:
        missing = sorted(set(MARKER_IDS) - set(markers))
        warnings.append(f"Marker {missing[0]} not detected; calibrating from 3 markers (slightly less accurate).")

    H, rms, _ = compute_homography(markers, px_per_mm)
    if rms > 1.0:
        warnings.append(f"Marker fit error is {rms:.2f} mm - the sheet may be bent, creased or a marker was misread.")
    src_ppm = _source_px_per_mm(H, px_per_mm)
    if src_ppm < 2.0:
        warnings.append(f"Low-resolution photo ({src_ppm:.1f} px/mm); outline accuracy is about +/-{1 / src_ppm:.1f} mm. Move closer.")

    size = int(round(WORKSPACE_MM * px_per_mm))
    img = image_bgr
    if src_ppm > 2 * px_per_mm:                       # pre-shrink to avoid aliasing in the warp
        s = 1.5 * px_per_mm / src_ppm
        img = cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        H = H @ np.linalg.inv(np.diag([s, s, 1.0]))
    warped = cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    scale_err = verify_scale(warped, px_per_mm)
    if scale_err is not None and abs(scale_err) > 1.5:
        warnings.append(f"Re-measured marker size differs from 25 mm by {scale_err:+.1f}% - check the print scale.")

    mask, poly, t, seg_warn = segment_item(warped, px_per_mm, method, threshold, min_area_mm2)
    return ScanResult(warped=warped, mask=mask, polygon=poly, px_per_mm=px_per_mm,
                      source_mm_per_px=1.0 / src_ppm, marker_ids=sorted(markers), reproj_rms_mm=rms,
                      scale_error_pct=scale_err, threshold=t, homography=H, warnings=warnings + seg_warn)


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes (JPEG/PNG/...) to BGR, honouring EXIF rotation."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ScanError("Could not decode that file as an image.")
    return img


# --------------------------------------------------------------------------- #
# Polygon processing (shapely)
# --------------------------------------------------------------------------- #
def _largest_polygon(geom) -> Polygon:
    if geom.is_empty:
        raise ScanError("outline is empty")
    if isinstance(geom, Polygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
    if not polys:
        raise ScanError("outline has no area")
    return max(polys, key=lambda p: p.area)


def _clean_polygon(poly) -> Polygon:
    if not poly.is_valid:
        poly = poly.buffer(0)
    return _largest_polygon(poly)


def _smooth_ring(ring_coords, sigma_mm: float, spacing_mm: float) -> np.ndarray:
    """Resample a closed ring every ``spacing_mm`` and low-pass it with a Gaussian
    (std ``sigma_mm`` measured along the outline, wrapping around).

    Unlike corner-cutting schemes the amount of rounding depends only on
    ``sigma_mm``, not on how long the polygon's edges happen to be.
    """
    import shapely
    pts = np.asarray(ring_coords, np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    line = LineString(np.vstack([pts, pts[:1]]))
    n = max(16, int(round(line.length / spacing_mm)))
    samples = shapely.line_interpolate_point(line, np.linspace(0, line.length, n, endpoint=False))
    xy = np.column_stack([shapely.get_x(samples), shapely.get_y(samples)])
    sigma = sigma_mm / (line.length / n)                       # in samples
    if sigma < 0.3:
        return xy
    half = min(int(math.ceil(3 * sigma)), n // 2 - 1)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.vstack([xy[-half:], xy, xy[:half]]) if half > 0 else xy
    return np.column_stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)])


def smooth_polygon(poly: Polygon, simplify_tol: float = 0.05, sigma_mm: float = 0.5,
                   spacing_mm: float = 0.25, min_hole_area_mm2: float = 40.0) -> Polygon:
    """Remove pixel staircase noise: uniform resample, Gaussian smoothing along
    the outline (``sigma_mm``; corners round by about that much), then a light
    Douglas-Peucker pass (``simplify_tol``) to keep the vertex count modest."""
    poly = _clean_polygon(poly)
    shell = _smooth_ring(poly.exterior.coords, sigma_mm, spacing_mm)
    holes = [_smooth_ring(h.coords, sigma_mm, spacing_mm) for h in poly.interiors if Polygon(h).area >= min_hole_area_mm2]
    out = _clean_polygon(Polygon(shell, holes))
    if simplify_tol > 0:
        out = _clean_polygon(out.simplify(simplify_tol, preserve_topology=True))
    return out


def expand_polygon(poly: Polygon, clearance_mm: float = 0.6) -> Polygon:
    """Grow the outline by ``clearance_mm`` on every side (holes shrink)."""
    if clearance_mm < 0:
        raise ValueError("clearance_mm must be >= 0")
    if clearance_mm == 0:
        return poly
    return _largest_polygon(poly.buffer(clearance_mm, quad_segs=8, join_style="round"))


def add_finger_scoops(poly: Polygon, radius_mm: float = 12.0,
                      sides: Sequence[str] = ("left", "right")) -> Polygon:
    """Union semicircular finger-pull scoops onto the outline.

    Each scoop is a circle centred where the outline crosses the middle of the
    bounding box on that side, so half of it bites into the surrounding block.
    ``sides``: any of 'left', 'right', 'top', 'bottom'.
    """
    if radius_mm <= 0:
        raise ValueError("radius_mm must be > 0")
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    result = poly
    for side in sides:
        if side not in ("left", "right", "top", "bottom"):
            raise ValueError(f"unknown scoop side {side!r}")
        horizontal = side in ("left", "right")
        ray = LineString([(minx - 1, cy), (maxx + 1, cy)]) if horizontal else LineString([(cx, miny - 1), (cx, maxy + 1)])
        hit = poly.intersection(ray)
        if hit.is_empty:
            edge = {"left": (minx, cy), "right": (maxx, cy), "bottom": (cx, miny), "top": (cx, maxy)}[side]
        else:
            hx0, hy0, hx1, hy1 = hit.bounds
            edge = {"left": (hx0, cy), "right": (hx1, cy), "bottom": (cx, hy0), "top": (cx, hy1)}[side]
        result = result.union(Point(*edge).buffer(radius_mm, quad_segs=24))
    return _largest_polygon(result)


def prepare_cutout(poly: Polygon, clearance_mm: float = 0.6, scoops: bool = False,
                   scoop_radius_mm: float = 12.0, scoop_sides: Sequence[str] = ("left", "right"),
                   smooth_sigma_mm: float = 0.5, simplify_tol: float = 0.05,
                   print_scale: float = 1.0) -> Polygon:
    """Scan outline -> cutter profile: scale correction, smoothing, clearance, scoops.

    ``print_scale`` = measured/nominal size of the printed sheet (e.g. 0.98 if the
    200 mm border measured 196 mm); the outline is rescaled about its centroid.
    """
    if print_scale <= 0:
        raise ValueError("print_scale must be > 0")
    p = _clean_polygon(poly)
    if abs(print_scale - 1.0) > 1e-9:
        p = affinity.scale(p, print_scale, print_scale, origin="centroid")
    p = smooth_polygon(p, simplify_tol, smooth_sigma_mm)
    p = expand_polygon(p, clearance_mm)
    p = _clean_polygon(p.simplify(0.03, preserve_topology=True))
    if scoops:
        p = add_finger_scoops(p, scoop_radius_mm, scoop_sides)
    return p


def center_polygon(poly: Polygon, cx: float = 0.0, cy: float = 0.0) -> Polygon:
    """Translate so the bounding-box centre sits at (cx, cy)."""
    minx, miny, maxx, maxy = poly.bounds
    return affinity.translate(poly, cx - (minx + maxx) / 2, cy - (miny + maxy) / 2)


# --------------------------------------------------------------------------- #
# 2D profile export (laser / CNC)
# --------------------------------------------------------------------------- #
def _rings(poly: Polygon) -> List[Tuple[str, List[Tuple[float, float]]]]:
    rings = [("CUT", list(poly.exterior.coords)[:-1])]
    rings += [("HOLE", list(h.coords)[:-1]) for h in poly.interiors]
    return rings


def polygon_to_svg(poly: Polygon, margin_mm: float = 2.0, stroke_mm: float = 0.1) -> str:
    """SVG in real millimetres (width/height carry ``mm`` units, viewBox in mm),
    y flipped so the drawing matches the top-down view of the block."""
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx + 2 * margin_mm, maxy - miny + 2 * margin_mm
    d = []
    for _, ring in _rings(poly):
        pts = [(x - minx + margin_mm, maxy - y + margin_mm) for x, y in ring]
        d.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in pts) + " Z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}mm" height="{h:.3f}mm" '
            f'viewBox="0 0 {w:.3f} {h:.3f}"><path d="{" ".join(d)}" fill="none" stroke="#000000" '
            f'stroke-width="{stroke_mm}" fill-rule="evenodd"/></svg>')


def polygon_to_dxf(poly: Polygon) -> bytes:
    """DXF (R2010, millimetres): closed LWPOLYLINEs, outline on layer CUT and
    islands on layer HOLE, translated so the bounding box starts at the origin."""
    import ezdxf
    minx, miny, _, _ = poly.bounds
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add("CUT", color=1)
    doc.layers.add("HOLE", color=5)
    msp = doc.modelspace()
    for layer, ring in _rings(poly):
        msp.add_lwpolyline([(x - minx, y - miny) for x, y in ring], close=True, dxfattribs={"layer": layer})
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# Synthetic photos (tests and the in-app demo)
# --------------------------------------------------------------------------- #
def demo_item_polygon() -> Polygon:
    """A pliers-like silhouette (~150 x 62 mm), centred on the origin, y up."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    handle_a = box(-75, -28, -5, -8).buffer(5, quad_segs=8)
    handle_b = box(-75, 8, -5, 28).buffer(5, quad_segs=8)
    head = Point(10, 0).buffer(22, quad_segs=16)
    jaw = box(20, -9, 70, 9).buffer(3, quad_segs=8)
    shape = unary_union([handle_a, handle_b, head, jaw])
    return center_polygon(_largest_polygon(shape))


def simulate_photo(item: Optional[Polygon] = None, px_per_mm: float = 6.0, out_size: Tuple[int, int] = (1600, 1200),
                   rotation_deg: float = 10.0, perspective: float = 0.12, item_bgr: Tuple[int, int, int] = (35, 35, 175),
                   noise_sigma: float = 3.0, gradient: float = 0.12, seed: int = 0,
                   return_transform: bool = False):
    """Render a photo-like image of ``item`` (centred on the sheet) with tilt,
    uneven lighting and noise.  ``item`` is in CAD mm (y up) centred on (0, 0).

    With ``return_transform`` also returns ``M`` mapping sheet mm (x, y down) to
    photo pixels, for ground-truth checks.
    """
    margin = 20.0
    sheet = render_calibration_image(px_per_mm, margin)
    if item is not None:
        c = WORKSPACE_MM / 2
        polys = [item] if isinstance(item, Polygon) else list(item.geoms)
        for p in polys:
            def tp(ring):
                return np.array([((margin + c + x) * px_per_mm, (margin + c - y) * px_per_mm) for x, y in ring], np.int32)
            cv2.fillPoly(sheet, [tp(p.exterior.coords)], item_bgr, lineType=cv2.LINE_AA)
            for hole in p.interiors:
                cv2.fillPoly(sheet, [tp(hole.coords)], (255, 255, 255), lineType=cv2.LINE_AA)
    sh, sw = sheet.shape[:2]
    W, Hh = out_size
    scale = 0.72 * Hh / sh
    cx, cy = W / 2, Hh / 2
    quad = np.array([(-sw / 2, -sh / 2), (sw / 2, -sh / 2), (sw / 2, sh / 2), (-sw / 2, sh / 2)], np.float64) * scale
    quad[:2, 0] *= (1.0 - perspective)                      # keystone: far edge narrower
    quad[:2, 1] *= (1.0 - perspective * 0.5)
    a = math.radians(rotation_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = (quad @ R.T) + (cx, cy)
    src = np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)], np.float64)
    P = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))

    rng = np.random.default_rng(seed)
    table = np.clip(rng.normal(95, 8, (Hh, W, 1)) + np.array([30, 45, 60]), 0, 255).astype(np.uint8)
    table = cv2.GaussianBlur(table, (0, 0), 6)
    warped = cv2.warpPerspective(sheet, P, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), P, (W, Hh), flags=cv2.INTER_LINEAR)[:, :, None] / 255.0
    photo = (table * (1 - alpha) + warped * alpha).astype(np.float32)
    gx = np.linspace(1 - gradient, 1 + gradient / 3, W)[None, :, None]
    gy = np.linspace(1 + gradient / 4, 1 - gradient / 2, Hh)[:, None, None]
    photo = photo * gx * gy + rng.normal(0, noise_sigma, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (0, 0), 0.8)

    if not return_transform:
        return photo
    to_px = np.array([[px_per_mm, 0, margin * px_per_mm], [0, px_per_mm, margin * px_per_mm], [0, 0, 1]])
    return photo, P @ to_px
