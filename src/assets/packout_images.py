"""Offline SVG thumbnails of Packout organizers (no image files needed)."""
from __future__ import annotations

import base64

RED = "#DB0007"
RED_DARK = "#9E0005"
RED_LIGHT = "#F2262D"
BLACK = "#0D0D0D"
CHARCOAL = "#2A2A2A"


def _poly(points, fill, stroke="none", sw=1.0, opacity=1.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'


def organizer_svg(width_mm: float, depth_mm: float, height_mm: float, selected: bool = False) -> str:
    """Oblique drawing of an organizer, proportional to its interior size."""
    W, D, H = width_mm, depth_mm, max(height_mm, 20.0)
    # Fit the drawing into a 320 x 190 viewBox; depth runs up-right at 0.5 scale.
    k = min(210.0 / (W + 0.5 * D), 105.0 / (H * 1.5 + 0.35 * D))
    wp, hp = W * k, H * k * 1.5
    dx, dy = 0.5 * D * k, 0.35 * D * k
    fx = (320 - (wp + dx)) / 2
    fy = (190 - (hp + dy)) / 2 + dy + 2      # front-face top-left y, vertically centred
    top = [(fx, fy), (fx + wp, fy), (fx + wp + dx, fy - dy), (fx + dx, fy - dy)]
    front = [(fx, fy), (fx + wp, fy), (fx + wp, fy + hp), (fx, fy + hp)]
    side = [(fx + wp, fy), (fx + wp + dx, fy - dy), (fx + wp + dx, fy - dy + hp), (fx + wp, fy + hp)]
    # recessed tray on the lid-side opening
    m = 0.07
    ix0, iy0 = fx + wp * m + dx * m, fy - dy * m
    iw, ih = wp * (1 - 2 * m), dy * (1 - 2 * m)
    tray = [(ix0, iy0), (ix0 + iw, iy0), (ix0 + iw + dx * (1 - 2 * m), iy0 - ih), (ix0 + dx * (1 - 2 * m), iy0 - ih)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 190" role="img" aria-label="Packout organizer">',
        f'<ellipse cx="{fx + (wp + dx) / 2:.1f}" cy="{fy + hp + 6:.1f}" rx="{(wp + dx) / 2 + 6:.1f}" ry="7" fill="#000" opacity="0.45"/>',
        _poly(front, RED, RED_DARK, 1.2),
        _poly(side, RED_DARK, "#6E0004", 1.2),
        _poly(top, RED_LIGHT, RED_DARK, 1.2),
        _poly(tray, BLACK, "#000", 1),
    ]
    # bin dividers inside the tray
    for t in (1 / 3, 2 / 3):
        a = (tray[0][0] + (tray[1][0] - tray[0][0]) * t, tray[0][1])
        b = (tray[3][0] + (tray[2][0] - tray[3][0]) * t, tray[3][1])
        parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="{CHARCOAL}" stroke-width="2"/>')
    # two flat latches along the top edge of the front face
    lw, lh = min(30.0, wp * 0.2), min(9.0, hp * 0.3)
    for t in (0.22, 0.78):
        parts.append(f'<rect x="{fx + wp * t - lw / 2:.1f}" y="{fy + hp * 0.14:.1f}" width="{lw:.1f}" height="{lh:.1f}" rx="2" fill="{BLACK}"/>')
    parts.append("</svg>")
    return "".join(parts)


def custom_svg(selected: bool = False) -> str:
    """Dashed outline with a ruler/'+' glyph for the Custom card."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 190" role="img" aria-label="Custom dimensions">'
        f'<rect x="60" y="45" width="200" height="100" rx="8" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="10 7"/>'
        f'<path d="M60 165 H260 M60 159 V171 M260 159 V171" stroke="#B0B0B0" stroke-width="2" fill="none"/>'
        f'<path d="M280 45 V145 M274 45 H286 M274 145 H286" stroke="#B0B0B0" stroke-width="2" fill="none"/>'
        f'<text x="160" y="105" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="30" font-weight="700" fill="#FFFFFF">L × W × D</text>'
        "</svg>"
    )


def svg_data_uri(svg: str) -> str:
    """data: URI so an SVG can be used as an <img> or st.image source."""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
