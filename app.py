"""Streamlit UI for the Packout insert generator.  Run: streamlit run app.py"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import html
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely import wkt
import streamlit as st

from cad_engine import PackoutOrganizer
from src import geometry as geo
from src import packout_db as pk
from src import cnc
from src import phone_db as ph
from src import printers as pr
from src import scanner as sc
from src.geometry import GRID
from src.theme import RED, apply_theme, style_figure

st.set_page_config(page_title="Packout Insert Builder", page_icon="\U0001F9F0", layout="wide")
apply_theme()
ss = st.session_state


class _TabAbort(Exception):
    """Stop rendering the current tab only (st.stop() would also blank the tabs after it)."""

# --------------------------------------------------------------------------- state
ss.setdefault("tmpdir", tempfile.mkdtemp(prefix="packout_"))
ss.setdefault("case_key", "8435")
_p0 = pr.PRINTERS[pr.DEFAULT_PRINTER]
for k, v in (("custom_x", 203.0), ("custom_y", 305.0), ("custom_z", 95.0),
             ("bed_x", _p0.x), ("bed_y", _p0.y), ("bed_z", _p0.z), ("bed_round", _p0.round_bed),
             ("bed_preset", _p0.label)):
    ss.setdefault(k, v)
ss.setdefault("interlock", True)
ss.setdefault("stl_items", [])
ss.setdefault("stl_preview", None)
ss.setdefault("scan_items", [])
ss.setdefault("scan_preview", None)

BED_PRESETS = {p.label: p for p in pr.PRINTERS.values()}
BED_PRESETS[pr.CUSTOM_LABEL] = None


def on_round_toggle():
    if not ss.bed_round:                       # Bed Y is not drawn for a round plate; give it a sane value back
        ss.bed_y = ss.bed_x


def on_bed_preset():
    p = BED_PRESETS[ss.bed_preset]
    if p:
        ss.bed_x, ss.bed_y, ss.bed_z, ss.bed_round = p.x, p.y, p.z, p.round_bed


def current_interior():
    if ss.case_key == pk.CUSTOM_KEY:
        return (ss.custom_x, ss.custom_y, ss.custom_z)
    return pk.CASES[ss.case_key].interior_mm


def save_upload(up):
    data = up.getvalue()
    h = hashlib.sha1(data).hexdigest()[:12]
    path = os.path.join(ss.tmpdir, f"{h}.stl")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path, h, os.path.splitext(up.name)[0]


@st.cache_data(show_spinner=False)
def cached_analysis(path: str, rot: tuple):
    return geo.analyze_stl(path, rot)


@st.cache_data(show_spinner=False)
def cached_preview_mesh(path: str, rot: tuple, max_faces: int = 15000):
    m = geo._load_oriented_mesh(path, rot)
    if len(m.faces) > max_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            pass
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m.vertices.copy(), m.faces.copy()


@st.cache_data(show_spinner=False)
def cached_sheet_pdf(paper: str) -> bytes:
    return sc.calibration_sheet_pdf(paper)


@st.cache_data(show_spinner=False)
def cached_sheet_png(paper: str) -> bytes:
    return sc.calibration_sheet_png(paper)


@st.cache_resource(show_spinner=False)
def publish_static_sheets() -> bool:
    """Write the sheets to ./static so they can be opened straight in a browser tab
    (served at /app/static/..., needs ``enableStaticServing``) without a download."""
    out = Path(__file__).resolve().parent / "static"
    out.mkdir(exist_ok=True)
    for paper in sc.PAPER_SIZES:
        (out / f"packout_calibration_sheet_{paper}.pdf").write_bytes(sc.calibration_sheet_pdf(paper))
        (out / f"packout_calibration_sheet_{paper}.png").write_bytes(sc.calibration_sheet_png(paper))
    return True


@st.cache_data(show_spinner=False)
def cached_demo_photo() -> bytes:
    ok, buf = cv2.imencode(".jpg", sc.simulate_photo(sc.demo_item_polygon()), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


@st.cache_data(show_spinner=False, max_entries=8)
def cached_scan(data: bytes, px_per_mm: float, method: str, threshold):
    return sc.scan_photo(sc.decode_image(data), px_per_mm=px_per_mm, method=method, threshold=threshold)


def mesh_figure(v, f, color=RED, height=460):
    """3D mesh with per-face Lambert shading computed here, so long thin triangles
    (e.g. a pocket floor with a many-vertex outline) do not get smoothed vertex normals."""
    v, f = np.asarray(v, float), np.asarray(f)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.35, -0.45, 0.82])
    shade = 0.32 + 0.68 * np.clip(n @ (light / np.linalg.norm(light)), 0, 1)
    base = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)
    rgb = np.clip(base[None, :] * shade[:, None], 0, 255).astype(int)
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                              facecolor=[f"rgb({r},{g},{b})" for r, g, b in rgb], flatshading=True,
                              lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0, roughness=1.0)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                                 bgcolor="#1b1b1b"), margin=dict(l=0, r=0, t=0, b=0))
    return style_figure(fig, height)


# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="pk-hero"><h1>Packout Insert Builder</h1>'
    "<p>Parametric Gridfinity-style inserts for Milwaukee Packout organizers, split to fit your printer bed.</p></div>",
    unsafe_allow_html=True)

tab_case, tab_phone, tab_stl, tab_scan, tab_out = st.tabs(
    ["1 · Case & printer", "2 · Phones & hub", "3 · Custom STL blocks", "📷 Photo Tool Scanner", "4 · Layout & export"])

# =========================================================================== TAB 1
with tab_case:
    st.subheader("Choose your Packout case")
    st.caption("Interior sizes are Milwaukee's published figures (dividers not included). Measure yours and use Custom if they differ.")
    cards = list(pk.CASES.values()) + [None]
    for row in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, case in zip(cols, cards[row:row + 3]):
            key = case.key if case else pk.CUSTOM_KEY
            sel = ss.case_key == key
            with col:
                if case:
                    L, W, H = case.dims_ldh_in
                    Lm, Wm, Hm = case.dims_ldh_mm
                    badge = '<span class="pk-badge est">est.</span>' if case.estimated else ""
                    body = (f'<div class="pk-name">{html.escape(case.name)}{badge}</div>'
                            f'<div class="pk-part">#{case.part_no}</div>'
                            f'<div class="pk-dims">{L:.1f} × {W:.1f} × {H:.1f} in'
                            f'<span>{Lm:.0f} × {Wm:.0f} × {Hm:.0f} mm (L × W × H)</span></div>')
                else:
                    body = ('<div class="pk-name">Custom dimensions</div><div class="pk-part">Any case or tray</div>'
                            '<div class="pk-dims">You set L × W × D<span>in millimetres</span></div>')
                uri = pk.photo_data_uri(key)             # real product photo; offline SVG drawing as the fallback
                img = (f'<div class="pk-img photo"><img src="{uri}" alt="{html.escape(case.label)}"/></div>' if uri
                       else f'<div class="pk-img">{pk.thumbnail_svg(case, sel)}</div>')
                st.markdown(f'<div class="pk-card{" selected" if sel else ""}">{img}{body}</div>', unsafe_allow_html=True)
                if st.button("✓ Selected" if sel else "Select", key=f"sel_{key}",
                             type="primary" if sel else "secondary", width="stretch"):
                    ss.case_key = key
                    st.rerun()
                if case is None and sel:
                    c = st.columns(3)
                    c[0].number_input("Width X (mm)", 50.0, 1000.0, step=1.0, key="custom_x")
                    c[1].number_input("Length Y (mm)", 50.0, 1000.0, step=1.0, key="custom_y")
                    c[2].number_input("Depth Z (mm)", 20.0, 500.0, step=1.0, key="custom_z")

    ix, iy, iz = current_interior()
    fp = pk.footprint((ix, iy, iz))
    m = st.columns(4)
    m[0].metric("Interior (X × Y × Z)", f"{ix:.0f} × {iy:.0f} × {iz:.0f} mm")
    m[1].metric("42 mm grid", f"{fp['cells_x']} × {fp['cells_y']} cells")
    m[2].metric("Unused margin", f"{fp['leftover_x']:.0f} × {fp['leftover_y']:.0f} mm")
    m[3].metric("Floor used by grid", f"{fp['usable_pct']:.0f}%")

    st.subheader("Printer bed")
    st.selectbox("Preset", list(BED_PRESETS), key="bed_preset", on_change=on_bed_preset)
    st.checkbox("Round bed (X is the diameter)", key="bed_round", on_change=on_round_toggle,
                help="Circular build plate such as the Prusa HT90. Tiles must then fit inside the circle diagonally.")
    c = st.columns(3)
    c[0].number_input("Bed diameter (mm)" if ss.bed_round else "Bed X (mm)", 50.0, 1000.0, step=5.0, key="bed_x")
    if ss.bed_round:
        c[1].number_input("Bed Z (mm)", 20.0, 1000.0, step=5.0, key="bed_z")
    else:
        c[1].number_input("Bed Y (mm)", 50.0, 1000.0, step=5.0, key="bed_y")
        c[2].number_input("Bed Z (mm)", 20.0, 1000.0, step=5.0, key="bed_z")

    st.subheader("Baseplates")
    st.checkbox("Baseplate-free: interlocking blocks (no baseplates to print)", key="interlock",
                help="Every block gets a flat underside with dovetail sockets on all its cell edges. Blocks that touch are "
                     "locked together with small loose clips, so you print only the blocks and a handful of clips. "
                     "Untick for Gridfinity feet on printed baseplate tiles instead.")

# =========================================================================== TAB 2
with tab_phone:
    st.subheader("Phone slots")
    names = ph.list_phones() + ["Custom…"]
    sel_name = st.selectbox("Phone model", names, key="phone_sel")
    if sel_name == "Custom…":
        c = st.columns(3)
        dims = (c[0].number_input("Length (mm)", 50.0, 300.0, 150.0, 0.1, key="ph_L"),
                c[1].number_input("Width (mm)", 30.0, 200.0, 72.0, 0.1, key="ph_W"),
                c[2].number_input("Thickness (mm)", 3.0, 30.0, 8.0, 0.1, key="ph_T"))
        spec = ph.PhoneSpec("custom", "Custom phone", "Custom", *dims)
    else:
        spec = ph.get_phone(sel_name)
        st.markdown(f'<div class="pk-spec"><b>{html.escape(spec.name)}</b> &nbsp;·&nbsp; {spec.length} × {spec.width} × '
                    f'{spec.thickness} mm (L × W × T) &nbsp;·&nbsp; recommended clearance +{spec.clearance} mm</div>',
                    unsafe_allow_html=True)
    c = st.columns(3)
    n_phones = c[0].number_input("Number of phones", 0, 40, 2, key="n_phones",
                                 help="All phones go into one rack block. A rack that would be longer than your printer bed "
                                      "is split into the fewest equal racks that each fit.")
    clearance = c[1].slider("Clearance per side (mm)", 0.2, 1.5, float(spec.clearance), 0.1, key=f"clr_{spec.key}",
                            help="Tolerance added around the phone. Recommended value comes from the phone database.")
    scoops = c[2].checkbox("Finger scoops", True, key="scoops")
    ix, iy, iz = current_interior()
    c = st.columns(3)
    rack_style = c[0].selectbox("Rack style", ["Auto (fits the case)", "Upright", "On its long edge", "Flat single slots"],
                                key="rack_style",
                                help="Upright: phones stand on their short edge like the Maker Club rack. On its long edge: "
                                     "much lower, for shallow cases. Flat single slots: one block per phone for very low cases. "
                                     "Auto picks the first that lets the phones clear the lid.")
    cable_groove = c[1].checkbox("Cable groove", True, key="cable_groove",
                                 help="Room under each phone (upright) or a notch in the end walls (on its long edge) for the plug and cable.")
    vent = c[2].checkbox("Airflow windows", True, key="vent", help="Arch-topped windows through the separators, as in the reference rack.")
    c = st.columns(3)
    hold = c[0].slider("Holder height (% of phone)", 40, 100, 60, 5, key="hold_pct",
                       help="How much of each phone the rack holds; the rest sticks out so it is easy to grab.") / 100.0
    sep = c[1].slider("Separator wall (mm)", 3.0, 10.0, 5.0, 0.5, key="sep_wall")
    manual_lean = c[2].checkbox("Set lean manually", False, key="manual_lean")

    def lean_for(stance_):
        return next((a_ for a_ in range(0, 61) if geo.rack_rise(spec.dims, clearance, stance_, a_, cable_groove) <= iz - 1.0), None)

    rack_flat, rack_stance, angle = False, "portrait", 0
    if rack_style.startswith("Auto"):
        pose = geo.choose_rack_pose(spec.dims, clearance, iz - 1.0, cable_groove)
        if pose is None:
            rack_flat = True
        else:
            rack_stance, angle = pose
    elif rack_style == "Flat single slots":
        rack_flat = True
    else:
        rack_stance = "portrait" if rack_style == "Upright" else "landscape"
        fit = lean_for(rack_stance)
        angle = fit if fit is not None else 60
        if fit is None and not manual_lean:
            st.warning("These phones cannot clear the lid in this case even leaning back 60°; they will stick out of the lid line.")
    if rack_flat:
        fit = geo.min_tilt_for_height(spec.dims, clearance, iz - 1.0)
        angle = fit if fit is not None else 85
        phone_h = float(min(70.0, max(20.0, int(iz - 2))))
        st.info(f"Flat single slots: one block per phone, tilted {angle}° from vertical"
                + ("" if fit is not None else " (these phones still stick out of the lid line)")
                + (". Chosen because no rack pose clears the lid in this case." if rack_style.startswith("Auto") else "."))
        if manual_lean:
            angle = c[2].slider("Tilt from vertical (°)", 0, 85, int(angle), key="angle_flat")
    else:
        phone_h = float(min(70.0, max(20.0, int(iz - 2))))
        if manual_lean:
            angle = st.slider("Lean back from vertical (°)", 0, 60, int(angle), key="angle_rack")
        pl = geo.phone_rack_plan(spec.dims, max(int(n_phones), 1), clearance, rack_stance, angle, sep, 6.0, hold, cable_groove)
        st.markdown(f'<div class="pk-spec"><b>{"Upright" if rack_stance == "portrait" else "On its long edge"}</b>, lean {angle}° &nbsp;·&nbsp; '
                    f'{int(n_phones)} phone(s) → {pl["nx"]} × {pl["ny"]} cells, {pl["H"]:.0f} mm tall block, phones reach {pl["rise"]:.0f} mm '
                    f'(case interior {iz:.0f} mm)</div>', unsafe_allow_html=True)

    st.subheader("USB-C hub")
    use_hub = st.checkbox("Include a USB-C hub block", True, key="use_hub")
    c = st.columns(4)
    hub_dims = (c[0].number_input("Hub length (mm)", 10.0, 300.0, 105.0, 0.5, key="hub_L"),
                c[1].number_input("Hub width (mm)", 10.0, 300.0, 60.0, 0.5, key="hub_W"),
                c[2].number_input("Hub height (mm)", 5.0, 100.0, 25.0, 0.5, key="hub_H"))
    cable_pt = c[3].checkbox("Cable channels", True, key="hub_cable",
                             help="Floor groove under the hub plus tunnels out of the sides.")

# =========================================================================== TAB 3
with tab_stl, contextlib.suppress(_TabAbort):
    st.subheader("Universal STL block")
    st.caption("Drop any STL (stand, charger, meter, tool, battery adapter). Print it on a grid base, or cut it out of a holder block with your chosen clearance.")
    up = st.file_uploader("Drag & drop an .stl file", type=["stl"], key="stl_up")
    if up is None:
        st.info("Upload an STL to analyse it, set its orientation, and size the block.")
    else:
        path, h, stem = save_upload(up)
        st.markdown("**Orientation** (rotations applied in X → Y → Z order)")
        c = st.columns(3)
        rot = (c[0].radio("Rotate X", [0, 90, 180, 270], horizontal=True, key=f"rx_{h}"),
               c[1].radio("Rotate Y", [0, 90, 180, 270], horizontal=True, key=f"ry_{h}"),
               c[2].radio("Rotate Z", [0, 90, 180, 270], horizontal=True, key=f"rz_{h}"))
        try:
            info = cached_analysis(path, rot)
        except Exception as e:
            st.error(f"Could not read that STL: {e}")
            raise _TabAbort
        m = st.columns(5)
        m[0].metric("X", f"{info['size'][0]:.1f} mm")
        m[1].metric("Y", f"{info['size'][1]:.1f} mm")
        m[2].metric("Z (height)", f"{info['size'][2]:.1f} mm")
        m[3].metric("Watertight", "Yes" if info["watertight"] else "No")
        m[4].metric("Triangles", f"{info['faces']:,}")
        for w in info["warnings"]:
            st.warning(w)

        left, right = st.columns([3, 2])
        with right:
            mode_label = st.radio(
                "What to make", ["Object on a base (print the object itself)", "Holder block (pocket cut out of a block)"],
                key=f"md_{h}", help="Object on a base fuses the model to a grid base so it sits in the case as a printed part. "
                                    "Holder block cuts the model out of a block so it can be set into it.")
            on_base = mode_label.startswith("Object")
            if on_base:
                base_t = st.slider("Base thickness (mm)", 1.5, 10.0, 3.0, 0.5, key=f"bt_{h}")
                sug = geo.suggest_grid_units(info["size"][:2], 0.0, 2.0)
                st.caption(f"Smallest base that fits: {sug[0]} × {sug[1]} cells")
                c = st.columns(2)
                gx = c[0].number_input("Grid X (cells)", 1, 10, sug[0], key=f"gbx_{h}_{rot}")
                gy = c[1].number_input("Grid Y (cells)", 1, 10, sug[1], key=f"gby_{h}_{rot}")
                name = st.text_input("Block name", stem[:24].replace(" ", "_") or "stl_object", key=f"nmb_{h}")
                clr, pocket, cable_side, cable_floor, height = 0.0, "silhouette", False, False, None
            else:
                base_t = 3.0
                fit_label = st.radio("Fit", ["Press fit (+0.4 mm)", "Loose fit (+0.6 mm)", "Custom"], key=f"fit_{h}")
                clr = {"Press fit (+0.4 mm)": 0.4, "Loose fit (+0.6 mm)": 0.6}.get(fit_label)
                if clr is None:
                    clr = st.slider("Clearance (mm)", 0.0, 1.5, 0.4, 0.05, key=f"clr_{h}")
                sug = geo.suggest_grid_units(info["size"][:2], clr)
                st.caption(f"Smallest grid that fits: {sug[0]} × {sug[1]} cells")
                c = st.columns(2)
                gx = c[0].number_input("Grid X (cells)", 1, 10, sug[0], key=f"gx_{h}_{rot}_{clr}")
                gy = c[1].number_input("Grid Y (cells)", 1, 10, sug[1], key=f"gy_{h}_{rot}_{clr}")
                pocket_label = st.radio(
                    "Pocket shape", ["Drop-in outline (recommended)", "Exact 3D negative"], key=f"pk_{h}",
                    help="Drop-in outline cuts the part's top-down silhouette straight down, so it always drops in and "
                         "stands, hooks or overhangs don't turn the block solid. Exact 3D negative is a mould of the part "
                         "and only works for parts without overhangs.")
                pocket = "exact" if pocket_label.startswith("Exact") else "silhouette"
                st.markdown("**Cable routing**")
                cable_side = st.checkbox("Side channel (exits front)", False, key=f"cs_{h}")
                cable_floor = st.checkbox("Floor groove (under the item)", False, key=f"cf_{h}")
                height_auto = st.checkbox("Block height = part height", True, key=f"ha_{h}")
                height = None if height_auto else st.slider("Block height (mm)", 15.0, 120.0, 40.0, 1.0, key=f"hh_{h}")
                name = st.text_input("Block name", stem[:24].replace(" ", "_") or "stl_block", key=f"nm_{h}")
        with left:
            v, f = cached_preview_mesh(path, rot)
            st.plotly_chart(mesh_figure(v, f), use_container_width=True)

        cfg = dict(name=name, path=path, rot=rot, clearance=float(clr), gx=int(gx), gy=int(gy),
                   cable_side=cable_side, cable_floor=cable_floor, height=height, pocket=pocket,
                   on_base=on_base, base_t=float(base_t))
        c = st.columns(2)
        if c[0].button("Preview generated block", key="prev_btn"):
            try:
                with st.spinner("Cutting block..."):
                    if on_base:
                        blk, nx, ny, binfo = geo.create_stl_on_base(
                            path, (cfg["gx"], cfg["gy"]), base_t, rotation_deg=rot, return_info=True)
                    else:
                        blk, nx, ny, binfo = geo.create_custom_stl_subtraction_block(
                            path, cfg["gx"], cfg["gy"], cfg["clearance"], cable_side, rotation_deg=rot,
                            cable_floor_groove=cable_floor, height_mm=height, pocket=pocket, return_info=True)
                    mesh = PackoutOrganizer.to_mesh(blk, 0.05, 0.3)
                ss.stl_preview = dict(v=mesh.vertices, f=mesh.faces, mode=binfo["mode"], cells=(nx, ny))
            except ValueError as e:
                ss.stl_preview = None
                st.error(str(e))
        if c[1].button("Add to project", type="primary", key="add_btn"):
            try:
                if on_base:                                   # validate fit before accepting (cheap: coarse mesh)
                    geo.create_stl_on_base(path, (cfg["gx"], cfg["gy"]), base_t, rotation_deg=rot, max_faces=1500)
                else:
                    geo.create_custom_stl_subtraction_block(path, cfg["gx"], cfg["gy"], cfg["clearance"], False,
                                                            rotation_deg=rot, pocket=pocket)
                base, n = cfg["name"] or "stl_block", 2
                taken = {i["name"] for i in ss.stl_items} | {i["name"] for i in ss.scan_items}
                while cfg["name"] in taken:
                    cfg["name"], n = f"{base}_{n}", n + 1
                ss.stl_items.append(cfg)
                st.success(f"Added {cfg['name']}. Generate the project in tab 4.")
            except ValueError as e:
                st.error(str(e))
        if ss.stl_preview:
            p = ss.stl_preview
            if p["mode"] == "convex_hull":
                st.warning("The exact STL shape could not be used; the pocket is its convex hull.")
            st.caption(f"Generated block: {p['cells'][0]} × {p['cells'][1]} cells")
            st.plotly_chart(mesh_figure(p["v"], p["f"], "#C8C8C8", 420), use_container_width=True)

    if ss.stl_items:
        st.subheader("Blocks in this project")
        for i, it in enumerate(list(ss.stl_items)):
            c = st.columns([4, 1])
            if it.get("on_base"):
                detail = f'object on {it["base_t"]:g} mm base · rot {it["rot"]}'
            else:
                detail = (f'+{it["clearance"]} mm · rot {it["rot"]} · '
                          f'{"exact pocket" if it.get("pocket") == "exact" else "drop-in outline"}'
                          f'{" · side channel" if it["cable_side"] else ""}{" · floor groove" if it["cable_floor"] else ""}')
            c[0].markdown(f'<div class="pk-spec"><b>{html.escape(it["name"])}</b> · {it["gx"]}×{it["gy"]} cells · {detail}</div>',
                          unsafe_allow_html=True)
            if c[1].button("Remove", key=f"rm_{i}_{it['name']}"):
                ss.stl_items.pop(i)
                st.rerun()

def cnc_panel(p: dict) -> None:
    """Shapeoko 4 XL G-code for the previewed scanned block."""
    m = cnc.MACHINES[cnc.DEFAULT_MACHINE]
    st.caption(f"Mills the pocket in stock (MDF, plywood, plastic, foam...) and optionally cuts the block out. "
               f"{m.label}. Work zero: lower-left corner of the stock, Z0 on the stock top.")
    c = st.columns(3)
    material = c[0].selectbox("Material", list(cnc.MATERIALS), key="cnc_material")
    tool_label = c[1].selectbox("Bit (flat endmill)", list(cnc.TOOL_SIZES), index=2, key="cnc_tool")
    tool = cnc.TOOL_SIZES[tool_label]
    stock_t = c[2].number_input("Stock thickness (mm)", 5.0, float(m.z), max(25.4, float(np.ceil(p["depth"] + 3))),
                                step=0.1, key="cnc_stock")
    c = st.columns(3)
    depth = c[0].number_input("Pocket depth (mm)", 1.0, float(m.z), float(p["depth"]), step=0.5, key="cnc_depth")
    cut_out = c[1].checkbox("Also cut the block out (with tabs)", value=True, key="cnc_cutout")
    tool_change = c[2].checkbox("Emit M6 tool change (Carbide Motion)", value=True, key="cnc_m6",
                                help="Carbide Motion pauses so you can fit the bit. Untick for senders that reject M6, e.g. plain GRBL.")
    d = cnc.material_defaults(material, tool)
    tag = f"{material}_{tool}"                                   # new defaults whenever material or bit changes
    with st.expander("Feeds, speeds and tabs"):
        c = st.columns(4)
        feed = c[0].number_input("Feed (mm/min)", 100.0, 6000.0, d["feed"], 50.0, key=f"cnc_feed_{tag}")
        plunge = c[1].number_input("Plunge / ramp (mm/min)", 50.0, 3000.0, d["plunge"], 50.0, key=f"cnc_plunge_{tag}")
        stepdown = c[2].number_input("Stepdown (mm)", 0.2, 25.0, d["stepdown"], 0.1, key=f"cnc_sd_{tag}")
        rpm = c[3].number_input("Spindle (rpm)", 1000.0, 30000.0, float(d["rpm"]), 500.0, key=f"cnc_rpm_{tag}")
        c = st.columns(4)
        stepover = c[0].number_input("Stepover (% of bit)", 10.0, 100.0, 40.0, 5.0, key="cnc_stepover")
        tabs = c[1].number_input("Tabs", 0, 12, 4, key="cnc_tabs")
        tab_h = c[2].number_input("Tab height (mm)", 0.5, 10.0, 2.5, 0.5, key="cnc_tabh")
        margin = c[3].number_input("Stock margin (mm)", 5.0, 100.0, max(15.0, tool + 2.0), 1.0, key="cnc_margin")
        flute = st.number_input("Bit flute length (mm)", 3.0, 80.0, 19.0, 1.0, key="cnc_flute",
                                help="Only used to warn when the pocket is deeper than the bit can cut.")
    job = cnc.CncJob(name=p["name"], tool_diameter=tool, stock_thickness=float(stock_t), pocket_depth=float(depth),
                     material=material, feed=feed, plunge=plunge, stepdown=stepdown, rpm=rpm, stepover_pct=stepover,
                     cut_out=bool(cut_out), margin=float(margin), tabs=int(tabs), tab_height=float(tab_h),
                     flute_length=float(flute), tool_change=bool(tool_change))
    try:
        res = cnc.generate_gcode(p["profile"], p["size"], job)
    except cnc.CncError as e:
        st.error(str(e))
        return
    mm = st.columns(4)
    mm[0].metric("Stock to cut", f"{res.stock[0]:.0f} × {res.stock[1]:.0f} mm", help=f"Thickness {res.stock[2]:.1f} mm")
    mm[1].metric("Est. run time", f"{res.minutes:.0f} min")
    mm[2].metric("Z levels", f"{res.layers}")
    mm[3].metric("Not reachable", f"{res.uncut_pct:.1f}%", help="Narrow gaps and sharp inside corners the bit cannot enter.")
    for w in res.warnings:
        st.warning(w)
    fig = go.Figure()
    sx, sy, _ = res.stock
    fig.add_shape(type="rect", x0=0, y0=0, x1=sx, y1=sy, line=dict(color="#777", width=1.5), fillcolor="rgba(255,255,255,0.04)")
    bx, by = res.block_origin
    fig.add_shape(type="rect", x0=bx, y0=by, x1=bx + p["size"][0], y1=by + p["size"][1],
                  line=dict(color="#B0B0B0", width=1, dash="dot"))
    for k, path in enumerate(res.pocket_paths):
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=RED, width=1.3),
                                 name="Pocket passes", legendgroup="p", showlegend=(k == 0), hoverinfo="skip"))
    for path in res.cutout_paths:
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color="#FFFFFF", width=1.5),
                                 name="Cut-out (bit centre)", hoverinfo="skip"))
    fig.update_xaxes(range=[-5, sx + 5], title="mm", showgrid=False, constrain="domain")
    fig.update_yaxes(range=[-5, sy + 5], scaleanchor="x", title="mm", showgrid=False, constrain="domain")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15, x=0))
    st.plotly_chart(style_figure(fig, 420), use_container_width=True)
    st.download_button("Download G-code (.nc)", res.gcode, file_name=f"{p['name']}_shapeoko4xl.nc",
                       mime="text/plain", type="primary", key="dl_cnc", on_click="ignore")
    st.caption("In Carbide Motion: zero X/Y at the lower-left corner of the stock and Z on its top surface. "
               "Feeds are starting points for a 1/4 in bit; test on scrap first.")


# =========================================================================== TAB: PHOTO SCANNER
with tab_scan:
    st.subheader("Photo tool scanner")
    st.caption("Place any item on a printed calibration sheet, photograph it, and get a matching cut-out block.")

    # ---- step 1: calibration sheet
    st.markdown('<span class="pk-step">1</span> **Print the calibration sheet**', unsafe_allow_html=True)
    paper = st.radio("Paper", ["A4", "Letter"], horizontal=True, key="scan_paper")
    c = st.columns(2)
    c[0].download_button("Download sheet (PDF)", cached_sheet_pdf(paper), file_name=f"packout_calibration_sheet_{paper}.pdf",
                         mime="application/pdf", type="primary", key="dl_sheet", on_click="ignore", width="stretch")
    c[1].download_button("Download sheet (PNG image)", cached_sheet_png(paper), file_name=f"packout_calibration_sheet_{paper}.png",
                         mime="image/png", key="dl_sheet_png", on_click="ignore", width="stretch")
    publish_static_sheets()
    c = st.columns(2)
    c[0].link_button("Open PDF in a new tab (print from there)", f"app/static/packout_calibration_sheet_{paper}.pdf",
                     width="stretch")
    c[1].link_button("Open PNG in a new tab", f"app/static/packout_calibration_sheet_{paper}.png", width="stretch")
    st.caption("Print at 100% (Actual size, not fit-to-page). The square border should measure 200 mm; "
               "if it does not, enter the measured size below. If a downloaded file will not open, use the "
               "\"Open in a new tab\" buttons and print from the browser.")
    with st.expander("Preview the sheet"):
        st.image(cached_sheet_png(paper), width="stretch")

    # ---- step 2: photo
    st.markdown('<span class="pk-step">2</span> **Upload a photo**', unsafe_allow_html=True)
    photo_up = st.file_uploader("Drag & drop a photo of the item on the sheet", type=["jpg", "jpeg", "png", "webp"], key="scan_up")
    use_demo = st.checkbox("Use a synthetic demo photo (pliers on the sheet)", value=False, key="scan_demo",
                           help="Lets you try the scanner without printing anything.")
    with st.expander("Scan settings"):
        c = st.columns(4)
        measured_border = c[0].number_input("Measured printed border (mm)", 150.0, 260.0, 200.0, 0.5, key="scan_border",
                                            help="Nominal 200 mm. Enter what a ruler says if your printer scaled the page.")
        scan_ppm = c[1].select_slider("Analysis resolution (px/mm)", [4.0, 5.0, 6.0, 8.0], 5.0, key="scan_ppm")
        scan_method = c[2].radio("Edge detection", ["otsu", "adaptive"], horizontal=True, key="scan_method",
                                 help="Adaptive copes better with very uneven lighting.")
        thr_manual = c[3].checkbox("Manual threshold", False, key="scan_thr_on")
        scan_thr = c[3].slider("Threshold", 10, 200, 60, key="scan_thr") if thr_manual else None

    data = photo_up.getvalue() if photo_up is not None else (cached_demo_photo() if use_demo else None)
    if data is None:
        st.info("Upload a photo (or tick the demo photo) to detect the outline.")
    else:
        try:
            with st.spinner("Detecting markers and outline..."):
                scan = cached_scan(data, scan_ppm, scan_method, scan_thr)
        except sc.ScanError as e:
            st.error(str(e))
            scan = None
        if scan is not None:
            for w in scan.warnings:
                st.warning(w)
            print_scale = measured_border / sc.WORKSPACE_MM
            sx, sy = (v * print_scale for v in (scan.polygon.bounds[2] - scan.polygon.bounds[0],
                                                scan.polygon.bounds[3] - scan.polygon.bounds[1]))
            m = st.columns(4)
            m[0].metric("Markers", f"{len(scan.marker_ids)} / 4")
            m[1].metric("Photo scale", f"{scan.source_mm_per_px:.3f} mm/px", f"fit error {scan.reproj_rms_mm:.2f} mm", delta_color="off")
            m[2].metric("Item size", f"{sx:.1f} × {sy:.1f} mm")
            m[3].metric("Edge threshold", f"{scan.threshold:.0f} / 255")

            left, right = st.columns([3, 2])
            with right:
                st.markdown('<span class="pk-step">3</span> **Cut-out settings**', unsafe_allow_html=True)
                ix, iy, iz = current_interior()
                max_depth = max(8.0, min(80.0, float(int(iz - 2 - geo.FOOT_H - 1.5))))
                depth = st.slider("Cutout depth Z (mm)", 5.0, max_depth, min(20.0, max_depth), 1.0, key="scan_depth",
                                  help=f"Pocket depth. The block is {geo.FOOT_H + 1.5:.2f} mm taller than this (feet + floor).")
                clr = st.slider("Clearance margin (mm)", 0.0, 2.0, 0.6, 0.05, key="scan_clr")
                scoops_on = st.checkbox("Finger-pull scoops", False, key="scan_scoops")
                scoop_r, scoop_sides = 12.0, ("left", "right")
                if scoops_on:
                    scoop_r = st.slider("Scoop radius (mm)", 6.0, 20.0, 12.0, 1.0, key="scan_scoop_r")
                    labels = st.multiselect("Scoop sides", ["Left", "Right", "Top", "Bottom"], ["Left", "Right"], key="scan_scoop_sides")
                    scoop_sides = tuple(s.lower() for s in labels)
                cutout = sc.prepare_cutout(scan.polygon, clr, scoops=bool(scoops_on and scoop_sides), scoop_radius_mm=scoop_r,
                                           scoop_sides=scoop_sides or ("left", "right"), print_scale=print_scale)
                bx0, by0, bx1, by1 = cutout.bounds
                need = (geo._cells_for(bx1 - bx0, 2.5), geo._cells_for(by1 - by0, 2.5))
                sig = hashlib.sha1(data).hexdigest()[:8]
                c = st.columns(2)
                gx = c[0].number_input("Grid X (cells)", 1, 10, need[0], key=f"sgx_{sig}_{need[0]}")
                gy = c[1].number_input("Grid Y (cells)", 1, 10, need[1], key=f"sgy_{sig}_{need[1]}")
                st.caption(f"Cut-out outline {bx1 - bx0:.1f} × {by1 - by0:.1f} mm; smallest grid {need[0]} × {need[1]}.")
                if gx < need[0] or gy < need[1]:
                    st.warning(f"Grid {gx} × {gy} is too small for this outline (needs at least {need[0]} × {need[1]}).")
                name = st.text_input("Block name", "scanned_tool", key="scan_name")
                c = st.columns(2)
                do_prev = c[0].button("Preview block", key="scan_prev")
                do_add = c[1].button("Add to project", type="primary", key="scan_add")

            with left:
                rgb = cv2.cvtColor(scan.warped, cv2.COLOR_BGR2RGB)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
                fig = go.Figure()
                fig.add_layout_image(source=uri, xref="x", yref="y", x=0, y=0, sizex=sc.WORKSPACE_MM, sizey=sc.WORKSPACE_MM,
                                     xanchor="left", yanchor="top", layer="below")

                def ring_xy(coords):
                    xs = [p[0] for p in coords]
                    ys = [sc.WORKSPACE_MM - p[1] for p in coords]            # CAD (y up) -> image (y down)
                    return xs, ys
                for k, ring in enumerate([scan.polygon.exterior] + list(scan.polygon.interiors)):
                    xs, ys = ring_xy(ring.coords)
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=RED, width=2.5),
                                             name="Detected edge", legendgroup="edge", showlegend=(k == 0),
                                             hovertemplate="x %{x:.1f} mm, y %{y:.1f} mm<extra>detected edge</extra>"))
                for k, ring in enumerate([cutout.exterior] + list(cutout.interiors)):
                    xs, ys = ring_xy(ring.coords)
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#FFFFFF", width=1.5, dash="dash"),
                                             name="With clearance", legendgroup="clr", showlegend=(k == 0), hoverinfo="skip"))
                fig.update_xaxes(range=[0, sc.WORKSPACE_MM], title="mm", constrain="domain", showgrid=False)
                fig.update_yaxes(range=[sc.WORKSPACE_MM, 0], scaleanchor="x", title="mm", showgrid=False)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12, x=0))
                st.plotly_chart(style_figure(fig, 600), use_container_width=True)
                st.caption("Perspective-corrected photo. Red = detected edge; dashed white = cut-out outline with clearance"
                           + (" and scoops." if scoops_on else "."))

            def build_scan_block(poly, gx_, gy_, depth_):
                blk, nx, ny, binfo = geo.create_scanned_contour_block(poly, int(gx_), int(gy_), float(depth_), return_info=True)
                return blk, nx, ny, binfo

            if do_prev:
                try:
                    with st.spinner("Cutting block..."):
                        blk, nx, ny, binfo = build_scan_block(cutout, gx, gy, depth)
                        mesh = PackoutOrganizer.to_mesh(blk, 0.05, 0.3)
                        stl_bytes = PackoutOrganizer.to_mesh(blk, 0.01, 0.1).export(file_type="stl")
                    ss.scan_preview = dict(v=mesh.vertices, f=mesh.faces, cells=(nx, ny), height=binfo["height_mm"], stl=stl_bytes,
                                           svg=sc.polygon_to_svg(binfo["profile"]), dxf=sc.polygon_to_dxf(binfo["profile"]),
                                           name=name or "scanned_tool", profile=binfo["profile"],
                                           size=(nx * GRID, ny * GRID), depth=float(depth))
                except ValueError as e:
                    ss.scan_preview = None
                    st.error(str(e))
            if do_add:
                try:
                    build_scan_block(cutout, gx, gy, depth)                  # validate before accepting
                    base, n = name or "scanned_tool", 2
                    nm = base
                    while nm in {i["name"] for i in ss.stl_items} | {i["name"] for i in ss.scan_items}:
                        nm, n = f"{base}_{n}", n + 1
                    ss.scan_items.append(dict(name=nm, wkt=cutout.wkt, gx=int(gx), gy=int(gy), depth=float(depth)))
                    st.success(f"Added {nm}. Generate the project in the Layout & export tab.")
                except ValueError as e:
                    st.error(str(e))

            p = ss.scan_preview
            if p:
                st.markdown(f"**Generated block** · {p['cells'][0]} × {p['cells'][1]} cells · {p['height']:.1f} mm tall")
                st.plotly_chart(mesh_figure(p["v"], p["f"], "#C8C8C8", 420), use_container_width=True)
                c = st.columns(3)
                c[0].download_button("Download STL", p["stl"], file_name=f"{p['name']}.stl", mime="model/stl", key="dl_scan_stl", on_click="ignore")
                c[1].download_button("Download profile SVG", p["svg"], file_name=f"{p['name']}.svg", mime="image/svg+xml", key="dl_scan_svg", on_click="ignore")
                c[2].download_button("Download profile DXF", p["dxf"], file_name=f"{p['name']}.dxf", mime="application/dxf", key="dl_scan_dxf", on_click="ignore")

                with st.expander("CNC program (Shapeoko 4 XL)"):
                    cnc_panel(p)

    if ss.scan_items:
        st.subheader("Scanned blocks in this project")
        for i, it in enumerate(list(ss.scan_items)):
            c = st.columns([4, 1])
            c[0].markdown(f'<div class="pk-spec"><b>{html.escape(it["name"])}</b> · {it["gx"]}×{it["gy"]} cells · '
                          f'depth {it["depth"]:.0f} mm</div>', unsafe_allow_html=True)
            if c[1].button("Remove", key=f"rm_scan_{i}_{it['name']}"):
                ss.scan_items.pop(i)
                st.rerun()

# =========================================================================== TAB 4
with tab_out:
    def project_sig() -> str:
        """Everything build() reads, so the results can be flagged as out of date."""
        return repr((current_interior(), ss.case_key, ss.bed_x, ss.get("bed_y"), ss.bed_z, bool(ss.bed_round), bool(ss.interlock),
                     int(n_phones), spec.key, spec.dims, clearance, angle, phone_h, scoops, rack_flat, rack_stance, sep, hold, cable_groove,
                     vent, use_hub, hub_dims, cable_pt,
                     [dict(i) for i in ss.stl_items], [dict(i) for i in ss.scan_items]))

    def build():
        ix, iy, iz = current_interior()
        bed = (ss.bed_x, ss.bed_x if ss.bed_round else ss.get('bed_y', ss.bed_x), ss.bed_z)
        custom = ss.case_key == pk.CUSTOM_KEY
        org = PackoutOrganizer(None if custom else pk.CASES[ss.case_key].label,
                               interior=(ix, iy, iz) if custom else None, bed=bed, bed_round=bool(ss.bed_round),
                               interlock=bool(ss.interlock))
        if n_phones and rack_flat:
            org.add_phone_slots(int(n_phones), phone=spec, clearance=clearance, angle=float(angle),
                                height_mm=float(phone_h), finger_scoops=scoops)
        elif n_phones:
            org.add_phone_rack(int(n_phones), phone=spec, clearance=clearance, stance=rack_stance, angle=float(angle),
                               separator=sep, hold_frac=hold, cable_groove=cable_groove, vent=vent, finger_scoops=scoops)
        for it in ss.stl_items:
            if it.get("on_base"):
                org.add_stl_on_base(it["path"], name=it["name"], grid_units=(it["gx"], it["gy"]),
                                    base_thickness=it["base_t"], rotation_deg=it["rot"])
                continue
            org.add_stl_insert(it["path"], name=it["name"], grid_units=(it["gx"], it["gy"]),
                               clearance_mm=it["clearance"], cable_channel=it["cable_side"],
                               rotation_deg=it["rot"], cable_floor_groove=it["cable_floor"], height_mm=it["height"],
                               pocket=it.get("pocket", "silhouette"))
        for it in ss.scan_items:
            org.add_scanned_block(wkt.loads(it["wkt"]), name=it["name"], grid_units=(it["gx"], it["gy"]), depth_mm=it["depth"])
        if use_hub:
            org.add_hub(hub_dims=hub_dims, cable_pass_through=cable_pt)
        ss.org, ss.parts, ss.meshes, ss.zip = org, org.build_parts(), {}, {}
        ss.built_sig = project_sig()
        ss.table = pd.DataFrame({"Item": [i.name for i in org.items], "Cells": [f"{i.nx}x{i.ny}" for i in org.items],
                                 "Pin": False, "X cell": 0, "Y cell": 0})
        ss.table_key = ss.get("table_key", 0) + 1

    st.subheader("Generate")
    st.caption(f"{ss.case_key if ss.case_key == pk.CUSTOM_KEY else pk.CASES[ss.case_key].label} · "
               f"{int(n_phones)} phone(s) · {len(ss.stl_items)} custom STL block(s) · {len(ss.scan_items)} scanned block(s) · hub {'on' if use_hub else 'off'}")
    if st.button("Generate project", type="primary", key="gen"):
        try:
            with st.spinner("Building CAD geometry..."):
                build()
        except ValueError as e:
            ss.pop("org", None)
            st.error(str(e))

    if "org" not in ss:
        st.info("Configure the tabs, then press **Generate project**.")
        st.stop()
    if ss.get("built_sig") != project_sig():
        st.warning("The project changed since it was last generated (blocks, phones, hub, case or printer settings). "
                   "Press **Generate project** to update the layout and export.")
        st.stop()

    org: PackoutOrganizer = ss.org
    parts = ss.parts
    lx, ly = org.leftover
    tiles = org.plan_tiles()
    m = st.columns(4)
    m[0].metric("Grid", f"{org.cells_x} × {org.cells_y} cells", f"{lx:.0f} × {ly:.0f} mm unused", delta_color="off")
    m[1].metric("Baseplate tiles", "none" if org.interlock else len(tiles))
    m[2].metric("Dovetail clips" if org.interlock else "Dovetail keys", org.seam_key_count())
    m[3].metric("Insert blocks", len(org.items))

    st.subheader("Layout canvas")
    problems = st.container()
    left, right = st.columns([1, 2])
    with left:
        st.caption("Tick **Pin** to fix an item at a cell (0,0 = front-left); the rest auto-pack.")
        edited = st.data_editor(
            ss.table, key=f"ed{ss.table_key}", hide_index=True, width="stretch", disabled=["Item", "Cells"],
            column_config={"X cell": st.column_config.NumberColumn(min_value=0, max_value=org.cells_x - 1, step=1),
                           "Y cell": st.column_config.NumberColumn(min_value=0, max_value=org.cells_y - 1, step=1)})
        for it, (_, row) in zip(org.items, edited.iterrows()):
            it.manual = bool(row["Pin"])
            it.pos = (int(row["X cell"]), int(row["Y cell"])) if it.manual else None
        layout_warnings = org.layout()
    unplaced = [it for it in org.items if it.pos is None]
    with problems:
        if unplaced:
            used = sum(it.nx * it.ny for it in org.items if it.pos is not None)
            st.error("**Not placed in the layout (no room left):** "
                     + ", ".join(f"{it.name} ({it.nx}×{it.ny} cells)" for it in unplaced)
                     + f". The {org.cells_x}×{org.cells_y} grid has {org.cells_x * org.cells_y - used} free cells. "
                     "Remove a phone slot or block, choose a larger case, or shrink the block's grid size. "
                     "Unplaced blocks are still exported, but not positioned.")
        for w in layout_warnings:
            if not any(w.startswith(f"{it.name}: ") and "does not fit" in w for it in unplaced):
                st.warning(w)
    with right:
        fig = go.Figure()
        W, D = org.cells_x * GRID, org.cells_y * GRID
        fig.add_shape(type="rect", x0=0, y0=0, x1=org.interior[0], y1=org.interior[1], line=dict(color="#888", dash="dot"))
        palette = [RED, "#F58518", "#4c78a8", "#54a24b", "#b279a2", "#72b7b2"]
        for t in ([] if org.interlock else tiles):
            fig.add_shape(type="rect", x0=t.cx * GRID, y0=t.cy * GRID, x1=(t.cx + t.nx) * GRID, y1=(t.cy + t.ny) * GRID,
                          line=dict(color="#EEE", width=3), fillcolor="rgba(255,255,255,0.05)")
        for i in range(org.cells_x + 1):
            fig.add_shape(type="line", x0=i * GRID, x1=i * GRID, y0=0, y1=D, line=dict(color="#555", width=1))
        for j in range(org.cells_y + 1):
            fig.add_shape(type="line", y0=j * GRID, y1=j * GRID, x0=0, x1=W, line=dict(color="#555", width=1))
        for k, it in enumerate(org.items):
            if it.pos is None:
                continue
            x0, y0 = it.pos[0] * GRID, it.pos[1] * GRID
            fig.add_shape(type="rect", x0=x0 + 1, y0=y0 + 1, x1=x0 + it.nx * GRID - 1, y1=y0 + it.ny * GRID - 1,
                          fillcolor=palette[k % len(palette)], opacity=0.85, line=dict(width=0))
            fig.add_annotation(x=x0 + it.nx * GRID / 2, y=y0 + it.ny * GRID / 2, text=it.name.replace("_", " "),
                               showarrow=False, font=dict(color="white", size=11))
        fig.update_xaxes(range=[-5, org.interior[0] + 5], title="X (mm)", constrain="domain", gridcolor="#333")
        fig.update_yaxes(range=[-5, org.interior[1] + 5], title="Y (mm)", scaleanchor="x", gridcolor="#333")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(style_figure(fig, 520), use_container_width=True)
        st.caption("Blocks interlock with loose dovetail clips where they touch (no baseplates); thin lines = 42 mm cells."
                   if org.interlock else
                   "Bright outlines = baseplate tiles (each fits the printer bed); thin lines = 42 mm cells.")

    st.subheader("Part preview")
    name = st.selectbox("Part", list(parts), key="part_sel")
    if name not in ss.meshes:
        with st.spinner("Tessellating..."):
            ss.meshes[name] = (PackoutOrganizer.to_mesh(parts[name], 0.05, 0.3), PackoutOrganizer.check_part(parts[name]))
    mesh, report = ss.meshes[name]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(mesh_figure(mesh.vertices, mesh.faces, RED, 520), use_container_width=True)
    with c2:
        st.markdown("**Print check**")
        st.write(f"Size: {report['size'][0]} × {report['size'][1]} × {report['size'][2]} mm")
        st.write(f"Volume: {report['volume_cm3']} cm³")
        st.write("Watertight: " + ("✅" if report["watertight"] else "❌"))
        st.write("Valid solid: " + ("✅" if report["valid"] else "❌"))
        st.write("Fits bed: " + ("✅" if org.fits_bed(parts[name]) else "❌"))

    st.subheader("Export")
    fmt = st.radio("Format", ["stl", "step", "both"], horizontal=True, key="fmt")
    if st.button("Prepare zip", key="zip_btn"):
        with st.spinner("Exporting..."):
            if org.interlock:                          # clip count follows the current (pinned) layout
                org.layout()
                parts = org.build_parts()
            ss.zip[fmt] = org.export_zip(parts, fmt)
    if fmt in ss.zip:
        st.download_button(f"Download packout_inserts_{fmt}.zip", ss.zip[fmt], file_name=f"packout_inserts_{fmt}.zip",
                           mime="application/zip", type="primary", on_click="ignore")
        st.caption("The zip's README.txt records the layout as of when it was prepared; prepare it again after changing pins.")
