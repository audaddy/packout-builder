"""Streamlit UI for the Packout insert generator.  Run: streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cad_engine import PackoutOrganizer
from src import geometry as geo
from src import packout_db as pk
from src import phone_db as ph
from src.geometry import GRID
from src.theme import RED, apply_theme, style_figure

st.set_page_config(page_title="Packout Insert Builder", page_icon="\U0001F9F0", layout="wide")
apply_theme()
ss = st.session_state

# --------------------------------------------------------------------------- state
ss.setdefault("tmpdir", tempfile.mkdtemp(prefix="packout_"))
ss.setdefault("case_key", "8435")
for k, v in (("custom_x", 203.0), ("custom_y", 305.0), ("custom_z", 95.0),
             ("bed_x", 220.0), ("bed_y", 220.0), ("bed_z", 250.0)):
    ss.setdefault(k, v)
ss.setdefault("stl_items", [])
ss.setdefault("stl_preview", None)

BED_PRESETS = {
    "Custom": None,
    "Ender 3 / generic 220 (220 x 220 x 250)": (220.0, 220.0, 250.0),
    "Prusa MK4 (250 x 210 x 220)": (250.0, 210.0, 220.0),
    "Bambu X1 / P1 (256 x 256 x 256)": (256.0, 256.0, 256.0),
    "Bambu A1 mini (180 x 180 x 180)": (180.0, 180.0, 180.0),
}


def on_bed_preset():
    v = BED_PRESETS[ss.bed_preset]
    if v:
        ss.bed_x, ss.bed_y, ss.bed_z = v


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


def mesh_figure(v, f, color=RED, height=460):
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                              color=color, flatshading=True,
                              lighting=dict(ambient=0.55, diffuse=0.85, specular=0.2)))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                                 bgcolor="#1b1b1b"), margin=dict(l=0, r=0, t=0, b=0))
    return style_figure(fig, height)


# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="pk-hero"><h1>Packout Insert Builder</h1>'
    "<p>Parametric Gridfinity-style inserts for Milwaukee Packout organizers, split to fit your printer bed.</p></div>",
    unsafe_allow_html=True)

tab_case, tab_phone, tab_stl, tab_out = st.tabs(
    ["1 · Case & printer", "2 · Phones & hub", "3 · Custom STL blocks", "4 · Layout & export"])

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
                st.markdown(f'<div class="pk-card{" selected" if sel else ""}"><div class="pk-img">'
                            f'{pk.thumbnail_svg(case, sel)}</div>{body}</div>', unsafe_allow_html=True)
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
    c = st.columns(3)
    c[0].number_input("Bed X (mm)", 50.0, 1000.0, step=5.0, key="bed_x")
    c[1].number_input("Bed Y (mm)", 50.0, 1000.0, step=5.0, key="bed_y")
    c[2].number_input("Bed Z (mm)", 20.0, 1000.0, step=5.0, key="bed_z")

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
    n_phones = c[0].number_input("Number of slots", 0, 12, 2, key="n_phones")
    clearance = c[1].slider("Clearance per side (mm)", 0.2, 1.5, float(spec.clearance), 0.1, key=f"clr_{spec.key}",
                            help="Tolerance added around the phone. Recommended value comes from the phone database.")
    scoops = c[2].checkbox("Side finger scoops", True, key="scoops")
    ix, iy, iz = current_interior()
    max_block_h = max(float(int(iz - 2)), 20.0)
    c = st.columns(3)
    auto_tilt = c[0].checkbox("Auto-tilt to fit case height", True, key="auto_tilt",
                              help="Picks the smallest tilt that lets the phone sit below the lid line.")
    fit = geo.min_tilt_for_height(spec.dims, clearance, iz - 1.0)
    if auto_tilt:
        angle = fit if fit is not None else 85
        c[1].metric("Tilt from vertical", f"{angle}°")
        if fit is None:
            st.warning("This phone cannot lie low enough for this case even at 85°; it will stick out of the lid line.")
    else:
        angle = c[1].slider("Tilt from vertical (°)", 0, 85, 15, key="angle")
    phone_h = c[2].slider("Block height (mm)", 25.0, max(max_block_h, 26.0), min(70.0, max_block_h), 1.0,
                          key=f"phone_h_{pk.CUSTOM_KEY if ss.case_key == pk.CUSTOM_KEY else ss.case_key}")

    st.subheader("USB-C hub")
    use_hub = st.checkbox("Include a USB-C hub block", True, key="use_hub")
    c = st.columns(4)
    hub_dims = (c[0].number_input("Hub length (mm)", 10.0, 300.0, 105.0, 0.5, key="hub_L"),
                c[1].number_input("Hub width (mm)", 10.0, 300.0, 60.0, 0.5, key="hub_W"),
                c[2].number_input("Hub height (mm)", 5.0, 100.0, 25.0, 0.5, key="hub_H"))
    cable_pt = c[3].checkbox("Cable channels", True, key="hub_cable",
                             help="Floor groove under the hub plus tunnels out of the sides.")

# =========================================================================== TAB 3
with tab_stl:
    st.subheader("Universal STL block")
    st.caption("Drop any STL (charger, meter, tool, battery adapter). It is cut out of a grid block with your chosen clearance.")
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
            st.stop()
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
            fit_label = st.radio("Fit", ["Press fit (+0.4 mm)", "Loose fit (+0.6 mm)", "Custom"], key=f"fit_{h}")
            clr = {"Press fit (+0.4 mm)": 0.4, "Loose fit (+0.6 mm)": 0.6}.get(fit_label)
            if clr is None:
                clr = st.slider("Clearance (mm)", 0.0, 1.5, 0.4, 0.05, key=f"clr_{h}")
            sug = geo.suggest_grid_units(info["size"][:2], clr)
            st.caption(f"Smallest grid that fits: {sug[0]} × {sug[1]} cells")
            c = st.columns(2)
            gx = c[0].number_input("Grid X (cells)", 1, 10, sug[0], key=f"gx_{h}_{rot}_{clr}")
            gy = c[1].number_input("Grid Y (cells)", 1, 10, sug[1], key=f"gy_{h}_{rot}_{clr}")
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
                   cable_side=cable_side, cable_floor=cable_floor, height=height)
        c = st.columns(2)
        if c[0].button("Preview generated block", key="prev_btn"):
            try:
                with st.spinner("Cutting block..."):
                    blk, nx, ny, binfo = geo.create_custom_stl_subtraction_block(
                        path, cfg["gx"], cfg["gy"], cfg["clearance"], cable_side, rotation_deg=rot,
                        cable_floor_groove=cable_floor, height_mm=height, return_info=True)
                    mesh = PackoutOrganizer.to_mesh(blk, 0.05, 0.3)
                ss.stl_preview = dict(v=mesh.vertices, f=mesh.faces, mode=binfo["mode"], cells=(nx, ny))
            except ValueError as e:
                ss.stl_preview = None
                st.error(str(e))
        if c[1].button("Add to project", type="primary", key="add_btn"):
            try:
                geo.create_custom_stl_subtraction_block(path, cfg["gx"], cfg["gy"], cfg["clearance"], False,
                                                        rotation_deg=rot)      # validate fit before accepting
                base, n = cfg["name"] or "stl_block", 2
                taken = {i["name"] for i in ss.stl_items}
                while cfg["name"] in taken:
                    cfg["name"], n = f"{base}_{n}", n + 1
                ss.stl_items.append(cfg)
                st.success(f"Added {cfg['name']}. Generate the project in tab 4.")
            except ValueError as e:
                st.error(str(e))
        if ss.stl_preview:
            p = ss.stl_preview
            if p["mode"] != "mesh":
                st.warning("The exact STL shape could not be used; the pocket is its convex hull.")
            st.caption(f"Generated block: {p['cells'][0]} × {p['cells'][1]} cells")
            st.plotly_chart(mesh_figure(p["v"], p["f"], "#C8C8C8", 420), use_container_width=True)

    if ss.stl_items:
        st.subheader("Blocks in this project")
        for i, it in enumerate(list(ss.stl_items)):
            c = st.columns([4, 1])
            c[0].markdown(f'<div class="pk-spec"><b>{html.escape(it["name"])}</b> · {it["gx"]}×{it["gy"]} cells · '
                          f'+{it["clearance"]} mm · rot {it["rot"]}'
                          f'{" · side channel" if it["cable_side"] else ""}{" · floor groove" if it["cable_floor"] else ""}</div>',
                          unsafe_allow_html=True)
            if c[1].button("Remove", key=f"rm_{i}_{it['name']}"):
                ss.stl_items.pop(i)
                st.rerun()

# =========================================================================== TAB 4
with tab_out:
    def build():
        ix, iy, iz = current_interior()
        bed = (ss.bed_x, ss.bed_y, ss.bed_z)
        custom = ss.case_key == pk.CUSTOM_KEY
        org = PackoutOrganizer(None if custom else pk.CASES[ss.case_key].label,
                               interior=(ix, iy, iz) if custom else None, bed=bed)
        if n_phones:
            org.add_phone_slots(int(n_phones), phone=spec, clearance=clearance, angle=float(angle),
                                height_mm=float(phone_h), finger_scoops=scoops)
        for it in ss.stl_items:
            org.add_stl_insert(it["path"], name=it["name"], grid_units=(it["gx"], it["gy"]),
                               clearance_mm=it["clearance"], cable_channel=it["cable_side"],
                               rotation_deg=it["rot"], cable_floor_groove=it["cable_floor"], height_mm=it["height"])
        if use_hub:
            org.add_hub(hub_dims=hub_dims, cable_pass_through=cable_pt)
        ss.org, ss.parts, ss.meshes, ss.zip = org, org.build_parts(), {}, {}
        ss.table = pd.DataFrame({"Item": [i.name for i in org.items], "Cells": [f"{i.nx}x{i.ny}" for i in org.items],
                                 "Pin": False, "X cell": 0, "Y cell": 0})
        ss.table_key = ss.get("table_key", 0) + 1

    st.subheader("Generate")
    st.caption(f"{ss.case_key if ss.case_key == pk.CUSTOM_KEY else pk.CASES[ss.case_key].label} · "
               f"{int(n_phones)} phone slot(s) · {len(ss.stl_items)} custom STL block(s) · hub {'on' if use_hub else 'off'}")
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

    org: PackoutOrganizer = ss.org
    parts = ss.parts
    lx, ly = org.leftover
    tiles = org.plan_tiles()
    m = st.columns(4)
    m[0].metric("Grid", f"{org.cells_x} × {org.cells_y} cells", f"{lx:.0f} × {ly:.0f} mm unused", delta_color="off")
    m[1].metric("Baseplate tiles", len(tiles))
    m[2].metric("Dovetail keys", org.seam_key_count())
    m[3].metric("Insert blocks", len(org.items))

    st.subheader("Layout canvas")
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
        for w in org.layout():
            st.warning(w)
    with right:
        fig = go.Figure()
        W, D = org.cells_x * GRID, org.cells_y * GRID
        fig.add_shape(type="rect", x0=0, y0=0, x1=org.interior[0], y1=org.interior[1], line=dict(color="#888", dash="dot"))
        palette = [RED, "#F58518", "#4c78a8", "#54a24b", "#b279a2", "#72b7b2"]
        for t in tiles:
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
        st.caption("Bright outlines = baseplate tiles (each fits the printer bed); thin lines = 42 mm cells.")

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
            ss.zip[fmt] = org.export_zip(parts, fmt)
    if fmt in ss.zip:
        st.download_button(f"Download packout_inserts_{fmt}.zip", ss.zip[fmt], file_name=f"packout_inserts_{fmt}.zip",
                           mime="application/zip", type="primary")
        st.caption("The zip's README.txt records the layout as of when it was prepared; prepare it again after changing pins.")
