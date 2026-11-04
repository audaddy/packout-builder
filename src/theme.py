"""Milwaukee red & black dark theme for the Streamlit UI."""
from __future__ import annotations

RED = "#DB0007"
RED_HOVER = "#F2262D"
BG = "#121212"
CARD = "#2A2A2A"
CARD_HI = "#333333"
TEXT = "#FFFFFF"
MUTED = "#B0B0B0"

CSS = f"""
:root {{ --mk-red:{RED}; --mk-red-hover:{RED_HOVER}; --mk-bg:{BG}; --mk-card:{CARD};
        --mk-card-hi:{CARD_HI}; --mk-text:{TEXT}; --mk-muted:{MUTED}; }}
.stApp {{ background: var(--mk-bg); color: var(--mk-text); }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: .02em; }}
h2, h3 {{ border-left: 4px solid var(--mk-red); padding-left: .6rem; }}
a {{ color: var(--mk-red-hover); }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--mk-muted); }}

/* hero banner */
.pk-hero {{ background: linear-gradient(100deg, var(--mk-red) 0%, #7d0004 38%, var(--mk-bg) 78%);
           border-radius: 10px; padding: 1.4rem 1.8rem; margin-bottom: 1rem; border: 1px solid #000; }}
.pk-hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; text-transform: uppercase; color:#fff; }}
.pk-hero p {{ margin: .25rem 0 0; color: #f0d0d0; }}

/* buttons */
.stButton > button, [data-testid="stDownloadButton"] > button, [data-testid="stBaseButton-secondary"] {{
    background: var(--mk-card); color: var(--mk-text); border: 1px solid #444; border-radius: 6px;
    font-weight: 600; transition: all .15s; }}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {{
    border-color: var(--mk-red); color: #fff; box-shadow: 0 0 0 1px var(--mk-red); }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
[data-testid="stDownloadButton"] > button[kind="primary"] {{
    background: var(--mk-red); border: 1px solid var(--mk-red); color: #fff; text-transform: uppercase; letter-spacing: .04em; }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    background: var(--mk-red-hover); border-color: var(--mk-red-hover); }}

/* tabs */
button[data-baseweb="tab"] {{ color: var(--mk-muted); font-weight: 600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: #fff; }}
[data-baseweb="tab-highlight"] {{ background-color: var(--mk-red) !important; height: 3px; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
    background: var(--mk-card) !important; border-color: #444 !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{ border-color: var(--mk-red) !important; }}
[data-testid="stNumberInput"] button {{ background: var(--mk-card-hi); }}

/* metric cards */
[data-testid="stMetric"] {{ background: var(--mk-card); border: 1px solid #3a3a3a; border-left: 4px solid var(--mk-red);
                           border-radius: 8px; padding: .8rem 1rem; }}
[data-testid="stMetricLabel"] {{ color: var(--mk-muted); }}

/* upload drop zone */
[data-testid="stFileUploaderDropzone"] {{ background: var(--mk-card); border: 2px dashed var(--mk-red);
    border-radius: 10px; padding: 1.6rem; transition: all .15s; }}
[data-testid="stFileUploaderDropzone"]:hover {{ background: var(--mk-card-hi); box-shadow: 0 0 14px rgba(219,0,7,.45); }}
[data-testid="stFileUploaderDropzone"] small {{ color: var(--mk-muted); }}

/* expanders, alerts, tables */
[data-testid="stExpander"] {{ background: var(--mk-card); border: 1px solid #3a3a3a; border-radius: 8px; }}
[data-testid="stAlert"] {{ border-radius: 8px; }}

/* case cards */
.pk-card {{ background: var(--mk-card); border: 2px solid #3a3a3a; border-radius: 12px; padding: .7rem .8rem .6rem;
           margin-bottom: .4rem; transition: all .15s; }}
.pk-card:hover {{ border-color: #777; }}
.pk-card.selected {{ border-color: var(--mk-red); box-shadow: 0 0 16px rgba(219,0,7,.45); }}
.pk-card .pk-img {{ background: #1b1b1b; border-radius: 8px; padding: .3rem; }}
.pk-card .pk-img svg {{ width: 100%; height: auto; display: block; }}
.pk-name {{ font-weight: 700; font-size: 1.02rem; margin-top: .5rem; }}
.pk-part {{ color: var(--mk-muted); font-size: .85rem; }}
.pk-dims {{ margin-top: .3rem; font-weight: 700; color: #fff; }}
.pk-dims span {{ display: block; color: var(--mk-muted); font-weight: 400; font-size: .85rem; }}
.pk-badge {{ display: inline-block; background: var(--mk-red); color: #fff; border-radius: 4px; padding: 0 .4rem;
            font-size: .7rem; font-weight: 700; margin-left: .4rem; vertical-align: middle; }}
.pk-badge.est {{ background: #555; }}

/* spec chips */
.pk-spec {{ background: var(--mk-card); border-radius: 8px; padding: .6rem .9rem; border: 1px solid #3a3a3a; }}
.pk-spec b {{ color: #fff; }}
"""


def apply_theme() -> None:
    """Inject the stylesheet (call once per run, after set_page_config)."""
    import streamlit as st
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def style_figure(fig, height=None):
    """Make a Plotly figure match the dark theme."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1b1b1b",
                      font=dict(color=TEXT), **({"height": height} if height else {}))
    return fig
