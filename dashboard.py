import os
import base64

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


st.set_page_config(
    page_title="CDC SSF UNS - Placement Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PALET WARNA SEMANTIK
# ---------------------------------------------------------------------------
COLOR_SIENNA = "#872408"       # negatif / ghosting / rejection
COLOR_COCOA = "#E2782F"        # aksen utama
COLOR_JASMINE = "#F7D475"      # highlight / pembanding netral
COLOR_DRAB_DARK = "#403314"    # teks / netral gelap
COLOR_SEAL_BROWN = "#4A230E"   # background aksen
COLOR_OLIVE = "#6B7A3D"        # sukses
COLOR_BG_CARD = "#FBF4E8"      # background card lembut


def tint(hex_color: str, factor: float) -> str:
    """Campur warna dengan putih; factor 0 = warna asli, 1 = putih penuh."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * factor)
    g = round(g + (255 - g) * factor)
    b = round(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


FUNNEL_STAGES = [
    "Selecting Student by Company",
    "Study Case",
    "CDC Briefing Student",
    "Interview User",
    "Final Interview",
    "Placement",
]
PALETTE_SEQUENTIAL = [
    COLOR_JASMINE,
    tint(COLOR_COCOA, 0.35),
    COLOR_COCOA,
    COLOR_SIENNA,
    COLOR_SEAL_BROWN,
    COLOR_DRAB_DARK,
]

# Aturan resmi FAQ: Ghosting diasumsikan dari PIHAK PERUSAHAAN,
# dihitung dari send_date per batch pengiriman (baris tracking_company).
FU1_DAYS = 7
FU2_DAYS = 14
FU3_DAYS = 21
GHOSTING_DAYS = 28
SYNC_STALE_DAYS = 14

# FAQ: kolom "eligible" = kolom "ketersediaan".
# Nilai riil di data: status = Active/Lulus/Inactive/Cuti,
# ketersediaan = Available/Placed/Tidak Aktif. Varian ID tetap diterima.
VAL_STATUS_AKTIF = {"active", "aktif"}
VAL_TERSEDIA = {"available", "tersedia"}
VAL_ADA = {"ada"}

REJECTION_STAGES = [
    "Rejection Screening CV",
    "Rejection Study Case",
    "Rejection Interview User",
    "Rejection Final Interview",
]

# Tahap terjauh yang dicapai kandidat (untuk funnel & sinyal respons).
STAGE_RANK = {s: i for i, s in enumerate(FUNNEL_STAGES)}
PROGRESS_TO_RANK = {
    **STAGE_RANK,
    "Finish": 5,      # sudah selesai magang = pernah placement
    "FU 1": 0, "FU 2": 0, "FU 3": 0, "Ghosting": 0,  # tak pernah lolos screening
}
REJECTION_TO_RANK = {
    "Rejection Screening CV": 0,
    "Rejection Study Case": 1,
    "Rejection Interview User": 3,
    "Rejection Final Interview": 4,
    "Placement": 5,
    "Ghosting": 0,
}

# ---------------------------------------------------------------------------
# STYLE - CSS Custom, gaya BENTO BOX
# - Font judul: Nohemi (Fontshare); body: Inter.
# - Background halaman: gradasi krem lembut.
# - Kartu KPI: flat gelap; kartu paling penting diberi OUTLINE GRADASI + glow.
# - Semua permukaan utama dipaksa !important -> tahan dark mode di device mana pun.
# ---------------------------------------------------------------------------
PAGE_BG_GRAD = "linear-gradient(165deg, #FBF6EC 0%, #F6EFE3 45%, #EEDFC8 100%)"
CARD_BG = "#FFFFFF"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

/* Nohemi tidak tersedia di Google Fonts/Fontshare - di-load dari folder
   static/ repo (butuh [server] enableStaticServing di config.toml).
   Urutan sumber: font ter-install lokal dulu (local), lalu beberapa
   kemungkinan nama file di static/. Kalau semuanya tidak ada, browser
   jatuh ke Space Grotesk (fallback paling mirip Nohemi). */
@font-face {{
    font-family: 'Nohemi';
    src: local('Nohemi SemiBold'), local('Nohemi SemBd'), local('Nohemi Bold'), local('Nohemi'),
         url('./app/static/Nohemi-SemiBold.woff2') format('woff2'),
         url('./app/static/Nohemi-Bold.woff2') format('woff2'),
         url('./app/static/Nohemi-SemiBold.otf') format('opentype'),
         url('./app/static/Nohemi-Bold.otf') format('opentype');
    font-weight: 500 800;
    font-display: swap;
}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

/* ===== paksa latar & teks dasar (dark-mode proof) =====
   background: MESH GRADIENT lembut dari palet - beberapa radial-gradient
   ditumpuk di atas warna dasar krem */
.stApp {{
    background:
        linear-gradient(90deg, #FBF2E0 0px, #FBF2E0 400px, rgba(251, 242, 224, 0) 540px),
        radial-gradient(at 38% 6%, rgba(247, 212, 117, 0.55) 0px, transparent 38%),
        radial-gradient(at 92% 5%, rgba(255, 154, 74, 0.42) 0px, transparent 42%),
        radial-gradient(at 88% 90%, rgba(214, 106, 60, 0.26) 0px, transparent 45%),
        radial-gradient(at 45% 96%, rgba(164, 176, 94, 0.30) 0px, transparent 40%),
        radial-gradient(at 55% 45%, rgba(255, 250, 240, 0.85) 0px, transparent 60%),
        #FBF2E0 !important;
    background-attachment: fixed !important;
}}
header[data-testid="stHeader"] {{ background: transparent !important; }}
div[data-testid="stMarkdownContainer"] p {{ color: {COLOR_DRAB_DARK} !important; }}
div[data-testid="stWidgetLabel"] p {{ color: {COLOR_SEAL_BROWN} !important; font-weight: 600; }}

/* ===== SIDEBAR navigasi: panel gelap membulat gaya bento ===== */
div[data-testid="stSidebarUserContent"] {{ padding-top: 0.6rem !important; }}
div[data-testid="stSidebarHeader"] {{ padding: 0 !important; height: 0 !important; }}
section[data-testid="stSidebar"] {{
    background:
        radial-gradient(at 88% 10%, rgba(247, 212, 117, 0.28) 0px, transparent 40%),
        radial-gradient(at 8% 42%, rgba(226, 120, 47, 0.35) 0px, transparent 45%),
        radial-gradient(at 82% 88%, rgba(164, 176, 94, 0.22) 0px, transparent 45%),
        linear-gradient(180deg, #A64B1A 0%, #7C3413 45%, {COLOR_SEAL_BROWN} 100%) !important;
    border-radius: 0 22px 22px 0;
}}
/* sidebar tidak bisa ditutup ataupun digeser lebarnya */
div[data-testid="stSidebarCollapseButton"],
button[data-testid="stExpandSidebarButton"],
div[data-testid="collapsedControl"],
div[data-testid="stSidebarResizeHandle"],
button[data-testid="stSidebarResizeHandle"] {{
    display: none !important;
}}
section[data-testid="stSidebar"] {{
    width: 255px !important;
    min-width: 255px !important;
    max-width: 255px !important;
}}
section[data-testid="stSidebar"] * {{ color: {tint(COLOR_JASMINE, 0.55)}; }}
.side-brand {{
    text-align: center;
    margin: 4px 0 14px 0;
}}
.side-brand .brand-name {{
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: {COLOR_JASMINE} !important;
    line-height: 1.15;
}}
.side-brand .brand-sub {{
    font-size: 0.78rem;
    color: {tint(COLOR_JASMINE, 0.5)} !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.side-sync {{
    position: fixed;
    bottom: 12px;
    left: 0;
    width: 255px;
    text-align: center;
    font-size: 0.72rem;
    color: {tint(COLOR_JASMINE, 0.45)} !important;
    z-index: 50;
}}
div[data-testid="stPopover"], .stPopover {{
    width: 100%;
}}
div[data-testid="stPopover"] > div, .stPopover > div {{
    width: 100%;
}}
div[data-testid="stPopover"] button, .stPopover button {{
    width: 100% !important;
}}
/* izinkan pill menu aktif menembus tepi kanan sidebar (efek tab menyatu);
   konten sidebar pendek sehingga scroll tidak dibutuhkan */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
    overflow: visible !important;
}}
/* logo + daun autumn di belakangnya */
.side-logo {{
    position: relative;
    text-align: center;
    margin: 0 0 2px 0;
}}
.side-leaves {{
    position: absolute;
    top: -16px;
    left: 50%;
    transform: translateX(-50%);
    width: 205px;
    max-width: 96%;
    opacity: 0.6;
    transform-origin: center;
    animation: leafSway 6s ease-in-out infinite;
}}
.side-logo-img {{
    position: relative;
    width: 158px;
    max-width: 82%;
}}
@keyframes leafSway {{
    0%, 100% {{ transform: translateX(-50%) rotate(-4deg); }}
    50% {{ transform: translateX(-50%) rotate(5deg) translateY(5px); }}
}}
.page-title {{
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: {COLOR_SEAL_BROWN};
    margin: 0;
}}
div[data-testid="stPageLink"] a {{
    border-radius: 12px;
    padding: 4px 10px;
}}
div[data-testid="stPageLink"] a p,
div[data-testid="stPageLink"] a span {{
    color: {tint(COLOR_JASMINE, 0.5)} !important;
    font-weight: 600;
}}
div[data-testid="stPageLink"] a:hover {{
    background: rgba(226, 120, 47, 0.22) !important;
}}
div[data-testid="stPageLink"] a[aria-current="page"] {{
    background: #FFF6E6 !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
}}
div[data-testid="stPageLink"] a[aria-current="page"] p,
div[data-testid="stPageLink"] a[aria-current="page"] span {{
    color: {COLOR_SEAL_BROWN} !important;
    font-weight: 700;
}}


/* input & multiselect */
div[data-baseweb="select"] > div {{
    background-color: {CARD_BG} !important;
    border-color: {tint(COLOR_COCOA, 0.55)} !important;
    color: {COLOR_DRAB_DARK} !important;
}}
span[data-baseweb="tag"] {{
    background-color: {COLOR_COCOA} !important;
    color: #FFFFFF !important;
}}
div[data-testid="stDateInput"] input {{
    background-color: {CARD_BG} !important;
    color: {COLOR_DRAB_DARK} !important;
}}

/* layout rapat: target muat satu layar tanpa scroll */
.block-container {{
    padding-top: 0.8rem !important;
    padding-bottom: 0.7rem !important;
    padding-left: 2rem !important;
    padding-right: 2.2rem !important;
}}
div[data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}

.dash-header {{
    padding: 12px 22px 13px 22px;
    border-radius: 14px;
    background: linear-gradient(135deg, {COLOR_SEAL_BROWN} 0%, {COLOR_SIENNA} 100%);
    margin-bottom: 8px;
}}
.dash-header h1 {{
    color: #FFF8EE !important;
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0;
}}
/* selector panjang: harus menang dari aturan stMarkdownContainer p di atas */
div[data-testid="stMarkdownContainer"] .dash-header p {{
    color: {tint(COLOR_JASMINE, 0.4)} !important;
    font-size: 0.84rem;
    margin: 2px 0 0 0;
}}

/* ===== KPI cards: FLAT; kartu terpenting diberi outline gradasi + glow ===== */
.kpi-row {{
    display: flex;
    gap: 14px;
    margin: 6px 0 18px 0;
    flex-wrap: wrap;
}}
/* kartu biasa: krem terang; kartu highlight: gelap + outline gradasi + glow
   -> satu kartu terpenting langsung menonjol di antara kartu terang */
.kpi-card {{
    flex: 1;
    min-width: 150px;
    background: #FFFFFF;
    border: 1px solid rgba(226, 120, 47, 0.16);
    border-radius: 14px;
    padding: 14px 12px 11px 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(74, 35, 14, 0.07);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
.kpi-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 24px rgba(74, 35, 14, 0.16);
}}
.kpi-card.kpi-hl {{
    border: 2px solid transparent;
    background:
        linear-gradient({COLOR_SEAL_BROWN}, {COLOR_SEAL_BROWN}) padding-box,
        linear-gradient(135deg, {COLOR_JASMINE} 0%, {COLOR_COCOA} 45%, {COLOR_SIENNA} 100%) border-box;
    box-shadow: 0 0 18px rgba(226, 120, 47, 0.55);
}}
.kpi-value {{
    color: {COLOR_SIENNA};
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
}}
.kpi-label {{
    color: {tint(COLOR_DRAB_DARK, 0.15)};
    font-size: 0.76rem;
    font-weight: 600;
    margin-top: 4px;
}}
.kpi-sub {{
    color: {tint(COLOR_DRAB_DARK, 0.4)};
    font-size: 0.7rem;
    margin-top: 2px;
}}
.kpi-card.kpi-hl .kpi-value {{ color: {COLOR_JASMINE}; }}
.kpi-card.kpi-hl .kpi-label {{ color: {tint(COLOR_JASMINE, 0.55)}; }}
.kpi-card.kpi-hl .kpi-sub {{ color: {tint(COLOR_COCOA, 0.35)}; }}

div[data-testid="stMetric"] {{
    background-color: #FFFFFF;
    border: 1px solid rgba(226, 120, 47, 0.16);
    border-radius: 12px;
    padding: 10px 14px 8px 14px;
}}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetric"] label p {{
    color: {COLOR_SEAL_BROWN} !important;
    font-weight: 600;
    font-size: 0.8rem;
}}
div[data-testid="stMetricValue"] {{
    color: {COLOR_SIENNA} !important;
    font-weight: 700;
    font-size: 1.5rem;
}}

.section-title {{
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: {COLOR_SEAL_BROWN};
    margin: 2px 0 1px 0;
}}
.section-caption {{
    color: {tint(COLOR_DRAB_DARK, 0.35)};
    font-size: 0.8rem;
    margin-bottom: 8px;
    line-height: 1.4;
}}

.insight-box {{
    border-left: 5px solid var(--accent);
    background-color: var(--bg);
    padding: 9px 14px;
    border-radius: 8px;
    margin: 4px 0 8px 0;
    font-size: 0.86rem;
    line-height: 1.45;
    color: {COLOR_DRAB_DARK} !important;
}}
.insight-box b {{ color: {COLOR_SEAL_BROWN}; }}

/* ===== kartu bento: putih, rounded besar, shadow lembut ===== */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 18px !important;
    border: 1px solid rgba(226, 120, 47, 0.14) !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(74, 35, 14, 0.09);
    transition: box-shadow 0.25s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    box-shadow: 0 12px 28px rgba(74, 35, 14, 0.14);
}}
div[data-testid="stExpander"] {{
    background-color: {CARD_BG} !important;
    border-radius: 16px !important;
    border: 1px solid {tint(COLOR_COCOA, 0.8)} !important;
}}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {{
    color: {COLOR_SEAL_BROWN} !important;
    font-weight: 600;
}}

div[data-testid="stAlertContainer"], div[data-testid="stAlert"] {{
    background-color: {COLOR_BG_CARD} !important;
    border: 1px solid {tint(COLOR_COCOA, 0.65)} !important;
    border-left: 5px solid {COLOR_COCOA} !important;
    border-radius: 10px !important;
}}
div[data-testid="stAlertContainer"] p, div[data-testid="stAlert"] p {{
    color: {COLOR_DRAB_DARK} !important;
}}

/* stagger antar kartu KPI (nama keyframe disuntik per-halaman di page_header) */
.kpi-card:nth-child(1) {{ animation-delay: 0.00s; }}
.kpi-card:nth-child(2) {{ animation-delay: 0.06s; }}
.kpi-card:nth-child(3) {{ animation-delay: 0.12s; }}
.kpi-card:nth-child(4) {{ animation-delay: 0.18s; }}
.kpi-card:nth-child(5) {{ animation-delay: 0.24s; }}

#MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# CSS tambahan: badge Business Task, kotak "Catatan Analis" (gaya Analyst's
# Note), delta KPI naik/turun, dan kartu profil tim.
st.markdown(f"""
<style>
.bt-badge {{
    display: inline-block;
    background: linear-gradient(135deg, {COLOR_COCOA} 0%, {COLOR_SIENNA} 100%);
    color: #FFF6E6 !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 3px 9px;
    border-radius: 999px;
    margin-left: 10px;
    vertical-align: middle;
    box-shadow: 0 2px 6px rgba(135, 36, 8, 0.25);
}}
.analis-box {{
    background: linear-gradient(160deg, {tint(COLOR_SEAL_BROWN, 0.02)} 0%, {COLOR_SIENNA} 120%);
    border-radius: 14px;
    padding: 13px 16px 11px 16px;
    box-shadow: 0 5px 14px rgba(74, 35, 14, 0.18);
    height: 100%;
}}
.analis-box .analis-title {{
    font-family: 'Nohemi', 'Space Grotesk', 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {COLOR_JASMINE} !important;
    margin: 0 0 7px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.analis-box .analis-item {{
    color: #FDF6EA !important;
    font-size: 0.82rem;
    line-height: 1.5;
    margin: 0 0 5px 0;
    padding-left: 14px;
    position: relative;
}}
.analis-box .analis-item::before {{
    content: "";
    position: absolute;
    left: 0; top: 8px;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {COLOR_JASMINE};
}}
.analis-box .analis-item b {{ color: #FFFFFF !important; }}
.kpi-delta {{
    font-size: 0.72rem;
    font-weight: 700;
    margin-top: 3px;
}}
.kpi-delta.up {{ color: {COLOR_OLIVE}; }}
.kpi-delta.down {{ color: {COLOR_SIENNA}; }}
.kpi-card.kpi-hl .kpi-delta.up {{ color: {tint(COLOR_OLIVE, 0.45)}; }}
.kpi-card.kpi-hl .kpi-delta.down {{ color: {tint(COLOR_SIENNA, 0.45)}; }}
.team-card {{
    background: #FFFFFF;
    border: 1px solid rgba(226, 120, 47, 0.16);
    border-radius: 16px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 5px 14px rgba(74, 35, 14, 0.08);
    height: 100%;
}}
.team-card .team-avatar {{
    width: 62px; height: 62px;
    border-radius: 50%;
    margin: 0 auto 10px auto;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Nohemi', 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: #FFF6E6;
    background: linear-gradient(135deg, {COLOR_JASMINE} 0%, {COLOR_COCOA} 50%, {COLOR_SIENNA} 100%);
}}
.team-card .team-name {{
    font-family: 'Nohemi', 'Space Grotesk', sans-serif;
    font-weight: 700; font-size: 1rem; color: {COLOR_SEAL_BROWN};
}}
.team-card .team-role {{ color: {COLOR_COCOA}; font-size: 0.78rem; font-weight: 600; margin-top: 2px; }}
.team-card .team-nim {{ color: {tint(COLOR_DRAB_DARK, 0.4)}; font-size: 0.74rem; margin-top: 4px; }}

/* ===== samakan TINGGI kartu berdampingan =====
   kolom Streamlit sudah stretch (flex), tapi isi kartu tingginya mengikuti
   konten. Paksa vertical-block + border-wrapper mengisi tinggi kolom supaya
   kotak Catatan Analis di sebelah chart tinggi (tidak menyisakan celah). */
div[data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{ height: 100%; }}
div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"] {{ height: 100%; }}
</style>
""", unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def insight(text: str, kind: str = "info"):
    accents = {
        "info": COLOR_COCOA,
        "success": COLOR_OLIVE,
        "warning": COLOR_JASMINE,
        "error": COLOR_SIENNA,
    }
    accent = accents.get(kind, COLOR_COCOA)
    st.markdown(
        f'<div class="insight-box" style="--accent:{accent}; --bg:{tint(accent, 0.88)};">'
        f'<b>Insight:</b> {text}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards):
    """Deret kartu KPI flat; beri "highlight": True pada SATU kartu terpenting
    untuk memberi outline gradasi + glow (pola aksen tunggal). Opsional
    "delta": angka (persen) yang dirender sebagai panah naik/turun."""
    html = '<div class="kpi-row">'
    for c in cards:
        cls = "kpi-card kpi-hl" if c.get("highlight") else "kpi-card"
        help_attr = f' title="{c["help"]}"' if c.get("help") else ""
        sub = f'<div class="kpi-sub">{c["sub"]}</div>' if c.get("sub") else ""
        delta_html = ""
        if c.get("delta") is not None:
            dv = c["delta"]
            arah = "up" if dv >= 0 else "down"
            panah = "▲" if dv >= 0 else "▼"  # segitiga (bukan emoji)
            lbl = c.get("delta_label", "vs tahun lalu")
            delta_html = f'<div class="kpi-delta {arah}">{panah} {abs(dv):.1f}% {lbl}</div>'
        html += (
            f'<div class="{cls}"{help_attr}>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div>{sub}{delta_html}</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def catatan_analis(points, title="Catatan Analis"):
    """Kotak narasi insight bergaya "Analyst's Note" (meniru dashboard juara).
    points: list string (boleh mengandung <b>). Ditaruh berdampingan dengan
    chart untuk memberi interpretasi langsung di halaman."""
    items = "".join(f'<div class="analis-item">{p}</div>' for p in points)
    st.markdown(
        f'<div class="analis-box"><div class="analis-title">{title}</div>'
        f'{items}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=300):
    has_title = bool(fig.layout.title.text)
    has_legend = len(fig.data) > 1 or any(tr.type == "pie" for tr in fig.data)
    top_margin = 38 if has_title else (34 if has_legend else 14)

    # legend dengan banyak kategori dipindah ke BAWAH chart supaya tidak
    # bertabrakan dengan judul (mis. rekap 18 program studi)
    n_legend = 0
    for tr in fig.data:
        if getattr(tr, "showlegend", True) is False:
            continue
        if tr.type == "pie":
            n_legend += len(set(tr.labels)) if tr.labels is not None else 1
        else:
            n_legend += 1
    legend_below = n_legend > 5
    if legend_below:
        legend_cfg = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                          yanchor="top", y=-0.18, xanchor="left", x=0)
        bottom_margin = 8
        height += 80
    else:
        legend_cfg = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                          yanchor="bottom", y=1.0, xanchor="right", x=1.0)
        bottom_margin = 8

    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=COLOR_DRAB_DARK, size=11),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=top_margin, b=bottom_margin),
        legend=legend_cfg,
        hoverlabel=dict(bgcolor="white", font_size=11),
        height=height,
    )
    # angka sumbu ditulis penuh dgn pemisah ribuan (mis. 8,000 bukan "8k")
    fig.update_yaxes(tickformat=",", separatethousands=True)
    fig.update_xaxes(separatethousands=True)
    # title_font hanya diset bila chart punya judul - plotly.js merender
    # teks "undefined" kalau properti title diisi tanpa title text.
    if has_title:
        fig.update_layout(title_font=dict(size=13, color=COLOR_SEAL_BROWN))
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}


def show_chart(fig, height=300):
    st.plotly_chart(style_fig(fig, height), width="stretch", config=PLOTLY_CONFIG)


def hitung_status_followup(hari_sejak_kirim, sudah_direspon) -> str:
    if sudah_direspon or pd.isna(hari_sejak_kirim):
        return "Direspon"
    if hari_sejak_kirim > GHOSTING_DAYS:
        return "Ghosting"
    if hari_sejak_kirim > FU3_DAYS:
        return "FU 3"
    if hari_sejak_kirim > FU2_DAYS:
        return "FU 2"
    if hari_sejak_kirim > FU1_DAYS:
        return "FU 1"
    return "Menunggu Respons (Normal)"


def resolve_col(df: pd.DataFrame, base: str, suffixes=("_student", "", "_status")):
    for suf in suffixes:
        cand = f"{base}{suf}"
        if cand in df.columns:
            return cand
    return None


def norm_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


BULAN_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12,
}


def parse_bulan_masuk(x):
    """Parse 'Februari 2023' (nama bulan Indonesia) menjadi Timestamp awal bulan."""
    try:
        b, t = str(x).strip().lower().split()
        return pd.Timestamp(int(t), BULAN_ID[b], 1)
    except (ValueError, KeyError):
        return pd.NaT


# ---------------------------------------------------------------------------
# LOAD + PREPARE DATA (SEKALI SAJA)
# Semua pembacaan CSV, pembersihan tipe, dan merge dilakukan dalam satu fungsi
# ber-cache_resource: rerun halaman tidak mengulang komputasi berat apa pun.
# Konsekuensi cache_resource: konsumen WAJIB .copy() sebelum mengubah frame.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menyiapkan data (hanya sekali)...")
def load_all() -> dict:
    company = pd.read_csv("cleaned_company.csv")
    talent_request = pd.read_csv("cleaned_talent_request.csv")
    tracking_company = pd.read_csv("cleaned_tracking_company.csv")
    tracking_student = pd.read_csv("cleaned_tracking_student.csv")
    student_all = pd.read_csv("cleaned_student_all.csv")
    status_student = pd.read_csv("cleaned_status_student.csv")

    for df in (company, talent_request, tracking_company, tracking_student, student_all, status_student):
        df.columns = df.columns.str.strip().str.lower()

    for df, col in [
        (company, "created_at"),
        (talent_request, "request_date"),
        (tracking_company, "request_date"),
        (tracking_company, "send_date"),
        (tracking_student, "last_update"),
        (status_student, "sync_date"),
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    key_cols = [
        (company, "id_company"), (talent_request, "id_talent_req"), (talent_request, "id_company"),
        (tracking_company, "id_tracking_company"), (tracking_company, "id_talent_req"), (tracking_company, "id_company"),
        (tracking_student, "id_tracking_student"), (tracking_student, "id_tracking_company"), (tracking_student, "nim"),
        (student_all, "nim"), (status_student, "id_status"), (status_student, "nim"),
    ]
    for df, col in key_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col, df in [("jumlah_dikirimkan", tracking_company), ("jumlah_permintaan", tracking_company),
                    ("headcount", talent_request), ("minimum_semester", talent_request),
                    ("ipk", status_student), ("semester", student_all),
                    ("internship_semester", tracking_student)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- tahap terjauh tiap kandidat (dasar funnel & sinyal respons) ----
    rank = tracking_student["progress_student"].map(PROGRESS_TO_RANK)
    mask_rejected = tracking_student["progress_student"].eq("Rejected")
    rank = rank.mask(mask_rejected, tracking_student["rejection"].map(REJECTION_TO_RANK))
    tracking_student["stage_reached"] = rank.fillna(0).astype(int)

    # ---- MASTER (prinsip ERD: nama perusahaan dari master COMPANY) ----
    master = tracking_student.merge(tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc"))
    master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
    master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
    master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
    master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))

    master["tahun_update"] = master["last_update"].dt.year
    if "send_date" in master.columns:
        master["lama_proses_hari"] = (master["last_update"] - master["send_date"]).dt.days

    # ---- BATCH PENGIRIMAN (basis ghosting BT-05) ----
    tc_base = tracking_company[tracking_company["send_date"].notna()].copy()
    tc_base = tc_base.merge(company[["id_company", "company_name"]], on="id_company", how="left")
    tc_base["tahun_tc"] = tc_base["send_date"].dt.year

    # Batch dianggap DIRESPON perusahaan jika minimal satu mahasiswanya
    # lolos ke tahap setelah screening, ATAU sudah ada keputusan
    # (Placement / Rejection apa pun - menolak juga bentuk respons).
    # Kolom progress di tracking_company sengaja TIDAK dipakai sebagai
    # sinyal respons: hasil validasi data menunjukkan nilainya tidak
    # berkorelasi dengan pergerakan kandidat.
    decided = tracking_student["rejection"].isin(["Placement"] + REJECTION_STAGES)
    resp = (
        tracking_student.assign(responded=(tracking_student["stage_reached"] >= 1) | decided)
        .groupby("id_tracking_company")["responded"].any()
    )
    tc_base["sudah_direspon"] = tc_base["id_tracking_company"].map(resp).fillna(False)

    default_ref = tracking_company["send_date"].max()
    if pd.isna(default_ref):
        default_ref = pd.Timestamp(datetime.now().date())

    # pool mahasiswa + status kesiapan (untuk matching & ML)
    pool = student_all.merge(status_student, on="nim", how="inner", suffixes=("_student", "_status"))
    prodi_col = resolve_col(pool, "program_studi") or "program_studi"
    semester_col = resolve_col(pool, "semester") or "semester"
    pool["_prodi_norm"] = norm_text(pool[prodi_col])
    pool["_eligible"] = (
        norm_text(pool["ketersediaan"]).isin(VAL_TERSEDIA)
        & norm_text(pool["status"]).isin(VAL_STATUS_AKTIF)
    )

    # dimensi waktu untuk analisis demand vs supply per bulan:
    # - supply: bulan_masuk mahasiswa (2019-2023), dijadikan stok kumulatif
    # - demand: request_date + headcount, di-explode per bidang studi
    if "bulan_masuk" in student_all.columns:
        student_all["masuk_dt"] = student_all["bulan_masuk"].map(parse_bulan_masuk)
    else:
        student_all["masuk_dt"] = pd.NaT

    # SUPPLY = estimasi kapan mahasiswa SIAP MAGANG (bukan saat masuk kuliah):
    # bulan masuk + (semester magang - 1) x 6 bulan. Semester magang diambil
    # dari histori magang tiap mahasiswa; kalau belum pernah magang, pakai
    # median seluruh data (5).
    med_sem_global = tracking_student["internship_semester"].median()
    if pd.isna(med_sem_global):
        med_sem_global = 5
    sem_per_student = tracking_student.groupby("nim")["internship_semester"].median()
    sem_est = student_all["nim"].map(sem_per_student).fillna(med_sem_global).clip(1, 14)
    offset_bulan = ((sem_est - 1) * 6).round().astype(int)
    masuk_period = pd.PeriodIndex(student_all["masuk_dt"], freq="M")
    student_all["siap_dt"] = (masuk_period + offset_bulan.to_numpy()).to_timestamp()

    dm = talent_request.loc[
        talent_request["request_date"].notna(),
        ["request_date", "headcount", "bidang_studi_dibutuhkan"],
    ].copy()
    dm["bulan"] = dm["request_date"].dt.to_period("M").dt.to_timestamp()

    # TOTAL demand per bulan: tiap posisi dihitung SEKALI (headcount apa adanya).
    # Dipakai saat filter "Semua bidang" -> total jujur, tidak menggelembung.
    demand_monthly_total = dm.groupby("bulan")["headcount"].sum().reset_index()

    # demand PER BIDANG: headcount PENUH tiap posisi yang menerima bidang itu.
    # Dunia nyata: 1 posisi (mis. butuh 3, terima "Informatika, Elektro") bisa
    # diisi mahasiswa dari bidang mana pun yang diterima -> tiap bidang terekspos
    # ke seluruh slot posisi itu. Konsekuensinya angka antar-bidang TIDAK
    # dijumlahkan (satu posisi muncul penuh di >1 bidang).
    dmb = dm.copy()
    dmb["bidang_norm"] = dmb["bidang_studi_dibutuhkan"].astype(str).str.split(",")
    dmb = dmb.explode("bidang_norm")
    dmb["bidang_norm"] = dmb["bidang_norm"].str.strip().str.lower()
    dmb = dmb[dmb["bidang_norm"] != ""]
    demand_monthly = dmb.groupby(["bulan", "bidang_norm"])["headcount"].sum().reset_index()

    # pemenuhan talent request (dipakai Overview & Mitra)
    tr_fulfill = talent_request.merge(
        tracking_company.groupby("id_talent_req")["jumlah_dikirimkan"].sum().reset_index(),
        on="id_talent_req", how="left",
    )
    tr_fulfill = tr_fulfill.merge(company[["id_company", "company_name"]], on="id_company", how="left")
    tr_fulfill["jumlah_dikirimkan"] = tr_fulfill["jumlah_dikirimkan"].fillna(0)
    tr_fulfill["belum_terpenuhi"] = tr_fulfill["headcount"] - tr_fulfill["jumlah_dikirimkan"]
    tr_fulfill["pemenuhan"] = (tr_fulfill["jumlah_dikirimkan"] / tr_fulfill["headcount"]).clip(0, 1)

    return {
        "company": company,
        "talent_request": talent_request,
        "tracking_company": tracking_company,
        "tracking_student": tracking_student,
        "student_all": student_all,
        "status_student": status_student,
        "master": master,
        "tc_base": tc_base,
        "pool": pool,
        "pool_prodi_col": prodi_col,
        "pool_semester_col": semester_col,
        "tr_fulfill": tr_fulfill,
        "demand_monthly": demand_monthly,
        "demand_monthly_total": demand_monthly_total,
        "default_ref_date": default_ref,
    }


DATA = load_all()
company = DATA["company"]
talent_request = DATA["talent_request"]
tracking_company = DATA["tracking_company"]
tracking_student = DATA["tracking_student"]
student_all = DATA["student_all"]
status_student = DATA["status_student"]
master = DATA["master"]
tc_base = DATA["tc_base"]
tr_fulfill = DATA["tr_fulfill"]
DEFAULT_REF_DATE = DATA["default_ref_date"]

COMPANY_NAME_COL = "company_name" if "company_name" in master.columns else "company"


# ---------------------------------------------------------------------------
# MATCH SUMMARY (VEKTORISASI)
# Jumlah kandidat per (prodi, ambang semester) dihitung SEKALI sebagai matriks
# lookup; tiap talent request tinggal menjumlahkan dari matriks itu.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Menghitung ringkasan kecocokan (hanya sekali)...")
def compute_match_summary() -> pd.DataFrame:
    pool = DATA["pool"]
    semester_col = DATA["pool_semester_col"]

    max_thr = 15
    sem = pd.to_numeric(pool[semester_col], errors="coerce").fillna(0).to_numpy()
    elig = pool["_eligible"].to_numpy()
    cnt_all, cnt_elig = {}, {}
    for p, idx in pool.groupby("_prodi_norm").indices.items():
        s = sem[idx]
        e = elig[idx]
        cnt_all[p] = np.array([(s >= t).sum() for t in range(max_thr)])
        cnt_elig[p] = np.array([((s >= t) & e).sum() for t in range(max_thr)])

    tr_named = talent_request.merge(company[["id_company", "company_name"]], on="id_company", how="left")

    rows = []
    for tr in tr_named.itertuples(index=False):
        bidang = [x.strip() for x in str(getattr(tr, "bidang_studi_dibutuhkan", "")).split(",") if x.strip()]
        bidang_norm = [b.lower() for b in bidang]
        min_sem = getattr(tr, "minimum_semester", 0)
        min_sem = 0 if pd.isna(min_sem) else int(min(max(min_sem, 0), max_thr - 1))

        n_prodi = sum(int(cnt_all[b][0]) for b in bidang_norm if b in cnt_all)
        n_sem = sum(int(cnt_all[b][min_sem]) for b in bidang_norm if b in cnt_all)
        n_final = sum(int(cnt_elig[b][min_sem]) for b in bidang_norm if b in cnt_elig)

        rows.append({
            "id_talent_req": tr.id_talent_req,
            "nama_posisi": getattr(tr, "nama_posisi", "-"),
            "company_name": getattr(tr, "company_name", "-"),
            "bidang_dibutuhkan": ", ".join(bidang) if bidang else "(kosong)",
            "min_semester": min_sem,
            "cocok_prodi": n_prodi,
            "cocok_prodi_semester": n_sem,
            "kandidat_final": n_final,
        })

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# FILTER - di dalam POPOVER per tab supaya hemat ruang vertikal (no-scroll)
# ---------------------------------------------------------------------------
FILTER_TAHUN = sorted(int(t) for t in master["tahun_update"].dropna().unique())
FILTER_PRODI = sorted(student_all["program_studi"].dropna().unique().tolist()) if "program_studi" in student_all.columns else []
FILTER_JENIS = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist()) if "jenis_penempatan" in talent_request.columns else []


def run_gsap_animations(slug: str):
    """Animasi interaktif via GSAP (dimuat dari CDN). Streamlit menyaring tag
    <script> di markdown, jadi JS dijalankan lewat components.html - iframe
    same-origin yang boleh memanipulasi DOM halaman induk. Kalau CDN gagal
    dimuat, halaman tetap tampil normal tanpa animasi (graceful fallback)."""
    # st.iframe menggantikan components.html (dihapus Streamlit per Jun 2026);
    # fallback ke components.html untuk versi lama.
    _html_embed = getattr(st, "iframe", components.html)
    try:
        _current = nav.url_path or ""
    except NameError:
        _current = ""
    _html_embed(
        _GSAP_HTML.replace("__SLUG__", slug).replace("__HREF__", _current),
        height=1,
    )


_GSAP_HTML = """
        <script>
        // page-marker: __SLUG__  (membuat konten iframe unik per halaman ->
        // komponen di-mount ulang -> animasi jalan lagi setiap pindah menu)
        (function () {
            const SLUG = "__SLUG__";
            const HREF = "__HREF__";
            const P = window.parent;
            const doc = P.document;

            // lidah tab aktif dibuat flush PERSIS ke tepi sidebar dengan
            // mengukur geometri asli di browser (bukan menebak padding)
            function fitActiveTab() {
                const sb = doc.querySelector("section[data-testid='stSidebar']");
                const a = doc.querySelector(
                    "section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='" + HREF + "']");
                if (!sb || !a) return false;
                // JS memegang PENUH geometri pill dengan inline !important -
                // kebal terhadap perbedaan CSS antar versi/deploy Streamlit
                a.style.setProperty("margin-left", "0px", "important");
                a.style.setProperty("margin-right", "0px", "important");
                a.style.setProperty("border-radius", "999px 0 0 999px", "important");
                a.style.setProperty("box-sizing", "border-box", "important");
                a.style.removeProperty("width");
                const r = a.getBoundingClientRect();
                const gap = sb.getBoundingClientRect().right - r.right;
                if (Math.abs(gap) > 0.5) a.style.setProperty("width", (r.width + gap) + "px", "important");
                return true;
            }

            function animate() {
                if (!fitActiveTab()) return false;
                const gsap = P.gsap;
                if (!gsap) return true;
                // Animasi masuk kartu/kontainer ditangani oleh CSS (lihat blok STYLE),
                // karena GSAP di dalam iframe tidak bisa menyentuh DOM halaman induk
                // ("GSAP target not found"). GSAP di sini hanya untuk count-up angka.

                // count-up angka KPI: "41,600", "152.3%", "51 hari"
                doc.querySelectorAll(".kpi-value").forEach(function (el) {
                    if (el.dataset.counted === SLUG) return;
                    const txt = el.textContent.trim();
                    const m = txt.match(/^([\\d.,]+)(.*)$/);
                    if (!m) return;
                    const target = parseFloat(m[1].replace(/,/g, ""));
                    if (isNaN(target)) return;
                    const suffix = m[2] || "";
                    const decimal = m[1].includes(".");
                    el.dataset.counted = SLUG;
                    const obj = { v: 0 };
                    gsap.to(obj, {
                        v: target, duration: 1.1, ease: "power2.out",
                        onUpdate: function () {
                            el.textContent = (decimal
                                ? obj.v.toFixed(1)
                                : Math.round(obj.v).toLocaleString("en-US")) + suffix;
                        },
                    });
                });
                return true;
            }

            function tryAnimate(attempt) {
                if (animate() !== false) return;
                if (attempt < 12) setTimeout(function () { tryAnimate(attempt + 1); }, 300);
            }

            P.addEventListener("resize", fitActiveTab);

            // fit lidah tab jalan mandiri - tidak menunggu / tergantung GSAP
            (function retryFit(i) {
                if (!fitActiveTab() && i < 15) setTimeout(function () { retryFit(i + 1); }, 300);
            })(0);

            if (P.gsap) {
                tryAnimate(0);
            } else if (!P._gsapLoading) {
                P._gsapLoading = true;
                const s = doc.createElement("script");
                s.src = "https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js";
                s.onload = function () { tryAnimate(0); };
                doc.head.appendChild(s);
            } else {
                setTimeout(function () { tryAnimate(0); }, 600);
            }
        })();
        </script>
"""


def page_header(title: str, key: str = None, with_prodi: bool = True, with_ref_date: bool = False, bt: str = ""):
    """Baris judul halaman: judul di kiri, tombol Filter (popover) di ujung
    kanan. Juga menyuntikkan animasi transisi dengan nama keyframe unik per
    halaman - nama yang berubah membuat animasi restart setiap pindah halaman.
    bt: kode Business Task yang dijawab halaman ini (mis. "BT-04") -> badge."""
    slug = "".join(ch for ch in (key or title).lower() if ch.isalnum())

    # ANIMASI PINDAH TAB (CSS, andal): nama keyframe UNIK per halaman. Saat pindah
    # tab, animation-name pada kartu/kontainer berubah -> browser WAJIB memutar
    # ulang animasi (baik elemen dibuat ulang maupun dipakai ulang oleh Streamlit).
    # Tidak butuh JS/GSAP (yang gagal menyentuh DOM dari iframe di Streamlit Cloud).
    st.markdown(
        "<style>"
        f"@keyframes sweep_{slug} {{ from {{ opacity: 0; transform: translateX(80px); }} "
        f"to {{ opacity: 1; transform: none; }} }} "
        f".kpi-card, div[data-testid='stVerticalBlockBorderWrapper'], "
        f"div[data-testid='stExpander'], div[data-testid='stDataFrame'], "
        f"div[data-testid='stPlotlyChart'] {{ "
        f"animation: sweep_{slug} 0.55s cubic-bezier(0.16, 1, 0.3, 1) both; }}"
        "</style>",
        unsafe_allow_html=True,
    )

    col_t, col_f = st.columns([8, 1], vertical_alignment="center")
    with col_t:
        badge = f'<span class="bt-badge">{bt}</span>' if bt else ""
        st.markdown(f'<div class="page-title">{title}{badge}</div>', unsafe_allow_html=True)

    tahun, prodi, jenis, ref_date = [], [], [], None
    if key is not None:
        with col_f:
            with st.popover(":material/tune: Filter"):
                tahun = st.multiselect("Tahun", FILTER_TAHUN, default=FILTER_TAHUN, key=f"f_tahun_{key}")
                if with_prodi:
                    prodi = st.multiselect("Program Studi", FILTER_PRODI, default=[],
                                           placeholder="Semua program studi", key=f"f_prodi_{key}")
                jenis = st.multiselect("Jenis Penempatan", FILTER_JENIS, default=[],
                                       placeholder="Semua jenis", key=f"f_jenis_{key}")
                if with_ref_date:
                    ref_raw = st.date_input(
                        "Tanggal Acuan Ghosting",
                        value=DEFAULT_REF_DATE.date() if hasattr(DEFAULT_REF_DATE, "date") else DEFAULT_REF_DATE,
                        key=f"f_ref_{key}",
                        help="Ghosting dihitung dari send_date sampai tanggal ini (aturan FAQ).",
                    )
                    ref_date = pd.Timestamp(ref_raw)
    run_gsap_animations(slug)
    return tahun, prodi, jenis, ref_date


def scope_master(tahun, prodi, jenis) -> pd.DataFrame:
    mm = master
    mask = mm["tahun_update"].isin(tahun) if tahun else pd.Series(True, index=mm.index)
    if prodi and "program_studi" in mm.columns:
        mask &= mm["program_studi"].isin(prodi)
    if jenis and "jenis_penempatan" in mm.columns:
        mask &= mm["jenis_penempatan"].isin(jenis)
    return mm[mask].copy()


def scope_tc(tahun, jenis) -> pd.DataFrame:
    tcc = tc_base
    mask = tcc["tahun_tc"].isin(tahun) if tahun else pd.Series(True, index=tcc.index)
    if jenis and "jenis_penempatan" in tcc.columns:
        mask &= tcc["jenis_penempatan"].isin(jenis)
    return tcc[mask].copy()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
# info sinkronisasi ditampilkan di bagian paling bawah sidebar
LAST_SYNC_TXT = ""
if "sync_date" in status_student.columns and status_student["sync_date"].notna().any():
    LAST_SYNC_TXT = f"Data sync terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}"



# ---------------------------------------------------------------------------
# TAB 1 - OVERVIEW
# ---------------------------------------------------------------------------
def page_overview():
    tahun_f, prodi_f, jenis_f, _ = page_header("Overview", key="overview")
    m = scope_master(tahun_f, prodi_f, jenis_f)
    tc_scope = scope_tc(tahun_f, jenis_f)

    total_dikirim_individu = m["id_tracking_student"].nunique()
    total_placement = int((m["rejection"] == "Placement").sum())
    success_rate = (total_placement / total_dikirim_individu * 100) if total_dikirim_individu > 0 else 0

    total_diminta = tc_scope["jumlah_permintaan"].sum() if "jumlah_permintaan" in tc_scope.columns else 0
    total_dikirim_batch = tc_scope["jumlah_dikirimkan"].sum() if "jumlah_dikirimkan" in tc_scope.columns else 0
    fulfillment_rate = (total_dikirim_batch / total_diminta * 100) if total_diminta and total_diminta > 0 else 0

    selesai = m[m["rejection"].isin(["Placement"] + REJECTION_STAGES)]
    lama_proses = selesai["lama_proses_hari"].mean() if "lama_proses_hari" in selesai.columns and selesai["lama_proses_hari"].notna().any() else None
    n_request_belum = int((tr_fulfill["belum_terpenuhi"] > 0).sum())

    # ---- Delta year-over-year: bandingkan 2 tahun PENUH terakhir ----
    # Tahun berjalan sering hanya terisi sebagian (data berhenti di tengah tahun),
    # jadi tahun dengan volume < 40% tahun tersibuk dianggap parsial dan dilewati
    # agar delta tidak menyesatkan (mis. 2025 yang datanya baru sampai Januari).
    vol_by_year = m.groupby("tahun_update")["id_tracking_student"].nunique()
    vol_by_year = vol_by_year[vol_by_year.index.notna()]
    tahun_penuh = sorted(int(y) for y in vol_by_year.index if vol_by_year[y] >= 0.4 * vol_by_year.max())
    yoy_pair = (tahun_penuh[-2], tahun_penuh[-1]) if len(tahun_penuh) >= 2 else None

    def _yoy(series_by_year):
        if not yoy_pair:
            return None
        prev, cur = series_by_year.get(yoy_pair[0], 0), series_by_year.get(yoy_pair[1], 0)
        return ((cur - prev) / prev * 100) if prev else None

    plc_by_year = m[m["rejection"] == "Placement"].groupby("tahun_update").size()
    kand_by_year = m.groupby("tahun_update")["id_tracking_student"].nunique()
    delta_plc = _yoy(plc_by_year)
    delta_kand = _yoy(kand_by_year)
    delta_lbl = f"vs {yoy_pair[0]}" if yoy_pair else "vs tahun lalu"

    kpi_row([
        {"value": f"{total_placement:,}", "label": "Placement Berhasil",
         "sub": f"{success_rate:.1f}% dari kandidat dikirim", "highlight": True,
         "delta": delta_plc, "delta_label": delta_lbl,
         "help": "Success rate = placement dibagi kandidat dikirim."},
        {"value": f"{total_dikirim_individu:,}", "label": "Kandidat Dikirim",
         "delta": delta_kand, "delta_label": delta_lbl,
         "help": "Jumlah proses seleksi kandidat pada rentang filter."},
        {"value": f"{fulfillment_rate:.1f}%", "label": "Fulfillment Rate",
         "help": "Jumlah dikirim vs diminta. Di atas 100% berarti kandidat dikirim melebihi kuota (wajar untuk shortlist)."},
        {"value": f"{lama_proses:.0f} hari" if lama_proses is not None else "-", "label": "Rata-rata Lama Proses",
         "help": "Dari batch dikirim sampai keputusan terakhir."},
        {"value": f"{n_request_belum:,}", "label": "Request Belum Terpenuhi",
         "sub": "data master, di luar filter"},
    ])

    TARGET_SUCCESS = 25.0  # target internal success rate (%) untuk garis acuan

    col_kiri, col_kanan = st.columns([2, 1])
    with col_kiri:
        with st.container(border=True):
            section("Tren Placement & Success Rate per Bulan", "Batang = jumlah placement. Garis = success rate (%).")
            dec = m[m["rejection"].isin(["Placement"] + REJECTION_STAGES)].copy()
            dec["bulan"] = dec["last_update"].dt.to_period("M")
            per_bulan = dec.groupby("bulan").agg(
                placement=("rejection", lambda s: (s == "Placement").sum()),
                keputusan=("rejection", "size"),
            ).reset_index().sort_values("bulan")
            per_bulan["success_pct"] = (per_bulan["placement"] / per_bulan["keputusan"] * 100).round(1)
            per_bulan["bulan_dt"] = per_bulan["bulan"].dt.to_timestamp()

            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_combo.add_trace(go.Bar(
                x=per_bulan["bulan_dt"], y=per_bulan["placement"], name="Placement",
                marker_color=COLOR_COCOA,
            ), secondary_y=False)
            fig_combo.add_trace(go.Scatter(
                x=per_bulan["bulan_dt"], y=per_bulan["success_pct"], name="Success Rate (%)",
                mode="lines+markers", line=dict(color=COLOR_SEAL_BROWN, width=2.5),
                marker=dict(color=COLOR_SIENNA, size=6),
            ), secondary_y=True)
            # garis target success rate (acuan evaluasi internal)
            if len(per_bulan) > 0:
                fig_combo.add_hline(
                    y=TARGET_SUCCESS, line_dash="dot", line_color=COLOR_OLIVE, line_width=2,
                    annotation_text=f"Target {TARGET_SUCCESS:.0f}%", annotation_position="top left",
                    annotation_font_color=COLOR_OLIVE, secondary_y=True,
                )
            fig_combo.update_yaxes(title_text=None, secondary_y=False)
            fig_combo.update_yaxes(title_text=None, secondary_y=True, rangemode="tozero")
            # elemen dinamis: tombol rentang waktu
            fig_combo.update_xaxes(rangeselector=dict(
                buttons=[
                    dict(count=6, label="6 bln", step="month", stepmode="backward"),
                    dict(count=12, label="1 thn", step="month", stepmode="backward"),
                    dict(step="all", label="Semua"),
                ],
                bgcolor=tint(COLOR_JASMINE, 0.6), activecolor=COLOR_COCOA,
                font=dict(size=10, color=COLOR_DRAB_DARK),
            ))
            show_chart(fig_combo, height=310)

    status_map = m["rejection"].map(
        lambda r: "Placement" if r == "Placement"
        else ("Ditolak" if r in REJECTION_STAGES
              else ("Ghosting" if r == "Ghosting" else "On Progress"))
    )
    with col_kanan:
        with st.container(border=True):
            section("Status Akhir Kandidat", "Komposisi hasil akhir seluruh proses seleksi.")
            status_counts = status_map.value_counts().reset_index()
            status_counts.columns = ["status", "jumlah"]
            fig_donut = px.pie(
                status_counts, names="status", values="jumlah", hole=0.55,
                color="status",
                color_discrete_map={
                    "Placement": COLOR_OLIVE, "Ditolak": COLOR_SIENNA,
                    "Ghosting": COLOR_SEAL_BROWN, "On Progress": COLOR_JASMINE,
                },
            )
            fig_donut.update_traces(textinfo="percent", textfont_size=11)
            show_chart(fig_donut, height=310)

    with st.container(border=True):
        section("Perjalanan Kandidat: dari Dikirim sampai Placement", "Berapa kandidat gugur di tiap titik hingga tersisa placement.")
        wf_vals = {
            "Rej. Screening CV": -int((m["rejection"] == "Rejection Screening CV").sum()),
            "Rej. Study Case": -int((m["rejection"] == "Rejection Study Case").sum()),
            "Rej. Interview User": -int((m["rejection"] == "Rejection Interview User").sum()),
            "Rej. Final Interview": -int((m["rejection"] == "Rejection Final Interview").sum()),
            "Ghosting": -int((m["rejection"] == "Ghosting").sum()),
            "Masih Berjalan": -int((m["rejection"] == "On Progress").sum()),
        }
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(wf_vals) + ["total"],
            x=["Kandidat Dikirim"] + list(wf_vals.keys()) + ["Placement"],
            y=[total_dikirim_individu] + list(wf_vals.values()) + [0],
            text=[f"{total_dikirim_individu:,}"] + [f"{v:,}" for v in wf_vals.values()] + [f"{total_placement:,}"],
            textposition="outside",
            connector={"line": {"color": tint(COLOR_DRAB_DARK, 0.6)}},
            increasing={"marker": {"color": COLOR_OLIVE}},
            decreasing={"marker": {"color": COLOR_SIENNA}},
            totals={"marker": {"color": COLOR_COCOA}},
        ))
        show_chart(fig_wf, height=290)

    # ---- SANKEY: alur bidang studi -> jenis penempatan -> hasil akhir ----
    with st.container(border=True):
        section("Alur Kandidat: Bidang Studi \u2192 Jenis Penempatan \u2192 Hasil Akhir")
        sk = m.copy()
        sk["hasil"] = status_map.values
        prodi_col_m = "program_studi" if "program_studi" in sk.columns else None
        jenis_col_m = "jenis_penempatan" if "jenis_penempatan" in sk.columns else None
        if prodi_col_m and jenis_col_m and len(sk) > 0:
            top_prodi = sk[prodi_col_m].value_counts().head(6).index.tolist()
            sk["prodi_grp"] = sk[prodi_col_m].where(sk[prodi_col_m].isin(top_prodi), "Bidang Lain")
            sk = sk.dropna(subset=[jenis_col_m])

            prodi_nodes = [p for p in top_prodi if p in sk["prodi_grp"].unique()]
            if "Bidang Lain" in sk["prodi_grp"].unique():
                prodi_nodes.append("Bidang Lain")
            jenis_nodes = sk[jenis_col_m].dropna().unique().tolist()
            hasil_order = ["Placement", "On Progress", "Ditolak", "Ghosting"]
            hasil_nodes = [h for h in hasil_order if h in sk["hasil"].unique()]

            labels = prodi_nodes + jenis_nodes + hasil_nodes
            idx = {l: i for i, l in enumerate(labels)}
            hasil_color = {"Placement": COLOR_OLIVE, "On Progress": COLOR_JASMINE,
                           "Ditolak": COLOR_SIENNA, "Ghosting": COLOR_SEAL_BROWN}
            node_colors = ([tint(COLOR_COCOA, 0.15)] * len(prodi_nodes)
                           + [tint(COLOR_DRAB_DARK, 0.45)] * len(jenis_nodes)
                           + [hasil_color.get(h, COLOR_COCOA) for h in hasil_nodes])

            s1 = sk.groupby(["prodi_grp", jenis_col_m]).size().reset_index(name="n")
            s2 = sk.groupby([jenis_col_m, "hasil"]).size().reset_index(name="n")
            src = [idx[r.prodi_grp] for r in s1.itertuples()] + [idx[getattr(r, jenis_col_m)] for r in s2.itertuples()]
            tgt = [idx[getattr(r, jenis_col_m)] for r in s1.itertuples()] + [idx[r.hasil] for r in s2.itertuples()]
            val = s1["n"].tolist() + s2["n"].tolist()
            link_colors = ([tint(COLOR_COCOA, 0.72)] * len(s1)
                           + [tint(hasil_color.get(r.hasil, COLOR_COCOA), 0.6) for r in s2.itertuples()])

            fig_sk = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(label=labels, color=node_colors, pad=14, thickness=14,
                          line=dict(color="rgba(0,0,0,0)", width=0)),
                link=dict(source=src, target=tgt, value=val, color=link_colors),
            ))
            fig_sk.update_layout(font=dict(size=10, color=COLOR_DRAB_DARK))
            show_chart(fig_sk, height=300)
        else:
            insight("Data bidang studi / jenis penempatan tidak lengkap untuk alur ini.", kind="error")

    top_reject = m[m["rejection"].isin(REJECTION_STAGES)]["rejection"].value_counts()
    top_reject_name = top_reject.index[0].replace("Rejection ", "") if len(top_reject) else "-"
    catatan_analis([
        f"Dari <b>{total_dikirim_individu:,}</b> kandidat dikirim, <b>{total_placement:,}</b> berhasil "
        f"placement (success rate <b>{success_rate:.1f}%</b>).",
        f"Success rate {'di atas' if success_rate >= TARGET_SUCCESS else 'masih di bawah'} "
        f"target internal {TARGET_SUCCESS:.0f}%.",
        f"Penolakan paling banyak terjadi di tahap <b>{top_reject_name}</b> - jadi fokus pendampingan.",
        f"Fulfillment <b>{fulfillment_rate:.1f}%</b>: kandidat dikirim melebihi kuota (wajar untuk shortlist), "
        "tapi tetap ada <b>request belum terpenuhi</b> yang perlu dikejar.",
    ])

# ---------------------------------------------------------------------------
# TAB 2 - FUNNEL & GHOSTING
# ---------------------------------------------------------------------------
def page_funnel():
    tahun_f, prodi_f, jenis_f, tanggal_acuan = page_header(
        "Funnel & Ghosting", key="funnel", with_ref_date=True)
    m = scope_master(tahun_f, prodi_f, jenis_f)

    tc_status = scope_tc(tahun_f, jenis_f)
    tc_status["hari_sejak_kirim"] = (tanggal_acuan - tc_status["send_date"]).dt.days
    tc_status["status_followup"] = [
        hitung_status_followup(h, r)
        for h, r in zip(tc_status["hari_sejak_kirim"], tc_status["sudah_direspon"])
    ]

    followup_counts = tc_status["status_followup"].value_counts()
    total_tc = len(tc_status)
    n_ghosting = int(followup_counts.get("Ghosting", 0))
    ghosting_rate = (n_ghosting / total_tc * 100) if total_tc > 0 else 0

    kpi_row([
        {"value": f"{n_ghosting:,}", "label": "Ghosting", "sub": f"{ghosting_rate:.1f}% dari batch terkirim",
         "highlight": True,
         "help": "Aturan FAQ: >28 hari sejak send_date tanpa respons perusahaan."},
        {"value": f"{total_tc:,}", "label": "Batch Terkirim"},
        {"value": f"{int(followup_counts.get('FU 1', 0)):,}", "label": "Butuh FU 1"},
        {"value": f"{int(followup_counts.get('FU 2', 0)):,}", "label": "Butuh FU 2"},
        {"value": f"{int(followup_counts.get('FU 3', 0)):,}", "label": "Butuh FU 3"},
    ])

    col_kiri, col_kanan = st.columns(2)
    with col_kiri:
        with st.container(border=True):
            section("Funnel Seleksi Kandidat", "Jumlah & persentase kandidat yang bertahan di tiap tahap.")
            funnel_counts = [int((m["stage_reached"] >= i).sum()) for i in range(len(FUNNEL_STAGES))]
            fig_funnel = go.Figure(go.Funnel(
                y=FUNNEL_STAGES, x=funnel_counts,
                marker={"color": PALETTE_SEQUENTIAL},
                textinfo="value+percent previous",
            ))
            show_chart(fig_funnel, height=330)

    with col_kanan:
        with st.container(border=True):
            section("Sebaran Penolakan per Tahap Seleksi", "Di tahap mana penolakan paling banyak terjadi.")
            rej = m[m["rejection"].isin(REJECTION_STAGES)]["rejection"].value_counts().reindex(REJECTION_STAGES).fillna(0).reset_index()
            rej.columns = ["tahap_rejection", "jumlah"]
            fig_rej = px.bar(rej, x="jumlah", y="tahap_rejection", orientation="h",
                             color_discrete_sequence=[COLOR_SIENNA])
            fig_rej.update_layout(yaxis_title=None, xaxis_title=None)
            fig_rej.update_yaxes(categoryorder="total ascending")
            show_chart(fig_rej, height=330)

    konversi = [
        (FUNNEL_STAGES[i], funnel_counts[i] / funnel_counts[i - 1] * 100)
        for i in range(1, len(FUNNEL_STAGES)) if funnel_counts[i - 1] > 0
    ]
    if konversi:
        tahap_bocor, rate_bocor = min(konversi, key=lambda t: t[1])
        insight(f"Konversi terendah ada di tahap <b>{tahap_bocor}</b> ({rate_bocor:.0f}% dari tahap sebelumnya) - "
                "prioritaskan pendampingan CDC di tahap ini. "
                f"Ghosting menurut aturan FAQ: <b>{n_ghosting:,} batch</b> ({ghosting_rate:.1f}%).", kind="warning")

    # ---- HEATMAP: kapan setiap tahap seleksi paling ramai? ----
    with st.container(border=True):
        section("Volume Aktivitas Seleksi per Bulan dan Tahap", "Makin gelap = makin banyak aktivitas di bulan dan tahap itu.")
        hm = m.dropna(subset=["last_update"]).copy()
        urutan_tahap = FUNNEL_STAGES + ["FU 1", "FU 2", "FU 3", "Ghosting", "Rejected", "Finish"]
        hm = hm[hm["progress_student"].isin(urutan_tahap)]
        if len(hm) > 0:
            hm["bulan"] = hm["last_update"].dt.to_period("M").astype(str)
            piv = hm.pivot_table(index="progress_student", columns="bulan",
                                 values="id_tracking_student", aggfunc="count", fill_value=0)
            piv = piv.reindex([t for t in urutan_tahap if t in piv.index])
            fig_hm = px.imshow(
                piv, aspect="auto", color_continuous_scale=["#FFF6E6", COLOR_JASMINE, COLOR_COCOA, COLOR_SIENNA],
                labels=dict(color="Jumlah"),
            )
            fig_hm.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_colorbar=dict(title=None))
            fig_hm.update_xaxes(tickfont=dict(size=9))
            show_chart(fig_hm, height=300)
        else:
            insight("Belum ada data aktivitas tahap pada rentang filter ini.", kind="error")

    tahap_bocor_txt = konversi and min(konversi, key=lambda t: t[1])[0] or "-"
    catatan_analis([
        f"Konversi paling bocor di tahap <b>{tahap_bocor_txt}</b> - dampingi kandidat lebih intensif di sini.",
        f"Ghosting (aturan FAQ, &gt;{GHOSTING_DAYS} hari sejak dikirim tanpa respons): "
        f"<b>{n_ghosting:,} batch</b> ({ghosting_rate:.1f}%).",
        f"Ada <b>{int(followup_counts.get('FU 1', 0)) + int(followup_counts.get('FU 2', 0)) + int(followup_counts.get('FU 3', 0)):,}</b> "
        "batch yang butuh follow-up bertahap sebelum jatuh ke ghosting.",
        "Heatmap menandai bulan-bulan puncak aktivitas: berguna untuk mengatur beban kerja tim CDC.",
    ])

    ghosted = tc_status[tc_status["status_followup"] == "Ghosting"]
    with st.expander("Analisis ghosting lanjutan (per perusahaan, per tahap, tren bulanan)", icon=":material/query_stats:"):
        col1, col2 = st.columns(2)
        with col1:
            ghosting_by_company = ghosted["company_name"].value_counts().head(10).reset_index()
            ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
            fig_gc = px.bar(ghosting_by_company, x="jumlah_ghosting", y="perusahaan", orientation="h",
                            title="Top Perusahaan Kontributor Ghosting",
                            color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_gc.update_layout(yaxis_title=None, xaxis_title=None)
            fig_gc.update_yaxes(categoryorder="total ascending")
            show_chart(fig_gc, height=290)
        with col2:
            ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
            ghosting_by_stage = ghosted_children["progress_student"].value_counts().head(8).reset_index()
            ghosting_by_stage.columns = ["tahap", "jumlah"]
            fig_gs = px.bar(ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
                            title="Kandidat pada Batch Ghosting - Tahap Terakhir",
                            color_discrete_sequence=[COLOR_SIENNA])
            fig_gs.update_layout(yaxis_title=None, xaxis_title=None)
            fig_gs.update_yaxes(categoryorder="total ascending")
            show_chart(fig_gs, height=290)

        tren_fu = tc_status.copy()
        tren_fu["bulan_kirim"] = tren_fu["send_date"].dt.to_period("M").astype(str)
        tren_fu = tren_fu.groupby(["bulan_kirim", "status_followup"]).size().reset_index(name="jumlah")
        fig_fu = px.bar(
            tren_fu.sort_values("bulan_kirim"), x="bulan_kirim", y="jumlah", color="status_followup",
            title="Komposisi Status Follow-up per Bulan Pengiriman",
            color_discrete_map={
                "Direspon": COLOR_OLIVE,
                "Menunggu Respons (Normal)": COLOR_JASMINE,
                "FU 1": tint(COLOR_COCOA, 0.35),
                "FU 2": COLOR_COCOA,
                "FU 3": COLOR_SIENNA,
                "Ghosting": COLOR_SEAL_BROWN,
            },
        )
        fig_fu.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
        show_chart(fig_fu, height=280)

        n_ghosting_tercatat = int((m["rejection"] == "Ghosting").sum())
        insight(
            f"Data mencatat {n_ghosting_tercatat:,} kandidat berlabel Ghosting - angka per kandidat itu tidak setara "
            "dengan aturan FAQ karena satu batch bisa berisi campuran kandidat ghosting dan kandidat yang tetap "
            "diproses. Untuk monitoring follow-up, gunakan angka aturan FAQ karena bisa dihitung ulang kapan pun.",
        )

    perlu_followup = tc_status[tc_status["status_followup"].isin(["FU 1", "FU 2", "FU 3", "Ghosting"])].sort_values("hari_sejak_kirim", ascending=False)
    with st.expander(f"Daftar {len(perlu_followup):,} batch yang butuh follow-up", icon=":material/list_alt:"):
        st.dataframe(
            perlu_followup[["id_tracking_company", "company_name", "posisi", "send_date", "hari_sejak_kirim", "status_followup"]],
            width="stretch", hide_index=True, height=320,
        )

# ---------------------------------------------------------------------------
# TAB 3 - MITRA
# ---------------------------------------------------------------------------
def page_mitra():
    tahun_f, prodi_f, jenis_f, _ = page_header("Mitra Perusahaan", key="mitra")
    m = scope_master(tahun_f, prodi_f, jenis_f)

    kpi_row([
        {"value": f"{m[COMPANY_NAME_COL].nunique():,}", "label": "Perusahaan Aktif (filter)", "highlight": True,
         "help": "Perusahaan yang punya proses seleksi berjalan pada rentang filter."},
        {"value": f"{company['id_company'].nunique():,}", "label": "Perusahaan Mitra"},
        {"value": f"{talent_request['id_talent_req'].nunique():,}", "label": "Total Talent Request"},
        {"value": f"{m['nama_posisi'].nunique():,}" if "nama_posisi" in m.columns else "-", "label": "Posisi Dibuka (filter)"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section("Top 10 Acceptance Rate Perusahaan", "Persentase kandidat yang diterima (min. 3 kandidat).")
            perf = m.groupby(COMPANY_NAME_COL).agg(
                total=("id_tracking_student", "count"),
                placement=("rejection", lambda x: (x == "Placement").sum()),
            ).reset_index()
            perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
            perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10).rename(columns={COMPANY_NAME_COL: "perusahaan"})
            fig_acc = px.bar(perf, x="acceptance_rate", y="perusahaan", orientation="h",
                             color_discrete_sequence=[COLOR_COCOA])
            fig_acc.update_layout(yaxis_title=None, xaxis_title="%")
            fig_acc.update_yaxes(categoryorder="total ascending")
            show_chart(fig_acc, height=310)
    with col2:
        with st.container(border=True):
            section("Top 10 Perusahaan berdasarkan Jumlah Talent Request")
            tr_named = talent_request.merge(company[["id_company", "company_name"]], on="id_company", how="left")
            volume = tr_named["company_name"].value_counts().head(10).reset_index()
            volume.columns = ["perusahaan", "jumlah_request"]
            fig_vol = px.treemap(volume, path=["perusahaan"], values="jumlah_request",
                                 color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_vol.update_layout(margin=dict(l=4, r=4, t=8, b=4))
            show_chart(fig_vol, height=310)

    n_belum = int((tr_fulfill["belum_terpenuhi"] > 0).sum())
    top_sektor = tr_named["industri_sektor"].value_counts().index[0] if "industri_sektor" in tr_named.columns and len(tr_named) else "-"
    catatan_analis([
        f"Ada <b>{company['id_company'].nunique():,}</b> perusahaan mitra dengan "
        f"<b>{talent_request['id_talent_req'].nunique():,}</b> talent request total.",
        f"Sektor industri paling banyak meminta talent: <b>{top_sektor}</b>.",
        f"<b>{n_belum:,}</b> talent request belum terpenuhi - lihat daftar prioritas di bawah, "
        "urut dari yang paling lama menunggu.",
        "Acceptance rate per perusahaan (kiri) memakai minimal 3 kandidat agar tidak bias oleh sampel kecil.",
    ])

    with st.expander("Profil mitra: sektor industri & tipe perusahaan", icon=":material/domain:"):
        col3, col4 = st.columns(2)
        with col3:
            sektor = tr_named["industri_sektor"].value_counts().head(10).reset_index() if "industri_sektor" in tr_named.columns else pd.DataFrame(columns=["a", "b"])
            sektor.columns = ["sektor", "jumlah"]
            fig_sektor = px.bar(sektor, x="jumlah", y="sektor", orientation="h",
                                title="Permintaan per Sektor Industri",
                                color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_sektor.update_layout(yaxis_title=None, xaxis_title=None)
            fig_sektor.update_yaxes(categoryorder="total ascending")
            show_chart(fig_sektor, height=290)
        with col4:
            tipe = company["company_type"].value_counts().reset_index() if "company_type" in company.columns else pd.DataFrame(columns=["a", "b"])
            tipe.columns = ["tipe", "jumlah"]
            fig_tipe = px.pie(tipe, names="tipe", values="jumlah", hole=0.5,
                              title="Komposisi Tipe Perusahaan",
                              color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_tipe.update_traces(textinfo="percent+label", textfont_size=10)
            show_chart(fig_tipe, height=290)

    prioritas = tr_fulfill[tr_fulfill["belum_terpenuhi"] > 0].sort_values("request_date")
    with st.expander(f"Prioritas {len(prioritas):,} talent request belum terpenuhi", icon=":material/priority_high:"):
        st.caption("Diurutkan dari request paling lama.")
        st.dataframe(
            prioritas[["id_talent_req", "company_name", "nama_posisi", "headcount",
                       "jumlah_dikirimkan", "belum_terpenuhi", "pemenuhan", "request_date"]],
            width="stretch", hide_index=True, height=320,
            column_config={
                "pemenuhan": st.column_config.ProgressColumn("Pemenuhan", format="percent", min_value=0, max_value=1),
                "request_date": st.column_config.DateColumn("Tanggal Request"),
            },
        )

# ---------------------------------------------------------------------------
# SEGMENTASI MITRA (RFM-style) - mengelompokkan perusahaan mitra untuk BT-03/BT-04
# ---------------------------------------------------------------------------
SEG_COLORS = {
    "Mitra Andalan": COLOR_OLIVE,
    "Mitra Aktif Baru": COLOR_COCOA,
    "Mitra Pasif": COLOR_JASMINE,
    "Mitra Dorman": COLOR_SEAL_BROWN,
}
SEG_DESC = {
    "Mitra Andalan": "sering minta talent & masih aktif belakangan ini",
    "Mitra Aktif Baru": "baru/aktif terkini tapi frekuensi masih sedikit",
    "Mitra Pasif": "dulu sering minta, tapi sudah lama tidak ada permintaan",
    "Mitra Dorman": "jarang minta dan sudah lama tidak aktif",
}


@st.cache_data(show_spinner="Menghitung segmentasi mitra (hanya sekali)...")
def compute_mitra_segments() -> pd.DataFrame:
    tr = talent_request.copy()
    ref = tr["request_date"].max()
    if pd.isna(ref):
        ref = pd.Timestamp(datetime.now().date())
    g = tr.groupby("id_company").agg(
        frekuensi=("id_talent_req", "count"),
        last_req=("request_date", "max"),
        volume=("headcount", "sum"),
    ).reset_index()
    g["recency_hari"] = (ref - g["last_req"]).dt.days

    # responsiveness dari master (acceptance & ghosting per perusahaan)
    if "id_company" in master.columns:
        resp = master.groupby("id_company").agg(
            total_proses=("id_tracking_student", "count"),
            placement=("rejection", lambda s: (s == "Placement").sum()),
            ghosting=("rejection", lambda s: (s == "Ghosting").sum()),
        ).reset_index()
        g = g.merge(resp, on="id_company", how="left")
    for c in ("total_proses", "placement", "ghosting"):
        if c not in g.columns:
            g[c] = 0
        g[c] = g[c].fillna(0)
    g["acceptance_rate"] = np.where(g["total_proses"] > 0, g["placement"] / g["total_proses"] * 100, 0).round(1)
    g["ghosting_rate"] = np.where(g["total_proses"] > 0, g["ghosting"] / g["total_proses"] * 100, 0).round(1)

    g = g.merge(company[["id_company", "company_name", "industry_sector", "skala_perusahaan"]],
                on="id_company", how="left")

    med_f = g["frekuensi"].median()
    med_r = g["recency_hari"].median()

    def _seg(row):
        recent = row["recency_hari"] <= med_r
        freq = row["frekuensi"] >= med_f
        if recent and freq:
            return "Mitra Andalan"
        if recent and not freq:
            return "Mitra Aktif Baru"
        if (not recent) and freq:
            return "Mitra Pasif"
        return "Mitra Dorman"

    g["segmen"] = g.apply(_seg, axis=1)
    return g


def page_segmentasi():
    page_header("Segmentasi Mitra")
    seg = compute_mitra_segments()

    counts = seg["segmen"].value_counts()
    n_andalan = int(counts.get("Mitra Andalan", 0))
    n_pasif = int(counts.get("Mitra Pasif", 0))
    n_bermasalah = int((seg["ghosting_rate"] >= 20).sum())

    kpi_row([
        {"value": f"{n_andalan:,}", "label": "Mitra Andalan", "highlight": True,
         "sub": "sering minta & masih aktif",
         "help": "Frekuensi request tinggi dan permintaan terbaru relatif baru."},
        {"value": f"{len(seg):,}", "label": "Total Mitra Bersegmen"},
        {"value": f"{n_pasif:,}", "label": "Mitra Pasif",
         "help": "Dulu sering minta talent, tapi sudah lama tidak ada permintaan - layak di-reengage."},
        {"value": f"{n_bermasalah:,}", "label": "Mitra Rawan Ghosting",
         "sub": "ghosting rate >= 20%"},
    ])

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            section("Peta Mitra: Frekuensi vs Recency Permintaan Talent",
                    "Sumbu Y dibalik: makin ke atas = permintaan makin baru. Ukuran titik = total headcount diminta.")
            plot = seg.copy()
            fig_seg = px.scatter(
                plot, x="frekuensi", y="recency_hari", size="volume", color="segmen",
                hover_name="company_name",
                hover_data={"acceptance_rate": True, "ghosting_rate": True, "volume": True, "segmen": False},
                color_discrete_map=SEG_COLORS, size_max=34,
            )
            fig_seg.update_yaxes(autorange="reversed", title="Recency (hari sejak request terakhir)")
            fig_seg.update_xaxes(title="Frekuensi request")
            fig_seg.update_layout(legend_title=None)
            show_chart(fig_seg, height=360)
    with col2:
        with st.container(border=True):
            section("Jumlah Mitra per Segmen")
            seg_ct = counts.reindex(list(SEG_COLORS.keys())).fillna(0).reset_index()
            seg_ct.columns = ["segmen", "jumlah"]
            fig_bar = px.bar(seg_ct, x="jumlah", y="segmen", orientation="h", color="segmen",
                             color_discrete_map=SEG_COLORS, text="jumlah")
            fig_bar.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
            fig_bar.update_yaxes(categoryorder="total ascending")
            show_chart(fig_bar, height=360)

    top_ghost = seg.sort_values("ghosting_rate", ascending=False).iloc[0] if len(seg) else None
    catatan_analis([
        f"<b>{n_andalan:,} Mitra Andalan</b> jadi tulang punggung: sering minta talent & masih aktif - "
        "jaga hubungan dan prioritaskan pemenuhannya.",
        f"<b>{n_pasif:,} Mitra Pasif</b> dulu aktif tapi lama menghilang - kandidat kuat untuk di-reengage "
        "(hubungi ulang, tawarkan batch baru).",
        f"<b>{n_bermasalah:,} mitra rawan ghosting</b> (ghosting rate &ge; 20%)"
        + (f", tertinggi <b>{top_ghost['company_name']}</b> ({top_ghost['ghosting_rate']:.0f}%)." if top_ghost is not None else ".")
        + " Perlu aturan follow-up lebih tegas.",
        "Segmentasi memakai prinsip RFM: Recency (kebaruan request), Frequency (jumlah request), "
        "dan volume headcount sebagai ukuran nilai mitra.",
    ])

    with st.expander("Tabel lengkap segmentasi mitra & unduh CSV", icon=":material/table_view:"):
        show = seg[["company_name", "industry_sector", "skala_perusahaan", "frekuensi", "recency_hari",
                    "volume", "acceptance_rate", "ghosting_rate", "segmen"]].sort_values(
            ["segmen", "frekuensi"], ascending=[True, False])
        st.dataframe(
            show, width="stretch", hide_index=True, height=330,
            column_config={
                "company_name": "Perusahaan", "industry_sector": "Sektor", "skala_perusahaan": "Skala",
                "frekuensi": "Frek. Request", "recency_hari": "Recency (hari)", "volume": "Total Headcount",
                "acceptance_rate": st.column_config.NumberColumn("Acceptance %", format="%.1f"),
                "ghosting_rate": st.column_config.NumberColumn("Ghosting %", format="%.1f"),
                "segmen": "Segmen",
            },
        )
        st.download_button("Unduh Segmentasi CSV", show.to_csv(index=False).encode("utf-8"),
                           "segmentasi_mitra.csv", "text/csv", icon=":material/download:")


# ---------------------------------------------------------------------------
# TAB 4 - KESIAPAN (data master terkini - tanpa filter)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_prodi_profile() -> pd.DataFrame:
    """Profil tiap program studi di 5 dimensi (untuk radar):
    demand, supply, success rate, rata-rata IPK, dan tingkat kesiapan."""
    sa = student_all
    ss = status_student
    supply = sa["program_studi"].value_counts().rename("supply")

    dm = talent_request[["bidang_studi_dibutuhkan", "headcount"]].dropna(subset=["bidang_studi_dibutuhkan"]).copy()
    dm["bidang"] = dm["bidang_studi_dibutuhkan"].str.split(",")
    dm = dm.explode("bidang")
    dm["bidang"] = dm["bidang"].str.strip()
    demand = dm.groupby("bidang")["headcount"].sum().rename("demand")

    prodi_norm_m = master["program_studi"] if "program_studi" in master.columns else pd.Series(dtype=object)
    succ = master.assign(_p=prodi_norm_m).groupby("_p").agg(
        total=("id_tracking_student", "count"),
        plc=("rejection", lambda s: (s == "Placement").sum()),
    )
    succ["success"] = np.where(succ["total"] > 0, succ["plc"] / succ["total"] * 100, 0)

    # status_student sudah memuat kolom program_studi (denormalisasi) -> pakai langsung
    prodi_ss = ss["program_studi"] if "program_studi" in ss.columns else ss["nim"].map(
        sa.set_index("nim")["program_studi"])
    ipk = ss.assign(_p=prodi_ss).groupby("_p")["ipk"].mean().rename("ipk")

    ss2 = ss.assign(_p=prodi_ss)
    ss2["_ready"] = (norm_text(ss2["status"]).isin(VAL_STATUS_AKTIF)
                     & norm_text(ss2["ketersediaan"]).isin(VAL_TERSEDIA))
    ready = ss2.groupby("_p")["_ready"].mean().mul(100).rename("kesiapan")

    prof = pd.concat([supply, demand, succ["success"].rename("success"), ipk, ready], axis=1)
    prof = prof.reindex(supply.index).fillna(0)
    prof.index.name = "program_studi"
    return prof.reset_index()


@st.cache_data(show_spinner=False)
def compute_skill_gap() -> pd.DataFrame:
    """Tools yang dikuasai mahasiswa (supply, data solid) vs tools yang disebut
    di deskripsi requirement perusahaan (demand, perkiraan dari teks)."""
    import re as _re
    tools = status_student["tools"].dropna().astype(str).str.split(",").explode().str.strip()
    tools = tools[tools != ""]
    supply = tools.value_counts()

    desc = talent_request["deskripsi_requirement"].dropna().astype(str)
    rows = []
    for tool, sup in supply.items():
        # word-boundary agar tool 1 huruf (mis. "R") tidak salah cocok di mana-mana
        pat = r"\b" + _re.escape(tool) + r"\b"
        dem = int(desc.str.contains(pat, case=False, regex=True).sum())
        rows.append({"tool": tool, "dikuasai": int(sup), "diminta": dem})
    return pd.DataFrame(rows)


def page_kesiapan():
    page_header("Kesiapan Mahasiswa")

    ss = status_student
    status_norm = norm_text(ss["status"])
    ketersediaan_norm = norm_text(ss["ketersediaan"])
    cv_norm = norm_text(ss["cv"])
    porto_norm = norm_text(ss["portofolio"])

    eligible_mask = (
        status_norm.isin(VAL_STATUS_AKTIF)
        & ketersediaan_norm.isin(VAL_TERSEDIA)
        & cv_norm.isin(VAL_ADA)
        & porto_norm.isin(VAL_ADA)
    )
    eligible = ss[eligible_mask].merge(student_all[["nim", "program_studi", "semester"]], on="nim", how="left")
    pernah_dikirim = set(tracking_student["nim"].unique())
    eligible_nganggur = eligible[~eligible["nim"].isin(pernah_dikirim)]

    kpi_row([
        {"value": f"{len(eligible_nganggur):,}", "label": "Layak tapi Belum Pernah Dikirim", "highlight": True,
         "sub": "supply belum tersalurkan",
         "help": "Prioritas untuk dicarikan penempatan."},
        {"value": f"{student_all['nim'].nunique():,}", "label": "Mahasiswa Terdaftar"},
        {"value": f"{len(eligible):,}", "label": "Layak Kirim Saat Ini",
         "help": "Status aktif + tersedia + CV ada + portofolio ada."},
        {"value": f"{(cv_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya CV"},
        {"value": f"{(porto_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya Portofolio"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section("Demand vs Supply dari Waktu ke Waktu", "Demand = jumlah kebutuhan (headcount) perusahaan. Supply = perkiraan mahasiswa yang siap magang.")
            col_prodi, col_gran = st.columns([3, 2])
            prodi_pilih = col_prodi.selectbox(
                "Fokus jurusan / bidang studi",
                ["Semua bidang"] + FILTER_PRODI,
                key="ks_prodi_ts",
            )
            granularitas = col_gran.radio(
                "Granularitas", ["Bulanan", "Kuartalan", "Tahunan"],
                horizontal=True, index=1, key="ks_gran",
            )
            freq = {"Bulanan": "M", "Kuartalan": "Q", "Tahunan": "Y"}[granularitas]

            sup_view = student_all[student_all["siap_dt"].notna()]
            if prodi_pilih != "Semua bidang":
                # per bidang: headcount penuh tiap posisi yang menerima bidang ini
                p_norm = prodi_pilih.strip().lower()
                dm_view = DATA["demand_monthly"]
                dm_view = dm_view[dm_view["bidang_norm"] == p_norm]
                sup_view = sup_view[norm_text(sup_view["program_studi"]) == p_norm]
            else:
                # semua bidang: total posisi (tiap posisi dihitung sekali)
                dm_view = DATA["demand_monthly_total"]

            d_series = dm_view.groupby(dm_view["bulan"].dt.to_period(freq))["headcount"].sum()
            s_series = sup_view.groupby(sup_view["siap_dt"].dt.to_period(freq)).size()

            if len(d_series) == 0 and len(s_series) == 0:
                insight("Tidak ada data demand maupun supply untuk bidang ini.", kind="error")
            else:
                periode_semua = list(d_series.index) + list(s_series.index)
                idx = pd.period_range(min(periode_semua), max(periode_semua), freq=freq)
                x = idx.to_timestamp()
                d_full = d_series.reindex(idx, fill_value=0)
                s_full = s_series.reindex(idx, fill_value=0)

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=x, y=s_full, name="Mahasiswa Siap Magang (est.)",
                    mode="lines", line=dict(color=COLOR_OLIVE, width=2, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(107, 122, 61, 0.30)",
                ))
                fig_ts.add_trace(go.Scatter(
                    x=x, y=d_full, name="Headcount Diminta",
                    mode="lines", line=dict(color=COLOR_COCOA, width=2.5, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(226, 120, 47, 0.35)",
                ))
                fig_ts.update_yaxes(rangemode="tozero")
                # fokuskan tampilan awal ke periode permintaan (zoom out tetap bisa)
                if len(d_series) > 0:
                    x_awal = d_series.index.min().to_timestamp() - pd.Timedelta(days=45)
                    x_akhir = pd.Timestamp(x[-1]) + pd.Timedelta(days=45)
                    fig_ts.update_xaxes(range=[x_awal, x_akhir])
                show_chart(fig_ts, height=280)
    with col2:
        with st.container(border=True):
            section("Ketersediaan & Status Keaktifan Mahasiswa", "Sebaran mahasiswa menurut ketersediaan dan status keaktifan.")
            elig_ct = ss.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
            fig_elig = px.bar(elig_ct, x="ketersediaan", y="jumlah", color="status",
                              color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_elig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
            show_chart(fig_elig, height=352)

    # ---- RADAR profil prodi + SKILL GAP tools (BT-01 technical matching) ----
    col_r, col_s = st.columns(2)
    with col_r:
        with st.container(border=True):
            section("Profil Program Studi (5 Dimensi)",
                    "Lima dimensi dinormalisasi 0-1 antar prodi. Pilih prodi untuk dibandingkan.")
            prof = compute_prodi_profile()
            dims = ["demand", "supply", "success", "ipk", "kesiapan"]
            dim_label = {"demand": "Demand", "supply": "Supply", "success": "Success Rate",
                         "ipk": "Rata IPK", "kesiapan": "Kesiapan"}
            prof_n = prof.copy()
            for d in dims:
                rng = prof_n[d].max() - prof_n[d].min()
                prof_n[d] = (prof_n[d] - prof_n[d].min()) / rng if rng > 0 else 0.0
            default_prodi = prof.sort_values("supply", ascending=False)["program_studi"].head(3).tolist()
            pilih = st.multiselect("Program studi", prof["program_studi"].tolist(),
                                   default=default_prodi, key="radar_prodi", max_selections=5)
            if pilih:
                fig_radar = go.Figure()
                pal = [COLOR_COCOA, COLOR_OLIVE, COLOR_SIENNA, COLOR_SEAL_BROWN, COLOR_JASMINE]
                for i, p in enumerate(pilih):
                    row = prof_n[prof_n["program_studi"] == p]
                    if row.empty:
                        continue
                    vals = [float(row[d].iloc[0]) for d in dims]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=[dim_label[d] for d in dims] + [dim_label[dims[0]]],
                        fill="toself", name=p, line=dict(color=pal[i % len(pal)]),
                        fillcolor="rgba(0,0,0,0)",
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)))
                show_chart(fig_radar, height=330)
            else:
                insight("Pilih minimal satu program studi untuk menampilkan radar.", kind="info")
    with col_s:
        with st.container(border=True):
            section("Tools: Dikuasai Mahasiswa vs Diminta Perusahaan",
                    "Dikuasai = jumlah mahasiswa yang menguasai tools itu. Diminta = perkiraan dari teks kebutuhan perusahaan.")
            gap = compute_skill_gap()
            top_gap = gap.sort_values("dikuasai", ascending=False).head(12)
            melt = top_gap.melt(id_vars="tool", value_vars=["dikuasai", "diminta"],
                                var_name="sisi", value_name="jumlah")
            fig_skill = px.bar(melt, x="jumlah", y="tool", color="sisi", orientation="h", barmode="group",
                               color_discrete_map={"dikuasai": COLOR_JASMINE, "diminta": COLOR_COCOA})
            fig_skill.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
            fig_skill.update_yaxes(categoryorder="total ascending")
            show_chart(fig_skill, height=330)

    _pool_ready = int(((norm_text(ss["status"]).isin(VAL_STATUS_AKTIF))
                       & (norm_text(ss["ketersediaan"]).isin(VAL_TERSEDIA))).sum())
    _gap = compute_skill_gap()
    _demand_top = _gap.sort_values("diminta", ascending=False).head(3)["tool"].tolist()
    _supply_top = _gap.sort_values("dikuasai", ascending=False).head(3)["tool"].tolist()
    catatan_analis([
        f"Dari <b>{student_all['nim'].nunique():,}</b> mahasiswa terdaftar, hanya <b>{_pool_ready:,}</b> "
        "berstatus aktif + tersedia (kolam nyata yang bisa dikirim).",
        f"Tools paling dikuasai: <b>{', '.join(_supply_top)}</b>. "
        f"Tools paling sering diminta di requirement: <b>{', '.join(_demand_top)}</b>.",
        "Radar membantu melihat prodi mana yang demand-nya tinggi tapi kesiapannya rendah - "
        "prioritas pembinaan CV/portofolio.",
        f"<b>{len(eligible_nganggur):,}</b> mahasiswa sudah layak tapi belum pernah dikirim - supply yang belum tersalurkan.",
    ])

    with st.expander("Gap total per bidang, distribusi IPK & semester", icon=":material/bar_chart:"):
        st.caption("Demand = total slot (headcount) posisi yang menerima bidang ini. Satu posisi bisa menerima "
                   "beberapa bidang, jadi angka demand antar-bidang tidak untuk dijumlahkan. Supply = mahasiswa terdaftar.")
        # Demand per bidang = headcount PENUH tiap posisi yang menerima bidang itu
        # (posisi minta segitu ya segitu; slot bisa diisi mahasiswa bidang mana pun
        # yang diterima). Supply = jumlah mahasiswa terdaftar.
        dm_tot = talent_request[["bidang_studi_dibutuhkan", "headcount"]].dropna(subset=["bidang_studi_dibutuhkan"]).copy()
        dm_tot["bidang_studi"] = dm_tot["bidang_studi_dibutuhkan"].str.split(",")
        dm_tot = dm_tot.explode("bidang_studi")
        dm_tot["bidang_studi"] = dm_tot["bidang_studi"].str.strip()
        demand = dm_tot.groupby("bidang_studi")["headcount"].sum().astype(int).reset_index()
        demand.columns = ["bidang_studi", "jumlah"]; demand["tipe"] = "Demand (headcount)"
        supply = student_all["program_studi"].dropna().value_counts().reset_index()
        supply.columns = ["bidang_studi", "jumlah"]; supply["tipe"] = "Supply (mahasiswa)"
        gap_melt = pd.concat([demand, supply], ignore_index=True)
        top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
        gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]
        fig_gap = px.bar(gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
                         title="Matching Gap Total: Demand vs Supply per Bidang Studi (satuan: orang)",
                         color_discrete_map={"Demand (headcount)": COLOR_COCOA, "Supply (mahasiswa)": COLOR_JASMINE})
        fig_gap.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        show_chart(fig_gap, height=320)

        col3, col4 = st.columns(2)
        with col3:
            fig_ipk = px.histogram(ss, x="ipk", nbins=20, title="Distribusi IPK",
                                   color_discrete_sequence=[COLOR_COCOA])
            fig_ipk.update_layout(xaxis_title="IPK", yaxis_title=None)
            show_chart(fig_ipk, height=270)
        with col4:
            sem_ct = student_all["semester"].value_counts().sort_index().reset_index()
            sem_ct.columns = ["semester", "jumlah"]
            fig_sem = px.bar(sem_ct, x="semester", y="jumlah", title="Distribusi Semester",
                             color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_sem.update_layout(xaxis_title="Semester", yaxis_title=None)
            show_chart(fig_sem, height=270)

    if len(eligible_nganggur) > 0:
        with st.expander(f"Daftar {len(eligible_nganggur):,} mahasiswa layak yang belum pernah dikirim", icon=":material/person_alert:"):
            nama_col_e = resolve_col(eligible_nganggur, "nama") or "nama"
            cols_show = [c for c in ["nim", nama_col_e, "program_studi", "semester", "ipk"] if c in eligible_nganggur.columns]
            st.dataframe(eligible_nganggur[cols_show], width="stretch", hide_index=True, height=320)

# ---------------------------------------------------------------------------
# TAB 5 - MATCHING TALENT (data master terkini - tanpa filter)
# ---------------------------------------------------------------------------
def page_matching():
    page_header("Matching Talent")
    match_summary = compute_match_summary()
    n_zero = int((match_summary["kandidat_final"] == 0).sum())

    pool_all = DATA["pool"]
    n_siap = int(pool_all["_eligible"].sum())
    total_slot = int(talent_request["headcount"].sum())
    rasio = total_slot / n_siap if n_siap else 0

    kpi_row([
        {"value": f"{n_siap:,}", "label": "Mahasiswa Siap & Tersedia", "highlight": True,
         "sub": f"vs {total_slot:,} slot dibuka",
         "help": "Mahasiswa aktif + tersedia - kolam nyata yang bisa dikirim, jauh di bawah total slot permintaan."},
        {"value": f"{len(match_summary):,}", "label": "Total Talent Request"},
        {"value": f"{rasio:.1f}x", "label": "Slot per Mahasiswa Siap",
         "help": f"{total_slot:,} slot diperebutkan oleh {n_siap:,} mahasiswa siap."},
        {"value": f"{student_all['nim'].nunique():,}", "label": "Total Mahasiswa Terdaftar"},
    ])

    # ---- Kecocokan lokasi (BT-01: domisili vs kota untuk WFO/Hybrid) ----
    with st.container(border=True):
        section("Kecocokan Lokasi: Mahasiswa Siap vs Slot WFO/Hybrid per Kota",
                "Mahasiswa Siap = mahasiswa aktif dan tersedia per kota domisili. "
                "Slot WFO/Hybrid = kebutuhan posisi WFO/Hybrid per kota perusahaan.")
        ss_loc = status_student.copy()
        ready_mask = (norm_text(ss_loc["status"]).isin(VAL_STATUS_AKTIF)
                      & norm_text(ss_loc["ketersediaan"]).isin(VAL_TERSEDIA))
        supply_kota = ss_loc[ready_mask]["domisili"].dropna().astype(str).str.strip().value_counts()

        tr_loc = talent_request.merge(company[["id_company", "kota"]], on="id_company", how="left")
        if "working_arrangement" in tr_loc.columns:
            tr_wfo = tr_loc[norm_text(tr_loc["working_arrangement"]).isin({"wfo", "hybrid"})]
        else:
            tr_wfo = tr_loc
        demand_kota = tr_wfo.groupby(tr_wfo["kota"].astype(str).str.strip())["headcount"].sum()

        kota_all = sorted(set(supply_kota.index) | set(demand_kota.index))
        loc = pd.DataFrame({"kota": kota_all})
        loc["Mahasiswa Siap"] = loc["kota"].map(supply_kota).fillna(0).astype(int)
        loc["Slot WFO/Hybrid"] = loc["kota"].map(demand_kota).fillna(0).astype(int)
        loc = loc[loc["kota"].str.lower() != "nan"]
        loc["total"] = loc["Mahasiswa Siap"] + loc["Slot WFO/Hybrid"]
        loc = loc.sort_values("total", ascending=False).head(12)
        loc_melt = loc.melt(id_vars="kota", value_vars=["Mahasiswa Siap", "Slot WFO/Hybrid"],
                            var_name="sisi", value_name="jumlah")
        fig_loc = px.bar(loc_melt, x="jumlah", y="kota", color="sisi", orientation="h", barmode="group",
                         color_discrete_map={"Mahasiswa Siap": COLOR_OLIVE, "Slot WFO/Hybrid": COLOR_COCOA})
        fig_loc.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
        fig_loc.update_yaxes(categoryorder="total ascending")
        show_chart(fig_loc, height=300)

        surplus_kota = loc.loc[loc["Mahasiswa Siap"] - loc["Slot WFO/Hybrid"] > 0, "kota"].head(1).tolist()
        defisit_kota = loc.loc[loc["Slot WFO/Hybrid"] - loc["Mahasiswa Siap"] > 0, "kota"].head(1).tolist()
        catatan_analis([
            "Posisi WFO/Hybrid paling lancar diisi kalau ada mahasiswa siap di kota yang sama.",
            (f"Kota dengan pasokan mahasiswa berlebih: <b>{surplus_kota[0]}</b> - cocok jadi sumber kandidat." if surplus_kota else "Belum ada kota dengan surplus pasokan yang jelas."),
            (f"Kota dengan slot WFO/Hybrid melebihi pasokan lokal: <b>{defisit_kota[0]}</b> - "
             "pertimbangkan kandidat bersedia relokasi atau posisi WFH." if defisit_kota else "Slot WFO/Hybrid relatif terpenuhi pasokan lokal."),
            "Lokasi melengkapi kriteria prodi, semester, IPK, dan tools di bawah untuk matching yang realistis.",
        ])

    with st.container(border=True):
        section("Kandidat Memenuhi Syarat per Talent Request")

        colp1, colp2 = st.columns(2)
        comp_counts = match_summary.groupby("company_name").size()
        comp_opts = sorted(match_summary["company_name"].dropna().unique().tolist())
        pilih_comp = colp1.selectbox(
            "Perusahaan", comp_opts,
            format_func=lambda c: f"{c} ({comp_counts.get(c, 0)} request)",
        )
        subset_req = match_summary[match_summary["company_name"] == pilih_comp].sort_values("kandidat_final", ascending=False)
        req_labels = {
            r.id_talent_req: f"{r.nama_posisi} ({r.id_talent_req}) - {r.kandidat_final} kandidat"
            for r in subset_req.itertuples(index=False)
        }
        selected_id = colp2.selectbox("Posisi / Request", list(req_labels.keys()), format_func=lambda k: req_labels[k])

        selected = talent_request[talent_request["id_talent_req"] == selected_id].iloc[0]
        bidang_dibutuhkan = [x.strip() for x in str(selected.get("bidang_studi_dibutuhkan", "")).split(",") if x.strip()]
        min_semester = selected.get("minimum_semester", 0)
        if pd.isna(min_semester):
            min_semester = 0

        pool = DATA["pool"]
        prodi_col = DATA["pool_prodi_col"]
        semester_col = DATA["pool_semester_col"]
        nama_col = resolve_col(pool, "nama") or "nama"

        bidang_norm = [b.lower() for b in bidang_dibutuhkan]
        cocok_prodi_df = pool[pool["_prodi_norm"].isin(bidang_norm)]
        cocok_semester_df = cocok_prodi_df[pd.to_numeric(cocok_prodi_df[semester_col], errors="coerce").fillna(0) >= min_semester]
        candidates = cocok_semester_df[cocok_semester_df["_eligible"]].copy()

        if len(candidates) == 0:
            if len(cocok_prodi_df) == 0:
                sebab = f"tidak ada mahasiswa dari bidang studi <b>{', '.join(bidang_dibutuhkan) or '(kosong)'}</b>."
            elif len(cocok_semester_df) == 0:
                sebab = f"ada {len(cocok_prodi_df):,} mahasiswa dari bidang yang cocok, tapi semuanya di bawah minimum semester ({int(min_semester)})."
            else:
                sebab = (f"ada {len(cocok_semester_df):,} mahasiswa cocok bidang & semester, "
                         "tapi belum ada yang berstatus aktif dan tersedia.")
            insight(f"Tidak ada kandidat yang memenuhi syarat - {sebab}", kind="error")
        else:
            candidates["jml_tools"] = candidates["tools"].fillna("").astype(str).apply(
                lambda s: len([x for x in s.split(",") if x.strip()])) if "tools" in candidates.columns else 0
            sem_num = pd.to_numeric(candidates[semester_col], errors="coerce").fillna(0)

            col_pop, col_n = st.columns(2, vertical_alignment="bottom")
            with col_pop.popover(":material/tune: Pilih Kriteria", width="stretch"):
                st.caption("Centang kriteria yang diprioritaskan. Kandidat diurutkan dari yang paling memenuhi kriteria terpilih.")
                pr_ipk = st.checkbox("IPK tinggi", value=True)
                pr_porto = st.checkbox("Punya portofolio", value=True)
                pr_tools = st.checkbox("Tools banyak", value=True)
                pr_cv = st.checkbox("Punya CV", value=False)
                pr_sem = st.checkbox("Semester tinggi", value=False)
            top_n = col_n.number_input("Teratas", min_value=5, max_value=100, value=25, step=5)

            def _norm01(s):
                s = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
                rng = s.max() - s.min()
                return (s - s.min()) / rng if rng > 0 else s * 0

            comps = []
            if pr_ipk:
                comps.append(_norm01(candidates["ipk"]))
            if pr_porto:
                comps.append(norm_text(candidates["portofolio"]).isin(VAL_ADA).astype(float))
            if pr_tools:
                comps.append(_norm01(candidates["jml_tools"]))
            if pr_cv:
                comps.append(norm_text(candidates["cv"]).isin(VAL_ADA).astype(float))
            if pr_sem:
                comps.append(_norm01(sem_num))

            base_cols = [c for c in ["nim", nama_col, prodi_col, semester_col, "ipk",
                                     "cv", "portofolio", "jml_tools", "domisili"] if c in candidates.columns]
            col_cfg = {
                "ipk": st.column_config.NumberColumn("IPK", format="%.2f"),
                "jml_tools": st.column_config.NumberColumn("Jml Tools"),
                semester_col: st.column_config.NumberColumn("Semester"),
            }
            if comps:
                candidates["skor_prioritas"] = sum(comps) / len(comps)
                view = candidates.sort_values("skor_prioritas", ascending=False).head(int(top_n))
                show_cols = base_cols + ["skor_prioritas"]
                col_cfg["skor_prioritas"] = st.column_config.ProgressColumn("Skor Prioritas", format="%.2f", min_value=0, max_value=1)
                ket = "skor prioritas (gabungan kriteria terpilih)"
            else:
                view = candidates.head(int(top_n))
                show_cols = base_cols
                ket = "urutan default"

            st.markdown(f"**{len(candidates):,} kandidat memenuhi syarat** menampilkan {len(view)} teratas menurut {ket}:")
            st.dataframe(view[show_cols], width="stretch", hide_index=True, height=430, column_config=col_cfg)

    with st.expander(f"Ringkasan jumlah kandidat memenuhi syarat - semua {len(match_summary):,} talent request", icon=":material/table_view:"):
        st.caption(
            "Kolom bertahap memperlihatkan di tahap mana kandidat menyusut: cocok prodi, lalu + minimum semester, "
            "lalu + aktif & tersedia (kandidat final)."
        )
        st.dataframe(
            match_summary.sort_values(["kandidat_final", "cocok_prodi"]).reset_index(drop=True),
            width="stretch", hide_index=True, height=320,
            column_config={
                "id_talent_req": "ID", "nama_posisi": "Posisi", "company_name": "Perusahaan",
                "bidang_dibutuhkan": "Bidang Dibutuhkan", "min_semester": "Min. Smt",
                "cocok_prodi": "Cocok Prodi", "cocok_prodi_semester": "+ Semester",
                "kandidat_final": "Kandidat Final",
            },
        )


# ---------------------------------------------------------------------------
# TAB 6 - LAPORAN & QUALITY CHECK
# ---------------------------------------------------------------------------
def page_laporan():
    tahun_f, prodi_f, jenis_f, _ = page_header("Laporan", key="laporan")
    m = scope_master(tahun_f, prodi_f, jenis_f)

    with st.container(border=True):
        section("Rekapitulasi Placement per Periode", "Dikelompokkan per program studi, perusahaan, atau jenis penempatan.")
        dim_options = {
            "Program Studi": "program_studi",
            "Perusahaan": COMPANY_NAME_COL,
            "Jenis Penempatan": "jenis_penempatan",
        }
        dims_pilihan = st.multiselect("Kelompokkan berdasarkan", list(dim_options.keys()), default=["Program Studi"])
        mode_animasi = st.toggle("Mode animasi", value=False,
                                 help="Putar perubahan komposisi placement antar kuartal (tombol play di bawah chart).")

        placement_only = m[m["rejection"] == "Placement"].copy()
        placement_only["periode"] = placement_only["last_update"].dt.to_period("Q").astype(str)

        if dims_pilihan:
            group_cols = ["periode"] + [dim_options[d] for d in dims_pilihan]
            recap = placement_only.groupby(group_cols).size().reset_index(name="jumlah_placement")
            dim_utama = dim_options[dims_pilihan[0]]
            n_kategori = recap[dim_utama].nunique()

            if mode_animasi and n_kategori <= 20:
                # lengkapi kombinasi kosong supaya tiap frame animasi konsisten
                pv = recap.pivot_table(index="periode", columns=dim_utama,
                                       values="jumlah_placement", aggfunc="sum").fillna(0)
                anim = pv.reset_index().melt(id_vars="periode", var_name=dim_utama, value_name="jumlah_placement")
                anim = anim.sort_values("periode")
                fig_recap = px.bar(
                    anim, x=dim_utama, y="jumlah_placement", color=dim_utama,
                    animation_frame="periode",
                    title=f"Placement per Kuartal berdasarkan {dims_pilihan[0]} (animasi)",
                    color_discrete_sequence=PALETTE_SEQUENTIAL,
                )
                fig_recap.update_layout(
                    showlegend=False, xaxis_title=None, yaxis_title=None,
                    yaxis_range=[0, float(anim["jumlah_placement"].max()) * 1.15],
                )
                show_chart(fig_recap, height=380)
            else:
                if mode_animasi and n_kategori > 20:
                    st.caption(f"Mode animasi dinonaktifkan: kategori terlalu banyak ({n_kategori}). Pilih dimensi dengan kategori lebih sedikit.")
                fig_recap = px.bar(
                    recap.sort_values("periode"), x="periode", y="jumlah_placement", color=dim_utama,
                    title=f"Placement per Kuartal berdasarkan {dims_pilihan[0]}",
                    color_discrete_sequence=PALETTE_SEQUENTIAL,
                )
                fig_recap.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
                show_chart(fig_recap, height=380)

            with st.expander("Tabel rekap & unduh CSV", icon=":material/table_view:"):
                st.dataframe(recap.sort_values("periode", ascending=False),
                             width="stretch", hide_index=True, height=300)
                st.download_button("Unduh Rekap CSV", recap.to_csv(index=False).encode("utf-8"),
                                   "rekap_placement.csv", "text/csv", icon=":material/download:")

    with st.container(border=True):
        section("Kesehatan dan Kesegaran Data Mahasiswa", "Memastikan data status mahasiswa selalu mutakhir.")
        ref_quality = status_student["sync_date"].max() if "sync_date" in status_student.columns else pd.Timestamp(datetime.now().date())
        stale_days = (ref_quality - status_student["sync_date"]).dt.days if "sync_date" in status_student.columns else pd.Series(dtype=float)
        n_stale = int((stale_days > SYNC_STALE_DAYS).sum()) if stale_days.notna().any() else 0
        # mahasiswa di STUDENT ALL yang belum punya record STATUS STUDENT
        nim_all = set(student_all["nim"].dropna().unique())
        nim_status = set(status_student["nim"].dropna().unique())
        n_belum_sync = len(nim_all - nim_status)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric(f"Data Status Usang (> {SYNC_STALE_DAYS} hari)", f"{n_stale:,}")
        mcol2.metric("Rata-rata Umur Sync", f"{stale_days.mean():.0f} hari" if stale_days.notna().any() else "-")
        mcol3.metric("Belum Punya Data Status", f"{n_belum_sync:,}")

        sync_bulan = status_student.copy()
        sync_bulan["bulan_sync"] = sync_bulan["sync_date"].dt.to_period("M").astype(str)
        sync_ct = sync_bulan["bulan_sync"].value_counts().sort_index().reset_index()
        sync_ct.columns = ["bulan", "jumlah"]
        fig_sync = px.bar(sync_ct, x="bulan", y="jumlah",
                          title="Sebaran Waktu Sinkronisasi Data Mahasiswa",
                          color_discrete_sequence=[COLOR_COCOA])
        fig_sync.update_layout(xaxis_title=None, yaxis_title=None)
        show_chart(fig_sync, height=280)

        catatan_analis([
            f"<b>{n_stale:,}</b> record status sudah usang (&gt;{SYNC_STALE_DAYS} hari sejak sync terakhir) - "
            "perlu disegarkan agar keputusan matching tidak salah.",
            f"<b>{n_belum_sync:,}</b> mahasiswa belum punya data status kesiapan sama sekali.",
            "Rekap placement bisa dipecah per program studi, perusahaan, dan jenis penempatan, lalu diunduh CSV.",
            "Kesegaran data adalah fondasi: semua analisis lain hanya seakurat data sync terakhir.",
        ])

# ---------------------------------------------------------------------------
# NAVIGASI SIDEBAR (menggantikan tabs) - logo di atas, teks CDC di bawah logo,
# lalu menu halaman. position="hidden" agar nav bawaan tidak dobel dengan
# menu custom di bawah.
# ---------------------------------------------------------------------------
PAGES = [
    st.Page(page_overview, title="Overview", icon=":material/monitoring:", default=True),
    st.Page(page_funnel, title="Funnel & Ghosting", icon=":material/filter_alt:"),
    st.Page(page_mitra, title="Mitra", icon=":material/handshake:"),
    st.Page(page_segmentasi, title="Segmentasi Mitra", icon=":material/scatter_plot:"),
    st.Page(page_kesiapan, title="Kesiapan", icon=":material/school:"),
    st.Page(page_matching, title="Matching Talent", icon=":material/person_search:"),
    st.Page(page_laporan, title="Laporan", icon=":material/description:"),
]
nav = st.navigation(PAGES, position="hidden")

# CSS penanda menu aktif disuntik dinamis berdasarkan halaman terpilih.
# Efek "tab menyatu": pill terang menempel ke tepi kanan sidebar dan
# menyambung ke area konten, dengan lekukan cekung di atas & bawah
# (pseudo-element lingkaran transparan + box-shadow berwarna latar konten).
# href yang dirender Streamlit: "" untuk halaman default, "page_xxx"
# (tanpa garis miring) untuk lainnya - hasil inspeksi DOM langsung.
_current_href = nav.url_path or ""
_ACTIVE_BG = "#FBF2E0"  # samakan dengan warna dasar latar konten
st.markdown(
    f"""<style>
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}'] {{
        position: relative;
        background: {_ACTIVE_BG} !important;
        border-radius: 999px 0 0 999px !important;
        width: calc(100% + 20px);
        box-sizing: border-box !important;
        margin-left: 0 !important;
        padding-right: 1.4rem !important;
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }}
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}'] p,
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}'] span {{
        color: {COLOR_SEAL_BROWN} !important;
        font-weight: 700;
    }}
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}']::before {{
        content: "";
        position: absolute;
        right: 0;
        top: -30px;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: transparent;
        box-shadow: 15px 15px 0 {_ACTIVE_BG};
    }}
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}']::after {{
        content: "";
        position: absolute;
        right: 0;
        bottom: -30px;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: transparent;
        box-shadow: 15px -15px 0 {_ACTIVE_BG};
    }}
    </style>""",
    unsafe_allow_html=True,
)

def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


with st.sidebar:
    leaves_html = ""
    logo_html = ""
    if os.path.exists("static/leaves.png"):
        leaves_html = f'<img class="side-leaves" src="data:image/png;base64,{_img_b64("static/leaves.png")}">'
    if os.path.exists("static/logo.png"):
        logo_html = f'<img class="side-logo-img" src="data:image/png;base64,{_img_b64("static/logo.png")}">'
    if leaves_html or logo_html:
        st.markdown(f'<div class="side-logo">{leaves_html}{logo_html}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-brand">'
        '<div class="brand-name">CDC</div>'
        '<div class="brand-sub">Career Development Center</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    for p in PAGES:
        st.page_link(p)
    if LAST_SYNC_TXT:
        st.markdown(f'<div class="side-sync">{LAST_SYNC_TXT}<br>SSDC2026025 - Makan Apaya</div>', unsafe_allow_html=True)

nav.run()
