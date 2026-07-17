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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

st.set_page_config(
    page_title="CDC SSF UNS — Placement Monitoring",
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
# STYLE — CSS Custom, gaya BENTO BOX
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

/* Nohemi tidak tersedia di Google Fonts/Fontshare — di-load dari folder
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
   background: MESH GRADIENT lembut dari palet — beberapa radial-gradient
   ditumpuk di atas warna dasar krem */
.stApp {{
    background:
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
section[data-testid="stSidebar"] {{
    background:
        radial-gradient(at 88% 10%, rgba(247, 212, 117, 0.28) 0px, transparent 40%),
        radial-gradient(at 8% 42%, rgba(226, 120, 47, 0.35) 0px, transparent 45%),
        radial-gradient(at 82% 88%, rgba(164, 176, 94, 0.22) 0px, transparent 45%),
        linear-gradient(180deg, #A64B1A 0%, #7C3413 45%, {COLOR_SEAL_BROWN} 100%) !important;
    border-radius: 0 22px 22px 0;
}}
/* sidebar tidak bisa ditutup */
div[data-testid="stSidebarCollapseButton"],
button[data-testid="stExpandSidebarButton"],
div[data-testid="collapsedControl"] {{
    display: none !important;
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
    margin-top: 7vh;
    text-align: center;
    font-size: 0.72rem;
    color: {tint(COLOR_JASMINE, 0.45)} !important;
}}
div[data-testid="stPopover"], .stPopover {{
    width: 100%;
    display: flex;
    justify-content: flex-end;
}}
div[data-testid="stPopover"] > div, .stPopover > div {{
    margin-left: auto !important;
}}
div[data-testid="stPopover"] button, .stPopover button {{
    margin-left: auto !important;
}}
div[data-testid="stColumn"]:has(div[data-testid="stPopover"]) div[data-testid="stVerticalBlock"] {{
    align-items: flex-end !important;
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
    margin: 10px 0 2px 0;
}}
.side-leaves {{
    position: absolute;
    top: -16px;
    left: 50%;
    transform: translateX(-50%);
    width: 240px;
    max-width: 96%;
    opacity: 0.6;
    transform-origin: center;
    animation: leafSway 6s ease-in-out infinite;
}}
.side-logo-img {{
    position: relative;
    width: 185px;
    max-width: 85%;
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
    padding-bottom: 0.6rem !important;
}}
div[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

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
    gap: 12px;
    margin: 2px 0 8px 0;
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
    margin-bottom: 4px;
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

#MainMenu, footer {{visibility: hidden;}}
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
    untuk memberi outline gradasi + glow (pola aksen tunggal)."""
    html = '<div class="kpi-row">'
    for c in cards:
        cls = "kpi-card kpi-hl" if c.get("highlight") else "kpi-card"
        help_attr = f' title="{c["help"]}"' if c.get("help") else ""
        sub = f'<div class="kpi-sub">{c["sub"]}</div>' if c.get("sub") else ""
        html += (
            f'<div class="{cls}"{help_attr}>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div>{sub}</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig, height=300):
    has_title = bool(fig.layout.title.text)
    has_legend = len(fig.data) > 1 or any(tr.type == "pie" for tr in fig.data)
    top_margin = 38 if has_title else (34 if has_legend else 14)

    # legend dengan banyak kategori dipindah ke BAWAH chart supaya tidak
    # bertabrakan dengan judul (mis. rekap 18 program studi)
    n_legend = sum(1 for tr in fig.data if getattr(tr, "showlegend", True) is not False)
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
    # title_font hanya diset bila chart punya judul — plotly.js merender
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
    # (Placement / Rejection apa pun — menolak juga bentuk respons).
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
    dm["bidang_norm"] = dm["bidang_studi_dibutuhkan"].astype(str).str.split(",")
    dm = dm.explode("bidang_norm")
    dm["bidang_norm"] = dm["bidang_norm"].str.strip().str.lower()
    dm = dm[dm["bidang_norm"] != ""]
    dm["bulan"] = dm["request_date"].dt.to_period("M").dt.to_timestamp()
    demand_monthly = dm.groupby(["bulan", "bidang_norm"])["headcount"].sum().reset_index()

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
# ML MODEL — MATCHING TALENT
# Label negatif = SEMUA nilai "Rejection ..." (di data tidak ada nilai
# "Rejected" pada kolom rejection). "On Progress" dan "Ghosting" dikeluarkan
# karena belum/bukan keputusan perusahaan atas kualitas kandidat.
# Metrik utama: ROC-AUC — lebih jujur daripada akurasi untuk kelas tak
# seimbang. AUC 0.5 berarti fitur profil tidak mengandung sinyal prediktif
# (kasus data sintetis kompetisi ini); pipeline yang sama akan belajar pola
# sebenarnya begitu dipakai pada data operasional riil.
# ---------------------------------------------------------------------------
ML_FEATURE_BASE = ["program_studi", "semester", "ipk", "cv", "portofolio", "domisili"]
MIN_TRAIN_ROWS = 15


@st.cache_resource(show_spinner="Melatih model rekomendasi (hanya sekali)...")
def train_matching_model():
    pool = DATA["pool"]
    base = pool.merge(
        tracking_student.loc[
            tracking_student["rejection"].isin(["Placement"] + REJECTION_STAGES),
            ["nim", "rejection"],
        ],
        on="nim", how="inner",
    )
    base["target"] = (base["rejection"] == "Placement").astype(int)

    resolved_cols = {}
    for feat in ML_FEATURE_BASE:
        col = resolve_col(base, feat)
        if col is not None:
            resolved_cols[feat] = col

    feature_cols = list(resolved_cols.values())
    if len(base) < MIN_TRAIN_ROWS or base["target"].nunique() < 2 or not feature_cols:
        return None

    X = base[feature_cols].copy()
    y = base["target"]

    encoders = {}
    numeric_feats = {"semester", "ipk"}
    for feat, col in resolved_cols.items():
        if feat in numeric_feats:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype(str).fillna("Unknown")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify,
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=4,
        random_state=42, class_weight="balanced", n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test) if len(X_test) > 0 else np.nan
    try:
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except ValueError:
        test_auc = np.nan
    baseline = float(max(y.mean(), 1 - y.mean()))

    feature_importance = pd.DataFrame({
        "fitur": list(resolved_cols.keys()),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model, "encoders": encoders, "feature_cols": feature_cols,
        "resolved_cols": resolved_cols, "numeric_feats": numeric_feats,
        "feature_importance": feature_importance,
        "test_accuracy": test_accuracy, "test_auc": test_auc,
        "baseline": baseline, "n_train": len(base),
    }


ml_bundle = train_matching_model()

# Bobot ML adaptif: w = (AUC - 0.5) x 2, dibatasi 0..1.
# AUC 0.5 (tidak ada sinyal, kasus data sintetis) -> ranking dipegang AHP;
# AUC 1.0 (sinyal sempurna) -> ranking sepenuhnya dari model.
# Dengan begitu skor rekomendasi selalu masuk akal, dan otomatis bergeser
# ke ML begitu pipeline dipakai pada data operasional riil.
if ml_bundle is not None and not np.isnan(ml_bundle["test_auc"]):
    ML_WEIGHT = float(min(1.0, max(0.0, (ml_bundle["test_auc"] - 0.5) * 2)))
else:
    ML_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# FILTER — di dalam POPOVER per tab supaya hemat ruang vertikal (no-scroll)
# ---------------------------------------------------------------------------
FILTER_TAHUN = sorted(int(t) for t in master["tahun_update"].dropna().unique())
FILTER_PRODI = sorted(student_all["program_studi"].dropna().unique().tolist()) if "program_studi" in student_all.columns else []
FILTER_JENIS = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist()) if "jenis_penempatan" in talent_request.columns else []


def run_gsap_animations():
    """Animasi interaktif via GSAP (dimuat dari CDN). Streamlit menyaring tag
    <script> di markdown, jadi JS dijalankan lewat components.html — iframe
    same-origin yang boleh memanipulasi DOM halaman induk. Kalau CDN gagal
    dimuat, halaman tetap tampil normal tanpa animasi (graceful fallback)."""
    # st.iframe menggantikan components.html (dihapus Streamlit per Jun 2026);
    # fallback ke components.html untuk versi lama.
    _html_embed = getattr(st, "iframe", components.html)
    _html_embed(
        """
        <script>
        (function () {
            const P = window.parent;
            const doc = P.document;

            function animate() {
                const gsap = P.gsap;
                if (!gsap) return;
                const cards = Array.from(doc.querySelectorAll(".kpi-card"));
                const tiles = Array.from(doc.querySelectorAll("div[data-testid='stVerticalBlockBorderWrapper']"));
                if (cards.length + tiles.length === 0) return false;

                gsap.fromTo(cards,
                    { opacity: 0, y: 26, scale: 0.96 },
                    { opacity: 1, y: 0, scale: 1, duration: 0.55, stagger: 0.08, ease: "power2.out", overwrite: "auto" });
                gsap.fromTo(tiles,
                    { opacity: 0, y: 30 },
                    { opacity: 1, y: 0, duration: 0.6, stagger: 0.1, delay: 0.15, ease: "power2.out", overwrite: "auto" });

                // count-up angka KPI: "41,600", "152.3%", "51 hari"
                doc.querySelectorAll(".kpi-value").forEach(function (el) {
                    if (el.dataset.counted === "1") return;
                    const txt = el.textContent.trim();
                    const m = txt.match(/^([\\d.,]+)(.*)$/);
                    if (!m) return;
                    const target = parseFloat(m[1].replace(/,/g, ""));
                    if (isNaN(target)) return;
                    const suffix = m[2] || "";
                    const decimal = m[1].includes(".");
                    el.dataset.counted = "1";
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
        """,
        height=1,
    )


def page_header(title: str, key: str = None, with_prodi: bool = True, with_ref_date: bool = False):
    """Baris judul halaman: judul di kiri, tombol Filter (popover) di ujung
    kanan. Juga menyuntikkan animasi transisi dengan nama keyframe unik per
    halaman — nama yang berubah membuat animasi restart setiap pindah halaman."""
    slug = "".join(ch for ch in (key or title).lower() if ch.isalnum())
    st.markdown(
        f"<style>"
        f"@keyframes pagein_{slug} {{ from {{ opacity: 0; transform: translateY(14px); }} "
        f"to {{ opacity: 1; transform: none; }} }} "
        f".block-container {{ animation: pagein_{slug} 0.5s ease; }}"
        f"</style>",
        unsafe_allow_html=True,
    )

    col_t, col_f = st.columns([8, 1], vertical_alignment="center")
    with col_t:
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)

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
    run_gsap_animations()
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
# TAB 1 — OVERVIEW
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

    kpi_row([
        {"value": f"{total_dikirim_individu:,}", "label": "Kandidat Dikirim",
         "help": "Jumlah proses seleksi kandidat (baris tracking_student) pada rentang filter."},
        {"value": f"{total_placement:,}", "label": "Placement Berhasil",
         "sub": f"{success_rate:.1f}% dari kandidat dikirim", "highlight": True,
         "help": "Placement dan success rate memang satu keluarga: success rate = placement dibagi kandidat dikirim, jadi keduanya digabung di satu kartu."},
        {"value": f"{fulfillment_rate:.1f}%", "label": "Fulfillment Rate",
         "help": "Jumlah dikirim vs diminta. Di atas 100% berarti CDC mengirim lebih banyak kandidat dari kuota — wajar untuk shortlist."},
        {"value": f"{lama_proses:.0f} hari" if lama_proses is not None else "-", "label": "Rata-rata Lama Proses",
         "help": "Dari batch dikirim (send_date) sampai keputusan terakhir."},
        {"value": f"{n_request_belum:,}", "label": "Request Belum Terpenuhi",
         "sub": "data master, di luar filter",
         "help": "Talent request yang jumlah kirimnya masih di bawah headcount."},
    ])

    col_kiri, col_kanan = st.columns([2, 1])
    with col_kiri:
        with st.container(border=True):
            section("Placement & Success Rate per Bulan",
                    "Batang = jumlah placement; garis = success rate. Gunakan tombol rentang untuk zoom.")
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

    with col_kanan:
        with st.container(border=True):
            section("Status Akhir Kandidat")
            status_map = m["rejection"].map(
                lambda r: "Placement" if r == "Placement"
                else ("Ditolak" if r in REJECTION_STAGES
                      else ("Ghosting" if r == "Ghosting" else "On Progress"))
            )
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
        section("Perjalanan Kandidat: dari Dikirim sampai Placement",
                "Waterfall: berapa kandidat hilang di tiap penyebab, dan berapa yang berakhir placement.")
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

# ---------------------------------------------------------------------------
# TAB 2 — FUNNEL & GHOSTING
# ---------------------------------------------------------------------------
def page_funnel():
    tahun_f, prodi_f, jenis_f, tanggal_acuan = page_header("Funnel & Ghosting", key="funnel", with_ref_date=True)
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
        {"value": f"{total_tc:,}", "label": "Batch Terkirim"},
        {"value": f"{int(followup_counts.get('FU 1', 0)):,}", "label": "Butuh FU 1"},
        {"value": f"{int(followup_counts.get('FU 2', 0)):,}", "label": "Butuh FU 2"},
        {"value": f"{int(followup_counts.get('FU 3', 0)):,}", "label": "Butuh FU 3"},
        {"value": f"{n_ghosting:,}", "label": "Ghosting", "sub": f"{ghosting_rate:.1f}% dari batch terkirim",
         "highlight": True,
         "help": "Aturan FAQ: >28 hari sejak send_date tanpa respons perusahaan."},
    ])

    col_kiri, col_kanan = st.columns(2)
    with col_kiri:
        with st.container(border=True):
            section("Funnel Seleksi Kandidat",
                    "Kandidat yang MENCAPAI tiap tahap; persentase = konversi dari tahap sebelumnya.")
            funnel_counts = [int((m["stage_reached"] >= i).sum()) for i in range(len(FUNNEL_STAGES))]
            fig_funnel = go.Figure(go.Funnel(
                y=FUNNEL_STAGES, x=funnel_counts,
                marker={"color": PALETTE_SEQUENTIAL},
                textinfo="value+percent previous",
            ))
            show_chart(fig_funnel, height=330)

    with col_kanan:
        with st.container(border=True):
            section("Rejection Breakdown per Tahap", "Di tahap mana kandidat paling banyak gagal.")
            rej = m[m["rejection"].isin(REJECTION_STAGES)]["rejection"].value_counts().reindex(REJECTION_STAGES).fillna(0).reset_index()
            rej.columns = ["tahap_rejection", "jumlah"]
            fig_rej = px.bar(rej, x="jumlah", y="tahap_rejection", orientation="h",
                             color_discrete_sequence=[COLOR_SIENNA])
            fig_rej.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_rej, height=330)

    konversi = [
        (FUNNEL_STAGES[i], funnel_counts[i] / funnel_counts[i - 1] * 100)
        for i in range(1, len(FUNNEL_STAGES)) if funnel_counts[i - 1] > 0
    ]
    if konversi:
        tahap_bocor, rate_bocor = min(konversi, key=lambda t: t[1])
        insight(f"Konversi terendah ada di tahap <b>{tahap_bocor}</b> ({rate_bocor:.0f}% dari tahap sebelumnya) — "
                "prioritaskan pendampingan CDC di tahap ini. "
                f"Ghosting menurut aturan FAQ: <b>{n_ghosting:,} batch</b> ({ghosting_rate:.1f}%).", kind="warning")

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
            show_chart(fig_gc, height=290)
        with col2:
            ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
            ghosting_by_stage = ghosted_children["progress_student"].value_counts().head(8).reset_index()
            ghosting_by_stage.columns = ["tahap", "jumlah"]
            fig_gs = px.bar(ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
                            title="Kandidat pada Batch Ghosting — Tahap Terakhir",
                            color_discrete_sequence=[COLOR_SIENNA])
            fig_gs.update_layout(yaxis_title=None, xaxis_title=None)
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
            f"Data mencatat {n_ghosting_tercatat:,} kandidat berlabel Ghosting — angka per kandidat itu tidak setara "
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
# TAB 3 — MITRA
# ---------------------------------------------------------------------------
def page_mitra():
    tahun_f, prodi_f, jenis_f, _ = page_header("Mitra Perusahaan", key="mitra")
    m = scope_master(tahun_f, prodi_f, jenis_f)

    kpi_row([
        {"value": f"{company['id_company'].nunique():,}", "label": "Perusahaan Mitra (master)"},
        {"value": f"{talent_request['id_talent_req'].nunique():,}", "label": "Total Talent Request (master)"},
        {"value": f"{m[COMPANY_NAME_COL].nunique():,}", "label": "Perusahaan Aktif (filter)", "highlight": True,
         "help": "Perusahaan yang punya proses seleksi berjalan pada rentang filter."},
        {"value": f"{m['nama_posisi'].nunique():,}" if "nama_posisi" in m.columns else "-", "label": "Posisi Dibuka (filter)"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section("Top 10 Acceptance Rate", "Minimal 3 kandidat dikirim; nama dari master COMPANY.")
            perf = m.groupby(COMPANY_NAME_COL).agg(
                total=("id_tracking_student", "count"),
                placement=("rejection", lambda x: (x == "Placement").sum()),
            ).reset_index()
            perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
            perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10).rename(columns={COMPANY_NAME_COL: "perusahaan"})
            fig_acc = px.bar(perf, x="acceptance_rate", y="perusahaan", orientation="h",
                             color_discrete_sequence=[COLOR_COCOA])
            fig_acc.update_layout(yaxis_title=None, xaxis_title="%")
            show_chart(fig_acc, height=310)
    with col2:
        with st.container(border=True):
            section("Top 10 Volume Talent Request")
            tr_named = talent_request.merge(company[["id_company", "company_name"]], on="id_company", how="left")
            volume = tr_named["company_name"].value_counts().head(10).reset_index()
            volume.columns = ["perusahaan", "jumlah_request"]
            fig_vol = px.treemap(volume, path=["perusahaan"], values="jumlah_request",
                                 color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_vol.update_layout(margin=dict(l=4, r=4, t=8, b=4))
            show_chart(fig_vol, height=310)

    with st.expander("Profil mitra: sektor industri & tipe perusahaan", icon=":material/domain:"):
        col3, col4 = st.columns(2)
        with col3:
            sektor = tr_named["industri_sektor"].value_counts().head(10).reset_index() if "industri_sektor" in tr_named.columns else pd.DataFrame(columns=["a", "b"])
            sektor.columns = ["sektor", "jumlah"]
            fig_sektor = px.bar(sektor, x="jumlah", y="sektor", orientation="h",
                                title="Permintaan per Sektor Industri",
                                color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_sektor.update_layout(yaxis_title=None, xaxis_title=None)
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
        st.caption("Diurutkan dari request paling lama. Data master, di luar filter.")
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
# TAB 4 — KESIAPAN (data master terkini — tanpa filter)
# ---------------------------------------------------------------------------
def page_kesiapan():
    page_header("Kesiapan Mahasiswa")
    st.caption("Tab ini memakai data master terkini (snapshot status mahasiswa), sehingga tidak memakai filter.")

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
        {"value": f"{student_all['nim'].nunique():,}", "label": "Mahasiswa Terdaftar"},
        {"value": f"{len(eligible):,}", "label": "Layak Kirim Saat Ini",
         "help": "Status aktif + tersedia + CV ada + portofolio ada. Sesuai FAQ, 'eligible' = kolom 'ketersediaan'."},
        {"value": f"{len(eligible_nganggur):,}", "label": "Layak tapi Belum Pernah Dikirim", "highlight": True,
         "sub": "supply belum tersalurkan",
         "help": "Prioritas untuk dicarikan penempatan."},
        {"value": f"{(cv_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya CV"},
        {"value": f"{(porto_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya Portofolio"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section("Demand vs Supply dari Waktu ke Waktu",
                    "Supply = estimasi mahasiswa SIAP MAGANG per periode (bulan masuk + (semester magang - 1) x 6 bulan, "
                    "semester magang dari histori tiap mahasiswa). Tampilan awal difokuskan ke periode permintaan; "
                    "zoom out untuk melihat semuanya.")
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

            dm_view = DATA["demand_monthly"]
            sup_view = student_all[student_all["siap_dt"].notna()]
            if prodi_pilih != "Semua bidang":
                p_norm = prodi_pilih.strip().lower()
                dm_view = dm_view[dm_view["bidang_norm"] == p_norm]
                sup_view = sup_view[norm_text(sup_view["program_studi"]) == p_norm]

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
            section("Eligibility Mahasiswa", "Sesuai FAQ: kolom 'eligible' = kolom 'ketersediaan'.")
            elig_ct = ss.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
            fig_elig = px.bar(elig_ct, x="ketersediaan", y="jumlah", color="status",
                              color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_elig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
            show_chart(fig_elig, height=330)

    with st.expander("Gap total per bidang, distribusi IPK & semester", icon=":material/bar_chart:"):
        demand = talent_request["bidang_studi_dibutuhkan"].dropna().str.split(",").explode().str.strip().value_counts().reset_index()
        demand.columns = ["bidang_studi", "jumlah"]; demand["tipe"] = "Demand"
        supply = student_all["program_studi"].dropna().value_counts().reset_index()
        supply.columns = ["bidang_studi", "jumlah"]; supply["tipe"] = "Supply"
        gap_melt = pd.concat([demand, supply], ignore_index=True)
        top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
        gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]
        fig_gap = px.bar(gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
                         title="Matching Gap Total: Demand vs Supply per Bidang Studi",
                         color_discrete_map={"Demand": COLOR_COCOA, "Supply": COLOR_JASMINE})
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
# TAB 5 — MATCHING TALENT (data master terkini — tanpa filter)
# ---------------------------------------------------------------------------
def page_matching():
    page_header("Matching Talent")
    st.caption("Tab ini memakai data master terkini, sehingga tidak memakai filter.")
    match_summary = compute_match_summary()
    n_zero = int((match_summary["kandidat_final"] == 0).sum())

    if ml_bundle is not None:
        model_label = "Hybrid"
        model_sub = f"ML {ML_WEIGHT*100:.0f}% + AHP {(1-ML_WEIGHT)*100:.0f}% (adaptif)"
        model_help = (f"Skor = gabungan probabilitas RandomForest ({ml_bundle['n_train']:,} histori keputusan) "
                      "dan skor AHP berbasis kriteria. Bobot ML mengikuti kualitas prediksi model (AUC) — "
                      "detail di expander 'Tentang model' di bawah.")
    else:
        model_label = "AHP"
        model_sub = f"histori < {MIN_TRAIN_ROWS} baris"
        model_help = "Histori keputusan belum cukup untuk melatih model — skor memakai AHP."

    kpi_row([
        {"value": f"{len(match_summary):,}", "label": "Total Talent Request"},
        {"value": f"{n_zero:,}", "label": "Request Tanpa Kandidat Cocok", "highlight": True,
         "help": "Paling kritis: tidak ada satu pun mahasiswa yang memenuhi syarat prodi + semester + eligible."},
        {"value": f"{ml_bundle['n_train']:,}" if ml_bundle else "0", "label": "Histori Keputusan (data latih)"},
        {"value": model_label, "label": "Metode Skoring", "sub": model_sub, "help": model_help},
    ])

    with st.container(border=True):
        section("Cari Kandidat Terbaik untuk Talent Request",
                "Pilih perusahaan lalu posisinya. Skor = hybrid adaptif ML + AHP (bobot mengikuti kualitas model).")

        colp1, colp2 = st.columns(2)
        comp_counts = match_summary.groupby("company_name").size()
        comp_opts = sorted(match_summary["company_name"].dropna().unique().tolist())
        pilih_comp = colp1.selectbox(
            "Perusahaan", comp_opts,
            format_func=lambda c: f"{c} ({comp_counts.get(c, 0)} request)",
        )
        subset_req = match_summary[match_summary["company_name"] == pilih_comp].sort_values("kandidat_final", ascending=False)
        req_labels = {
            r.id_talent_req: f"{r.nama_posisi} ({r.id_talent_req}) — {r.kandidat_final} kandidat"
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

        if len(candidates) > 0:
            # skor AHP selalu dihitung (kriteria transparan: IPK, portofolio,
            # semester, CV; bobot prodi 0.40 selalu penuh karena kandidat di
            # sini sudah lolos filter prodi)
            sem_num = pd.to_numeric(candidates[semester_col], errors="coerce").fillna(0)
            ipk_max = candidates["ipk"].max()
            candidates["skor_ahp"] = (
                0.40 * 1.0
                + 0.30 * (candidates["ipk"] / ipk_max if ipk_max and ipk_max > 0 else 0)
                + 0.15 * norm_text(candidates["portofolio"]).isin(VAL_ADA).astype(int)
                + 0.10 * (sem_num / sem_num.max() if sem_num.max() > 0 else 0)
                + 0.05 * norm_text(candidates["cv"]).isin(VAL_ADA).astype(int)
            )

            if ml_bundle is not None:
                model = ml_bundle["model"]
                encoders = ml_bundle["encoders"]
                resolved_cols = ml_bundle["resolved_cols"]
                numeric_feats = ml_bundle["numeric_feats"]
                feature_cols = ml_bundle["feature_cols"]

                X_cand = pd.DataFrame(index=candidates.index)
                for feat, train_col in resolved_cols.items():
                    cand_col = resolve_col(candidates, feat) or train_col
                    if cand_col not in candidates.columns:
                        X_cand[train_col] = 0
                        continue
                    if feat in numeric_feats:
                        vals = pd.to_numeric(candidates[cand_col], errors="coerce")
                        X_cand[train_col] = vals.fillna(vals.median() if vals.notna().any() else 0)
                    else:
                        le = encoders.get(train_col)
                        vals = candidates[cand_col].astype(str).fillna("Unknown")
                        if le is not None:
                            known = set(le.classes_)
                            vals = vals.map(lambda v: v if v in known else le.classes_[0])
                            X_cand[train_col] = le.transform(vals)
                        else:
                            X_cand[train_col] = 0

                X_cand = X_cand[feature_cols]
                candidates["skor_ml"] = model.predict_proba(X_cand)[:, 1]
                candidates["recommendation_score"] = (
                    ML_WEIGHT * candidates["skor_ml"] + (1 - ML_WEIGHT) * candidates["skor_ahp"]
                )
                candidates["metode_skor"] = f"Hybrid (ML {ML_WEIGHT*100:.0f}% + AHP {(1-ML_WEIGHT)*100:.0f}%)"
            else:
                candidates["recommendation_score"] = candidates["skor_ahp"]
                candidates["metode_skor"] = "AHP"

            candidates = candidates.sort_values("recommendation_score", ascending=False)

        col_hasil, col_dist = st.columns([3, 2])
        with col_hasil:
            st.markdown(f"**{len(candidates):,} kandidat cocok — 50 teratas:**")
            if len(candidates) == 0:
                if len(cocok_prodi_df) == 0:
                    sebab = f"tidak ada mahasiswa dari bidang studi <b>{', '.join(bidang_dibutuhkan) or '(kosong)'}</b>."
                elif len(cocok_semester_df) == 0:
                    sebab = f"ada {len(cocok_prodi_df):,} mahasiswa dari bidang yang cocok, tapi semuanya di bawah minimum semester ({int(min_semester)})."
                else:
                    sebab = (f"ada {len(cocok_semester_df):,} mahasiswa cocok bidang & semester, "
                             "tapi belum ada yang berstatus aktif dan tersedia.")
                insight(f"Tidak ada kandidat yang memenuhi syarat — {sebab}", kind="error")
            else:
                show_cols = [c for c in ["nim", nama_col, prodi_col, semester_col, "ipk", "cv",
                                         "portofolio", "domisili", "recommendation_score", "metode_skor"]
                             if c in candidates.columns]
                st.dataframe(
                    candidates[show_cols].head(50), width="stretch", hide_index=True, height=320,
                    column_config={
                        "recommendation_score": st.column_config.ProgressColumn("Skor", format="%.2f", min_value=0, max_value=1),
                    },
                )
        with col_dist:
            if len(candidates) > 0:
                fig_score = px.histogram(candidates, x="recommendation_score", nbins=20,
                                         title="Distribusi Skor Kandidat",
                                         color_discrete_sequence=[COLOR_COCOA])
                fig_score.update_layout(xaxis_title="Skor", yaxis_title=None)
                show_chart(fig_score, height=320)

    with st.expander(f"Ringkasan kecocokan semua talent request ({n_zero:,} request tanpa kandidat)", icon=":material/table_view:"):
        st.caption(
            "Kolom bertahap memperlihatkan di filter mana kandidat berkurang: prodi, lalu semester, lalu status "
            "aktif+tersedia. Diurutkan dari yang paling kritis."
        )
        st.dataframe(
            match_summary.sort_values(["kandidat_final", "cocok_prodi"]).reset_index(drop=True),
            width="stretch", hide_index=True, height=320,
            column_config={
                "id_talent_req": "ID",
                "nama_posisi": "Posisi",
                "company_name": "Perusahaan",
                "bidang_dibutuhkan": "Bidang Dibutuhkan",
                "min_semester": "Min. Smt",
                "cocok_prodi": "Cocok Prodi",
                "cocok_prodi_semester": "+ Semester",
                "kandidat_final": "Kandidat Final",
            },
        )

    if ml_bundle is not None:
        with st.expander("Tentang model: skoring hybrid, feature importance & evaluasi", icon=":material/psychology:"):
            auc = ml_bundle["test_auc"]
            insight(
                f"Skor rekomendasi = <b>{ML_WEIGHT*100:.0f}% ML + {(1-ML_WEIGHT)*100:.0f}% AHP</b>, dengan bobot ML "
                f"dihitung otomatis dari kualitas prediksi model: w = (AUC − 0.5) × 2. Evaluasi RandomForest pada "
                f"{ml_bundle['n_train']:,} histori keputusan menghasilkan AUC {auc:.2f} — artinya pada data kompetisi "
                "(sintetis) ini keputusan placement tidak berkorelasi dengan profil mahasiswa, sehingga ranking "
                "dipegang AHP yang kriterianya transparan (IPK, portofolio, semester, CV). Begitu pipeline yang sama "
                "dijalankan pada data operasional riil dan AUC naik, bobot ML ikut naik otomatis tanpa mengubah kode.",
                kind="info",
            )
            fig_fi = px.bar(ml_bundle["feature_importance"], x="importance", y="fitur", orientation="h",
                            color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_fi.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_fi, height=250)

# ---------------------------------------------------------------------------
# TAB 6 — LAPORAN & QUALITY CHECK
# ---------------------------------------------------------------------------
def page_laporan():
    tahun_f, prodi_f, jenis_f, _ = page_header("Laporan", key="laporan")
    m = scope_master(tahun_f, prodi_f, jenis_f)

    with st.container(border=True):
        section("Rekapitulasi Placement", "Laporan periodik untuk evaluasi institusi.")
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

    with st.expander("Kualitas data & sinkronisasi", icon=":material/rule:"):
        st.caption("Konsistensi STUDENT ALL vs STATUS STUDENT. Data master, di luar filter.")
        merged_check = student_all.merge(status_student[["nim", "sync_date"]], on="nim", how="left", indicator=True)
        belum_sync = merged_check[merged_check["_merge"] == "left_only"]

        ref_quality = status_student["sync_date"].max() if "sync_date" in status_student.columns else pd.Timestamp(datetime.now().date())
        stale_days = (ref_quality - status_student["sync_date"]).dt.days if "sync_date" in status_student.columns else pd.Series(dtype=float)
        n_stale = int((stale_days > SYNC_STALE_DAYS).sum()) if stale_days.notna().any() else 0

        col_q1, col_q2 = st.columns([2, 3])
        with col_q1:
            st.metric("Mahasiswa Belum Ada Data Status", f"{len(belum_sync):,}")
            st.metric(f"Data Status Usang (> {SYNC_STALE_DAYS} hari)", f"{n_stale:,}")
            st.metric("Rata-rata Umur Sync", f"{stale_days.mean():.0f} hari" if stale_days.notna().any() else "-")
        with col_q2:
            sync_bulan = status_student.copy()
            sync_bulan["bulan_sync"] = sync_bulan["sync_date"].dt.to_period("M").astype(str)
            sync_ct = sync_bulan["bulan_sync"].value_counts().sort_index().reset_index()
            sync_ct.columns = ["bulan", "jumlah"]
            fig_sync = px.bar(sync_ct, x="bulan", y="jumlah",
                              title="Sebaran Waktu Sinkronisasi Data Mahasiswa",
                              color_discrete_sequence=[COLOR_COCOA])
            fig_sync.update_layout(xaxis_title=None, yaxis_title=None)
            show_chart(fig_sync, height=270)

        if len(belum_sync) > 0:
            st.dataframe(belum_sync[["nim", "nama", "program_studi"]].head(20),
                         width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# NAVIGASI SIDEBAR (menggantikan tabs) — logo di atas, teks CDC di bawah logo,
# lalu menu halaman. position="hidden" agar nav bawaan tidak dobel dengan
# menu custom di bawah.
# ---------------------------------------------------------------------------
PAGES = [
    st.Page(page_overview, title="Overview", icon=":material/monitoring:", default=True),
    st.Page(page_funnel, title="Funnel & Ghosting", icon=":material/filter_alt:"),
    st.Page(page_mitra, title="Mitra", icon=":material/handshake:"),
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
# (tanpa garis miring) untuk lainnya — hasil inspeksi DOM langsung.
_current_href = nav.url_path or ""
_ACTIVE_BG = "#FBF2E0"  # samakan dengan warna dasar latar konten
st.markdown(
    f"""<style>
    section[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink'][href='{_current_href}'] {{
        position: relative;
        background: {_ACTIVE_BG} !important;
        border-radius: 999px 0 0 999px !important;
        margin-right: -1.6rem !important;
        padding-right: 1.4rem !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
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
        st.markdown(f'<div class="side-sync">{LAST_SYNC_TXT}<br>build v8</div>', unsafe_allow_html=True)

nav.run()
