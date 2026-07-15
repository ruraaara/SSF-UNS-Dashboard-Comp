import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="SSDC 2026 — Student Placement Dashboard",
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
    """Campur warna dengan putih; factor 0 = warna asli, 1 = putih penuh.
    Semua background turunan dihitung dari palet utama supaya kalau palet
    diganti, seluruh dashboard ikut berubah konsisten."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * factor)
    g = round(g + (255 - g) * factor)
    b = round(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


# 6 tahap funnel butuh 6 warna — urut terang ke gelap
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

# Aturan resmi FAQ BT-05: Ghosting diasumsikan dari PIHAK PERUSAHAAN,
# dihitung dari send_date per batch pengiriman (baris tracking_company).
FU1_DAYS = 7          # > 1 minggu tanpa respons -> FU 1
FU2_DAYS = 14         # > 2 minggu tanpa respons -> FU 2
FU3_DAYS = 21         # > 3 minggu tanpa respons -> FU 3
GHOSTING_DAYS = 28    # > 4 minggu tanpa respons -> Ghosting
SYNC_STALE_DAYS = 14

# FAQ: kolom "eligible" = kolom "ketersediaan". Data bisa berbahasa
# Inggris ("Active"/"Available") atau Indonesia ("Aktif"/"Tersedia"),
# jadi keduanya diterima setelah normalisasi lowercase.
VAL_STATUS_AKTIF = {"active", "aktif"}
VAL_TERSEDIA = {"available", "tersedia"}
VAL_ADA = {"ada"}

REJECTION_STAGES = [
    "Rejection Screening CV",
    "Rejection Study Case",
    "Rejection Interview User",
    "Rejection Final Interview",
]

# ---------------------------------------------------------------------------
# STYLE — CSS Custom
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.dash-header {{
    padding: 18px 24px;
    border-radius: 14px;
    background: linear-gradient(135deg, {COLOR_SEAL_BROWN} 0%, {COLOR_SIENNA} 100%);
    margin-bottom: 18px;
}}
.dash-header h1 {{
    color: #FFF8EE;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}}
.dash-header p {{
    color: {tint(COLOR_JASMINE, 0.4)};
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}}

div[data-testid="stMetric"] {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid {tint(COLOR_COCOA, 0.65)};
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
}}
div[data-testid="stMetricLabel"] {{ color: {COLOR_SEAL_BROWN}; font-weight: 600; font-size: 0.82rem; }}
div[data-testid="stMetricValue"] {{ color: {COLOR_SIENNA}; font-weight: 700; }}

.section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {COLOR_SEAL_BROWN};
    margin: 4px 0 2px 0;
}}
.section-caption {{
    color: {tint(COLOR_DRAB_DARK, 0.35)};
    font-size: 0.85rem;
    margin-bottom: 10px;
}}

.insight-box {{
    border-left: 5px solid var(--accent);
    background-color: var(--bg);
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0 20px 0;
    font-size: 0.92rem;
    line-height: 1.5;
    color: {COLOR_DRAB_DARK};
}}
.insight-box b {{ color: {COLOR_SEAL_BROWN}; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
    border-color: {tint(COLOR_COCOA, 0.65)} !important;
    background-color: #FFFDF9;
}}

button[data-baseweb="tab"] {{
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

st.markdown("""
<div class="dash-header">
    <h1>SSDC 2026 — Student Placement Dashboard</h1>
    <p>Ringkasan performa penempatan mahasiswa: dari permintaan perusahaan sampai placement.</p>
</div>
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


def style_fig(fig, height=None):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=COLOR_DRAB_DARK, size=12),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=42, b=8),
        title_font=dict(size=14, color=COLOR_SEAL_BROWN),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    if height:
        fig.update_layout(height=height)
    return fig


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
    """Cari nama kolom yang benar setelah merge dengan suffixes."""
    for suf in suffixes:
        cand = f"{base}{suf}"
        if cand in df.columns:
            return cand
    return None


def norm_text(series: pd.Series) -> pd.Series:
    """Normalisasi teks untuk matching: strip spasi + lowercase."""
    return series.astype(str).str.strip().str.lower()


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def _clean_key(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


@st.cache_data
def load_data():
    company = pd.read_csv("cleaned_company.csv")
    talent_request = pd.read_csv("cleaned_talent_request.csv")
    tracking_company = pd.read_csv("cleaned_tracking_company.csv")
    tracking_student = pd.read_csv("cleaned_tracking_student.csv")
    student_all = pd.read_csv("cleaned_student_all.csv")
    status_student = pd.read_csv("cleaned_status_student.csv")

    company = _normalize_columns(company)
    talent_request = _normalize_columns(talent_request)
    tracking_company = _normalize_columns(tracking_company)
    tracking_student = _normalize_columns(tracking_student)
    student_all = _normalize_columns(student_all)
    status_student = _normalize_columns(status_student)

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
        _clean_key(df, col)

    for col, df in [("jumlah_dikirimkan", tracking_company), ("jumlah_permintaan", tracking_company),
                    ("headcount", talent_request), ("minimum_semester", talent_request),
                    ("ipk", status_student), ("semester", student_all)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return company, talent_request, tracking_company, tracking_student, student_all, status_student


company, talent_request, tracking_company, tracking_student, student_all, status_student = load_data()

# ---------------------------------------------------------------------------
# DATA MERGING
# Prinsip ERD: nama perusahaan SELALU diambil dari master COMPANY via
# id_company, bukan dari kolom teks denormalisasi (nama_perusahaan/company)
# yang rawan beda penulisan antar tabel.
# ---------------------------------------------------------------------------
master = tracking_student.merge(tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc"))
master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))

master["tahun_update"] = master["last_update"].dt.year
# lama proses per kandidat: dari batch dikirim sampai update terakhir
if "send_date" in master.columns:
    master["lama_proses_hari"] = (master["last_update"] - master["send_date"]).dt.days

COMPANY_NAME_COL = "company_name" if "company_name" in master.columns else "company"

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("### :material/tune: Filter Laporan")

tahun_list = sorted(master["tahun_update"].dropna().unique().tolist())
tahun_pilihan = st.sidebar.multiselect("Tahun", tahun_list, default=tahun_list)

prodi_list = sorted(student_all["program_studi"].dropna().unique().tolist()) if "program_studi" in student_all.columns else []
prodi_pilihan = st.sidebar.multiselect("Program Studi", prodi_list, default=[])

jenis_list = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist()) if "jenis_penempatan" in talent_request.columns else []
jenis_pilihan = st.sidebar.multiselect("Jenis Penempatan", jenis_list, default=[])

st.sidebar.caption(
    "Filter berlaku untuk data proses seleksi (tracking). "
    "Tab Kesiapan & Matching memakai data master terkini sehingga tidak terpengaruh filter."
)

m = master[master["tahun_update"].isin(tahun_pilihan)].copy() if tahun_pilihan else master.copy()
if prodi_pilihan and "program_studi" in m.columns:
    m = m[m["program_studi"].isin(prodi_pilihan)]
if jenis_pilihan and "jenis_penempatan" in m.columns:
    m = m[m["jenis_penempatan"].isin(jenis_pilihan)]

tc_scope = tracking_company.merge(
    talent_request[["id_talent_req", "jenis_penempatan", "headcount"]]
    if "jenis_penempatan" in talent_request.columns else talent_request[["id_talent_req"]],
    on="id_talent_req", how="left",
)
tc_scope["tahun_tc"] = tc_scope["send_date"].dt.year if "send_date" in tc_scope.columns else tc_scope.get("request_date").dt.year
if tahun_pilihan:
    tc_scope = tc_scope[tc_scope["tahun_tc"].isin(tahun_pilihan)]
if jenis_pilihan and "jenis_penempatan" in tc_scope.columns:
    tc_scope = tc_scope[tc_scope["jenis_penempatan"].isin(jenis_pilihan)]

st.sidebar.markdown("---")
if "sync_date" in status_student.columns and status_student["sync_date"].notna().any():
    last_sync = status_student["sync_date"].max()
    st.sidebar.caption(f"Data sync terakhir: {last_sync.strftime('%d %B %Y')}")

default_ref_date = tracking_company["send_date"].max()
if pd.isna(default_ref_date):
    default_ref_date = datetime.now()
tanggal_acuan = st.sidebar.date_input(
    "Tanggal Acuan Monitoring Ghosting",
    value=default_ref_date.date() if hasattr(default_ref_date, "date") else default_ref_date,
)
tanggal_acuan = pd.Timestamp(tanggal_acuan)

# ---------------------------------------------------------------------------
# GHOSTING LOGIC — RULE-BASED sesuai FAQ BT-05:
# Ghosting diasumsikan dari PIHAK PERUSAHAAN, dihitung dari send_date,
# per BATCH pengiriman (satu baris tracking_company).
# Batch dianggap "direspon" jika minimal satu mahasiswa di batch itu sudah
# bergerak melewati tahap awal, atau sudah ada keputusan Placement/Rejected.
# ---------------------------------------------------------------------------
respon_per_tc = tracking_student.groupby("id_tracking_company").agg(
    ada_progress_lanjut=("progress_student", lambda s: (s != "Selecting Student by Company").any()),
    ada_keputusan=("rejection", lambda s: s.isin(["Placement", "Rejected"]).any()),
).reset_index()
respon_per_tc["sudah_direspon"] = respon_per_tc["ada_progress_lanjut"] | respon_per_tc["ada_keputusan"]

tc_status = tracking_company.merge(respon_per_tc[["id_tracking_company", "sudah_direspon"]], on="id_tracking_company", how="left")
tc_status["sudah_direspon"] = tc_status["sudah_direspon"].fillna(False)
tc_status["hari_sejak_kirim"] = (tanggal_acuan - tc_status["send_date"]).dt.days
tc_status["status_followup"] = tc_status.apply(
    lambda r: hitung_status_followup(r["hari_sejak_kirim"], r["sudah_direspon"]), axis=1
)

tc_status = tc_status.merge(
    talent_request[["id_talent_req", "jenis_penempatan"]] if "jenis_penempatan" in talent_request.columns else talent_request[["id_talent_req"]],
    on="id_talent_req", how="left", suffixes=("", "_tr2"),
)
# nama perusahaan resmi dari master COMPANY
tc_status = tc_status.merge(company[["id_company", "company_name"]], on="id_company", how="left")
tc_status["tahun_tc"] = tc_status["send_date"].dt.year
if tahun_pilihan:
    tc_status = tc_status[tc_status["tahun_tc"].isin(tahun_pilihan)]
jenis_col_status = "jenis_penempatan_tr2" if "jenis_penempatan_tr2" in tc_status.columns else "jenis_penempatan"
if jenis_pilihan and jenis_col_status in tc_status.columns:
    tc_status = tc_status[tc_status[jenis_col_status].isin(jenis_pilihan)]

# ---------------------------------------------------------------------------
# ML MODEL — MATCHING TALENT (RandomForest, prediksi peluang placement)
# ---------------------------------------------------------------------------
ML_FEATURE_BASE = ["program_studi", "semester", "ipk", "cv", "portofolio", "domisili"]
MIN_TRAIN_ROWS = 50


@st.cache_resource
def train_matching_model(student_all_df: pd.DataFrame, status_student_df: pd.DataFrame, tracking_student_df: pd.DataFrame):
    base = student_all_df.merge(status_student_df, on="nim", how="inner", suffixes=("_student", "_status"))
    base = base.merge(
        tracking_student_df[["nim", "rejection"]].dropna(subset=["rejection"]),
        on="nim", how="inner",
    )
    base = base[base["rejection"].isin(["Placement", "Rejected"])].copy()
    base["target"] = (base["rejection"] == "Placement").astype(int)

    resolved_cols = {}
    for feat in ML_FEATURE_BASE:
        col = resolve_col(base, feat)
        if col is not None:
            resolved_cols[feat] = col

    feature_cols = list(resolved_cols.values())
    if len(base) < MIN_TRAIN_ROWS or base["target"].nunique() < 2 or not feature_cols:
        return None  # data belum cukup -> caller fallback ke AHP

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test) if len(X_test) > 0 else np.nan

    feature_importance = pd.DataFrame({
        "fitur": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "resolved_cols": resolved_cols,
        "numeric_feats": numeric_feats,
        "feature_importance": feature_importance,
        "test_accuracy": test_accuracy,
        "n_train": len(base),
    }


ml_bundle = train_matching_model(student_all, status_student, tracking_student)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    ":material/monitoring: Overview",
    ":material/filter_alt: Funnel & Ghosting",
    ":material/handshake: Mitra",
    ":material/school: Kesiapan",
    ":material/person_search: Matching Talent",
    ":material/description: Laporan",
])

# ---------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    # Sumber tunggal untuk success rate: tracking_student (1 baris = 1
    # kandidat dikirim ke 1 perusahaan). Pembilang & penyebut satu tabel,
    # tidak mencampur dengan field manual jumlah_dikirimkan.
    total_dikirim_individu = m["id_tracking_student"].nunique()
    total_placement = int((m["rejection"] == "Placement").sum())
    success_rate = (total_placement / total_dikirim_individu * 100) if total_dikirim_individu > 0 else 0

    total_diminta = tc_scope["headcount"].sum() if "headcount" in tc_scope.columns else tc_scope.get("jumlah_permintaan", pd.Series(dtype=float)).sum()
    total_dikirim_batch = tc_scope["jumlah_dikirimkan"].sum() if "jumlah_dikirimkan" in tc_scope.columns else 0
    fulfillment_rate = (total_dikirim_batch / total_diminta * 100) if total_diminta and total_diminta > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kandidat Dikirim (proses seleksi)", f"{total_dikirim_individu:,}")
    col2.metric("Placement Berhasil", f"{total_placement:,}")
    col3.metric("Success Rate", f"{success_rate:.1f}%",
                help="Placement / kandidat dikirim, dihitung dari tracking_student (sumber yang sama).")
    col4.metric("Fulfillment Rate", f"{fulfillment_rate:.1f}%",
                help="Jumlah dikirim vs headcount yang diminta perusahaan (BT-03).")

    col5, col6, col7 = st.columns(3)
    col5.metric("Total Perusahaan Mitra", f"{company['id_company'].nunique():,}", help="Data master, di luar filter.")
    col6.metric("Total Mahasiswa Terdaftar", f"{student_all['nim'].nunique():,}", help="Data master, di luar filter.")
    selesai = m[m["rejection"].isin(["Placement"] + REJECTION_STAGES)]
    if "lama_proses_hari" in selesai.columns and selesai["lama_proses_hari"].notna().any():
        col7.metric("Rata-rata Lama Proses", f"{selesai['lama_proses_hari'].mean():.0f} hari",
                    help="Dari batch dikirim (send_date) sampai keputusan terakhir.")
    else:
        col7.metric("Rata-rata Lama Proses", "—")

    with st.container(border=True):
        placement_df = m[m["rejection"] == "Placement"].copy()
        placement_df["bulan"] = placement_df["last_update"].dt.to_period("M").astype(str)
        tren = placement_df.groupby("bulan").size().reset_index(name="jumlah_placement").sort_values("bulan")

        fig_tren = px.line(tren, x="bulan", y="jumlah_placement", markers=True, title="Tren Placement Berhasil per Bulan")
        fig_tren.update_traces(line_color=COLOR_COCOA, marker_color=COLOR_SEAL_BROWN, line_width=3)
        st.plotly_chart(style_fig(fig_tren), use_container_width=True)

        if len(tren) >= 2:
            delta = tren["jumlah_placement"].iloc[-1] - tren["jumlah_placement"].iloc[-2]
            arah = "naik" if delta > 0 else ("turun" if delta < 0 else "stagnan")
            kind = "success" if delta >= 0 else "warning"
            insight(f"Placement bulan <b>{tren['bulan'].iloc[-1]}</b> {arah} {abs(int(delta))} dibanding bulan sebelumnya.", kind=kind)
        else:
            insight("Data belum cukup untuk membandingkan tren antar bulan pada rentang filter ini.")

# ---------------------------------------------------------------------------
# TAB 2 — FUNNEL & GHOSTING
# ---------------------------------------------------------------------------
with tab2:
    with st.container(border=True):
        section("Funnel Seleksi Kandidat", "Persentase menunjukkan konversi dari tahap sebelumnya — tahap dengan drop terbesar adalah titik bocor utama.")
        stage_rank = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}
        m_funnel = m[m["progress_student"].isin(FUNNEL_STAGES)].copy()
        m_funnel["stage_rank"] = m_funnel["progress_student"].map(stage_rank)
        funnel_counts = [(m_funnel["stage_rank"] >= i).sum() for i in range(len(FUNNEL_STAGES))]

        fig_funnel = go.Figure(go.Funnel(
            y=FUNNEL_STAGES, x=funnel_counts,
            marker={"color": PALETTE_SEQUENTIAL},
            textinfo="value+percent previous",
        ))
        st.plotly_chart(style_fig(fig_funnel), use_container_width=True)

        konversi = [
            (FUNNEL_STAGES[i], funnel_counts[i] / funnel_counts[i - 1] * 100)
            for i in range(1, len(FUNNEL_STAGES)) if funnel_counts[i - 1] > 0
        ]
        if konversi:
            tahap_bocor, rate_bocor = min(konversi, key=lambda t: t[1])
            insight(
                f"Konversi terendah ada di tahap <b>{tahap_bocor}</b> ({rate_bocor:.0f}% dari tahap sebelumnya) — "
                "prioritaskan pendampingan CDC di tahap ini.",
                kind="warning",
            )

    with st.container(border=True):
        section("Rejection Breakdown per Tahap", "Di tahap mana mahasiswa paling banyak gagal (BT-04).")
        rej = m[m["rejection"].isin(REJECTION_STAGES)]["rejection"].value_counts().reindex(REJECTION_STAGES).fillna(0).reset_index()
        rej.columns = ["tahap_rejection", "jumlah"]
        fig_rej = px.bar(rej, x="jumlah", y="tahap_rejection", orientation="h",
                         title="Jumlah Rejection per Tahap", color_discrete_sequence=[COLOR_SIENNA])
        st.plotly_chart(style_fig(fig_rej), use_container_width=True)

    with st.container(border=True):
        section(
            "Deteksi Ghosting Perusahaan",
            "Sesuai FAQ BT-05: Ghosting diasumsikan dari pihak perusahaan, dihitung dari send_date per batch pengiriman — "
            ">7 hari FU1, >14 FU2, >21 FU3, >28 Ghosting (dari tanggal acuan di sidebar).",
        )
        total_tc_diproses = len(tc_status)
        total_ghosting_tc = int((tc_status["status_followup"] == "Ghosting").sum())
        ghosting_rate = (total_ghosting_tc / total_tc_diproses * 100) if total_tc_diproses > 0 else 0
        st.metric("Ghosting Rate (per batch pengiriman)", f"{ghosting_rate:.1f}%")

        ghosted = tc_status[tc_status["status_followup"] == "Ghosting"]
        col1, col2 = st.columns(2)
        with col1:
            ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
            ghosting_by_stage = ghosted_children["progress_student"].value_counts().reset_index()
            ghosting_by_stage.columns = ["tahap", "jumlah"]
            fig_gs = px.bar(ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
                            title="Ghosting berdasarkan Tahap Terakhir", color_discrete_sequence=[COLOR_SIENNA])
            st.plotly_chart(style_fig(fig_gs), use_container_width=True)
        with col2:
            ghosting_by_company = ghosted["company_name"].value_counts().head(10).reset_index()
            ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
            fig_gc = px.bar(ghosting_by_company, x="perusahaan", y="jumlah_ghosting",
                            title="Top Perusahaan Kontributor Ghosting", color_discrete_sequence=[COLOR_SIENNA])
            st.plotly_chart(style_fig(fig_gc), use_container_width=True)

    with st.container(border=True):
        section("Status Follow-up Saat Ini")
        followup_counts = tc_status["status_followup"].value_counts().reindex(
            ["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting"]).fillna(0).astype(int)
        kolom_status = st.columns(5)
        for col_widget, label in zip(kolom_status, ["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting"]):
            col_widget.metric(label.replace(" (Normal)", ""), f"{followup_counts.get(label, 0):,}")

        perlu_followup = tc_status[tc_status["status_followup"].isin(["FU 1", "FU 2", "FU 3", "Ghosting"])].sort_values("hari_sejak_kirim", ascending=False)
        with st.expander(f"Lihat {len(perlu_followup)} batch yang butuh follow-up", icon=":material/list_alt:"):
            st.dataframe(
                perlu_followup[["id_tracking_company", "company_name", "posisi", "send_date", "hari_sejak_kirim", "status_followup"]],
                use_container_width=True, hide_index=True,
            )

# ---------------------------------------------------------------------------
# TAB 3 — MITRA
# ---------------------------------------------------------------------------
with tab3:
    with st.container(border=True):
        section("Performa Perusahaan Mitra", "Kedua chart memakai nama resmi dari master COMPANY agar identitas perusahaan konsisten.")
        col1, col2 = st.columns(2)
        with col1:
            perf = m.groupby(COMPANY_NAME_COL).agg(
                total=("id_tracking_student", "count"),
                placement=("rejection", lambda x: (x == "Placement").sum()),
            ).reset_index()
            perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
            perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10).rename(columns={COMPANY_NAME_COL: "perusahaan"})
            fig_acc = px.bar(perf, x="acceptance_rate", y="perusahaan", orientation="h",
                             title="Top 10 Acceptance Rate (min. 3 kandidat)", color_discrete_sequence=[COLOR_COCOA])
            st.plotly_chart(style_fig(fig_acc), use_container_width=True)
        with col2:
            tr_named = talent_request.merge(company[["id_company", "company_name"]], on="id_company", how="left")
            volume = tr_named["company_name"].value_counts().head(10).reset_index()
            volume.columns = ["perusahaan", "jumlah_request"]
            fig_vol = px.treemap(volume, path=["perusahaan"], values="jumlah_request",
                                 title="Top 10 Volume Talent Request", color_discrete_sequence=PALETTE_SEQUENTIAL)
            st.plotly_chart(style_fig(fig_vol), use_container_width=True)

    with st.container(border=True):
        section("Prioritas Talent Request", "Diurutkan dari request paling lama (BT-03).")
        tr_fulfill = talent_request.merge(
            tracking_company.groupby("id_talent_req")["jumlah_dikirimkan"].sum().reset_index(),
            on="id_talent_req", how="left",
        )
        tr_fulfill = tr_fulfill.merge(company[["id_company", "company_name"]], on="id_company", how="left")
        tr_fulfill["jumlah_dikirimkan"] = tr_fulfill["jumlah_dikirimkan"].fillna(0)
        tr_fulfill["belum_terpenuhi"] = tr_fulfill["headcount"] - tr_fulfill["jumlah_dikirimkan"]
        prioritas = tr_fulfill[tr_fulfill["belum_terpenuhi"] > 0].sort_values("request_date")
        st.dataframe(
            prioritas[["id_talent_req", "company_name", "nama_posisi", "headcount", "jumlah_dikirimkan", "belum_terpenuhi", "request_date"]],
            use_container_width=True, hide_index=True,
        )

# ---------------------------------------------------------------------------
# TAB 4 — KESIAPAN  (data master terkini — tidak terpengaruh filter sidebar)
# ---------------------------------------------------------------------------
with tab4:
    st.caption("Tab ini memakai data master terkini (snapshot status mahasiswa), di luar filter sidebar.")

    ss = status_student.copy()
    ss["_status_norm"] = norm_text(ss["status"]) if "status" in ss.columns else ""
    ss["_ketersediaan_norm"] = norm_text(ss["ketersediaan"]) if "ketersediaan" in ss.columns else ""
    ss["_cv_norm"] = norm_text(ss["cv"]) if "cv" in ss.columns else ""
    ss["_porto_norm"] = norm_text(ss["portofolio"]) if "portofolio" in ss.columns else ""

    with st.container(border=True):
        section("Matching Gap: Demand vs Supply Bidang Studi")
        demand = talent_request["bidang_studi_dibutuhkan"].dropna().str.split(",").explode().str.strip().value_counts().reset_index()
        demand.columns = ["bidang_studi", "jumlah"]; demand["tipe"] = "Demand"
        supply = student_all["program_studi"].dropna().value_counts().reset_index()
        supply.columns = ["bidang_studi", "jumlah"]; supply["tipe"] = "Supply"

        gap_melt = pd.concat([demand, supply], ignore_index=True)
        top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
        gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]

        # dua warna netral setara — Demand bukan hal negatif, jangan pakai warna warning
        fig_gap = px.bar(gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
                         title="Top 10 Bidang Studi Demand vs Supply",
                         color_discrete_map={"Demand": COLOR_COCOA, "Supply": COLOR_JASMINE})
        st.plotly_chart(style_fig(fig_gap), use_container_width=True)

    with st.container(border=True):
        section("Kesiapan Dokumen & Mahasiswa Layak Kirim",
                "Sesuai FAQ: kolom 'eligible' = kolom 'ketersediaan'.")
        col1, col2 = st.columns(2)
        with col1:
            elig = status_student.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
            fig_elig = px.bar(elig, x="ketersediaan", y="jumlah", color="status",
                              title="Eligibility Mahasiswa", color_discrete_sequence=PALETTE_SEQUENTIAL)
            st.plotly_chart(style_fig(fig_elig), use_container_width=True)
        with col2:
            fig_ipk = px.histogram(status_student, x="ipk", nbins=20,
                                   title="Distribusi IPK Mahasiswa", color_discrete_sequence=[COLOR_COCOA])
            st.plotly_chart(style_fig(fig_ipk), use_container_width=True)

        eligible = ss[
            ss["_status_norm"].isin(VAL_STATUS_AKTIF)
            & ss["_ketersediaan_norm"].isin(VAL_TERSEDIA)
            & ss["_cv_norm"].isin(VAL_ADA)
            & ss["_porto_norm"].isin(VAL_ADA)
        ].merge(student_all[["nim", "program_studi", "semester"]], on="nim", how="left")

        pernah_dikirim = set(tracking_student["nim"].unique())
        eligible_nganggur = eligible[~eligible["nim"].isin(pernah_dikirim)]

        col_a, col_b = st.columns(2)
        col_a.metric("Mahasiswa Layak Kirim Saat Ini", f"{len(eligible):,}")
        col_b.metric("Layak tapi Belum Pernah Dikirim", f"{len(eligible_nganggur):,}",
                     help="Supply yang belum tersalurkan — prioritas untuk dicarikan penempatan (BT-06).")
        if len(eligible_nganggur) > 0:
            with st.expander(f"Lihat {len(eligible_nganggur)} mahasiswa layak yang belum pernah dikirim", icon=":material/person_alert:"):
                nama_col_e = resolve_col(eligible_nganggur, "nama") or "nama"
                cols_show = [c for c in ["nim", nama_col_e, "program_studi", "semester", "ipk"] if c in eligible_nganggur.columns]
                st.dataframe(eligible_nganggur[cols_show], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5 — MATCHING TALENT (data master terkini — di luar filter sidebar)
# ---------------------------------------------------------------------------
@st.cache_data
def compute_match_summary(talent_request_df: pd.DataFrame, student_all_df: pd.DataFrame, status_student_df: pd.DataFrame) -> pd.DataFrame:
    """Hitung jumlah kandidat cocok per talent request secara bertahap
    (prodi -> semester -> status & ketersediaan) supaya kelihatan di tahap
    mana kandidat hilang."""
    pool = student_all_df.merge(status_student_df, on="nim", how="inner", suffixes=("_student", "_status"))
    prodi_col = resolve_col(pool, "program_studi") or "program_studi"
    semester_col = resolve_col(pool, "semester") or "semester"

    pool["_prodi_norm"] = norm_text(pool[prodi_col])
    pool["_ketersediaan_norm"] = norm_text(pool["ketersediaan"]) if "ketersediaan" in pool.columns else ""
    pool["_status_norm"] = norm_text(pool["status"]) if "status" in pool.columns else ""

    rows = []
    for _, tr in talent_request_df.iterrows():
        bidang = [x.strip() for x in str(tr.get("bidang_studi_dibutuhkan", "")).split(",") if x.strip() != ""]
        bidang_norm = [b.lower() for b in bidang]
        min_sem = tr.get("minimum_semester", 0)
        if pd.isna(min_sem):
            min_sem = 0

        cocok_prodi = pool[pool["_prodi_norm"].isin(bidang_norm)] if bidang_norm else pool.iloc[0:0]
        cocok_semester = cocok_prodi[cocok_prodi[semester_col] >= min_sem]
        cocok_final = cocok_semester[
            cocok_semester["_ketersediaan_norm"].isin(VAL_TERSEDIA)
            & cocok_semester["_status_norm"].isin(VAL_STATUS_AKTIF)
        ]

        rows.append({
            "id_talent_req": tr.get("id_talent_req"),
            "nama_posisi": tr.get("nama_posisi", "-"),
            "nama_perusahaan": tr.get("nama_perusahaan", "-"),
            "bidang_dibutuhkan": ", ".join(bidang) if bidang else "(kosong)",
            "min_semester": min_sem,
            "cocok_prodi": len(cocok_prodi),
            "cocok_prodi_semester": len(cocok_semester),
            "kandidat_final": len(cocok_final),
        })

    return pd.DataFrame(rows)


with tab5:
    st.caption("Tab ini memakai data master terkini, di luar filter sidebar.")
    match_summary = compute_match_summary(talent_request, student_all, status_student)

    with st.container(border=True):
        section(
            "Ringkasan Kecocokan — Semua Talent Request",
            "Diurutkan dari yang paling kritis kandidat.",
        )
        n_zero = int((match_summary["kandidat_final"] == 0).sum())
        col_a, col_b = st.columns(2)
        col_a.metric("Total Talent Request", f"{len(match_summary):,}")
        col_b.metric("Request Tanpa Kandidat Cocok", f"{n_zero:,}")

        if n_zero > 0:
            insight(
                f"Ada <b>{n_zero} talent request</b> dengan 0 kandidat cocok. "
                "Cek kolom bertahap di tabel: kalau 'Cocok Prodi' sudah 0, tidak ada mahasiswa "
                "dari bidang studi yang diminta. Kalau baru drop di 'Kandidat Final', "
                "penyebabnya mahasiswa yang cocok belum berstatus aktif/tersedia.",
                kind="warning" if n_zero < len(match_summary) else "error",
            )
        else:
            insight("Semua talent request punya minimal 1 kandidat cocok.", kind="success")

        display_summary = match_summary.sort_values("kandidat_final").reset_index(drop=True)

        def _highlight_zero(row):
            return ["color: %s; font-weight: 700;" % COLOR_SIENNA if row["kandidat_final"] == 0 else "" for _ in row]

        st.dataframe(
            display_summary.style.apply(_highlight_zero, axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "id_talent_req": "ID",
                "nama_posisi": "Posisi",
                "nama_perusahaan": "Perusahaan",
                "bidang_dibutuhkan": "Bidang Dibutuhkan",
                "min_semester": "Min. Semester",
                "cocok_prodi": "Cocok Prodi",
                "cocok_prodi_semester": "Cocok Prodi + Semester",
                "kandidat_final": "Kandidat Final",
            },
        )

        if int((match_summary["cocok_prodi"] == 0).sum()) == len(match_summary):
            with st.expander("Debug: kenapa 'Cocok Prodi' 0 di semua baris?", icon=":material/troubleshoot:"):
                st.caption(
                    "Kalau nilai program_studi mahasiswa tidak pernah muncul persis di "
                    "bidang_studi_dibutuhkan (walau artinya sama), berarti penulisannya beda "
                    "(typo, singkatan, urutan kata). Samakan dulu nilainya di sumber data, baru refresh."
                )
                col_x, col_y = st.columns(2)
                with col_x:
                    st.markdown("**Nilai unik `program_studi` (mahasiswa):**")
                    st.dataframe(
                        pd.Series(sorted(student_all["program_studi"].dropna().unique()), name="program_studi"),
                        use_container_width=True, hide_index=True,
                    )
                with col_y:
                    st.markdown("**Nilai unik `bidang_studi_dibutuhkan` (talent request):**")
                    bidang_unik = (
                        talent_request["bidang_studi_dibutuhkan"]
                        .dropna().str.split(",").explode().str.strip()
                        .drop_duplicates().sort_values()
                    )
                    st.dataframe(bidang_unik.rename("bidang_studi_dibutuhkan"), use_container_width=True, hide_index=True)

        if int((match_summary["cocok_prodi"] > 0).sum()) > 0 and int((match_summary["kandidat_final"] == 0).sum()) == len(match_summary):
            with st.expander("Debug: kenapa 'Kandidat Final' 0 padahal 'Cocok Prodi' > 0?", icon=":material/troubleshoot:"):
                st.caption(
                    "Berarti masalahnya di kolom `ketersediaan` / `status` — nilainya tidak termasuk "
                    "Available/Tersedia dan Active/Aktif. Cek daftar nilai unik di bawah."
                )
                col_x, col_y = st.columns(2)
                with col_x:
                    st.markdown("**Nilai unik `ketersediaan`:**")
                    st.dataframe(status_student["ketersediaan"].value_counts().rename("jumlah"), use_container_width=True)
                with col_y:
                    st.markdown("**Nilai unik `status`:**")
                    st.dataframe(status_student["status"].value_counts().rename("jumlah"), use_container_width=True)

    with st.container(border=True):
        section(
            "Cari Kandidat Terbaik untuk Talent Request",
            "Skor kandidat = probabilitas placement dari model ML (RandomForest), dilatih dari histori "
            "keputusan Placement/Rejected. Kalau histori belum cukup, otomatis fallback ke skor AHP.",
        )

        if ml_bundle is not None:
            insight(
                f"Model aktif: RandomForest, dilatih dari {ml_bundle['n_train']:,} baris histori keputusan, "
                f"akurasi test sekitar {ml_bundle['test_accuracy']*100:.1f}%.",
                kind="success",
            )
        else:
            insight(
                "Histori keputusan (Placement/Rejected) belum cukup atau hanya 1 kelas — "
                "dashboard fallback ke skor AHP (bukan ML) untuk tab ini.",
                kind="warning",
            )

        tr_sorted = match_summary.sort_values("kandidat_final", ascending=False).copy()
        tr_sorted["label"] = (
            tr_sorted["nama_posisi"].fillna("-")
            + " — "
            + tr_sorted["nama_perusahaan"].fillna("-")
            + " ("
            + tr_sorted["id_talent_req"].astype(str)
            + ") — "
            + tr_sorted["kandidat_final"].astype(str)
            + " kandidat"
        )

        pilihan_label = st.selectbox("Pilih Talent Request", tr_sorted["label"].tolist())

        selected_id = tr_sorted.loc[tr_sorted["label"] == pilihan_label, "id_talent_req"].iloc[0]
        selected = talent_request[talent_request["id_talent_req"] == selected_id].iloc[0]

        bidang_dibutuhkan = [
            x.strip() for x in str(selected.get("bidang_studi_dibutuhkan", "")).split(",") if x.strip() != ""
        ]
        min_semester = selected.get("minimum_semester", 0)
        if pd.isna(min_semester):
            min_semester = 0

        candidates = student_all.merge(status_student, on="nim", how="inner", suffixes=("_student", "_status"))
        prodi_col = resolve_col(candidates, "program_studi") or "program_studi"
        semester_col = resolve_col(candidates, "semester") or "semester"
        nama_col = resolve_col(candidates, "nama") or "nama"

        bidang_dibutuhkan_norm = [b.lower() for b in bidang_dibutuhkan]
        candidates["_prodi_norm"] = norm_text(candidates[prodi_col])
        candidates["_ketersediaan_norm"] = norm_text(candidates["ketersediaan"])
        candidates["_status_norm"] = norm_text(candidates["status"])

        cocok_prodi_df = candidates[candidates["_prodi_norm"].isin(bidang_dibutuhkan_norm)]
        cocok_semester_df = cocok_prodi_df[cocok_prodi_df[semester_col] >= min_semester]
        candidates = cocok_semester_df[
            cocok_semester_df["_ketersediaan_norm"].isin(VAL_TERSEDIA)
            & cocok_semester_df["_status_norm"].isin(VAL_STATUS_AKTIF)
        ].copy()

        if len(candidates) > 0:
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
                            vals = vals.apply(lambda v: v if v in known else le.classes_[0])
                            X_cand[train_col] = le.transform(vals)
                        else:
                            X_cand[train_col] = 0

                X_cand = X_cand[feature_cols]
                candidates["recommendation_score"] = model.predict_proba(X_cand)[:, 1]
                candidates["metode_skor"] = "ML (placement probability)"
            else:
                candidates["prodi_score"] = 1.0
                candidates["ipk_score"] = (
                    candidates["ipk"] / candidates["ipk"].max() if candidates["ipk"].max() > 0 else 0
                )
                candidates["semester_score"] = candidates[semester_col] / candidates[semester_col].max()
                candidates["cv_score"] = candidates["cv"].astype(str).str.lower().eq("ada").astype(int)
                candidates["portfolio_score"] = candidates["portofolio"].astype(str).str.lower().eq("ada").astype(int)

                w_prodi, w_ipk, w_portfolio, w_semester, w_cv = 0.40, 0.30, 0.15, 0.10, 0.05
                candidates["recommendation_score"] = (
                    w_prodi * candidates["prodi_score"]
                    + w_ipk * candidates["ipk_score"]
                    + w_portfolio * candidates["portfolio_score"]
                    + w_semester * candidates["semester_score"]
                    + w_cv * candidates["cv_score"]
                )
                candidates["metode_skor"] = "AHP (fallback)"

            candidates = candidates.sort_values("recommendation_score", ascending=False)

        st.markdown(f"**{len(candidates)} kandidat cocok ditemukan:**")

        if len(candidates) == 0:
            if len(cocok_prodi_df) == 0:
                sebab = f"tidak ada mahasiswa terdaftar dari bidang studi <b>{', '.join(bidang_dibutuhkan) or '(kosong)'}</b>."
            elif len(cocok_semester_df) == 0:
                sebab = f"ada {len(cocok_prodi_df)} mahasiswa dari bidang yang cocok, tapi semuanya di bawah minimum semester ({int(min_semester)})."
            else:
                sebab = (
                    f"ada {len(cocok_semester_df)} mahasiswa yang cocok bidang & semester, "
                    "tapi belum ada yang berstatus aktif dan tersedia di data status kesiapan."
                )
            insight(f"Tidak ada kandidat yang memenuhi syarat — {sebab}", kind="error")
        else:
            show_cols = [
                c for c in [
                    "nim", nama_col, prodi_col, semester_col, "ipk", "cv",
                    "portofolio", "domisili", "recommendation_score", "metode_skor",
                ]
                if c in candidates.columns
            ]
            st.dataframe(candidates[show_cols].head(50), use_container_width=True, hide_index=True)

    if ml_bundle is not None:
        with st.container(border=True):
            section("Feature Importance Model", "Fitur yang paling berpengaruh terhadap peluang placement menurut model.")
            fig_fi = px.bar(
                ml_bundle["feature_importance"],
                x="importance", y="fitur", orientation="h",
                title="Feature Importance — RandomForest",
                color_discrete_sequence=[COLOR_SEAL_BROWN],
            )
            st.plotly_chart(style_fig(fig_fi), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 6 — LAPORAN & QUALITY CHECK
# ---------------------------------------------------------------------------
with tab6:
    with st.container(border=True):
        section("Rekapitulasi Placement", "Laporan periodik untuk evaluasi institusi (BT-07).")
        dim_options = {
            "Program Studi": "program_studi",
            "Perusahaan": COMPANY_NAME_COL,
            "Jenis Penempatan": "jenis_penempatan",
        }
        dims_pilihan = st.multiselect("Kelompokkan berdasarkan", list(dim_options.keys()), default=["Program Studi"])

        placement_only = m[m["rejection"] == "Placement"].copy()
        placement_only["periode"] = placement_only["last_update"].dt.to_period("Q").astype(str)

        if dims_pilihan:
            group_cols = ["periode"] + [dim_options[d] for d in dims_pilihan]
            recap = placement_only.groupby(group_cols).size().reset_index(name="jumlah_placement").sort_values("periode", ascending=False)

            # visual dulu, tabel belakangan — tren per periode langsung terbaca
            dim_utama = dim_options[dims_pilihan[0]]
            fig_recap = px.bar(
                recap.sort_values("periode"), x="periode", y="jumlah_placement", color=dim_utama,
                title=f"Placement per Kuartal berdasarkan {dims_pilihan[0]}",
                color_discrete_sequence=PALETTE_SEQUENTIAL,
            )
            st.plotly_chart(style_fig(fig_recap), use_container_width=True)

            st.dataframe(recap, use_container_width=True, hide_index=True)
            st.download_button("Unduh Rekap CSV", recap.to_csv(index=False).encode("utf-8"),
                               "rekap_placement.csv", "text/csv", icon=":material/download:")

    with st.container(border=True):
        section("Kualitas Data & Sinkronisasi", "BT-08: konsistensi STUDENT ALL vs STATUS STUDENT.")
        merged_check = student_all.merge(status_student[["nim", "sync_date"]], on="nim", how="left", indicator=True)
        belum_sync = merged_check[merged_check["_merge"] == "left_only"]

        stale_days = (tanggal_acuan - status_student["sync_date"]).dt.days if "sync_date" in status_student.columns else pd.Series(dtype=float)
        n_stale = int((stale_days > SYNC_STALE_DAYS).sum()) if stale_days.notna().any() else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mahasiswa Belum Ada Data Status", f"{len(belum_sync):,}")
        col_b.metric(f"Data Status Usang (> {SYNC_STALE_DAYS} hari)", f"{n_stale:,}")
        col_c.metric("Rata-rata Umur Sync", f"{stale_days.mean():.0f} hari" if stale_days.notna().any() else "—")

        if len(belum_sync) > 0:
            st.dataframe(belum_sync[["nim", "nama", "program_studi"]].head(20), use_container_width=True, hide_index=True)
