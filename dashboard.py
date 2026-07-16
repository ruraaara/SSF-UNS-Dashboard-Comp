import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="SSDC 2026 — Student Placement Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
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

# Aturan resmi FAQ BT-05: Ghosting diasumsikan dari PIHAK PERUSAHAAN,
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
# progress_student punya nilai terminal (Placement/Finish/Rejected/Ghosting/FU x)
# di luar 6 tahap funnel, jadi dipetakan eksplisit ke peringkat tahap.
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
# STYLE — CSS Custom
# Tema dikunci light via .streamlit/config.toml supaya render sama di semua
# device; !important di bawah ini pengaman kalau ada yang override manual.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

/* layout lebih rapat */
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
}}
div[data-testid="stVerticalBlock"] {{ gap: 0.6rem; }}

.dash-header {{
    padding: 14px 22px;
    border-radius: 14px;
    background: linear-gradient(135deg, {COLOR_SEAL_BROWN} 0%, {COLOR_SIENNA} 100%);
    margin-bottom: 12px;
}}
.dash-header h1 {{
    color: #FFF8EE !important;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0;
}}
.dash-header p {{
    color: {tint(COLOR_JASMINE, 0.4)} !important;
    font-size: 0.85rem;
    margin: 2px 0 0 0;
}}

/* KPI cards gaya referensi: angka besar di atas, label di bawah */
.kpi-row {{
    display: flex;
    gap: 12px;
    margin: 4px 0 10px 0;
    flex-wrap: wrap;
}}
.kpi-card {{
    flex: 1;
    min-width: 150px;
    background: linear-gradient(160deg, {COLOR_SEAL_BROWN} 0%, #2E1608 100%);
    border-radius: 14px;
    padding: 16px 12px 13px 12px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(74, 35, 14, 0.25);
}}
.kpi-value {{
    color: {COLOR_JASMINE};
    font-size: 1.65rem;
    font-weight: 800;
    line-height: 1.1;
}}
.kpi-label {{
    color: {tint(COLOR_JASMINE, 0.55)};
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 5px;
}}

div[data-testid="stMetric"] {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid {tint(COLOR_COCOA, 0.65)};
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
    font-size: 1.55rem;
}}

.section-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {COLOR_SEAL_BROWN};
    margin: 2px 0 1px 0;
}}
.section-caption {{
    color: {tint(COLOR_DRAB_DARK, 0.35)};
    font-size: 0.82rem;
    margin-bottom: 6px;
}}

.insight-box {{
    border-left: 5px solid var(--accent);
    background-color: var(--bg);
    padding: 10px 14px;
    border-radius: 8px;
    margin: 6px 0 10px 0;
    font-size: 0.88rem;
    line-height: 1.45;
    color: {COLOR_DRAB_DARK};
}}
.insight-box b {{ color: {COLOR_SEAL_BROWN}; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 14px !important;
    border-color: {tint(COLOR_COCOA, 0.65)} !important;
    background-color: #FFFDF9;
}}

button[data-baseweb="tab"] {{ font-weight: 600; }}

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
    """Deret kartu KPI gaya referensi: angka besar di atas, label di bawah."""
    html = '<div class="kpi-row">'
    for c in cards:
        help_attr = f' title="{c["help"]}"' if c.get("help") else ""
        html += (
            f'<div class="kpi-card"{help_attr}>'
            f'<div class="kpi-value">{c["value"]}</div>'
            f'<div class="kpi-label">{c["label"]}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def style_fig(fig, height=300):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=COLOR_DRAB_DARK, size=11),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=38, b=8),
        title_font=dict(size=13, color=COLOR_SEAL_BROWN),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hoverlabel=dict(bgcolor="white", font_size=11),
        height=height,
    )
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


# ---------------------------------------------------------------------------
# LOAD + PREPARE DATA (SEKALI SAJA)
# Semua pembacaan CSV, pembersihan tipe, DAN merge master dilakukan di dalam
# satu fungsi ber-cache_resource, sehingga rerun halaman (ganti filter, ganti
# tab) tidak mengulang komputasi berat sama sekali. cache_resource dipilih
# (bukan cache_data) supaya hasilnya tidak di-copy ulang tiap rerun; sebagai
# gantinya, SEMUA konsumen wajib .copy() sebelum mengubah frame.
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
                    ("ipk", status_student), ("semester", student_all)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- tahap terjauh tiap kandidat (dasar funnel & sinyal respons) ----
    rank = tracking_student["progress_student"].map(PROGRESS_TO_RANK)
    mask_rejected = tracking_student["progress_student"].eq("Rejected")
    rank = rank.mask(mask_rejected, tracking_student["rejection"].map(REJECTION_TO_RANK))
    tracking_student["stage_reached"] = rank.fillna(0).astype(int)

    # ---- MASTER: tracking_student -> tracking_company -> company ->
    #      talent_request -> student_all -> status_student (prinsip ERD,
    #      nama perusahaan selalu dari master COMPANY via id_company) ----
    master = tracking_student.merge(tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc"))
    master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
    master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
    master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
    master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))

    master["tahun_update"] = master["last_update"].dt.year
    if "send_date" in master.columns:
        master["lama_proses_hari"] = (master["last_update"] - master["send_date"]).dt.days

    # ---- BATCH PENGIRIMAN (basis ghosting BT-05) ----
    # Hanya batch yang benar-benar sudah dikirim yang dimonitor ghosting-nya.
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
DEFAULT_REF_DATE = DATA["default_ref_date"]

COMPANY_NAME_COL = "company_name" if "company_name" in master.columns else "company"


# ---------------------------------------------------------------------------
# MATCH SUMMARY (VEKTORISASI)
# Versi lama me-loop 12.000 talent request dan memfilter 25.000 mahasiswa di
# tiap iterasi — inilah penyebab utama loading lama. Sekarang jumlah kandidat
# per (prodi, minimum semester) dihitung SEKALI sebagai matriks lookup, lalu
# tiap request tinggal menjumlahkan angka dari matriks itu.
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
# Perbaikan penting: label negatif diambil dari SEMUA nilai "Rejection ..."
# (di data tidak ada nilai "Rejected" pada kolom rejection — inilah yang
# membuat model lama selalu fallback ke AHP). "On Progress" dan "Ghosting"
# dikeluarkan karena belum/bukan keputusan perusahaan atas kualitas kandidat.
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

    feature_importance = pd.DataFrame({
        "fitur": [f for f in resolved_cols.keys()],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": model, "encoders": encoders, "feature_cols": feature_cols,
        "resolved_cols": resolved_cols, "numeric_feats": numeric_feats,
        "feature_importance": feature_importance,
        "test_accuracy": test_accuracy, "n_train": len(base),
    }


ml_bundle = train_matching_model()

# ---------------------------------------------------------------------------
# FILTER BAR PER TAB (bukan sidebar — tiap tab bisa dipilih independen)
# ---------------------------------------------------------------------------
FILTER_TAHUN = sorted(int(t) for t in master["tahun_update"].dropna().unique())
FILTER_PRODI = sorted(student_all["program_studi"].dropna().unique().tolist()) if "program_studi" in student_all.columns else []
FILTER_JENIS = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist()) if "jenis_penempatan" in talent_request.columns else []


def filter_bar(key: str, with_prodi: bool = True, with_ref_date: bool = False):
    n_cols = 2 + int(with_prodi) + int(with_ref_date)
    with st.container(border=True):
        cols = st.columns(n_cols)
        i = 0
        tahun = cols[i].multiselect("Tahun", FILTER_TAHUN, default=FILTER_TAHUN, key=f"f_tahun_{key}")
        i += 1
        prodi = []
        if with_prodi:
            prodi = cols[i].multiselect("Program Studi", FILTER_PRODI, default=[],
                                        placeholder="Semua program studi", key=f"f_prodi_{key}")
            i += 1
        jenis = cols[i].multiselect("Jenis Penempatan", FILTER_JENIS, default=[],
                                    placeholder="Semua jenis", key=f"f_jenis_{key}")
        i += 1
        ref_date = None
        if with_ref_date:
            ref_raw = cols[i].date_input(
                "Tanggal Acuan Ghosting",
                value=DEFAULT_REF_DATE.date() if hasattr(DEFAULT_REF_DATE, "date") else DEFAULT_REF_DATE,
                key=f"f_ref_{key}",
                help="Ghosting dihitung dari send_date sampai tanggal ini (aturan FAQ BT-05).",
            )
            ref_date = pd.Timestamp(ref_raw)
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
last_sync_txt = ""
if "sync_date" in status_student.columns and status_student["sync_date"].notna().any():
    last_sync_txt = f" &nbsp;|&nbsp; Data sync terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}"

st.markdown(f"""
<div class="dash-header">
    <h1>SSDC 2026 — Student Placement Dashboard</h1>
    <p>Ringkasan performa penempatan mahasiswa: dari permintaan perusahaan sampai placement.{last_sync_txt}</p>
</div>
""", unsafe_allow_html=True)

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
    tahun_f, prodi_f, jenis_f, _ = filter_bar("overview")
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

    kpi_row([
        {"value": f"{total_dikirim_individu:,}", "label": "Kandidat Dikirim",
         "help": "Jumlah proses seleksi kandidat (baris tracking_student) pada rentang filter."},
        {"value": f"{total_placement:,}", "label": "Placement Berhasil"},
        {"value": f"{success_rate:.1f}%", "label": "Success Rate",
         "help": "Placement dibagi kandidat dikirim — pembilang & penyebut sama-sama dari tracking_student."},
        {"value": f"{fulfillment_rate:.1f}%", "label": "Fulfillment Rate",
         "help": "Jumlah dikirim vs diminta (BT-03). Di atas 100% berarti CDC mengirim lebih banyak kandidat dari kuota — wajar untuk shortlist."},
        {"value": f"{lama_proses:.0f} hari" if lama_proses is not None else "-", "label": "Rata-rata Lama Proses",
         "help": "Dari batch dikirim (send_date) sampai keputusan terakhir."},
    ])

    col_kiri, col_kanan = st.columns([2, 1])
    with col_kiri:
        with st.container(border=True):
            section("Placement & Success Rate per Bulan",
                    "Batang = jumlah placement; garis = success rate dari keputusan yang keluar di bulan itu.")
            dec = m[m["rejection"].isin(["Placement"] + REJECTION_STAGES)].copy()
            dec["bulan"] = dec["last_update"].dt.to_period("M").astype(str)
            per_bulan = dec.groupby("bulan").agg(
                placement=("rejection", lambda s: (s == "Placement").sum()),
                keputusan=("rejection", "size"),
            ).reset_index().sort_values("bulan")
            per_bulan["success_pct"] = (per_bulan["placement"] / per_bulan["keputusan"] * 100).round(1)

            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_combo.add_trace(go.Bar(
                x=per_bulan["bulan"], y=per_bulan["placement"], name="Placement",
                marker_color=COLOR_COCOA,
            ), secondary_y=False)
            fig_combo.add_trace(go.Scatter(
                x=per_bulan["bulan"], y=per_bulan["success_pct"], name="Success Rate (%)",
                mode="lines+markers", line=dict(color=COLOR_SEAL_BROWN, width=2.5),
                marker=dict(color=COLOR_SIENNA, size=6),
            ), secondary_y=True)
            fig_combo.update_yaxes(title_text="Placement", secondary_y=False)
            fig_combo.update_yaxes(title_text="%", secondary_y=True, rangemode="tozero")
            show_chart(fig_combo, height=330)

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
            show_chart(fig_donut, height=330)

    with st.container(border=True):
        section("Perjalanan Kandidat: dari Dikirim sampai Placement",
                "Waterfall: dari semua kandidat yang dikirim, berapa yang hilang di tiap penyebab, dan berapa yang berakhir placement.")
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
        show_chart(fig_wf, height=360)

        if len(m) > 0:
            on_progress_pct = -wf_vals["Masih Berjalan"] / len(m) * 100
            insight(
                f"Dari <b>{total_dikirim_individu:,}</b> kandidat dikirim, <b>{total_placement:,}</b> berakhir placement "
                f"({success_rate:.1f}%), sementara {on_progress_pct:.0f}% masih berjalan prosesnya.",
            )

# ---------------------------------------------------------------------------
# TAB 2 — FUNNEL & GHOSTING
# ---------------------------------------------------------------------------
with tab2:
    tahun_f, prodi_f, jenis_f, tanggal_acuan = filter_bar("funnel", with_ref_date=True)
    m = scope_master(tahun_f, prodi_f, jenis_f)

    tc_status = scope_tc(tahun_f, jenis_f)
    tc_status["hari_sejak_kirim"] = (tanggal_acuan - tc_status["send_date"]).dt.days
    tc_status["status_followup"] = [
        hitung_status_followup(h, r)
        for h, r in zip(tc_status["hari_sejak_kirim"], tc_status["sudah_direspon"])
    ]

    col_kiri, col_kanan = st.columns(2)
    with col_kiri:
        with st.container(border=True):
            section("Funnel Seleksi Kandidat",
                    "Jumlah kandidat yang MENCAPAI tiap tahap (persentase = konversi dari tahap sebelumnya).")
            funnel_counts = [int((m["stage_reached"] >= i).sum()) for i in range(len(FUNNEL_STAGES))]
            fig_funnel = go.Figure(go.Funnel(
                y=FUNNEL_STAGES, x=funnel_counts,
                marker={"color": PALETTE_SEQUENTIAL},
                textinfo="value+percent previous",
            ))
            show_chart(fig_funnel, height=360)

    with col_kanan:
        with st.container(border=True):
            section("Rejection Breakdown per Tahap", "Di tahap mana kandidat paling banyak gagal (BT-04).")
            rej = m[m["rejection"].isin(REJECTION_STAGES)]["rejection"].value_counts().reindex(REJECTION_STAGES).fillna(0).reset_index()
            rej.columns = ["tahap_rejection", "jumlah"]
            fig_rej = px.bar(rej, x="jumlah", y="tahap_rejection", orientation="h",
                             color_discrete_sequence=[COLOR_SIENNA])
            fig_rej.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_rej, height=360)

    konversi = [
        (FUNNEL_STAGES[i], funnel_counts[i] / funnel_counts[i - 1] * 100)
        for i in range(1, len(FUNNEL_STAGES)) if funnel_counts[i - 1] > 0
    ]
    if konversi:
        tahap_bocor, rate_bocor = min(konversi, key=lambda t: t[1])
        insight(f"Konversi terendah ada di tahap <b>{tahap_bocor}</b> ({rate_bocor:.0f}% dari tahap sebelumnya) — "
                "prioritaskan pendampingan CDC di tahap ini.", kind="warning")

    with st.container(border=True):
        section("Deteksi Ghosting Perusahaan (BT-05)",
                "Sesuai FAQ: Ghosting dari pihak perusahaan, dihitung dari send_date per batch pengiriman — "
                ">7 hari FU1, >14 FU2, >21 FU3, >28 Ghosting. Filter Program Studi tidak berlaku di bagian ini (level batch).")

        followup_counts = tc_status["status_followup"].value_counts()
        total_tc = len(tc_status)
        n_ghosting = int(followup_counts.get("Ghosting", 0))
        ghosting_rate = (n_ghosting / total_tc * 100) if total_tc > 0 else 0

        kpi_row([
            {"value": f"{total_tc:,}", "label": "Batch Terkirim"},
            {"value": f"{int(followup_counts.get('FU 1', 0)):,}", "label": "Butuh FU 1"},
            {"value": f"{int(followup_counts.get('FU 2', 0)):,}", "label": "Butuh FU 2"},
            {"value": f"{int(followup_counts.get('FU 3', 0)):,}", "label": "Butuh FU 3"},
            {"value": f"{n_ghosting:,}", "label": f"Ghosting ({ghosting_rate:.1f}%)"},
        ])

        col1, col2 = st.columns(2)
        ghosted = tc_status[tc_status["status_followup"] == "Ghosting"]
        with col1:
            ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
            ghosting_by_stage = ghosted_children["progress_student"].value_counts().head(8).reset_index()
            ghosting_by_stage.columns = ["tahap", "jumlah"]
            fig_gs = px.bar(ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
                            title="Kandidat pada Batch Ghosting — Tahap Terakhir",
                            color_discrete_sequence=[COLOR_SIENNA])
            fig_gs.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_gs, height=300)
        with col2:
            ghosting_by_company = ghosted["company_name"].value_counts().head(10).reset_index()
            ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
            fig_gc = px.bar(ghosting_by_company, x="jumlah_ghosting", y="perusahaan", orientation="h",
                            title="Top Perusahaan Kontributor Ghosting",
                            color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_gc.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_gc, height=300)

        # tren komposisi status follow-up per bulan pengiriman
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
        show_chart(fig_fu, height=300)

        n_ghosting_tercatat = int((m["rejection"] == "Ghosting").sum())
        insight(
            f"Berdasarkan aturan FAQ (per batch, dari send_date), <b>{n_ghosting:,} batch</b> berstatus Ghosting "
            f"({ghosting_rate:.1f}% dari batch terkirim) pada tanggal acuan. Sebagai pembanding, data mencatat "
            f"{n_ghosting_tercatat:,} kandidat dengan label Ghosting — angka per kandidat itu tidak setara dengan "
            "aturan FAQ karena satu batch bisa berisi campuran kandidat ghosting dan kandidat yang tetap diproses. "
            "Untuk monitoring follow-up, gunakan angka aturan FAQ karena konsisten dan bisa dihitung ulang kapan pun.",
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
with tab3:
    tahun_f, prodi_f, jenis_f, _ = filter_bar("mitra")
    m = scope_master(tahun_f, prodi_f, jenis_f)

    kpi_row([
        {"value": f"{company['id_company'].nunique():,}", "label": "Perusahaan Mitra (master)"},
        {"value": f"{talent_request['id_talent_req'].nunique():,}", "label": "Total Talent Request (master)"},
        {"value": f"{m[COMPANY_NAME_COL].nunique():,}", "label": "Perusahaan Aktif (sesuai filter)"},
        {"value": f"{m['nama_posisi'].nunique():,}" if "nama_posisi" in m.columns else "-", "label": "Posisi Dibuka (sesuai filter)"},
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
            show_chart(fig_acc, height=320)
    with col2:
        with st.container(border=True):
            section("Top 10 Volume Talent Request")
            tr_named = talent_request.merge(company[["id_company", "company_name"]], on="id_company", how="left")
            volume = tr_named["company_name"].value_counts().head(10).reset_index()
            volume.columns = ["perusahaan", "jumlah_request"]
            fig_vol = px.treemap(volume, path=["perusahaan"], values="jumlah_request",
                                 color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_vol.update_layout(margin=dict(l=4, r=4, t=8, b=4))
            show_chart(fig_vol, height=320)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section("Permintaan per Sektor Industri", "Sektor mana yang paling banyak mencari talent.")
            sektor = tr_named["industri_sektor"].value_counts().head(10).reset_index() if "industri_sektor" in tr_named.columns else pd.DataFrame(columns=["a", "b"])
            sektor.columns = ["sektor", "jumlah"]
            fig_sektor = px.bar(sektor, x="jumlah", y="sektor", orientation="h",
                                color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_sektor.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_sektor, height=300)
    with col4:
        with st.container(border=True):
            section("Profil Perusahaan Mitra", "Komposisi tipe dan skala dari master COMPANY.")
            tipe = company["company_type"].value_counts().reset_index() if "company_type" in company.columns else pd.DataFrame(columns=["a", "b"])
            tipe.columns = ["tipe", "jumlah"]
            fig_tipe = px.pie(tipe, names="tipe", values="jumlah", hole=0.5,
                              color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_tipe.update_traces(textinfo="percent+label", textfont_size=10)
            show_chart(fig_tipe, height=300)

    with st.container(border=True):
        section("Prioritas Talent Request Belum Terpenuhi",
                "Diurutkan dari request paling lama (BT-03). Data master, di luar filter.")
        tr_fulfill = talent_request.merge(
            tracking_company.groupby("id_talent_req")["jumlah_dikirimkan"].sum().reset_index(),
            on="id_talent_req", how="left",
        )
        tr_fulfill = tr_fulfill.merge(company[["id_company", "company_name"]], on="id_company", how="left")
        tr_fulfill["jumlah_dikirimkan"] = tr_fulfill["jumlah_dikirimkan"].fillna(0)
        tr_fulfill["belum_terpenuhi"] = tr_fulfill["headcount"] - tr_fulfill["jumlah_dikirimkan"]
        tr_fulfill["pemenuhan"] = (tr_fulfill["jumlah_dikirimkan"] / tr_fulfill["headcount"]).clip(0, 1)
        prioritas = tr_fulfill[tr_fulfill["belum_terpenuhi"] > 0].sort_values("request_date")
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
with tab4:
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
        {"value": f"{len(eligible_nganggur):,}", "label": "Layak tapi Belum Pernah Dikirim",
         "help": "Supply belum tersalurkan — prioritas dicarikan penempatan (BT-06)."},
        {"value": f"{(cv_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya CV"},
        {"value": f"{(porto_norm.isin(VAL_ADA).mean() * 100):.0f}%", "label": "Punya Portofolio"},
    ])

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            section("Matching Gap: Demand vs Supply Bidang Studi")
            demand = talent_request["bidang_studi_dibutuhkan"].dropna().str.split(",").explode().str.strip().value_counts().reset_index()
            demand.columns = ["bidang_studi", "jumlah"]; demand["tipe"] = "Demand"
            supply = student_all["program_studi"].dropna().value_counts().reset_index()
            supply.columns = ["bidang_studi", "jumlah"]; supply["tipe"] = "Supply"
            gap_melt = pd.concat([demand, supply], ignore_index=True)
            top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
            gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]
            fig_gap = px.bar(gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
                             color_discrete_map={"Demand": COLOR_COCOA, "Supply": COLOR_JASMINE})
            fig_gap.update_layout(yaxis_title=None, xaxis_title=None, legend_title=None)
            show_chart(fig_gap, height=340)
    with col2:
        with st.container(border=True):
            section("Eligibility Mahasiswa", "Sesuai FAQ: kolom 'eligible' = kolom 'ketersediaan'.")
            elig_ct = ss.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
            fig_elig = px.bar(elig_ct, x="ketersediaan", y="jumlah", color="status",
                              color_discrete_sequence=PALETTE_SEQUENTIAL)
            fig_elig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
            show_chart(fig_elig, height=340)

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            section("Distribusi IPK")
            fig_ipk = px.histogram(ss, x="ipk", nbins=20, color_discrete_sequence=[COLOR_COCOA])
            fig_ipk.update_layout(xaxis_title="IPK", yaxis_title=None)
            show_chart(fig_ipk, height=280)
    with col4:
        with st.container(border=True):
            section("Distribusi Semester Mahasiswa")
            sem_ct = student_all["semester"].value_counts().sort_index().reset_index()
            sem_ct.columns = ["semester", "jumlah"]
            fig_sem = px.bar(sem_ct, x="semester", y="jumlah", color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_sem.update_layout(xaxis_title="Semester", yaxis_title=None)
            show_chart(fig_sem, height=280)

    if len(eligible_nganggur) > 0:
        with st.expander(f"Daftar {len(eligible_nganggur):,} mahasiswa layak yang belum pernah dikirim", icon=":material/person_alert:"):
            nama_col_e = resolve_col(eligible_nganggur, "nama") or "nama"
            cols_show = [c for c in ["nim", nama_col_e, "program_studi", "semester", "ipk"] if c in eligible_nganggur.columns]
            st.dataframe(eligible_nganggur[cols_show], width="stretch", hide_index=True, height=320)

# ---------------------------------------------------------------------------
# TAB 5 — MATCHING TALENT (data master terkini — tanpa filter)
# ---------------------------------------------------------------------------
with tab5:
    st.caption("Tab ini memakai data master terkini, sehingga tidak memakai filter.")
    match_summary = compute_match_summary()
    n_zero = int((match_summary["kandidat_final"] == 0).sum())

    if ml_bundle is not None:
        model_label = f"ML aktif (akurasi {ml_bundle['test_accuracy']*100:.0f}%)"
        model_help = f"RandomForest dilatih dari {ml_bundle['n_train']:,} histori keputusan Placement vs Rejection."
    else:
        model_label = "Fallback AHP"
        model_help = f"Histori keputusan < {MIN_TRAIN_ROWS} baris atau hanya 1 kelas — skor memakai AHP."

    kpi_row([
        {"value": f"{len(match_summary):,}", "label": "Total Talent Request"},
        {"value": f"{n_zero:,}", "label": "Request Tanpa Kandidat Cocok"},
        {"value": f"{ml_bundle['n_train']:,}" if ml_bundle else "0", "label": "Histori Keputusan (data latih)"},
        {"value": model_label, "label": "Metode Skoring", "help": model_help},
    ])

    with st.container(border=True):
        section("Ringkasan Kecocokan — Semua Talent Request",
                "Kolom bertahap memperlihatkan di filter mana kandidat berkurang: prodi lalu semester lalu status aktif+tersedia. "
                "Diurutkan dari yang paling kritis.")
        if n_zero > 0:
            insight(
                f"Ada <b>{n_zero:,} talent request</b> dengan 0 kandidat cocok. Kalau kolom 'Cocok Prodi' sudah 0, tidak ada "
                "mahasiswa dari bidang studi yang diminta; kalau baru habis di 'Kandidat Final', mahasiswanya ada tapi belum "
                "aktif/tersedia.", kind="warning",
            )
        st.dataframe(
            match_summary.sort_values(["kandidat_final", "cocok_prodi"]).reset_index(drop=True),
            width="stretch", hide_index=True, height=300,
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

    with st.container(border=True):
        section("Cari Kandidat Terbaik untuk Talent Request",
                "Pilih perusahaan lalu posisinya. Skor = probabilitas placement dari model RandomForest "
                "(fallback otomatis ke AHP bila histori belum cukup).")

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
                candidates["recommendation_score"] = model.predict_proba(X_cand)[:, 1]
                candidates["metode_skor"] = "ML (placement probability)"
            else:
                sem_num = pd.to_numeric(candidates[semester_col], errors="coerce").fillna(0)
                candidates["ipk_score"] = candidates["ipk"] / candidates["ipk"].max() if candidates["ipk"].max() > 0 else 0
                candidates["semester_score"] = sem_num / sem_num.max() if sem_num.max() > 0 else 0
                candidates["cv_score"] = norm_text(candidates["cv"]).isin(VAL_ADA).astype(int)
                candidates["portfolio_score"] = norm_text(candidates["portofolio"]).isin(VAL_ADA).astype(int)
                candidates["recommendation_score"] = (
                    0.40 * 1.0
                    + 0.30 * candidates["ipk_score"]
                    + 0.15 * candidates["portfolio_score"]
                    + 0.10 * candidates["semester_score"]
                    + 0.05 * candidates["cv_score"]
                )
                candidates["metode_skor"] = "AHP (fallback)"

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
                    candidates[show_cols].head(50), width="stretch", hide_index=True, height=340,
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
                show_chart(fig_score, height=340)

    if ml_bundle is not None:
        with st.container(border=True):
            section("Feature Importance Model",
                    "Fitur yang paling berpengaruh terhadap peluang placement menurut RandomForest.")
            fig_fi = px.bar(ml_bundle["feature_importance"], x="importance", y="fitur", orientation="h",
                            color_discrete_sequence=[COLOR_SEAL_BROWN])
            fig_fi.update_layout(yaxis_title=None, xaxis_title=None)
            show_chart(fig_fi, height=260)

# ---------------------------------------------------------------------------
# TAB 6 — LAPORAN & QUALITY CHECK
# ---------------------------------------------------------------------------
with tab6:
    tahun_f, prodi_f, jenis_f, _ = filter_bar("laporan")
    m = scope_master(tahun_f, prodi_f, jenis_f)

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
            recap = placement_only.groupby(group_cols).size().reset_index(name="jumlah_placement")

            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                dim_utama = dim_options[dims_pilihan[0]]
                fig_recap = px.bar(
                    recap.sort_values("periode"), x="periode", y="jumlah_placement", color=dim_utama,
                    title=f"Placement per Kuartal berdasarkan {dims_pilihan[0]}",
                    color_discrete_sequence=PALETTE_SEQUENTIAL,
                )
                fig_recap.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
                show_chart(fig_recap, height=360)
            with col_table:
                st.dataframe(recap.sort_values("periode", ascending=False),
                             width="stretch", hide_index=True, height=330)
                st.download_button("Unduh Rekap CSV", recap.to_csv(index=False).encode("utf-8"),
                                   "rekap_placement.csv", "text/csv", icon=":material/download:")

    with st.container(border=True):
        section("Kualitas Data & Sinkronisasi",
                "BT-08: konsistensi STUDENT ALL vs STATUS STUDENT. Data master, di luar filter.")
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
            show_chart(fig_sync, height=280)

        if len(belum_sync) > 0:
            st.dataframe(belum_sync[["nim", "nama", "program_studi"]].head(20),
                         width="stretch", hide_index=True)
