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
COLOR_SIENNA = "#872408"       # warning keras / negatif / ghosting / gap
COLOR_COCOA = "#E2782F"        # aksen utama / positif / umum
COLOR_JASMINE = "#F7D475"      # highlight / supply
COLOR_DRAB_DARK = "#403314"    # teks / netral gelap
COLOR_SEAL_BROWN = "#4A230E"   # background aksen
COLOR_OLIVE = "#6B7A3D"        # sukses, senada earthy — pengganti hijau default
COLOR_BG_CARD = "#FBF4E8"      # background card lembut

PALETTE_SEQUENTIAL = [COLOR_JASMINE, COLOR_COCOA, COLOR_SIENNA, COLOR_SEAL_BROWN, COLOR_DRAB_DARK]

FUNNEL_STAGES = [
    "Selecting Student by Company",
    "Study Case",
    "CDC Briefing Student",
    "Interview User",
    "Final Interview",
    "Placement",
]

# Aturan resmi status follow-up / Ghosting (BT-05) — RULE-BASED, bukan ML.
# Definisi ini datang dari panitia, jadi tidak perlu (dan tidak boleh) dimodelkan.
FU1_DAYS = 7          # > 1 minggu tanpa respons -> FU 1
FU2_DAYS = 14         # > 2 minggu tanpa respons -> FU 2
FU3_DAYS = 21         # > 3 minggu tanpa respons -> FU 3
GHOSTING_DAYS = 28    # > 4 minggu tanpa respons -> Ghosting
SYNC_STALE_DAYS = 14

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
    color: #F3D9BE;
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}}

div[data-testid="stMetric"] {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid #EAD9BB;
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
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-caption {{
    color: #7A6449;
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
    border-color: #EAD9BB !important;
    background-color: #FFFDF9;
}}

button[data-baseweb="tab"] {{
    font-weight: 600;
}}

/* Reskin semua native alert (st.info/warning/success/error) biar nyatu
   sama tema earthy, bukan biru/hijau default Streamlit */
div[data-testid="stAlertContainer"], div[data-testid="stAlert"] {{
    background-color: {COLOR_BG_CARD} !important;
    border: 1px solid #EAD9BB !important;
    border-left: 5px solid {COLOR_COCOA} !important;
    border-radius: 10px !important;
}}
div[data-testid="stAlertContainer"] p, div[data-testid="stAlert"] p {{
    color: {COLOR_DRAB_DARK} !important;
}}

.match-summary-row-zero {{
    color: {COLOR_SIENNA} !important;
    font-weight: 700;
}}

#MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dash-header">
    <h1>🍂 SSDC 2026 — Student Placement Dashboard</h1>
    <p>Ringkasan performa penempatan mahasiswa: dari permintaan perusahaan sampai placement.</p>
</div>
""", unsafe_allow_html=True)


def section(title: str, caption: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


def insight(text: str, kind: str = "info"):
    styles = {
        "info":    {"accent": COLOR_COCOA,      "bg": "#FDF1E7", "icon": "💡"},
        "success": {"accent": COLOR_OLIVE,       "bg": "#F1F4E9", "icon": "✅"},
        "warning": {"accent": COLOR_JASMINE,     "bg": "#FFF8E1", "icon": "⚠️"},
        "error":   {"accent": COLOR_SIENNA,      "bg": "#FBE9E4", "icon": "🔴"},
    }
    s = styles.get(kind, styles["info"])
    st.markdown(
        f'<div class="insight-box" style="--accent:{s["accent"]}; --bg:{s["bg"]};">'
        f'{s["icon"]} <b>Insight:</b> {text}</div>',
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
    """
    Cari nama kolom yang benar setelah merge dengan suffixes.
    Ini yang bikin KeyError kalau dipanggil pakai nama kolom mentah
    (mis. 'program_studi') padahal hasil merge-nya jadi 'program_studi_student'.
    """
    for suf in suffixes:
        cand = f"{base}{suf}"
        if cand in df.columns:
            return cand
    return None


def norm_text(series: pd.Series) -> pd.Series:
    """
    Normalisasi teks untuk matching: strip spasi + lowercase.
    .isin() pandas itu exact-match & case-sensitive, jadi kalau data mentah
    ada beda kapitalisasi/spasi ('Sistem Informasi ' vs 'sistem informasi'),
    semua match bakal gagal walau isinya "sama" secara makna. Ini nyebabin
    SEMUA talent request keliatan 0 kandidat padahal datanya sebenarnya ada.
    """
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
# ---------------------------------------------------------------------------
master = tracking_student.merge(tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc"))
master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))

master["tahun_update"] = master["last_update"].dt.year

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🍂 Filter Laporan")

tahun_list = sorted(master["tahun_update"].dropna().unique().tolist())
tahun_pilihan = st.sidebar.multiselect("Tahun", tahun_list, default=tahun_list)

prodi_list = sorted(student_all["program_studi"].dropna().unique().tolist()) if "program_studi" in student_all.columns else []
prodi_pilihan = st.sidebar.multiselect("Program Studi", prodi_list, default=[])

jenis_list = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist()) if "jenis_penempatan" in talent_request.columns else []
jenis_pilihan = st.sidebar.multiselect("Jenis Penempatan", jenis_list, default=[])

m = master[master["tahun_update"].isin(tahun_pilihan)].copy() if tahun_pilihan else master.copy()
if prodi_pilihan and "program_studi" in m.columns:
    m = m[m["program_studi"].isin(prodi_pilihan)]
if jenis_pilihan and "jenis_penempatan" in m.columns:
    m = m[m["jenis_penempatan"].isin(jenis_pilihan)]

tc_scope = tracking_company.merge(
    talent_request[["id_talent_req", "jenis_penempatan"]] if "jenis_penempatan" in talent_request.columns else talent_request[["id_talent_req"]],
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

st.sidebar.markdown("---")
st.sidebar.caption("Peta Business Task → Tab: Overview · Ghosting · Mitra · Kesiapan · Matching (ML) · Laporan")

# ---------------------------------------------------------------------------
# GHOSTING LOGIC (RULE-BASED — BT-05, TIDAK PAKAI ML)
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
tc_status["tahun_tc"] = tc_status["send_date"].dt.year
if tahun_pilihan:
    tc_status = tc_status[tc_status["tahun_tc"].isin(tahun_pilihan)]
jenis_col_status = "jenis_penempatan_tr2" if "jenis_penempatan_tr2" in tc_status.columns else "jenis_penempatan"
if jenis_pilihan and jenis_col_status in tc_status.columns:
    tc_status = tc_status[tc_status[jenis_col_status].isin(jenis_pilihan)]

# ---------------------------------------------------------------------------
# ML MODEL — MATCHING TALENT (RandomForest, prediksi peluang placement)
# Dilatih dari histori keputusan (Placement vs Rejected). Kalau data histori
# belum cukup / hanya 1 kelas, otomatis fallback ke skor AHP (weighted sum)
# supaya tab tetap jalan dan tidak error.
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
# TABS DECLARATION
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔻 Funnel & Ghosting", "🤝 Mitra", "🎓 Kesiapan", "🔍 Matching Talent (ML)", "📁 Laporan"
])

# ---------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    total_company = company["id_company"].nunique()
    total_mahasiswa = student_all["nim"].nunique()
    total_placement = int((m["rejection"] == "Placement").sum())
    total_dikirimkan = tc_scope["jumlah_dikirimkan"].sum() if "jumlah_dikirimkan" in tc_scope.columns else 0
    success_rate = (total_placement / total_dikirimkan * 100) if total_dikirimkan > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Company", f"{total_company:,}")
    col2.metric("Total Mahasiswa Terdaftar", f"{total_mahasiswa:,}")
    col3.metric("Total Placement Berhasil", f"{total_placement:,}")
    col4.metric("Overall Success Rate", f"{success_rate:.1f}%")

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
            insight(f"Placement bulan **{tren['bulan'].iloc[-1]}** {arah} {abs(int(delta))} dibanding bulan sebelumnya.", kind=kind)
        else:
            insight("Data belum cukup untuk membandingkan tren antar bulan pada rentang filter ini.")

# ---------------------------------------------------------------------------
# TAB 2 — FUNNEL & GHOSTING (rule-based)
# ---------------------------------------------------------------------------
with tab2:
    with st.container(border=True):
        section("Funnel Seleksi Kandidat")
        stage_rank = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}
        m_funnel = m[m["progress_student"].isin(FUNNEL_STAGES)].copy()
        m_funnel["stage_rank"] = m_funnel["progress_student"].map(stage_rank)
        funnel_counts = [(m_funnel["stage_rank"] >= i).sum() for i in range(len(FUNNEL_STAGES))]

        fig_funnel = go.Figure(go.Funnel(
            y=FUNNEL_STAGES, x=funnel_counts,
            marker={"color": PALETTE_SEQUENTIAL[:len(FUNNEL_STAGES)] if len(PALETTE_SEQUENTIAL) >= len(FUNNEL_STAGES) else COLOR_COCOA},
            textinfo="value+percent initial",
        ))
        st.plotly_chart(style_fig(fig_funnel), use_container_width=True)

    with st.container(border=True):
        section("Deteksi Ghosting Perusahaan", "Rule resmi BT-05: >7 hari FU1, >14 FU2, >21 FU3, >28 Ghosting (dihitung dari tanggal acuan di sidebar).")
        total_tc_diproses = len(tc_status)
        total_ghosting_tc = int((tc_status["status_followup"] == "Ghosting").sum())
        ghosting_rate = (total_ghosting_tc / total_tc_diproses * 100) if total_tc_diproses > 0 else 0
        st.metric("Ghosting Rate", f"{ghosting_rate:.1f}%")

        ghosted = tc_status[tc_status["status_followup"] == "Ghosting"]
        col1, col2 = st.columns(2)
        with col1:
            ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
            ghosting_by_stage = ghosted_children["progress_student"].value_counts().reset_index()
            ghosting_by_stage.columns = ["tahap", "jumlah"]
            fig_gs = px.bar(ghosting_by_stage, x="jumlah", y="tahap", orientation="h", title="Ghosting berdasarkan Tahap Terakhir", color_discrete_sequence=[COLOR_SIENNA])
            st.plotly_chart(style_fig(fig_gs), use_container_width=True)
        with col2:
            ghosting_by_company = ghosted["nama_perusahaan"].value_counts().head(10).reset_index()
            ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
            fig_gc = px.bar(ghosting_by_company, x="perusahaan", y="jumlah_ghosting", title="Top Perusahaan Kontributor Ghosting", color_discrete_sequence=[COLOR_SIENNA])
            st.plotly_chart(style_fig(fig_gc), use_container_width=True)

    with st.container(border=True):
        section("Status Follow-up Saat Ini")
        followup_counts = tc_status["status_followup"].value_counts().reindex(["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting"]).fillna(0).astype(int)
        kolom_status = st.columns(5)
        for col_widget, label in zip(kolom_status, ["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting"]):
            col_widget.metric(label.replace(" (Normal)", ""), f"{followup_counts.get(label, 0):,}")

        perlu_followup = tc_status[tc_status["status_followup"].isin(["FU 1", "FU 2", "FU 3", "Ghosting"])].sort_values("hari_sejak_kirim", ascending=False)
        with st.expander(f"📋 Lihat {len(perlu_followup)} batch yang butuh follow-up"):
            st.dataframe(perlu_followup[["id_tracking_company", "nama_perusahaan", "posisi", "send_date", "hari_sejak_kirim", "status_followup"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 — MITRA
# ---------------------------------------------------------------------------
with tab3:
    with st.container(border=True):
        section("Performa Perusahaan Mitra")
        col1, col2 = st.columns(2)
        with col1:
            perf = m.groupby("company").agg(total=("id_tracking_student", "count"), placement=("rejection", lambda x: (x == "Placement").sum())).reset_index()
            perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
            perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10).rename(columns={"company": "perusahaan"})
            fig_acc = px.bar(perf, x="acceptance_rate", y="perusahaan", orientation="h", title="Top 10 Acceptance Rate (min. 3 kandidat)", color_discrete_sequence=[COLOR_COCOA])
            st.plotly_chart(style_fig(fig_acc), use_container_width=True)
        with col2:
            volume = talent_request["nama_perusahaan"].value_counts().head(10).reset_index()
            volume.columns = ["perusahaan", "jumlah_request"]
            fig_vol = px.treemap(volume, path=["perusahaan"], values="jumlah_request", title="Top 10 Volume Talent Request", color_discrete_sequence=PALETTE_SEQUENTIAL)
            st.plotly_chart(style_fig(fig_vol), use_container_width=True)

    with st.container(border=True):
        section("Prioritas Talent Request")
        tr_fulfill = talent_request.merge(tracking_company.groupby("id_talent_req")["jumlah_dikirimkan"].sum().reset_index(), on="id_talent_req", how="left")
        tr_fulfill["jumlah_dikirimkan"] = tr_fulfill["jumlah_dikirimkan"].fillna(0)
        tr_fulfill["belum_terpenuhi"] = tr_fulfill["headcount"] - tr_fulfill["jumlah_dikirimkan"]
        prioritas = tr_fulfill[tr_fulfill["belum_terpenuhi"] > 0].sort_values("request_date")
        st.dataframe(prioritas[["id_talent_req", "nama_perusahaan", "nama_posisi", "headcount", "jumlah_dikirimkan", "belum_terpenuhi", "request_date"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4 — KESIAPAN
# ---------------------------------------------------------------------------
with tab4:
    with st.container(border=True):
        section("Matching Gap: Demand vs Supply Bidang Studi")
        demand = talent_request["bidang_studi_dibutuhkan"].dropna().str.split(",").explode().str.strip().value_counts().reset_index()
        demand.columns = ["bidang_studi", "jumlah"]; demand["tipe"] = "Demand"
        supply = student_all["program_studi"].dropna().value_counts().reset_index()
        supply.columns = ["bidang_studi", "jumlah"]; supply["tipe"] = "Supply"

        gap_melt = pd.concat([demand, supply], ignore_index=True)
        top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
        gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]

        fig_gap = px.bar(gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group", title="Top 10 Bidang Studi Demand vs Supply", color_discrete_map={"Demand": COLOR_SIENNA, "Supply": COLOR_JASMINE})
        st.plotly_chart(style_fig(fig_gap), use_container_width=True)

    with st.container(border=True):
        section("Kesiapan Dokumen & Mahasiswa Layak Kirim")
        col1, col2 = st.columns(2)
        with col1:
            elig = status_student.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
            fig_elig = px.bar(elig, x="ketersediaan", y="jumlah", color="status", title="Eligibility Mahasiswa", color_discrete_sequence=PALETTE_SEQUENTIAL)
            st.plotly_chart(style_fig(fig_elig), use_container_width=True)
        with col2:
            fig_ipk = px.histogram(status_student, x="ipk", nbins=20, title="Distribusi IPK Mahasiswa", color_discrete_sequence=[COLOR_COCOA])
            st.plotly_chart(style_fig(fig_ipk), use_container_width=True)

        eligible = status_student[(status_student["status"] == "Aktif") & (status_student["ketersediaan"] == "Tersedia") & (status_student["cv"] == "Ada") & (status_student["portofolio"] == "Ada")].merge(student_all[["nim", "program_studi", "semester"]], on="nim", how="left")
        st.metric("Total Mahasiswa Layak Kirim Saat Ini", f"{len(eligible):,}")

# ---------------------------------------------------------------------------
# TAB 5 — MATCHING TALENT (ML: placement probability)
# ---------------------------------------------------------------------------
@st.cache_data
def compute_match_summary(talent_request_df: pd.DataFrame, student_all_df: pd.DataFrame, status_student_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung jumlah kandidat cocok untuk SEMUA talent request sekaligus, bertahap
    (cocok prodi -> cocok semester -> cocok status & ketersediaan), supaya kalau
    hasil akhirnya 0 kita tau persis di tahap mana kandidat hilang.
    """
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
            (cocok_semester["_ketersediaan_norm"] == "tersedia") & (cocok_semester["_status_norm"] == "aktif")
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
    match_summary = compute_match_summary(talent_request, student_all, status_student)

    with st.container(border=True):
        section(
            "Ringkasan Kecocokan — Semua Talent Request",
            "Biar nggak perlu klik satu-satu buat tau mana yang 0 kandidat. Urut dari yang paling krisis kandidat.",
        )
        n_zero = int((match_summary["kandidat_final"] == 0).sum())
        col_a, col_b = st.columns(2)
        col_a.metric("Total Talent Request", f"{len(match_summary):,}")
        col_b.metric("Request Tanpa Kandidat Cocok", f"{n_zero:,}")

        if n_zero > 0:
            insight(
                f"Ada **{n_zero} talent request** dengan 0 kandidat cocok. "
                "Cek kolom bertahap di tabel: kalau 'cocok_prodi' sudah 0, berarti tidak ada mahasiswa "
                "dari bidang studi yang diminta. Kalau baru drop di 'kandidat_final', "
                "biasanya penyebabnya mahasiswa yang cocok prodi & semester belum submit status "
                "kesiapan (belum 'Aktif' / belum 'Tersedia').",
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
            with st.expander("🔍 Debug: kenapa 'Cocok Prodi' 0 di semua baris?"):
                st.caption(
                    "Kalau daftar nilai program_studi mahasiswa TIDAK PERNAH muncul persis "
                    "di daftar bidang_studi_dibutuhkan (walau kelihatan 'sama' secara arti), "
                    "berarti penulisannya beda (typo, singkatan, urutan kata, dll) — bukan bug matching. "
                    "Samakan dulu nilainya di sumber data (CSV/Sheet), baru refresh dashboard."
                )
                _prodi_col_dbg = resolve_col(
                    student_all.merge(status_student, on="nim", how="inner", suffixes=("_student", "_status")),
                    "program_studi",
                ) or "program_studi"
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
            with st.expander("🔍 Debug: kenapa 'Kandidat Final' 0 padahal 'Cocok Prodi' > 0?"):
                st.caption(
                    "Berarti masalahnya di kolom `ketersediaan` / `status` — nilainya bukan persis "
                    "'Tersedia' / 'Aktif'. Cek daftar nilai unik di bawah."
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
            "Skor kandidat = probabilitas placement dari model ML (RandomForest), "
            "dilatih dari histori keputusan Placement/Rejected. Kalau histori belum cukup, "
            "otomatis fallback ke skor AHP (weighted sum) supaya tetap ada hasil.",
        )

        if ml_bundle is not None:
            insight(
                f"Model aktif: RandomForest, dilatih dari {ml_bundle['n_train']:,} baris histori "
                f"keputusan, akurasi test ≈ {ml_bundle['test_accuracy']*100:.1f}%.",
                kind="success",
            )
        else:
            insight(
                "Histori keputusan (Placement/Rejected) belum cukup atau hanya 1 kelas — "
                "dashboard fallback ke skor AHP (bukan ML) untuk tab ini.",
                kind="warning",
            )

        # Urutkan pilihan: yang kandidatnya paling banyak di atas, biar nggak
        # nyasar milih request yang bakal 0 hasil.
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
            + tr_sorted["kandidat_final"].apply(lambda x: " ⚠️" if x == 0 else "")
        )

        pilihan_label = st.selectbox(
            "Pilih Talent Request",
            tr_sorted["label"].tolist()
        )

        selected_id = tr_sorted.loc[tr_sorted["label"] == pilihan_label, "id_talent_req"].iloc[0]
        selected = talent_request[talent_request["id_talent_req"] == selected_id].iloc[0]

        bidang_dibutuhkan = [
            x.strip()
            for x in str(
                selected.get("bidang_studi_dibutuhkan", "")
            ).split(",")
            if x.strip() != ""
        ]

        min_semester = selected.get("minimum_semester", 0)
        if pd.isna(min_semester):
            min_semester = 0

        candidates = student_all.merge(
            status_student,
            on="nim",
            how="inner",
            suffixes=("_student", "_status")
        )

        # resolve_col menangani KeyError akibat suffix _student/_status setelah merge
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
            (cocok_semester_df["_ketersediaan_norm"] == "tersedia")
            & (cocok_semester_df["_status_norm"] == "aktif")
        ].copy()

        if len(candidates) > 0:
            if ml_bundle is not None:
                # ---- Skor ML: placement_probability ----
                model = ml_bundle["model"]
                encoders = ml_bundle["encoders"]
                resolved_cols = ml_bundle["resolved_cols"]
                numeric_feats = ml_bundle["numeric_feats"]
                feature_cols = ml_bundle["feature_cols"]

                X_cand = pd.DataFrame(index=candidates.index)
                for feat, train_col in resolved_cols.items():
                    cand_col = resolve_col(candidates, feat) or train_col
                    if cand_col not in candidates.columns:
                        # fitur tidak tersedia di candidates -> isi netral
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
                # ---- Fallback AHP kalau data histori belum cukup ----
                candidates["prodi_score"] = 1.0
                candidates["ipk_score"] = (
                    candidates["ipk"] / candidates["ipk"].max()
                    if candidates["ipk"].max() > 0
                    else 0
                )
                candidates["semester_score"] = (
                    candidates[semester_col] / candidates[semester_col].max()
                )
                candidates["cv_score"] = (
                    candidates["cv"].astype(str).str.lower().eq("ada").astype(int)
                )
                candidates["portfolio_score"] = (
                    candidates["portofolio"].astype(str).str.lower().eq("ada").astype(int)
                )

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
                sebab = f"tidak ada mahasiswa terdaftar dari bidang studi **{', '.join(bidang_dibutuhkan) or '(kosong)'}**."
            elif len(cocok_semester_df) == 0:
                sebab = f"ada {len(cocok_prodi_df)} mahasiswa dari bidang yang cocok, tapi semuanya di bawah minimum semester ({int(min_semester)})."
            else:
                sebab = (
                    f"ada {len(cocok_semester_df)} mahasiswa yang cocok bidang & semester, "
                    "tapi belum ada yang berstatus 'Aktif' dan 'Tersedia' di data status kesiapan."
                )
            insight(f"Tidak ada kandidat yang memenuhi syarat — {sebab}", kind="error")
        else:
            show_cols = [
                c for c in [
                    "nim",
                    nama_col,
                    prodi_col,
                    semester_col,
                    "ipk",
                    "cv",
                    "portofolio",
                    "domisili",
                    "recommendation_score",
                    "metode_skor",
                ]
                if c in candidates.columns
            ]

            st.dataframe(
                candidates[show_cols].head(50),
                use_container_width=True,
                hide_index=True
            )

    if ml_bundle is not None:
        with st.container(border=True):
            section("Feature Importance Model", "Fitur mana yang paling berpengaruh terhadap peluang placement menurut model.")
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
        section("Rekapitulasi Placement")
        dim_options = {"Program Studi": "program_studi", "Perusahaan": "company", "Jenis Penempatan": "jenis_penempatan"}
        dims_pilihan = st.multiselect("Kelompokkan berdasarkan", list(dim_options.keys()), default=["Program Studi"])

        placement_only = m[m["rejection"] == "Placement"].copy()
        placement_only["periode"] = placement_only["last_update"].dt.to_period("Q").astype(str)

        if dims_pilihan:
            group_cols = ["periode"] + [dim_options[d] for d in dims_pilihan]
            recap = placement_only.groupby(group_cols).size().reset_index(name="jumlah_placement").sort_values("periode", ascending=False)
            st.dataframe(recap, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Unduh Rekap CSV", recap.to_csv(index=False).encode("utf-8"), "rekap_placement.csv", "text/csv")

    with st.container(border=True):
        section("Kualitas Data & Sinkronisasi")
        merged_check = student_all.merge(status_student[["nim", "sync_date"]], on="nim", how="left", indicator=True)
        belum_sync = merged_check[merged_check["_merge"] == "left_only"]
        st.metric("Mahasiswa Belum Ada Data Status (Belum Sync)", f"{len(belum_sync):,}")
        if len(belum_sync) > 0:
            st.dataframe(belum_sync[["nim", "nama", "program_studi"]].head(20), use_container_width=True, hide_index=True)
