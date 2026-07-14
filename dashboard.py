import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


st.set_page_config(
    page_title="SSDC 2026 — Student Placement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# PALET WARNA SEMANTIK
# ---------------------------------------------------------------------------
COLOR_SIENNA = "#872408"       # warning / negatif / ghosting / gap
COLOR_COCOA = "#E2782F"        # aksen utama / positif / umum
COLOR_JASMINE = "#F7D475"      # highlight / supply
COLOR_DRAB_DARK = "#403314"    # teks / netral gelap
COLOR_SEAL_BROWN = "#4A230E"   # background aksen

PALETTE_SEQUENTIAL = [COLOR_JASMINE, COLOR_COCOA, COLOR_SIENNA, COLOR_SEAL_BROWN, COLOR_DRAB_DARK]

FUNNEL_STAGES = [
    "Selecting Student by Company",
    "Study Case",
    "CDC Briefing Student",
    "Interview User",
    "Final Interview",
    "Placement",
]

# --- Aturan resmi status follow-up / Ghosting (BT-05), dari pihak PERUSAHAAN,
#     dihitung dari tracking_company.send_date tanpa respons:
FU1_DAYS = 7          # > 1 minggu tanpa respons -> FU 1
FU2_DAYS = 14         # > 2 minggu tanpa respons -> FU 2
FU3_DAYS = 21         # > 3 minggu tanpa respons -> FU 3
GHOSTING_DAYS = 28    # > 4 minggu tanpa respons -> Ghosting

# NB: ini bukan bagian dari aturan Ghosting di atas — cuma default terpisah utk
# menandai data mahasiswa yang "belum di-sync ulang" di BT-08. Ganti sesuai kebutuhan.
SYNC_STALE_DAYS = 14


def insight(text: str, kind: str = "info"):
    """Kotak analisis singkat di bawah/atas chart. kind: info | warning | error | success"""
    fn = {"info": st.info, "warning": st.warning, "error": st.error, "success": st.success}[kind]
    fn(f"**Insight:** {text}")


def hitung_status_followup(hari_sejak_kirim, sudah_direspon) -> str:
    """Aturan resmi BT-05: dihitung dari send_date tanpa respons pihak perusahaan.
    >1 minggu -> FU 1, >2 minggu -> FU 2, >3 minggu -> FU 3, >4 minggu -> Ghosting."""
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


# ---------------------------------------------------------------------------
# LOAD & NORMALISASI DATA
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
        (company, "id_company"),
        (talent_request, "id_talent_req"),
        (talent_request, "id_company"),
        (tracking_company, "id_tracking_company"),
        (tracking_company, "id_talent_req"),
        (tracking_company, "id_company"),
        (tracking_student, "id_tracking_student"),
        (tracking_student, "id_tracking_company"),
        (tracking_student, "nim"),
        (student_all, "nim"),
        (status_student, "id_status"),
        (status_student, "nim"),
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
# MASTER TABLE (relasi sesuai ERD)
# tracking_student -> tracking_company -> company
#                                       -> talent_request
# tracking_student -> nim -> student_all
# tracking_student -> nim -> status_student
# tracking_student punya kolom "company" sendiri (denormalized)
# ---------------------------------------------------------------------------
master = tracking_student.merge(tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc"))
master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))

master["tahun_update"] = master["last_update"].dt.year


# ---------------------------------------------------------------------------
# SIDEBAR — FILTER
# ---------------------------------------------------------------------------
st.sidebar.title("🍂 SSDC 2026")
st.sidebar.markdown("**Filter Laporan**")

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
    talent_request[["id_talent_req", "jenis_penempatan"]] if "jenis_penempatan" in talent_request.columns
    else talent_request[["id_talent_req"]],
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

st.sidebar.markdown("---")
st.sidebar.caption("Peta Business Task → Tab:\n"
                    "BT-01 → Matching · BT-02/05 → Funnel · BT-03/04 → Mitra · "
                    "BT-06 → Kesiapan · BT-07/08 → Laporan & Data")

st.sidebar.markdown("---")
default_ref_date = tracking_company["send_date"].max()
if pd.isna(default_ref_date):
    default_ref_date = datetime.now()
tanggal_acuan = st.sidebar.date_input(
    "Tanggal Acuan Monitoring Ghosting",
    value=default_ref_date.date() if hasattr(default_ref_date, "date") else default_ref_date,
    help="Status FU 1/2/3/Ghosting dihitung dari send_date sampai tanggal ini.",
)
tanggal_acuan = pd.Timestamp(tanggal_acuan)

# ---------------------------------------------------------------------------
# STATUS FOLLOW-UP / GHOSTING per tracking_company (BT-05, aturan resmi)
# Ghosting dari sisi PERUSAHAAN: dihitung dari send_date tanpa respons.
# "Respons" = ada tracking_student turunannya yang progress_student sudah
# lewat tahap awal, ATAU sudah ada keputusan definitif (Placement/Rejected).
# ---------------------------------------------------------------------------
respon_per_tc = tracking_student.groupby("id_tracking_company").agg(
    ada_progress_lanjut=("progress_student", lambda s: (s != "Selecting Student by Company").any()),
    ada_keputusan=("rejection", lambda s: s.isin(["Placement", "Rejected"]).any()),
).reset_index()
respon_per_tc["sudah_direspon"] = respon_per_tc["ada_progress_lanjut"] | respon_per_tc["ada_keputusan"]

tc_status = tracking_company.merge(
    respon_per_tc[["id_tracking_company", "sudah_direspon"]], on="id_tracking_company", how="left"
)
tc_status["sudah_direspon"] = tc_status["sudah_direspon"].fillna(False)
tc_status["hari_sejak_kirim"] = (tanggal_acuan - tc_status["send_date"]).dt.days
tc_status["status_followup"] = tc_status.apply(
    lambda r: hitung_status_followup(r["hari_sejak_kirim"], r["sudah_direspon"]), axis=1
)

# Terapkan filter scope (tahun & jenis penempatan) yang sama seperti tc_scope
tc_status = tc_status.merge(
    talent_request[["id_talent_req", "jenis_penempatan"]] if "jenis_penempatan" in talent_request.columns
    else talent_request[["id_talent_req"]],
    on="id_talent_req", how="left", suffixes=("", "_tr2"),
)
tc_status["tahun_tc"] = tc_status["send_date"].dt.year
if tahun_pilihan:
    tc_status = tc_status[tc_status["tahun_tc"].isin(tahun_pilihan)]
jenis_col_status = "jenis_penempatan_tr2" if "jenis_penempatan_tr2" in tc_status.columns else "jenis_penempatan"
if jenis_pilihan and jenis_col_status in tc_status.columns:
    tc_status = tc_status[tc_status[jenis_col_status].isin(jenis_pilihan)]


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔻 Funnel & Monitoring", "🤝 Mitra & Talent Request",
    "🎓 Kesiapan & Kelayakan", "🔍 Matching Talent", "📁 Laporan & Data Quality",
])


# ---------------------------------------------------------------------------
# TAB 1 — OVERVIEW  (BT-04 ringkas)
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

    st.markdown("---")

    placement_df = m[m["rejection"] == "Placement"].copy()
    placement_df["bulan"] = placement_df["last_update"].dt.to_period("M").astype(str)
    tren = placement_df.groupby("bulan").size().reset_index(name="jumlah_placement").sort_values("bulan")

    fig_tren = px.line(
        tren, x="bulan", y="jumlah_placement", markers=True,
        title="Tren Placement Berhasil per Bulan",
    )
    fig_tren.update_traces(line_color=COLOR_COCOA, marker_color=COLOR_SEAL_BROWN)
    fig_tren.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Placement")
    st.plotly_chart(fig_tren, use_container_width=True)

    if len(tren) >= 2:
        delta = tren["jumlah_placement"].iloc[-1] - tren["jumlah_placement"].iloc[-2]
        arah = "naik" if delta > 0 else ("turun" if delta < 0 else "stagnan")
        kind = "success" if delta >= 0 else "warning"
        insight(
            f"Placement bulan **{tren['bulan'].iloc[-1]}** {arah} sebanyak **{abs(int(delta))}** "
            f"dibanding bulan sebelumnya ({tren['bulan'].iloc[-2]}: {int(tren['jumlah_placement'].iloc[-2])} → "
            f"{tren['bulan'].iloc[-1]}: {int(tren['jumlah_placement'].iloc[-1])}). "
            f"Success rate keseluruhan saat ini **{success_rate:.1f}%** dari total kandidat yang dikirimkan.",
            kind=kind,
        )
    else:
        insight("Data belum cukup untuk membandingkan tren antar bulan pada rentang filter ini.")

    if "sync_date" in status_student.columns and status_student["sync_date"].notna().any():
        st.caption(f"ℹ️ Sinkronisasi data terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}")


# ---------------------------------------------------------------------------
# TAB 2 — FUNNEL & MONITORING  (BT-02, BT-05)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Funnel Seleksi")

    stage_rank = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}
    m_funnel = m[m["progress_student"].isin(FUNNEL_STAGES)].copy()
    m_funnel["stage_rank"] = m_funnel["progress_student"].map(stage_rank)

    funnel_counts = [(m_funnel["stage_rank"] >= i).sum() for i in range(len(FUNNEL_STAGES))]

    fig_funnel = go.Figure(go.Funnel(
        y=FUNNEL_STAGES, x=funnel_counts,
        marker={"color": PALETTE_SEQUENTIAL[:len(FUNNEL_STAGES)] if len(PALETTE_SEQUENTIAL) >= len(FUNNEL_STAGES) else COLOR_COCOA},
        textinfo="value+percent initial",
    ))
    fig_funnel.update_layout(title="Funnel Seleksi Kandidat")
    st.plotly_chart(fig_funnel, use_container_width=True)

    if len(funnel_counts) >= 2 and funnel_counts[0] > 0:
        drop_pct = [(funnel_counts[i] - funnel_counts[i + 1]) / funnel_counts[i] * 100 if funnel_counts[i] > 0 else 0
                    for i in range(len(funnel_counts) - 1)]
        worst_idx = int(np.argmax(drop_pct))
        insight(
            f"Drop-off terbesar terjadi antara tahap **{FUNNEL_STAGES[worst_idx]}** ke "
            f"**{FUNNEL_STAGES[worst_idx + 1]}**, kehilangan **{drop_pct[worst_idx]:.1f}%** kandidat "
            f"pada tahap tersebut. Ini titik prioritas untuk evaluasi proses seleksi.",
            kind="warning" if drop_pct[worst_idx] > 30 else "info",
        )

    total_tc_diproses = len(tc_status)
    total_ghosting_tc = int((tc_status["status_followup"] == "Ghosting").sum())
    ghosting_rate = (total_ghosting_tc / total_tc_diproses * 100) if total_tc_diproses > 0 else 0
    st.metric("Ghosting Rate (per batch pengiriman ke perusahaan)", f"{ghosting_rate:.1f}%")
    st.caption(
        f"Dihitung dari `tracking_company.send_date` s.d. tanggal acuan ({tanggal_acuan.strftime('%d %B %Y')}): "
        f"belum direspon > {GHOSTING_DAYS} hari (4 minggu) = Ghosting."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    ghosted = tc_status[tc_status["status_followup"] == "Ghosting"]

    with col1:
        # Tahap terakhir sesaat sebelum di-ghosting: ambil dari tracking_student
        # turunan tiap tracking_company yang berstatus Ghosting.
        ghosted_children = tracking_student[tracking_student["id_tracking_company"].isin(ghosted["id_tracking_company"])]
        ghosting_by_stage = ghosted_children["progress_student"].value_counts().reset_index()
        ghosting_by_stage.columns = ["tahap", "jumlah"]
        fig_gs = px.bar(
            ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
            title="Ghosting berdasarkan Tahap Terakhir",
            color_discrete_sequence=[COLOR_SIENNA],
        )
        fig_gs.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gs, use_container_width=True)

    with col2:
        ghosting_by_company = ghosted["nama_perusahaan"].value_counts().head(10).reset_index()
        ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
        fig_gc = px.bar(
            ghosting_by_company, x="perusahaan", y="jumlah_ghosting",
            title="Top Perusahaan Kontributor Ghosting Terbanyak",
            color_discrete_sequence=[COLOR_SIENNA],
        )
        st.plotly_chart(fig_gc, use_container_width=True)

    if not ghosting_by_company.empty:
        top_ghost_company = ghosting_by_company.iloc[0]["perusahaan"]
        top_ghost_stage = (ghosting_by_stage.sort_values("jumlah", ascending=False).iloc[0]["tahap"]
                            if not ghosting_by_stage.empty else "tidak diketahui")
        insight(
            f"Ghosting rate keseluruhan **{ghosting_rate:.1f}%** (>{GHOSTING_DAYS} hari tanpa respons perusahaan), "
            f"titik kehilangan terbanyak pada tahap **{top_ghost_stage}**, dengan **{top_ghost_company}** sebagai "
            f"kontributor terbesar. Rekomendasi: mulai eskalasi follow-up sejak status FU 1 (>{FU1_DAYS} hari), "
            f"jangan tunggu sampai Ghosting.",
            kind="error",
        )
    else:
        st.info("Belum ada batch pengiriman yang mencapai status Ghosting pada rentang filter ini.")

    st.markdown("---")
    st.subheader("Monitoring Status Follow-up (BT-02)")
    st.caption("Status dihitung otomatis per batch pengiriman ke perusahaan, mengikuti aturan resmi di atas.")

    followup_counts = tc_status["status_followup"].value_counts().reindex(
        ["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting", "Direspon"]
    ).fillna(0).astype(int)
    kolom_status = st.columns(5)
    for col_widget, label in zip(kolom_status, ["Menunggu Respons (Normal)", "FU 1", "FU 2", "FU 3", "Ghosting"]):
        col_widget.metric(label, f"{followup_counts.get(label, 0):,}")

    perlu_followup = tc_status[tc_status["status_followup"].isin(["FU 1", "FU 2", "FU 3", "Ghosting"])].sort_values(
        "hari_sejak_kirim", ascending=False
    )
    cols_show = [c for c in ["id_tracking_company", "nama_perusahaan", "posisi", "send_date",
                              "hari_sejak_kirim", "status_followup"] if c in perlu_followup.columns]
    st.dataframe(perlu_followup[cols_show].head(50), use_container_width=True, hide_index=True)

    if not perlu_followup.empty:
        insight(
            f"Ada **{len(perlu_followup)}** batch pengiriman yang butuh follow-up (status FU 1 s.d. Ghosting) "
            f"per tanggal acuan **{tanggal_acuan.strftime('%d %B %Y')}**. Prioritaskan yang sudah di status FU 3 "
            f"agar tidak jatuh ke Ghosting.",
            kind="warning",
        )
    else:
        st.success("Tidak ada batch pengiriman yang butuh follow-up saat ini.")


# ---------------------------------------------------------------------------
# TAB 3 — MITRA & TALENT REQUEST  (BT-03, BT-04)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Performa Perusahaan Mitra")
    col1, col2 = st.columns(2)

    with col1:
        perf = m.groupby("company").agg(
            total=("id_tracking_student", "count"),
            placement=("rejection", lambda x: (x == "Placement").sum()),
        ).reset_index()
        perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
        perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10)
        perf = perf.rename(columns={"company": "perusahaan"})

        fig_acc = px.bar(
            perf, x="acceptance_rate", y="perusahaan", orientation="h",
            title="Top 10 Perusahaan — Acceptance Rate (min. 3 kandidat)",
            color_discrete_sequence=[COLOR_COCOA],
        )
        fig_acc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        volume = talent_request["nama_perusahaan"].value_counts().head(10).reset_index()
        volume.columns = ["perusahaan", "jumlah_request"]
        fig_vol = px.treemap(
            volume, path=["perusahaan"], values="jumlah_request",
            title="Top 10 Perusahaan — Volume Talent Request",
            color_discrete_sequence=PALETTE_SEQUENTIAL,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    if not perf.empty:
        top_perf = perf.iloc[0]
        insight(
            f"**{top_perf['perusahaan']}** punya acceptance rate tertinggi (**{top_perf['acceptance_rate']}%** "
            f"dari {int(top_perf['total'])} kandidat diproses) — kandidat kuat untuk mitra strategis jangka panjang.",
            kind="success",
        )

    col3, col4 = st.columns(2)
    with col3:
        jenis_dist = talent_request["jenis_penempatan"].value_counts().reset_index()
        jenis_dist.columns = ["jenis_penempatan", "jumlah"]
        fig_jenis = px.pie(
            jenis_dist, values="jumlah", names="jenis_penempatan", hole=0.45,
            title="Jenis Penempatan yang Diminta",
            color_discrete_sequence=PALETTE_SEQUENTIAL,
        )
        st.plotly_chart(fig_jenis, use_container_width=True)

    with col4:
        sektor_dist = company["industry_sector"].value_counts().head(10).reset_index()
        sektor_dist.columns = ["sektor", "jumlah"]
        fig_sektor = px.bar(
            sektor_dist, x="jumlah", y="sektor", orientation="h",
            title="Top 10 Sektor Industri Mitra",
            color_discrete_sequence=[COLOR_SEAL_BROWN],
        )
        fig_sektor.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_sektor, use_container_width=True)

    st.markdown("---")
    st.subheader("Manajemen & Prioritas Talent Request (BT-03)")

    tr_fulfill = talent_request.merge(
        tracking_company.groupby("id_talent_req")["jumlah_dikirimkan"].sum().reset_index(),
        on="id_talent_req", how="left",
    )
    tr_fulfill["jumlah_dikirimkan"] = tr_fulfill["jumlah_dikirimkan"].fillna(0)
    tr_fulfill["fulfillment_rate"] = np.where(
        tr_fulfill["headcount"] > 0,
        (tr_fulfill["jumlah_dikirimkan"] / tr_fulfill["headcount"] * 100).round(1),
        np.nan,
    )
    tr_fulfill["belum_terpenuhi"] = tr_fulfill["headcount"] - tr_fulfill["jumlah_dikirimkan"]
    tr_fulfill = tr_fulfill.sort_values("request_date", ascending=True)

    cols_show = [c for c in ["id_talent_req", "nama_perusahaan", "nama_posisi", "jenis_penempatan",
                              "headcount", "jumlah_dikirimkan", "fulfillment_rate", "belum_terpenuhi",
                              "request_date"] if c in tr_fulfill.columns]
    prioritas = tr_fulfill[tr_fulfill["belum_terpenuhi"] > 0].sort_values("request_date")
    st.caption("Request yang belum terpenuhi, diurutkan dari yang paling lama menunggu (prioritas tertinggi):")
    st.dataframe(prioritas[cols_show].head(20), use_container_width=True, hide_index=True)

    if not prioritas.empty:
        oldest = prioritas.iloc[0]
        insight(
            f"Request **{oldest.get('nama_posisi', '-')}** dari **{oldest.get('nama_perusahaan', '-')}** "
            f"masih kurang **{int(oldest['belum_terpenuhi'])}** kandidat dan sudah menunggu sejak "
            f"**{oldest['request_date'].strftime('%d %B %Y') if pd.notna(oldest.get('request_date')) else '-'}** — prioritaskan pengiriman kandidat ke sini.",
            kind="warning",
        )
    else:
        st.success("Semua talent request pada rentang filter sudah terpenuhi headcount-nya.")


# ---------------------------------------------------------------------------
# TAB 4 — KESIAPAN & KELAYAKAN  (BT-06)
# ---------------------------------------------------------------------------
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        elig = status_student.groupby(["ketersediaan", "status"]).size().reset_index(name="jumlah")
        fig_elig = px.bar(
            elig, x="ketersediaan", y="jumlah", color="status", barmode="stack",
            title="Distribusi Eligibility / Ketersediaan Mahasiswa",
            color_discrete_sequence=PALETTE_SEQUENTIAL,
        )
        st.plotly_chart(fig_elig, use_container_width=True)

    with col2:
        cv_ada = int((status_student["cv"] == "Ada").sum())
        cv_tidak = int((status_student["cv"] == "Tidak Ada").sum())
        porto_ada = int((status_student["portofolio"] == "Ada").sum())
        porto_tidak = int((status_student["portofolio"] == "Tidak Ada").sum())
        kelengkapan = pd.DataFrame({
            "dokumen": ["CV", "CV", "Portofolio", "Portofolio"],
            "status": ["Ada", "Tidak Ada", "Ada", "Tidak Ada"],
            "jumlah": [cv_ada, cv_tidak, porto_ada, porto_tidak],
        })
        fig_doc = px.bar(
            kelengkapan, x="dokumen", y="jumlah", color="status", barmode="group",
            title="Kelengkapan Dokumen Mahasiswa",
            color_discrete_map={"Ada": COLOR_COCOA, "Tidak Ada": COLOR_SIENNA},
        )
        st.plotly_chart(fig_doc, use_container_width=True)

    incomplete = int(((status_student["cv"] != "Ada") | (status_student["portofolio"] != "Ada")).sum())
    insight(
        f"**{incomplete}** mahasiswa masih punya dokumen tidak lengkap (CV/portofolio) — mereka tidak boleh "
        f"dikirim ke perusahaan sampai dokumen dilengkapi.",
        kind="warning" if incomplete > 0 else "success",
    )

    st.markdown("---")
    demand = (talent_request["bidang_studi_dibutuhkan"].dropna().str.split(",").explode().str.strip()
              .value_counts().reset_index())
    demand.columns = ["bidang_studi", "jumlah"]
    demand["tipe"] = "Demand"

    supply = student_all["program_studi"].dropna().value_counts().reset_index()
    supply.columns = ["bidang_studi", "jumlah"]
    supply["tipe"] = "Supply"

    gap_wide = demand.set_index("bidang_studi")["jumlah"].to_frame("demand").join(
        supply.set_index("bidang_studi")["jumlah"].to_frame("supply"), how="outer"
    ).fillna(0)
    gap_wide["gap"] = gap_wide["supply"] - gap_wide["demand"]

    gap_melt = pd.concat([demand, supply], ignore_index=True)
    top_bidang = gap_melt.groupby("bidang_studi")["jumlah"].sum().sort_values(ascending=False).head(10).index
    gap_melt = gap_melt[gap_melt["bidang_studi"].isin(top_bidang)]

    fig_gap = px.bar(
        gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
        title="Matching Gap: Bidang Studi Dicari (Demand) vs Tersedia (Supply)",
        color_discrete_map={"Demand": COLOR_SIENNA, "Supply": COLOR_JASMINE},
    )
    fig_gap.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_gap, use_container_width=True)

    if not gap_wide.empty:
        biggest_shortage = gap_wide.sort_values("gap").iloc[0]
        insight(
            f"Bidang studi dengan kesenjangan terbesar (demand jauh lebih tinggi dari supply): "
            f"**{biggest_shortage.name}** — demand {int(biggest_shortage['demand'])} vs supply {int(biggest_shortage['supply'])}. "
            f"Perlu strategi rekrutmen mahasiswa dari prodi terkait atau edukasi ulang ke perusahaan soal ketersediaan talent.",
            kind="warning",
        )

    fig_ipk = px.histogram(
        status_student, x="ipk", nbins=20,
        title="Distribusi IPK Mahasiswa",
        color_discrete_sequence=[COLOR_COCOA],
    )
    st.plotly_chart(fig_ipk, use_container_width=True)

    st.markdown("---")
    st.subheader("Daftar Mahasiswa Layak Dikirim (BT-06)")
    st.caption("Kriteria: status Aktif, ketersediaan Tersedia, dan dokumen (CV & portofolio) lengkap.")

    eligible = status_student[
        (status_student["status"] == "Aktif")
        & (status_student["ketersediaan"] == "Tersedia")
        & (status_student["cv"] == "Ada")
        & (status_student["portofolio"] == "Ada")
    ].merge(student_all[["nim", "program_studi", "semester"]], on="nim", how="left")

    st.metric("Mahasiswa Layak Dikirim Saat Ini", f"{len(eligible):,}")
    cols_show = [c for c in ["nim", "nama", "program_studi", "semester", "ipk", "ketersediaan", "domisili"] if c in eligible.columns]
    st.dataframe(eligible[cols_show].sort_values("ipk", ascending=False), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 5 — MATCHING TALENT  (BT-01)
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Cari Kandidat Terbaik untuk Talent Request")
    st.caption("Sistem mencocokkan mahasiswa berdasarkan bidang studi, minimum semester, ketersediaan, dan IPK.")

    tr_options = talent_request.copy()
    tr_options["label"] = (
        tr_options["nama_posisi"].fillna("-") + " — " + tr_options["nama_perusahaan"].fillna("-")
        + " (" + tr_options["id_talent_req"].astype(str) + ")"
    )
    pilihan_label = st.selectbox("Pilih Talent Request", tr_options["label"].tolist())
    selected = tr_options[tr_options["label"] == pilihan_label].iloc[0]

    bidang_dibutuhkan = [b.strip() for b in str(selected.get("bidang_studi_dibutuhkan", "")).split(",")]
    min_semester = selected.get("minimum_semester", 0)
    if pd.isna(min_semester):
        min_semester = 0

    colr1, colr2, colr3 = st.columns(3)
    colr1.metric("Bidang Studi Dibutuhkan", ", ".join(bidang_dibutuhkan) if bidang_dibutuhkan else "-")
    colr2.metric("Min. Semester", f"{int(min_semester)}")
    colr3.metric("Headcount", f"{int(selected.get('headcount', 0)) if pd.notna(selected.get('headcount')) else '-'}")

    candidates = student_all.merge(status_student, on="nim", how="inner", suffixes=("", "_ss"))
    candidates = candidates[
        candidates["program_studi"].isin(bidang_dibutuhkan)
        & (candidates["semester"] >= min_semester)
        & (candidates["ketersediaan"] == "Tersedia")
        & (candidates["status"] == "Aktif")
    ].sort_values(["cv", "portofolio", "ipk"], ascending=[False, False, False])

    st.markdown(f"**{len(candidates)} kandidat cocok ditemukan:**")
    cols_show = [c for c in ["nim", "nama", "program_studi", "semester", "ipk", "cv", "portofolio", "domisili"] if c in candidates.columns]
    st.dataframe(candidates[cols_show].head(30), use_container_width=True, hide_index=True)

    if not candidates.empty:
        top_candidate = candidates.iloc[0]
        insight(
            f"Kandidat paling siap: **{top_candidate.get('nama', '-')}** (IPK {top_candidate.get('ipk', '-')}, "
            f"dokumen {'lengkap' if top_candidate.get('cv') == 'Ada' and top_candidate.get('portofolio') == 'Ada' else 'belum lengkap'}) "
            f"— prioritaskan untuk dikirim duluan ke perusahaan ini.",
            kind="success",
        )
    else:
        insight(
            "Tidak ada mahasiswa yang cocok dengan kriteria request ini saat ini. "
            "Kemungkinan gap supply-demand di bidang studi tersebut — cek Tab Kesiapan.",
            kind="warning",
        )


# ---------------------------------------------------------------------------
# TAB 6 — LAPORAN PERIODIK & DATA QUALITY  (BT-07, BT-08)
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Laporan Rekapitulasi Placement (BT-07)")

    dim_options = {
        "Program Studi": "program_studi",
        "Perusahaan": "company",
        "Jenis Penempatan": "jenis_penempatan",
    }
    dims_pilihan = st.multiselect("Kelompokkan berdasarkan", list(dim_options.keys()), default=["Program Studi"])

    placement_only = m[m["rejection"] == "Placement"].copy()
    placement_only["semester_laporan"] = placement_only["last_update"].dt.to_period("Q").astype(str)

    if dims_pilihan:
        group_cols = ["semester_laporan"] + [dim_options[d] for d in dims_pilihan]
        recap = placement_only.groupby(group_cols).size().reset_index(name="jumlah_placement")
        recap = recap.sort_values("semester_laporan", ascending=False)
        st.dataframe(recap, use_container_width=True, hide_index=True)

        csv = recap.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh Rekap CSV", csv, "rekap_placement_ssdc2026.csv", "text/csv")

        if not recap.empty:
            top_row = recap.sort_values("jumlah_placement", ascending=False).iloc[0]
            label_parts = " / ".join(str(top_row[dim_options[d]]) for d in dims_pilihan)
            insight(
                f"Kontributor placement terbesar pada rekap ini: **{label_parts}** dengan "
                f"**{int(top_row['jumlah_placement'])}** placement pada periode {top_row['semester_laporan']}.",
            )
    else:
        st.caption("Pilih minimal satu dimensi untuk membuat rekap.")

    st.markdown("---")
    st.subheader("Kualitas & Sinkronisasi Data Mahasiswa (BT-08)")

    merged_check = student_all.merge(status_student[["nim", "sync_date"]], on="nim", how="left", indicator=True)
    belum_sync = merged_check[merged_check["_merge"] == "left_only"]

    latest_ref_date = status_student["sync_date"].max() if status_student["sync_date"].notna().any() else pd.Timestamp.now()
    stale_sync = status_student[
        (latest_ref_date - status_student["sync_date"]).dt.days > SYNC_STALE_DAYS
    ]

    colq1, colq2, colq3 = st.columns(3)
    colq1.metric("Total Mahasiswa (student_all)", f"{student_all['nim'].nunique():,}")
    colq2.metric("Belum Ada Data status_student", f"{len(belum_sync):,}")
    colq3.metric(f"Sync Lebih dari {SYNC_STALE_DAYS} Hari", f"{len(stale_sync):,}")

    if len(belum_sync) > 0:
        cols_show = [c for c in ["nim", "nama", "program_studi"] if c in belum_sync.columns]
        st.caption("Mahasiswa di student_all tapi belum punya record di status_student:")
        st.dataframe(belum_sync[cols_show].head(30), use_container_width=True, hide_index=True)

    insight(
        f"**{len(belum_sync)}** mahasiswa belum tersinkron sama sekali antara student_all dan status_student, "
        f"dan **{len(stale_sync)}** data sudah lebih dari {SYNC_STALE_DAYS} hari sejak sync terakhir. "
        f"Rekomendasi: jadwalkan sinkronisasi rutin mingguan agar data kelayakan (Tab Kesiapan) akurat.",
        kind="warning" if (len(belum_sync) + len(stale_sync)) > 0 else "success",
    )
