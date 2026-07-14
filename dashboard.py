import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="SSDC 2026 — Student Placement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)



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



def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom ke lower-case + strip whitespace agar konsisten dgn ERD."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def _clean_key(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Samakan tipe data (string), strip whitespace, lower-case isi key column."""
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

    # Numerik penting
    if "jumlah_dikirimkan" in tracking_company.columns:
        tracking_company["jumlah_dikirimkan"] = pd.to_numeric(
            tracking_company["jumlah_dikirimkan"], errors="coerce"
        ).fillna(0)
    if "ipk" in status_student.columns:
        status_student["ipk"] = pd.to_numeric(status_student["ipk"], errors="coerce")

    return company, talent_request, tracking_company, tracking_student, student_all, status_student


company, talent_request, tracking_company, tracking_student, student_all, status_student = load_data()



master = tracking_student.merge(
    tracking_company, on="id_tracking_company", how="left", suffixes=("", "_tc")
)
master = master.merge(company, on="id_company", how="left", suffixes=("", "_co"))
master = master.merge(talent_request, on="id_talent_req", how="left", suffixes=("", "_tr"))
master = master.merge(student_all, on="nim", how="left", suffixes=("", "_sa"))
master = master.merge(status_student, on="nim", how="left", suffixes=("", "_ss"))


master["tahun_update"] = master["last_update"].dt.year



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
    st.sidebar.caption(f"Data sync terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}")


tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔻 Funnel", "🤝 Mitra", "🎓 Kesiapan"])


#tab1
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
        color_discrete_sequence=[COLOR_COCOA],
    )
    fig_tren.update_traces(line_color=COLOR_COCOA, marker_color=COLOR_SEAL_BROWN)
    fig_tren.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Placement")
    st.plotly_chart(fig_tren, use_container_width=True)

    if "sync_date" in status_student.columns and status_student["sync_date"].notna().any():
        st.caption(f"ℹ️ Sinkronisasi data terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}")


#tab2
with tab2:
    # Ranking tahap agar bisa dihitung kumulatif (candidate yg berada di tahap k
    # dianggap sudah melewati tahap 0..k)
    stage_rank = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}
    m_funnel = m[m["progress_student"].isin(FUNNEL_STAGES)].copy()
    m_funnel["stage_rank"] = m_funnel["progress_student"].map(stage_rank)

    funnel_counts = []
    for i, stage in enumerate(FUNNEL_STAGES):
        count_reached = (m_funnel["stage_rank"] >= i).sum()
        funnel_counts.append(count_reached)

    fig_funnel = go.Figure(go.Funnel(
        y=FUNNEL_STAGES,
        x=funnel_counts,
        marker={"color": PALETTE_SEQUENTIAL[:len(FUNNEL_STAGES)] if len(PALETTE_SEQUENTIAL) >= len(FUNNEL_STAGES) else COLOR_COCOA},
        textinfo="value+percent initial",
    ))
    fig_funnel.update_layout(title="Funnel Seleksi Kandidat")
    st.plotly_chart(fig_funnel, use_container_width=True)

    total_diproses = len(m)
    total_ghosting = int((m["rejection"] == "Ghosting").sum())
    ghosting_rate = (total_ghosting / total_diproses * 100) if total_diproses > 0 else 0
    st.metric("Ghosting Rate", f"{ghosting_rate:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        ghosting_by_stage = (
            m[m["rejection"] == "Ghosting"]["progress_student"]
            .value_counts().reset_index()
        )
        ghosting_by_stage.columns = ["tahap", "jumlah"]
        fig_gs = px.bar(
            ghosting_by_stage, x="jumlah", y="tahap", orientation="h",
            title="Ghosting berdasarkan Tahap Terakhir",
            color_discrete_sequence=[COLOR_SIENNA],
        )
        fig_gs.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gs, use_container_width=True)

    with col2:
        ghosting_by_company = (
            m[m["rejection"] == "Ghosting"]["company"]
            .value_counts().head(10).reset_index()
        )
        ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
        fig_gc = px.bar(
            ghosting_by_company, x="perusahaan", y="jumlah_ghosting",
            title="Top Perusahaan Kontributor Ghosting Terbanyak",
            color_discrete_sequence=[COLOR_SIENNA],
        )
        st.plotly_chart(fig_gc, use_container_width=True)

    st.markdown("---")
    if not ghosting_by_company.empty:
        top_ghost_company = ghosting_by_company.iloc[0]["perusahaan"]
        top_ghost_stage = (
            ghosting_by_stage.sort_values("jumlah", ascending=False).iloc[0]["tahap"]
            if not ghosting_by_stage.empty else "tidak diketahui"
        )
        st.error(
            f"**Catatan Analis:** Ghosting rate keseluruhan tercatat **{ghosting_rate:.1f}%**, "
            f"dengan titik kehilangan terbanyak pada tahap **{top_ghost_stage}** dan perusahaan "
            f"**{top_ghost_company}** sebagai kontributor terbesar. Disarankan menerapkan SOP "
            f"follow-up maksimal 2x24 jam pasca-tahap tersebut sebelum status diubah menjadi Ghosting."
        )
    else:
        st.info("Belum ada data ghosting pada rentang filter yang dipilih.")


#tab3
with tab3:
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


#tab4
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

    st.markdown("---")

    demand = (
        talent_request["bidang_studi_dibutuhkan"]
        .dropna().str.split(",").explode().str.strip()
        .value_counts().reset_index()
    )
    demand.columns = ["bidang_studi", "jumlah"]
    demand["tipe"] = "Demand"

    supply = student_all["program_studi"].dropna().value_counts().reset_index()
    supply.columns = ["bidang_studi", "jumlah"]
    supply["tipe"] = "Supply"

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

    fig_ipk = px.histogram(
        status_student, x="ipk", nbins=20,
        title="Distribusi IPK Mahasiswa",
        color_discrete_sequence=[COLOR_COCOA],
    )
    st.plotly_chart(fig_ipk, use_container_width=True)
