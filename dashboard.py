
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="SSDC 2026 — Student Placement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLOR_SIENNA = "#872408"       # merah-coklat gelap — warning/negatif
COLOR_COCOA = "#E2782F"        # oranye — aksen utama
COLOR_JASMINE = "#F7D475"      # kuning — highlight/positif
COLOR_DRAB_DARK = "#403314"    # coklat gelap — teks/netral gelap
COLOR_SEAL_BROWN = "#4A230E"   # coklat sangat gelap — background aksen

PALETTE_SEQUENTIAL = [COLOR_JASMINE, COLOR_COCOA, COLOR_SIENNA, COLOR_SEAL_BROWN, COLOR_DRAB_DARK]


def load_data():
   company = pd.read_csv("cleaned_company.csv", parse_dates=["created_at"])
status_student = pd.read_csv("cleaned_status_student.csv", parse_dates=["sync_date"])
student_all = pd.read_csv("cleaned_student_all.csv")
talent_request = pd.read_csv("cleaned_talent_request.csv", parse_dates=["request_date"])
tracking_company = pd.read_csv("cleaned_tracking_company.csv", parse_dates=["request_date", "send_date"])
tracking_student = pd.read_csv("cleaned_tracking_student.csv", parse_dates=["last_update"])

    
    for df, col in [
        (company, "id_company"), (talent_request, "id_talent_req"), (talent_request, "id_company"),
        (tracking_company, "id_tracking_company"), (tracking_company, "id_talent_req"),
        (tracking_company, "id_company"), (tracking_student, "id_tracking_student"),
        (tracking_student, "id_tracking_company"), (student_all, "NIM"),
        (status_student, "NIM"), (tracking_student, "NIM"),
    ]:
        df[col] = df[col].astype(str).str.strip()

    return company, status_student, student_all, talent_request, tracking_company, tracking_student


company, status_student, student_all, talent_request, tracking_company, tracking_student = load_data()


st.sidebar.title("🍂 SSDC 2026")
st.sidebar.markdown("**Filter Laporan**")

tracking_student["tahun_update"] = tracking_student["last_update"].dt.year
tahun_list = sorted(tracking_student["tahun_update"].dropna().unique().tolist())
tahun_pilihan = st.sidebar.multiselect("Tahun", tahun_list, default=tahun_list)

prodi_list = sorted(student_all["program_studi"].dropna().unique().tolist())
prodi_pilihan = st.sidebar.multiselect("Program Studi", prodi_list, default=[])

jenis_list = sorted(talent_request["jenis_penempatan"].dropna().unique().tolist())
jenis_pilihan = st.sidebar.multiselect("Jenis Penempatan", jenis_list, default=[])

ts_filtered = tracking_student[tracking_student["tahun_update"].isin(tahun_pilihan)].copy()


if prodi_pilihan:
    nim_prodi = student_all[student_all["program_studi"].isin(prodi_pilihan)]["NIM"]
    ts_filtered = ts_filtered[ts_filtered["NIM"].isin(nim_prodi)]


if jenis_pilihan:
    ts_filtered = ts_filtered[ts_filtered["jenis_penempatan"].isin(jenis_pilihan)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Data sync terakhir: {status_student['sync_date'].max().strftime('%d %B %Y')}")


tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔻 Funnel", "🤝 Mitra", "🎓 Kesiapan"])


with tab1:
    st.header("Ringkasan Program Placement")

    total_company = company["id_company"].nunique()
    total_mahasiswa = student_all["NIM"].nunique()
    total_placement = (ts_filtered["rejection"] == "Placement").sum()
    total_diproses = len(ts_filtered)
    success_rate = (total_placement / total_diproses * 100) if total_diproses > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Perusahaan Mitra", f"{total_company:,}")
    col2.metric("Total Mahasiswa Terdaftar", f"{total_mahasiswa:,}")
    col3.metric("Total Placement Berhasil", f"{total_placement:,}")
    col4.metric("Success Rate", f"{success_rate:.1f}%")

    st.markdown("---")

   
    placement_df = ts_filtered[ts_filtered["rejection"] == "Placement"].copy()
    placement_df["bulan"] = placement_df["last_update"].dt.to_period("M").astype(str)
    tren = placement_df.groupby("bulan").size().reset_index(name="jumlah_placement")

    fig_tren = px.line(
        tren, x="bulan", y="jumlah_placement", markers=True,
        title="Tren Placement Berhasil per Bulan",
        color_discrete_sequence=[COLOR_COCOA],
    )
    fig_tren.update_layout(xaxis_title="Bulan", yaxis_title="Jumlah Placement")
    st.plotly_chart(fig_tren, use_container_width=True)


with tab2:
    st.header("Alur Seleksi & Titik Kehilangan Kandidat")

    st.caption(
        "Catatan metodologi: data progress_student merepresentasikan status TERKINI "
        "tiap kandidat (snapshot), bukan riwayat lengkap tiap tahap. Distribusi di bawah "
        "menunjukkan sebaran status saat ini, bukan funnel kumulatif historis."
    )

    stage_order = [
        "Selecting Student by Company", "Study Case", "CDC Briefing Student",
        "Interview User", "Final Interview", "Placement",
        "FU 1", "FU 2", "FU 3", "Ghosting", "Rejected", "Finish",
    ]
    dist = ts_filtered["progress_student"].value_counts().reindex(stage_order).fillna(0).reset_index()
    dist.columns = ["tahap", "jumlah"]

    fig_dist = px.bar(
        dist, x="jumlah", y="tahap", orientation="h",
        title="Distribusi Status Kandidat Saat Ini",
        color_discrete_sequence=[COLOR_COCOA],
    )
    fig_dist.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        ghosting_rate = (ts_filtered["rejection"] == "Ghosting").mean() * 100
        st.metric("Ghosting Rate", f"{ghosting_rate:.1f}%")

        ghosting_by_company = (
            ts_filtered[ts_filtered["rejection"] == "Ghosting"]["company"]
            .value_counts().head(10).reset_index()
        )
        ghosting_by_company.columns = ["perusahaan", "jumlah_ghosting"]
        fig_gc = px.bar(
            ghosting_by_company, x="jumlah_ghosting", y="perusahaan", orientation="h",
            title="Top 10 Perusahaan dengan Ghosting Terbanyak",
            color_discrete_sequence=[COLOR_SIENNA],
        )
        fig_gc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gc, use_container_width=True)

    with col2:
        ghosting_by_stage = (
            ts_filtered[ts_filtered["progress_student"] == "Ghosting"]["jenis_penempatan"]
            .value_counts().reset_index()
        )
        ghosting_by_stage.columns = ["jenis_penempatan", "jumlah"]
        fig_gs = px.pie(
            ghosting_by_stage, values="jumlah", names="jenis_penempatan",
            title="Ghosting berdasarkan Jenis Penempatan",
            color_discrete_sequence=PALETTE_SEQUENTIAL,
        )
        st.plotly_chart(fig_gs, use_container_width=True)

        if not ghosting_by_company.empty:
            top_ghost_company = ghosting_by_company.iloc[0]["perusahaan"]
            st.info(
                f"**Catatan Analis:** Ghosting rate keseluruhan sebesar **{ghosting_rate:.1f}%**. "
                f"Perusahaan **{top_ghost_company}** tercatat sebagai kontributor ghosting terbanyak. "
                "Perlu SOP follow-up yang lebih ketat untuk perusahaan dengan pola ghosting berulang."
            )


with tab3:
    st.header("Performa Perusahaan Mitra")

    # Acceptance rate per perusahaan
    perf = ts_filtered.groupby("company").agg(
        total=("id_tracking_student", "count"),
        placement=("rejection", lambda x: (x == "Placement").sum()),
    ).reset_index()
    perf["acceptance_rate"] = (perf["placement"] / perf["total"] * 100).round(1)
    perf = perf[perf["total"] >= 3].sort_values("acceptance_rate", ascending=False).head(10)

    col1, col2 = st.columns(2)

    with col1:
        fig_acc = px.bar(
            perf, x="acceptance_rate", y="company", orientation="h",
            title="Top 10 Perusahaan — Acceptance Rate (min. 3 kandidat diproses)",
            color_discrete_sequence=[COLOR_COCOA],
        )
        fig_acc.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        volume = talent_request["nama_perusahaan"].value_counts().head(10).reset_index()
        volume.columns = ["perusahaan", "jumlah_request"]
        fig_vol = px.bar(
            volume, x="jumlah_request", y="perusahaan", orientation="h",
            title="Top 10 Perusahaan — Volume Talent Request",
            color_discrete_sequence=[COLOR_JASMINE],
        )
        fig_vol.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_vol, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        jenis_dist = talent_request["jenis_penempatan"].value_counts().reset_index()
        jenis_dist.columns = ["jenis_penempatan", "jumlah"]
        fig_jenis = px.pie(
            jenis_dist, values="jumlah", names="jenis_penempatan",
            title="Proporsi Jenis Penempatan Diminta",
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


with tab4:
    st.header("Kesiapan & Kelayakan Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        ketersediaan_dist = status_student["ketersediaan"].value_counts().reset_index()
        ketersediaan_dist.columns = ["status", "jumlah"]
        fig_ket = px.pie(
            ketersediaan_dist, values="jumlah", names="status",
            title="Distribusi Status Ketersediaan Mahasiswa",
            color_discrete_sequence=PALETTE_SEQUENTIAL,
        )
        st.plotly_chart(fig_ket, use_container_width=True)

    with col2:
        cv_ada = (status_student["CV"] == "Ada").sum()
        cv_tidak = (status_student["CV"] == "Tidak Ada").sum()
        porto_ada = (status_student["portofolio"] == "Ada").sum()
        porto_tidak = (status_student["portofolio"] == "Tidak Ada").sum()
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
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .reset_index()
    )
    demand.columns = ["bidang_studi", "demand"]

    supply = student_all["program_studi"].value_counts().reset_index()
    supply.columns = ["bidang_studi", "supply"]

    gap = pd.merge(demand, supply, on="bidang_studi", how="outer").fillna(0)
    gap = gap.sort_values("demand", ascending=False).head(10)
    gap_melt = gap.melt(id_vars="bidang_studi", value_vars=["demand", "supply"], var_name="tipe", value_name="jumlah")

    fig_gap = px.bar(
        gap_melt, x="jumlah", y="bidang_studi", color="tipe", orientation="h", barmode="group",
        title="Gap: Bidang Studi Dibutuhkan (Demand) vs Tersedia (Supply)",
        color_discrete_map={"demand": COLOR_SIENNA, "supply": COLOR_JASMINE},
    )
    fig_gap.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_gap, use_container_width=True)

    # Distribusi IPK
    fig_ipk = px.histogram(
        status_student, x="IPK", nbins=20,
        title="Distribusi IPK Mahasiswa",
        color_discrete_sequence=[COLOR_COCOA],
    )
    st.plotly_chart(fig_ipk, use_container_width=True)
