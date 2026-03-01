import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Analisis Hasil Simulasi Siswa", layout="wide")
st.title("📊 Dashboard Analisis Hasil Simulasi Siswa")
st.markdown("Analisis performa siswa berdasarkan 20 butir soal simulasi.")

# ==========================================================
# 2. LOAD DATA
# ==========================================================
# Menyesuaikan dengan file CSV yang diunggah
df = pd.read_csv("data_simulasi_50_siswa_20_soal.xlsx - Sheet1.csv")

# Mengambil semua kolom (Soal_1 sampai Soal_20)
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# 3. KPI PERFORMA (IKM VERSI SISWA)
# ==========================================================
# Rata-rata skor dari seluruh soal (Skala 1-4)
mean_scores = indikator.mean()
skor_global = (mean_scores.mean() / 4) * 100  # Konversi ke persen

def kategori_performa(x):
    if x >= 85: return "Sangat Tinggi"
    elif x >= 70: return "Tinggi"
    elif x >= 55: return "Cukup"
    else: return "Perlu Bimbingan"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rata-rata Skor Global", f"{mean_scores.mean():.2f} / 4.0")
with col2:
    st.metric("Persentase Capaian", f"{skor_global:.1f}%")
with col3:
    st.subheader(f"Status: {kategori_performa(skor_global)}")

st.divider()

# ==========================================================
# 4. ANALISIS PER SOAL (BAR CHART)
# ==========================================================
st.header("📈 Analisis Performa per Butir Soal")
fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
mean_scores.plot(kind="bar", color="skyblue", ax=ax_bar)
ax_bar.set_ylabel("Rata-rata Skor")
ax_bar.set_xlabel("Nomor Soal")
ax_bar.set_ylim(0, 4)
st.pyplot(fig_bar)

# ==========================================================
# 5. SEGMENTASI SISWA (K-MEANS CLUSTERING)
# ==========================================================
st.header("👥 Segmentasi Kemampuan Siswa")
st.info("Mengelompokkan siswa ke dalam 3 level kemampuan berdasarkan pola jawaban.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Menentukan label cluster berdasarkan rata-rata total
cluster_profile = df.groupby("Cluster").mean().mean(axis=1).sort_values(ascending=False)
mapping = {
    cluster_profile.index[0]: "Kemampuan Tinggi",
    cluster_profile.index[1]: "Kemampuan Sedang",
    cluster_profile.index[2]: "Perlu Intervensi"
}
df["Segmentasi"] = df["Cluster"].map(mapping)

# Visualisasi Distribusi Cluster
col_left, col_right = st.columns(2)

with col_left:
    st.write("### Jumlah Siswa per Segmen")
    st.table(df["Segmentasi"].value_counts())

with col_right:
    fig_pie, ax_pie = plt.subplots()
    df["Segmentasi"].value_counts().plot(kind="pie", autopct='%1.1f%%', colors=["#2ecc71", "#f1c40f", "#e74c3c"], ax=ax_pie)
    ax_pie.set_ylabel("")
    st.pyplot(fig_pie)

# ==========================================================
# 6. DETAIL DATA
# ==========================================================
st.header("📑 Data Mentah & Hasil Segmentasi")
st.dataframe(df)
