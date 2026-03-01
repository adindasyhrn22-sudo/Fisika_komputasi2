import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Siswa", layout="wide")
st.title("📊 Dashboard Analisis Hasil Ujian Siswa")
st.markdown("Visualisasi dan analisis data 50 siswa - 20 soal")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")

# Pastikan hanya kolom numerik (skor soal)
data_nilai = df.select_dtypes(include=np.number)

# ==========================================================
# KPI UTAMA
# ==========================================================
st.header("📌 Statistik Umum")

rata2_total = data_nilai.mean(axis=1).mean()
nilai_maks = data_nilai.max().max()
nilai_min = data_nilai.min().min()

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Nilai", f"{rata2_total:.2f}")
col2.metric("🏆 Nilai Tertinggi", nilai_maks)
col3.metric("📉 Nilai Terendah", nilai_min)

st.divider()

# ==========================================================
# DISTRIBUSI NILAI TOTAL
# ==========================================================
st.header("📊 Distribusi Nilai Total Siswa")

total_siswa = data_nilai.sum(axis=1)

fig1, ax1 = plt.subplots()
ax1.hist(total_siswa, bins=10)
ax1.set_title("Histogram Nilai Total")
ax1.set_xlabel("Nilai Total")
ax1.set_ylabel("Jumlah Siswa")

st.pyplot(fig1)

st.divider()

# ==========================================================
# RATA-RATA PER SOAL
# ==========================================================
st.header("📈 Analisis Per Soal")

rata_soal = data_nilai.mean()

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(rata_soal.index, rata_soal.values)
ax2.set_title("Rata-rata Skor per Soal")
ax2.set_xticklabels(rata_soal.index, rotation=45)

st.pyplot(fig2)

soal_tersulit = rata_soal.idxmin()
soal_termudah = rata_soal.idxmax()

st.warning(f"🔥 Soal Tersulit: {soal_tersulit}")
st.success(f"⭐ Soal Termudah: {soal_termudah}")

st.divider()

# ==========================================================
# RANKING SISWA
# ==========================================================
st.header("🏆 Ranking Siswa")

df["Total_Nilai"] = total_siswa
ranking = df.sort_values("Total_Nilai", ascending=False)

st.dataframe(ranking[["Total_Nilai"]].head(10))

st.divider()

# ==========================================================
# SEGMENTASI SISWA
# ==========================================================
st.header("🧠 Segmentasi Kemampuan Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_nilai)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("Cluster")["Total_Nilai"].mean()

fig3, ax3 = plt.subplots()
ax3.bar(cluster_mean.index.astype(str), cluster_mean.values)
ax3.set_title("Rata-rata Nilai per Cluster")
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Rata-rata Nilai")

st.pyplot(fig3)

st.success("📌 Segmentasi siswa berhasil dilakukan")
