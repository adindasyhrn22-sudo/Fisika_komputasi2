import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Dashboard Analisis Hasil Belajar", layout="wide")

st.title("🎓 Dashboard Analisis Hasil Belajar Siswa")
st.write("Dashboard interaktif lengkap untuk analisis performa siswa")

# =========================
# 1️⃣ Upload Data
# =========================
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
    df["Total_Nilai"] = df.sum(axis=1)

    # =========================
    # 2️⃣ KPI
    # =========================
    st.header("📊 Key Performance Indicator")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Siswa", len(df))
    col2.metric("Rata-rata Kelas", round(df["Total_Nilai"].mean(),2))
    col3.metric("Nilai Tertinggi", df["Total_Nilai"].max())
    col4.metric("Nilai Terendah", df["Total_Nilai"].min())

    # =========================
    # 3️⃣ Distribusi Nilai
    # =========================
    st.header("📈 Distribusi Total Nilai")

    fig_hist = px.histogram(df, x="Total_Nilai")
    st.plotly_chart(fig_hist, use_container_width=True)

    # =========================
    # 4️⃣ Analisis Kesulitan Soal
    # =========================
    st.header("📚 Analisis Tingkat Kesulitan Soal")

    mean_per_soal = df.drop(columns=["Total_Nilai"]).mean()
    soal_tersulit = mean_per_soal.idxmin()
    soal_termudah = mean_per_soal.idxmax()

    st.write(f"Soal paling sulit: **{soal_tersulit}**")
    st.write(f"Soal paling mudah: **{soal_termudah}**")

    fig_bar = px.bar(mean_per_soal,
                     labels={"value":"Rata-rata Skor", "index":"Soal"},
                     title="Rata-rata Skor Tiap Soal")
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # 5️⃣ Korelasi
    # =========================
    st.header("🔗 Korelasi Antar Soal")

    corr = df.drop(columns=["Total_Nilai"]).corr()
    fig_corr = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)

    # =========================
    # 6️⃣ Regresi Linear
    # =========================
    st.header("📈 Analisis Regresi Linear")

    X = df.drop(columns=["Total_Nilai"])
    y = df["Total_Nilai"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.write(f"Nilai R² Model: **{round(r2,3)}**")

    coef_df = pd.DataFrame({
        "Soal": X.columns,
        "Koefisien": model.coef_
    }).sort_values(by="Koefisien", ascending=False)

    fig_coef = px.bar(coef_df, x="Soal", y="Koefisien",
                      title="Kontribusi Soal terhadap Total Nilai")
    st.plotly_chart(fig_coef, use_container_width=True)

    # =========================
    # 7️⃣ Clustering
    # =========================
    st.header("🎯 Segmentasi Performa Siswa")

    k = st.slider("Jumlah Cluster", 2, 5, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig_cluster = px.scatter(df,
                             x="Total_Nilai",
                             y=df.index,
                             color="Cluster")
    st.plotly_chart(fig_cluster, use_container_width=True)

    # =========================
    # 8️⃣ Top 5 Siswa
    # =========================
    st.header("🏆 Top 5 Siswa")

    top5 = df.sort_values(by="Total_Nilai", ascending=False).head()
    st.dataframe(top5)

else:
    st.info("Silakan upload file Excel terlebih dahulu.")