import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# 1. Konfigurasi Halaman (Layout Wide agar bisa dibagi kolom)
st.set_page_config(page_title="AI Profiling SMAN 1", layout="wide", page_icon="🎓")

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Fungsi Memuat Model & Data (Optimasi untuk CSV Semicolon)
@st.cache_resource
def load_resource():
    # Load Model AI (Pastikan file ini sudah ada di GitHub)
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    
    # Load Label
    class_names = []
    with open("labels.txt", "r") as f:
        for line in f.readlines():
            item = line.strip()
            if item:
                # Menghapus angka di depan nama (misal "0 Brian" jadi "Brian")
                parts = item.split(' ', 1)
                class_names.append(parts[1].strip() if len(parts) > 1 else parts[0])
    
    # Load Database CSV (Pemisah Titik Koma)
    df = pd.read_csv("Data_X-2.csv", sep=";")
    return model, class_names, df

model, class_names, df_siswa = load_resource()

# 3. Logika Rekomendasi Karir (RIASEC Mapping)
def get_recommendation(riasec_code):
    riasec_code = str(riasec_code).upper()
    mapping = {
        "R": ("Teknik, Robotik, Pertanian", "Engineer, Arsitek, Teknisi", "Teknik Mesin, Teknik Sipil"),
        "I": ("Sains, Riset, Kedokteran", "Peneliti, Dokter, Data Scientist", "Kedokteran, Fisika, Informatika"),
        "A": ("Desain, Seni, Komunikasi", "Desainer, Penulis, Seniman", "DKV, Arsitektur, Sastra"),
        "S": ("Pendidikan, Psikologi, Hukum", "Guru, Psikolog, Pengacara", "Psikologi, Hukum, Keguruan"),
        "E": ("Bisnis, Manajemen, Politik", "Entrepreneur, Manager, Diplomat", "Manajemen, Ekonomi, Hubungan Internasional"),
        "C": ("Akuntansi, Administrasi, Data", "Akuntan, Notaris, Auditor", "Akuntansi, Perpajakan, Statistika")
    }
    
    # Ambil huruf pertama dari kode RIASEC (Misal 'R-I' ambil 'R')
    first_letter = riasec_code[0] if riasec_code else "S"
    return mapping.get(first_letter, ("Umum", "Konsultan", "Ilmu Komunikasi"))

# 4. Fungsi Prediksi Gambar
def predict_image(img):
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.empty((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

# --- TAMPILAN UTAMA ---
st.title("🎓 AI Student Profiling - SMAN 1 Balikpapan")
st.write("Arahkan wajah ke kamera untuk melihat profil akademik dan saran karir secara instan.")
st.divider()

# Layout Kolom: Kiri (1 bagian) untuk Kamera, Kanan (2 bagian) untuk Hasil
col_kamera, col_hasil = st.columns([1, 2])

with col_kamera:
    st.subheader("📷 Webcam Scan")
    input_camera = st.camera_input("Posisikan wajah di tengah frame")

with col_hasil:
    st.subheader("📊 Hasil Analisis Profil")
    
    if input_camera:
        with st.spinner('Sedang mencocokkan wajah dengan database...'):
            image = Image.open(input_camera).convert("RGB")
            nama_prediksi, skor = predict_image(image)
        
        # Ambang batas akurasi 80%
        if skor > 0.8:
            st.success(f"**Identitas Terdeteksi:** {nama_prediksi} ({skor*100:.1f}%)")
            
            # Cari data di CSV
            data_match = df_siswa[df_siswa['NAMA PESERTA DIDIK'].str.contains(nama_prediksi, case=False, na=False)]
            
            if not data_match.empty:
                siswa = data_match.iloc[0]
                # Ambil kolom sesuai nama di CSV kamu
                riasec = siswa['PEMETAAN MINAT DAN BAKAT DENGAN METODE RIASEC']
                kecerdasan = siswa['TES KECERDASAN MAJEMUK']
                gaya_belajar = siswa['ANGKET GAYA BELAJAR']
                
                prodi, profesi, jurusan = get_recommendation(riasec)
                
                # Baris 1: Informasi Dasar (Metric agar besar dan menonjol)
                m1, m2, m3 = st.columns(3)
                m1.metric("Gaya Belajar", gaya_belajar)
                m2.metric("Tipe RIASEC", riasec)
                m3.metric("Kecerdasan", "Dominan")
                
                st.info(f"💡 **Detail Kecerdasan:** {kecerdasan}")
                
                # Baris 2: Rekomendasi Masa Depan
                st.markdown("---")
                st.subheader("🚀 Rekomendasi Karir & Pendidikan")
                
                res1, res2 = st.columns(2)
                with res1:
                    st.write("**Rumpun Prodi Cocok:**")
                    st.warning(prodi)
                    st.write("**Pilihan Jurusan Kuliah:**")
                    st.warning(jurusan)
                
                with res2:
                    st.write("**Prospek Profesi:**")
                    st.success(profesi)
                    st.write("**Catatan:**")
                    st.caption("Hasil ini didasarkan pada integrasi data psikologi RIASEC dengan pengenalan wajah berbasis AI.")
            else:
                st.error("Wajah dikenali, namun data profil di CSV tidak ditemukan. Pastikan nama di label dan CSV sama.")
        else:
            st.warning("Wajah kurang jelas atau tidak terdaftar. Coba atur pencahayaan atau posisi wajah.")
    else:
        st.info("Menunggu input kamera... Silakan nyalakan kamera di sebelah kiri.")

st.divider()
st.caption("© 2026 Kelompok 6 - Informatika SMAN 1 Balikpapan")
