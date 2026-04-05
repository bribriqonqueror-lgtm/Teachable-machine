import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="AI Student Profiling - SMAN 1", layout="wide", page_icon="🎓")

# Desain Header
st.markdown("<h1 style='text-align: center;'>🎓 Sistem Rekomendasi Prodi Berbasis AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Identifikasi profil siswa melalui scan wajah untuk saran karir terbaik.</p>", unsafe_allow_html=True)
st.divider()

# 2. Fungsi Memuat Model & Data (Optimasi untuk CSV Semicolon)
@st.cache_resource
def load_resource():
    # Load Model AI
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    
    # Load Label
    class_names = []
    with open("labels.txt", "r") as f:
        for line in f.readlines():
            item = line.strip()
            if item:
                # Ambil nama setelah angka (misal: "0 Brian" -> "Brian")
                parts = item.split(' ', 1)
                class_names.append(parts[1].strip() if len(parts) > 1 else parts[0])
    
    # Load Database CSV (Sesuai format asli: pemisah titik koma ';')
    target_file = "Data_X-2.csv"
    if os.path.exists(target_file):
        df = pd.read_csv(target_file, sep=';')
        # Menyesuaikan nama kolom agar mudah diakses
        df.columns = ['No', 'Nama', 'Kelas', 'Kecerdasan', 'Gaya_Belajar', 'RIASEC']
        return model, class_names, df
    else:
        st.error("❌ Database 'Data_X-2.csv' tidak ditemukan!")
        st.stop()

model, class_names, df_siswa = load_resource()

# 3. Logika Rekomendasi Jurusan/Prodi (Expanded RIASEC)
def get_recommendation(riasec_code):
    if pd.isna(riasec_code) or riasec_code == "":
        return "Umum", "Silakan melakukan tes minat bakat lebih lanjut.", "Semua Jurusan"
    
    code = str(riasec_code).split('-')[0].strip()[0].upper() # Ambil huruf pertama
    
    data = {
        'R': ("Teknik, Ilmu Komputer, Pertanian", "Insinyur, Ahli IT, Pilot", "Teknik Mesin, Elektro, Informatika"),
        'I': ("Kedokteran, MIPA, Psikologi Forensik", "Dokter, Peneliti, Data Scientist", "Kedokteran, Biologi, Kimia"),
        'A': ("DKV, Arsitektur, Seni Musik", "Desainer, Arsitek, Seniman", "Desain Produk, Film & Televisi"),
        'S': ("Keguruan, Ilmu Komunikasi, Psikologi", "Guru, Perawat, Konselor", "Pendidikan, Sosiologi, Humas"),
        'E': ("Manajemen, Bisnis, Hukum", "Pengusaha, Pengacara, Manajer", "Akuntansi, Manajemen Bisnis, Hukum"),
        'C': ("Statistika, Administrasi, Perbankan", "Akuntan, Notaris, Auditor", "Ilmu Perpustakaan, Administrasi Negara")
    }
    return data.get(code, ("Umum", "Konsultasi dengan Guru BK", "Semua Jurusan"))

# 4. Fungsi Prediksi Gambar
def predict_image(img):
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    normalized_image = (img_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image
    
    prediction = model.predict(data)
    idx = np.argmax(prediction)
    return class_names[idx], prediction[0][idx]

# 5. UI: Pemilihan Metode Input
tab1, tab2 = st.tabs(["📸 Gunakan Kamera", "📁 Unggah Foto"])

with tab1:
    input_camera = st.camera_input("Scan Wajah Siswa")
    final_img = input_camera if input_camera else None

with tab2:
    input_file = st.file_uploader("Pilih file foto siswa (JPG/PNG)", type=['jpg', 'png', 'jpeg'])
    if input_file:
        st.image(input_file, caption="Foto Berhasil Diunggah", width=300)
        final_img = input_file

# 6. Menampilkan Hasil & Data Siswa
if final_img:
    with st.spinner('Menganalisis wajah...'):
        image = Image.open(final_img).convert("RGB")
        nama_prediksi, skor = predict_image(image)
        
    if skor > 0.8: # Ambang batas akurasi 80%
        st.success(f"### Wajah Dikenali: **{nama_prediksi}**")
        
        # Cari data di CSV
        data_match = df_siswa[df_siswa['Nama'].str.contains(nama_prediksi, case=False, na=False)]
        
        if not data_match.empty:
            siswa = data_match.iloc[0]
            prodi, profesi, jurusan = get_recommendation(siswa['RIASEC'])
            
            # Tampilan Grid Data
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📌 Profil Akademik")
                st.write(f"**Kelas:** {siswa['Kelas']}")
                st.info(f"**Kecerdasan Utama:**\n{siswa['Kecerdasan']}")
                st.warning(f"**Gaya Belajar:** {siswa['Gaya_Belajar']}")
            
            with col2:
                st.subheader("💡 Rekomendasi Masa Depan")
                st.success(f"**Prodi Cocok:**\n{prodi}")
                st.success(f"**Pilihan Jurusan Kuliah:**\n{jurusan}")
                st.success(f"**Prospek Karir:**\n{profesi}")
                st.write(f"*Berdasarkan Tipe RIASEC: {siswa['RIASEC']}*")
        else:
            st.warning("Wajah terdeteksi, namun data detail siswa tidak ditemukan di CSV.")
    else:
        st.error("Wajah tidak dikenali dengan cukup jelas. Harap coba lagi dengan pencahayaan lebih baik.")

# Footer
st.divider()
st.caption("Aplikasi Klasifikasi Profil Siswa - Dikembangkan untuk SMAN 1")
