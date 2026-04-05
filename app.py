import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="AI Profil Siswa SMAN 1", layout="wide")
st.title("🎓 Sistem Rekomendasi Karir & Prodi")
st.write("Selamat datang! Silakan scan wajah untuk melihat profil dan rekomendasi.")

# 2. Fungsi Memuat Model & Data (Anti-Gagal)
@st.cache_resource
def load_resource():
    # Load Model AI
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    
    # Load Label (PASTIKAN MENJOROK KE DALAM SEPERTI INI)
    with open("labels.txt", "r") as f:
        class_names = []
        for line in f.readlines():
            item = line.strip()
            if item: # Hanya ambil baris yang ada isinya
                class_names.append(item)
    
    # Mencari File CSV (Cek otomatis nama file)
    target_file = "Data_X-2.csv"
    if not os.path.exists(target_file):
        # Jika nama file tidak pas, cari file apa saja yang berakhiran .csv
        all_files = os.listdir('.')
        csv_files = [f for f in all_files if f.endswith('.csv')]
        if csv_files:
            target_file = csv_files[0] 
        else:
            st.error("❌ File CSV tidak ditemukan di GitHub!")
            st.stop()
# Versi lebih kuat: Mengabaikan baris yang kolomnya berantakan
    df = pd.read_csv(target_file, on_bad_lines='skip', sep=None, engine='python')
    
    # Bersihkan nama kolom agar seragam
    # Sesuaikan urutan kolom: No(0), Nama(1), Kelas(2), Kecerdasan(3), Gaya(4), RIASEC(5)
    df = df.iloc[:, [1, 2, 3, 4, 5]]
    df.columns = ['Nama', 'Kelas', 'Kecerdasan', 'Gaya_Belajar', 'RIASEC']
    
    # Baris return ini harus sejajar dengan kode di dalam def
    return model, class_names, df

# Eksekusi Pemuatan
try:
    model, class_names, df_siswa = load_resource()
    st.sidebar.success("✅ Berhasil terhubung dengan Database Siswa")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
    st.stop()

# 3. Logika Pemetaan Karir Berdasarkan RIASEC
def get_recommendation(ria_code):
    first_letter = str(ria_code)[0].upper() if pd.notna(ria_code) else "U"
    
    dict_recom = {
        'R': ("Teknik, Robotika, Pertanian", "Insinyur, Pilot, Ahli Pertanian"),
        'I': ("Kedokteran, Data Science, MIPA", "Peneliti, Dokter, Programmer"),
        'A': ("DKV, Seni, Sastra, Arsitektur", "Desainer, Penulis, Seniman"),
        'S': ("Psikologi, Hukum, Keguruan", "Guru, Konselor, Pengacara"),
        'E': ("Manajemen, Bisnis, Komunikasi", "Pengusaha, Politikus, Manajer"),
        'C': ("Akuntansi, Statistik, Perbankan", "Akuntan, Auditor, Administrator")
    }
    return dict_recom.get(first_letter, ("Umum", "Silakan konsultasi dengan Guru BK"))

# 4. Fitur Kamera
foto = st.camera_input("Silakan Ambil Foto Wajah Siswa")

if foto:
    # Proses Gambar untuk AI
    image = Image.open(foto).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prediksi Wajah
    prediction = model.predict(data)
    index = np.argmax(prediction)
    score = prediction[0][index]
    
    # Mengambil nama dari label (memotong angka di depan misal "0 Brian" jadi "Brian")
    nama_terdeteksi = class_names[index].split(' ', 1)[-1].strip()

    if score > 0.7:
        st.success(f"Wajah Dikenali: **{nama_terdeteksi}** (Akurasi: {score*100:.1f}%)")
        
        # Cari Data Siswa di Database
        match = df_siswa[df_siswa['Nama'].str.contains(nama_terdeteksi, case=False, na=False)]
        
        if not match.empty:
            data_siswa = match.iloc[0]
            prodi, karir = get_recommendation(data_siswa['RIASEC'])
            
            # Tampilan Hasil
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📌 Profil Siswa")
                st.write(f"**Nama:** {data_siswa['Nama']}")
                st.write(f"**Gaya Belajar:** {data_siswa['Gaya_Belajar']}")
                st.write(f"**Dominasi Kecerdasan:** {data_siswa['Kecerdasan']}")
            with col2:
                st.subheader("💡 Rekomendasi Karir")
                st.success(f"**Saran Prodi:** {prodi}")
                st.success(f"**Saran Profesi:** {karir}")
        else:
            st.warning(f"Nama '{nama_terdeteksi}' tidak ditemukan di CSV. Pastikan nama di labels.txt sama dengan di file CSV.")
    else:
        st.error("Wajah tidak terdaftar atau pencahayaan kurang baik.")
