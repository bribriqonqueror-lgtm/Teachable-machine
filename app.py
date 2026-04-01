import streamlit as st
import pandas as pd

# 1. Judul Aplikasi
st.set_page_config(page_title="Sistem AI SMAN 1", layout="wide")
st.title("🎓 Sistem Rekomendasi Karir SMAN 1 Balikpapan")

# 2. Fungsi Baca CSV (Sudah disesuaikan dengan baris kosong di filemu)
@st.cache_data
def load_data():
    # Filemu butuh skip 11 baris karena ada Kop Surat
    df = pd.read_csv('Data_X-2.csv', skiprows=11)
    # Ambil kolom yang penting saja
    df = df.iloc[:, [1, 2, 3, 4, 5]] 
    df.columns = ['Nama', 'Kelas', 'Kecerdasan', 'Gaya_Belajar', 'RIASEC']
    # Hapus baris yang Namanya kosong
    df = df.dropna(subset=['Nama'])
    return df

# 3. Menjalankan Website
try:
    df_siswa = load_data()
    st.success("✅ Data Siswa Berhasil Dimuat!")
    
    # Pilih Nama Siswa (Sebagai pengganti Face Recognition sementara)
    nama_siswa = st.selectbox("Cari Nama Siswa:", df_siswa['Nama'].unique())
    
    if nama_siswa:
        data = df_siswa[df_siswa['Nama'] == nama_siswa].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("📌 Profil Akademik")
            st.write(f"**Nama:** {data['Nama']}")
            st.write(f"**Kelas:** {data['Kelas']}")
            st.write(f"**Gaya Belajar:** {data['Gaya_Belajar']}")
        
        with col2:
            st.success("💡 Rekomendasi")
            st.write(f"**Kode RIASEC:** {data['RIASEC']}")
            st.write("Sistem siap dihubungkan dengan AI Face Recognition.")

except Exception as e:
    st.error(f"Ada masalah: {e}")