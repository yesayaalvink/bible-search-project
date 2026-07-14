import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests  # Menggunakan requests untuk mengunduh database & inference
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. ATUR ALAMAT REPOSITORI HUGGING FACE
# ==========================================
REPO_ID = "YesayaAlvink/bible-search-project"
HF_TOKEN = st.secrets["HF_TOKEN"] 

# URL Resmi Baru 2026 (Sistem Router Baru Hugging Face)
API_URL = f"https://router.huggingface.co/hf-inference/models/{REPO_ID}/pipeline/feature-extraction"


# ==========================================
# 2. PROSES MEMUAT DATABASE DARI CLOUD (MENGGUNAKAN REQUESTS - BEBAS CACHE RUSAK)
# ==========================================
@st.cache_resource
def load_database():
    # Unduh file pkl langsung via HTTP Resolve URL (Bypass library HF SDK & Cache rusak)
    url_database = f"https://huggingface.co/{REPO_ID}/resolve/main/database_ta.pkl"
    local_filename = "database_ta.pkl"
    
    # Proses unduh secara bertahap (chunking) agar hemat memori RAM Streamlit
    with requests.get(url_database, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    # Membaca database hasil unduhan bersih
    with open(local_filename, "rb") as f:
        data = pickle.load(f)
    return data["tabel_ayat"], data["vektor_ayat"]

df_alkitab, vektor_seluruh_ayat = load_database()


# ==========================================
# 3. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        # Mengirimkan permintaan langsung ke router baru Hugging Face
        response = requests.post(API_URL, headers=headers, json={"inputs": pertanyaan})
        
        if response.status_code == 200:
            return np.array(response.json())
        elif response.status_code == 503:
            # Model sedang dibangunkan (cold start) di server Hugging Face
            st.warning("Model AI sedang dibangunkan (loading). Silakan tunggu sekitar 15 detik lalu klik Cari lagi.")
            return None
        else:
            st.error(f"Gagal memproses kalimat. Status: {response.status_code}, Detail: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("Terjadi masalah koneksi internet sementara pada server. Silakan klik Cari lagi beberapa saat lagi.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan: {repr(e)}. Silakan coba klik Cari lagi.")
        return None


# ==========================================
# 4. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.title("Pencarian Semantik Alkitab (IndoBERT)")
st.write("Cari ayat berdasarkan makna cerita, bukan sekadar kata kunci.")

pertanyaan = st.text_input("Masukkan pencarian:", placeholder="Contoh: Daniel dilemparkan ke singa")

if st.button("Cari"):
    if pertanyaan:
        with st.spinner("AI sedang mencari ayat yang cocok..."):
            vektor_tanya = get_vektor_pertanyaan(pertanyaan)
            
            if vektor_tanya is not None:
                # Samakan dimensi angka (syarat perhitungan matematika)
                if len(vektor_tanya.shape) == 1:
                    vektor_tanya = vektor_tanya.reshape(1, -1)
                elif len(vektor_tanya.shape) == 3:
                    vektor_tanya = vektor_tanya[0][0].reshape(1, -1)
                    
                # Hitung kemiripan dengan 31.000 ayat
                skor_kemiripan = cosine_similarity(vektor_tanya, vektor_seluruh_ayat)[0]
                
                # Ambil 3 ayat paling mirip
                top_k = 3
                indeks_teratas = np.argsort(skor_kemiripan)[::-1][:top_k]
                
                st.success(f"Ditemukan {top_k} ayat yang paling relevan!")
                
                # Tampilkan hasilnya untuk 3 versi Alkitab sekaligus
                for idx in indeks_teratas:
                    baris = df_alkitab.iloc[idx]
                    skor = skor_kemiripan[idx]
                    
                    with st.expander(f"{baris['kitab']} {baris['pasal']}:{baris['ayat']} (Tingkat kecocokan: {skor:.2f})", expanded=True):
                        st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                        st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                        st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
