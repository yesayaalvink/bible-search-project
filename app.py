import os
# ==========================================
# TRIK RAHASIA UNTUK MENGAKALI BUG TOKEN HUGGING FACE
# ==========================================
# 1. Ambil nilai token Anda secara aman di awal
import streamlit as st
HF_TOKEN_VAL = st.secrets["HF_TOKEN"]

# 2. Hapus token dari memori OS agar Hugging Face tidak membacanya secara otomatis
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.pop("HUGGINGFACE_TOKEN", None)

# 3. Matikan fitur Xet dan token implisit agar proses unduh publik 100% anonim
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 4. Baru kita impor library setelah memori dibersihkan
import pandas as pd
import numpy as np
import pickle
import requests  # Menggunakan requests standar untuk menghindari bug routing HF
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

# ==========================================
# 1. ATUR ALAMAT REPOSITORI HUGGING FACE
# ==========================================
# Menggunakan repositori tunggal milik akun Anda
REPO_ID = "YesayaAlvink/bible-search-project"

# URL Resmi Baru 2026 (Sistem Router Baru Hugging Face)
API_URL = f"https://router.huggingface.co/hf-inference/models/{REPO_ID}/pipeline/feature-extraction"


# ==========================================
# 2. PROSES MEMUAT DATABASE DARI CLOUD
# ==========================================
@st.cache_resource
def load_database():
    # Mengunduh database secara publik (pasti berhasil karena memori token sudah dibersihkan)
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="database_ta.pkl"
    )
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["tabel_ayat"], data["vektor_ayat"]

df_alkitab, vektor_seluruh_ayat = load_database()


# ==========================================
# 3. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        # Kirim token secara manual lewat header HTTP (ini aman & tidak memicu bug)
        headers = {"Authorization": f"Bearer {HF_TOKEN_VAL}"}
        response = requests.post(API_URL, headers=headers, json={"inputs": pertanyaan})
        
        if response.status_code == 200:
            return np.array(response.json())
        elif response.status_code == 503:
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
