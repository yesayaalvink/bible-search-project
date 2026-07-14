import os
# Paksa matikan fitur Xet agar proses unduh lancar di server cloud
os.environ["HF_HUB_DISABLE_XET"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download, InferenceClient

# ==========================================
# 1. ATUR ALAMAT REPOSITORI HUGGING FACE
# ==========================================
# Repositori tunggal yang menampung file model AI dan database_ta.pkl
REPO_ID = "YesayaAlvink/bible-search-project"

# Token rahasia Anda untuk memproses pencarian lewat API
HF_TOKEN = st.secrets["HF_TOKEN"] 

# Inisialisasi client resmi Hugging Face
client = InferenceClient(token=HF_TOKEN)


# ==========================================
# 2. PROSES MEMUAT DATABASE DARI CLOUD
# ==========================================
@st.cache_resource
def load_database():
    # Mengunduh database secara publik (tanpa token) untuk menghindari error Signature
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
        # Meminta representasi vektor kalimat menggunakan model yang sudah di-upload ke repo yang sama
        embedding = client.feature_extraction(
            pertanyaan, 
            model=REPO_ID
        )
        return np.array(embedding)
    except Exception as e:
        st.error(f"Gagal memproses kalimat: {repr(e)}. Silakan coba klik Cari lagi.")
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
