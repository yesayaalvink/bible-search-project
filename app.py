import os
# PAKSA MATIKAN FITUR XET YANG SERING MACET/HANG DI SERVER CLOUD
os.environ["HF_HUB_DISABLE_XET"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download, InferenceClient

# GANTI DENGAN DETAIL AKUN HUGGING FACE ANDA
NAMA_AKUN_HF = "YesayaAlvink" 
NAMA_MODEL_HF = "bible-search-project"  # Pastikan nama model ini sudah persis dengan yang di HF
HF_TOKEN = st.secrets["HF_TOKEN"] 

# Inisialisasi client resmi Hugging Face
client = InferenceClient(token=HF_TOKEN)

# 2. Otomatis mengunduh database_ta.pkl dari Hugging Face ke server Streamlit
@st.cache_resource
def load_database():
    file_path = hf_hub_download(
        repo_id=f"{NAMA_AKUN_HF}/{NAMA_MODEL_HF}",
        filename="database_ta.pkl",
        token=HF_TOKEN
    )
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data["tabel_ayat"], data["vektor_ayat"]

df_alkitab, vektor_seluruh_ayat = load_database()

# Fungsi baru untuk meminta vektor dari Hugging Face secara resmi lewat router terbaru
def get_vektor_pertanyaan(pertanyaan):
    try:
        embedding = client.feature_extraction(
            pertanyaan, 
            model=f"{NAMA_AKUN_HF}/{NAMA_MODEL_HF}"
        )
        return np.array(embedding)
    except Exception as e:
        st.error(f"Gagal memproses kalimat: {e}. Silakan tunggu beberapa saat lalu coba klik Cari lagi.")
        return None

# 3. Tampilan Aplikasi
st.title("Pencarian Semantik Alkitab (IndoBERT)")
st.write("Cari ayat berdasarkan makna cerita, bukan sekadar kata kunci.")

pertanyaan = st.text_input("Masukkan pencarian:", placeholder="Contoh: Daniel dilemparkan ke singa")

if st.button("Cari"):
    if pertanyaan:
        with st.spinner("AI sedang mencari ayat yang cocok..."):
            # Ubah pertanyaan jadi angka vektor
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
                indeks_teratas = np.argsort(skor_kemiripan)[::-1][:3]
                
                st.success("Ditemukan 3 ayat yang paling relevan!")
                
                # Tampilkan hasilnya
                for idx in indeks_teratas:
                    baris = df_alkitab.iloc[idx]
                    skor = skor_kemiripan[idx]
                    
                    with st.expander(f"{baris['kitab']} {baris['pasal']}:{baris['ayat']} (Tingkat kecocokan: {skor:.2f})", expanded=True):
                        st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                        st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                        st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
