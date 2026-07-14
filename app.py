import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Memuat model langsung di server Streamlit

# ==========================================
# 1. ATUR ALAMAT REPOSITORI GITHUB
# ==========================================
REPO_ID = "YesayaAlvinK/bible-search-project"


# ==========================================
# 2. PROSES MEMUAT DATABASE DARI GITHUB RELEASES
# ==========================================
@st.cache_resource
def load_database():
    # Mengunduh database dari GitHub Releases (Bebas blokir IP CDN, stabil, dan kencang)
    url_database = f"https://github.com/{REPO_ID}/releases/download/v1.0.0/database_ta.pkl"
    local_filename = "database_ta.pkl"
    
    try:
        with requests.get(url_database, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except requests.exceptions.HTTPError as err:
        st.error(f"Gagal mengunduh database dari GitHub Releases. Kode Status: {r.status_code}. URL: {url_database}")
        raise err
                    
    with open(local_filename, "rb") as f:
        data = pickle.load(f)
    return data["tabel_ayat"], data["vektor_ayat"]

df_alkitab, vektor_seluruh_ayat = load_database()


# ==========================================
# 3. PROSES MEMUAT MODEL AI SECARA LOKAL (BYPASS HUGGING FACE TOTAL)
# ==========================================
@st.cache_resource
def load_model():
    model_dir = "local_model"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "1_Pooling"), exist_ok=True)
    
    # Daftar 7 file utama model yang harus didownload dari GitHub Releases
    files_to_download = [
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    try:
        # Download semua file utama secara bertahap agar hemat RAM
        for filename in files_to_download:
            local_path = os.path.join(model_dir, filename)
            if not os.path.exists(local_path):
                url = f"https://github.com/{REPO_ID}/releases/download/v1.0.0/{filename}"
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                
        # Membuat file 1_Pooling/config.json secara otomatis lewat kode
        pooling_config_path = os.path.join(model_dir, "1_Pooling", "config.json")
        if not os.path.exists(pooling_config_path):
            pooling_data = {
                "word_embedding_dimension": 768,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken_tokens": False
            }
            with open(pooling_config_path, "w") as f:
                json.dump(pooling_data, f)
                
    except Exception as err:
        st.error(f"Gagal menyusun folder model lokal di server. Detail: {err}")
        raise err
        
    # Memuat model secara lokal dari folder "local_model" di server Streamlit
    return SentenceTransformer(model_dir)

model = load_model()


# ==========================================
# 4. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        vektor = model.encode([pertanyaan])
        return np.array(vektor)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses kalimat: {repr(e)}. Silakan coba klik Cari lagi.")
        return None


# ==========================================
# 5. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.title("Pencarian Semantik Alkitab (IndoBERT)")
st.write("Cari ayat berdasarkan makna cerita, bukan sekadar kata kunci.")

pertanyaan = st.text_input("Masukkan pencarian:", placeholder="Contoh: Daniel dilemparkan ke singa")

if st.button("Cari"):
    if pertanyaan:
        with st.spinner("AI sedang mencari ayat yang cocok..."):
            vektor_tanya = get_vektor_pertanyaan(pertanyaan)
            
            if vektor_tanya is not None:
                # Samakan dimensi angka
                if len(vektor_tanya.shape) == 1:
                    vektor_tanya = vektor_tanya.reshape(1, -1)
                elif len(vektor_tanya.shape) == 3:
                    vektor_tanya = vektor_tanya[0][0].reshape(1, -1)
                    
                # Hitung kemiripan dengan 31.000 ayat
                skor_kemiripan = cosine_similarity(vektor_tanya, vektor_seluruh_ayat)[0]
                
                top_k = 3
                indeks_teratas = np.argsort(skor_kemiripan)[::-1][:top_k]
                
                st.success(f"Ditemukan {top_k} ayat yang paling relevan!")
                
                for idx in indeks_teratas:
                    baris = df_alkitab.iloc[idx]
                    skor = skor_kemiripan[idx]
                    
                    with st.expander(f"{baris['kitab']} {baris['pasal']}:{baris['ayat']} (Tingkat kecocokan: {skor:.2f})", expanded=True):
                        st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                        st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                        st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
