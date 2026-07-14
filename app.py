import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Memuat model langsung di server Streamlit
import plotly.express as px  # Library grafik interaktif

# ==========================================
# 1. ATUR ALAMAT REPOSITORI GITHUB
# ==========================================
REPO_ID = "YesayaAlvinK/bible-search-project"


# ==========================================
# 2. PROSES MEMUAT DATABASE DARI GITHUB RELEASES
# ==========================================
@st.cache_resource
def load_database():
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
# 3. PROSES MEMUAT MODEL AI DI SERVER STREAMLIT
# ==========================================
@st.cache_resource
def load_model():
    model_dir = "local_model"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "1_Pooling"), exist_ok=True)
    
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
                                
        pooling_config_path = os.path.join(model_dir, "1_Pooling", "config.json")
        if not os.path.exists(pooling_config_path):
            pooling_data = {
                "word_embedding_dimension": 768,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False
            }
            with open(pooling_config_path, "w") as f:
                json.dump(pooling_data, f)
                
    except Exception as err:
        st.error(f"Gagal menyusun folder model lokal di server. Detail: {err}")
        raise err
        
    return SentenceTransformer(model_dir)

model = load_model()


# ==========================================
# 4. HITUNG PROYEKSI KOORDINAT 2D (PCA) SECARA CACHED
# ==========================================
@st.cache_resource
def hitung_proyeksi_pca(vektor_ayat):
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=2)
    koordinat_2d = pca_model.fit_transform(vektor_ayat)
    return koordinat_2d, pca_model

koordinat_2d, pca_model = hitung_proyeksi_pca(vektor_seluruh_ayat)
df_alkitab['x'] = koordinat_2d[:, 0]
df_alkitab['y'] = koordinat_2d[:, 1]


# ==========================================
# 5. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        vektor = model.encode([pertanyaan])
        return np.array(vektor)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses kalimat: {repr(e)}. Silakan coba klik Cari lagi.")
        return None


# ==========================================
# 6. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="Pencarian Semantik Alkitab", layout="wide")
st.title("📖 Pencarian Semantik Alkitab (IndoBERT)")
st.write("Cari ayat berdasarkan makna cerita, bukan sekadar kata kunci.")

# --- BARIS FILTER DROP-DOWN ---
st.markdown("### 🔍 Penyaringan Metadata (Opsional)")
col1, col2 = st.columns(2)

# Mengambil urutan kitab unik untuk membagi PL dan PB secara dinamis
kitab_unik = list(df_alkitab['kitab'].unique())
kitab_pl = kitab_unik[:39]
kitab_pb = kitab_unik[39:]

with col1:
    perjanjian_filter = st.selectbox(
        "Pilih Bagian Alkitab:", 
        ["Seluruh Alkitab", "Perjanjian Lama (PL)", "Perjanjian Baru (PB)"]
    )

with col2:
    if perjanjian_filter == "Perjanjian Lama (PL)":
        pilihan_kitab = ["Semua Kitab PL"] + kitab_pl
    elif perjanjian_filter == "Perjanjian Baru (PB)":
        pilihan_kitab = ["Semua Kitab PB"] + kitab_pb
    else:
        pilihan_kitab = ["Semua Kitab"] + kitab_unik
        
    kitab_filter = st.selectbox("Pilih Kitab Spesifik:", pilihan_kitab)

st.markdown("---")

# Mempersiapkan mask penyaringan data
if perjanjian_filter == "Perjanjian Lama (PL)":
    if kitab_filter == "Semua Kitab PL":
        mask_filter = df_alkitab['kitab'].isin(kitab_pl)
    else:
        mask_filter = df_alkitab['kitab'] == kitab_filter
elif perjanjian_filter == "Perjanjian Baru (PB)":
    if kitab_filter == "Semua Kitab PB":
        mask_filter = df_alkitab['kitab'].isin(kitab_pb)
    else:
        mask_filter = df_alkitab['kitab'] == kitab_filter
else:
    if kitab_filter == "Semua Kitab":
        mask_filter = pd.Series([True] * len(df_alkitab))
    else:
        mask_filter = df_alkitab['kitab'] == kitab_filter

# Menyaring DataFrame dan vektor sesuai pilihan drop-down
df_tersaring = df_alkitab[mask_filter].reset_index(drop=True)
vektor_tersaring = vektor_seluruh_ayat[mask_filter.values]

# --- KOLOM UTAMA PENCARIAN & GRAFIK ---
pertanyaan = st.text_input("Masukkan pencarian makna cerita:", placeholder="Contoh: Daniel dilemparkan ke singa")

if st.button("Mulai Cari"):
    if pertanyaan:
        with st.spinner("AI sedang memproses pencarian dan peta kluster matematika..."):
            vektor_tanya = get_vektor_pertanyaan(pertanyaan)
            
            if vektor_tanya is not None:
                # Menghitung kemiripan semantik
                if len(vektor_tanya.shape) == 1:
                    vektor_tanya = vektor_tanya.reshape(1, -1)
                elif len(vektor_tanya.shape) == 3:
                    vektor_tanya = vektor_tanya[0][0].reshape(1, -1)
                    
                skor_kemiripan = cosine_similarity(vektor_tanya, vektor_tersaring)[0]
                
                top_k = min(3, len(df_tersaring))
                if top_k == 0:
                    st.warning("Tidak ada ayat yang cocok dengan filter yang Anda pilih.")
                else:
                    indeks_teratas = np.argsort(skor_kemiripan)[::-1][:top_k]
                    
                    # Layout Kolom Kiri (Hasil) dan Kolom Kanan (Grafik Visualisasi)
                    kol_hasil, col_grafik = st.columns([1.1, 1])
                    
                    with kol_hasil:
                        st.success(f"Ditemukan {top_k} ayat paling relevan sesuai filter!")
                        for idx in indeks_teratas:
                            baris = df_tersaring.iloc[idx]
                            skor = skor_kemiripan[idx]
                            
                            with st.expander(f"📍 {baris['kitab']} {baris['pasal']}:{baris['ayat']} (Kemiripan: {skor:.2f})", expanded=True):
                                st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                                st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                                st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
                                
                    with col_grafik:
                        st.subheader("Peta Kluster Vektor 2D")
                        
                        # Ambil sampel 300 ayat lain untuk latar belakang peta sebaran agar tidak lemot
                        df_latar = df_alkitab.sample(n=min(300, len(df_alkitab))).copy()
                        df_latar['Tipe'] = "Ayat Alkitab Lain"
                        df_latar['Ukuran_Titik'] = 6
                        
                        # Ambil ayat hasil relevan pencarian
                        df_relevan = df_tersaring.iloc[indeks_teratas].copy()
                        df_relevan['Tipe'] = "Hasil Relevan"
                        df_relevan['Ukuran_Titik'] = 14
                        
                        # Hitung proyeksi koordinat 2D untuk pertanyaan dosen
                        proyeksi_tanya = pca_model.transform(vektor_tanya)
                        df_tanya = pd.DataFrame([{
                            'kitab': "Pencarian Anda",
                            'pasal': "",
                            'ayat': "",
                            'teks_tb': pertanyaan,
                            'x': proyeksi_tanya[0, 0],
                            'y': proyeksi_tanya[0, 1],
                            'Tipe': "Pertanyaan Anda",
                            'Ukuran_Titik': 18
                        }])
                        
                        # Menggabungkan data untuk digambar di Plotly
                        df_plot = pd.concat([df_latar, df_relevan, df_tanya], ignore_index=True)
                        
                        fig = px.scatter(
                            df_plot, 
                            x='x', 
                            y='y', 
                            color='Tipe',
                            size='Ukuran_Titik',
                            hover_data=['kitab', 'pasal', 'ayat', 'teks_tb'],
                            color_discrete_map={
                                "Ayat Alkitab Lain": "lightgrey",
                                "Hasil Relevan": "blue",
                                "Pertanyaan Anda": "red"
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
