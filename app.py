import os
import streamlit as st

# ==========================================
# TRIK RAHASIA: AMBIL DAN BERSIHKAN TOKEN DARI MEMORI SISTEM
# ==========================================
HF_TOKEN_VAL = st.secrets.get("HF_TOKEN", "")
GEMINI_API_KEY_VAL = st.secrets.get("GEMINI_API_KEY", "")

os.environ.pop("HF_TOKEN", None)
os.environ.pop("HF_HUB_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ.pop("HUGGINGFACE_TOKEN", None)
os.environ.pop("HUGGINGFACE_CO_TOKEN", None)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer  # Memuat model langsung di server Streamlit

# IMPORT LIBRARY RESMI GOOGLE GENAI SESUAI DOKUMENTASI GOOGLE AI STUDIO
from google import genai
from google.genai import types

# ==========================================
# 1. ATUR ALAMAT REPOSITORI & API GEMINI
# ==========================================
REPO_ID = "YesayaAlvinK/bible-search-project"

# Inisialisasi Google GenAI Client resmi menggunakan API Key Anda dari Secrets
client_gemini = genai.Client(api_key=GEMINI_API_KEY_VAL)


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
                                
        import json
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
# 4. FUNGSI RAG GENERATOR (GEMINI 3.5 FLASH - METODE TERBARU & CEPAT)
# ==========================================
def panggil_gemini_rag(prompt):
    try:
        # Menggunakan metode standar generate_content yang resmi dan bebas dari error Unmarshaller
        response = client_gemini.models.generate_content(
            model="gemini-3.5-flash",  # Model Flash resmi yang sangat cepat tanpa delay berpikir
            contents=prompt
        )
        return response.text
    except Exception as e:
        # Mengembalikan string error asli agar bisa dibaca di layar
        return f"ERROR_DETAIL: {repr(e)}"


# ==========================================
# 5. GENERATOR RAG CADANGAN (QWEN 1.5B -> MICROSOFT PHI-3)
# ==========================================
def panggil_hf_model_rag(prompt, model_id):
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN_VAL}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "return_full_text": False}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, list) and len(res) > 0:
                return res[0].get('generated_text', '')
        return f"ERROR_DETAIL: HTTP Status {response.status_code} - {response.text}"
    except Exception as e:
        return f"ERROR_DETAIL: {repr(e)}"


# ==========================================
# 6. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        vektor = model.encode([pertanyaan])
        return np.array(vektor)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses kalimat: {repr(e)}")
        return None


# ==========================================
# 7. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="Pencarian Semantik Alkitab", layout="wide")

# --- SIDEBAR (PANE SAMPING) UNTUK FILTERING ---
st.sidebar.title("📖 Filter Alkitab Global")
st.sidebar.write("Penyaringan ini otomatis berlaku untuk kedua fitur pencarian.")

kitab_unik = list(df_alkitab['kitab'].unique())
kitab_pl = kitab_unik[:39]
kitab_pb = kitab_unik[39:]

perjanjian_filter = st.sidebar.selectbox(
    "Pilih Perjanjian:", 
    ["Seluruh Alkitab", "Perjanjian Lama (PL)", "Perjanjian Baru (PB)"]
)

if perjanjian_filter == "Perjanjian Lama (PL)":
    pilihan_kitab = ["Semua Kitab PL"] + kitab_pl
elif perjanjian_filter == "Perjanjian Baru (PB)":
    pilihan_kitab = ["Semua Kitab PB"] + kitab_pb
else:
    pilihan_kitab = ["Semua Kitab"] + kitab_unik
    
kitab_filter = st.sidebar.selectbox("Pilih Kitab Spesifik:", pilihan_kitab)

# Mempersiapkan mask penyaringan data secara global
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

# Data tersaring secara global
df_tersaring = df_alkitab[mask_filter].reset_index(drop=True)
vektor_tersaring = vektor_seluruh_ayat[mask_filter.values]


# --- TAMPILAN TAB UTAMA ---
st.title("📖 Sistem Pencarian Semantik Alkitab (IndoBERT)")
st.write("Silakan gunakan tab di bawah ini untuk mengakses fitur yang berbeda.")

tab1, tab2 = st.tabs(["🔍 Pencarian Semantik (Teks)", "🔄 Cari Ayat Serupa (Verse-to-Verse)"])

# ------------------------------------------
# TAB 1: PENCARIAN SEMANTIK + RAG HIERARKI
# ------------------------------------------
with tab1:
    st.subheader("Cari Ayat Alkitab Berdasarkan Topik Makna Kalimat")
    pertanyaan = st.text_input("Masukkan pencarian makna cerita:", placeholder="Contoh: kasih tuhan kepada manusia")

    if st.button("Mulai Cari", key="btn_pencarian"):
        if pertanyaan:
            with st.spinner("AI sedang mencocokkan makna ayat teologis..."):
                vektor_tanya = get_vektor_pertanyaan(pertanyaan)
                
                if vektor_tanya is not None:
                    if len(vektor_tanya.shape) == 1:
                        vektor_tanya = vektor_tanya.reshape(1, -1)
                        
                    skor_kemiripan = cosine_similarity(vektor_tanya, vektor_tersaring)[0]
                    
                    top_k = min(3, len(df_tersaring))
                    if top_k == 0:
                        st.warning("Tidak ada ayat yang cocok dengan filter aktif di sidebar.")
                    else:
                        indeks_teratas = np.argsort(skor_kemiripan)[::-1][:top_k]
                        
                        st.success(f"Ditemukan {top_k} ayat paling relevan!")
                        
                        # Tampilkan hasil ayat
                        daftar_ayat_terpilih = []
                        konteks_ayat = ""
                        for idx in indeks_teratas:
                            baris = df_tersaring.iloc[idx]
                            skor = skor_kemiripan[idx]
                            daftar_ayat_terpilih.append(baris)
                            konteks_ayat += f"\n- {baris['kitab']} {baris['pasal']}:{baris['ayat']} -> {baris['teks_tb']}"
                            
                            with st.expander(f"📍 {baris['kitab']} {baris['pasal']}:{baris['ayat']} (Tingkat Kemiripan: {skor:.2f})", expanded=True):
                                st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                                st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                                st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
                        
                        # Jalankan RAG menggunakan Hierarki 3 AI
                        st.markdown("---")
                        st.markdown("### 🤖 Analisis Teologis AI Generatif (RAG)")
                        
                        # Susun prompt untuk RAG
                        prompt_rag = (
                            f"Anda adalah seorang asisten Teologi Kristen yang ahli dalam penafsiran Alkitab. "
                            f"Pengguna sedang mencari topik: '{pertanyaan}'.\n\n"
                            f"Berikut adalah 3 ayat relevan yang ditemukan dari Alkitab:\n{konteks_ayat}\n\n"
                            f"Berikan penjelasan singkat teologis (maksimal 3-4 kalimat) dalam bahasa Indonesia, "
                            f"yang menjelaskan korelasi makna teologis antara topik pencarian dengan ayat-ayat di atas."
                        )
                        
                        # 1. Coba AI Utama (Gemini - Sangat Cepat!)
                        success_rag = False
                        with st.spinner("Menghubungi AI Utama (Gemini 3.5 flash)..."):
                            analisis_rag = panggil_gemini_rag(prompt_rag)
                            if analisis_rag and not analisis_rag.startswith("ERROR_DETAIL"):
                                st.info(f"### 🤖 Analisis Teologis AI (Gemini 3.5 flash):\n\n{analisis_rag}")
                                success_rag = True
                            else:
                                st.warning(f"⚠️ AI Utama (Gemini) Gagal. Detail: {analisis_rag}")
                        
                        # 2. Coba AI Cadangan 1 (Qwen 1.5B) jika Gemini gagal
                        if not success_rag:
                            st.warning("Menghubungi AI Cadangan 1 (Qwen 2.5 1.5B)...")
                            with st.spinner("Menghubungi AI Cadangan 1 (Qwen 2.5 1.5B)..."):
                                analisis_rag = panggil_hf_model_rag(prompt_rag, "Qwen/Qwen2.5-1.5B-Instruct")
                                if analisis_rag and not analisis_rag.startswith("ERROR_DETAIL"):
                                    st.info(f"### 🤖 Analisis Teologis AI (Qwen 2.5 1.5B):\n\n{analisis_rag}")
                                    success_rag = True
                                else:
                                    st.warning(f"⚠️ AI Cadangan 1 (Qwen 2.5 1.5B) Gagal. Detail: {analisis_rag}")
                                    
                        # 3. Coba AI Cadangan 2 (Microsoft Phi-3) jika Qwen gagal
                        if not success_rag:
                            st.warning("Menghubungi AI Cadangan 2 (Microsoft Phi-3)...")
                            with st.spinner("Menghubungi AI Cadangan 2 (Microsoft Phi-3)..."):
                                analisis_rag = panggil_hf_model_rag(prompt_rag, "microsoft/Phi-3-mini-4k-instruct")
                                if analisis_rag and not analisis_rag.startswith("ERROR_DETAIL"):
                                    st.info(f"### 🤖 Analisis Teologis AI (Microsoft Phi-3):\n\n{analisis_rag}")
                                    success_rag = True
                                else:
                                    st.warning(f"⚠️ AI Cadangan 2 (Microsoft Phi-3) Gagal. Detail: {analisis_rag}")
                                    
                        # 4. Jika semua gagal
                        if not success_rag:
                            st.error("⚠️ Semua sistem AI cadangan sedang sibuk/limit. Silakan coba klik Cari lagi.")

# ------------------------------------------
# TAB 2: CARI AYAT SERUPA (VERSE-TO-VERSE)
# ------------------------------------------
with tab2:
    st.subheader("Cari Ayat yang Memiliki Kedekatan Makna Paling Serupa")
    st.write("Pilih salah satu ayat Alkitab di bawah ini, AI akan mencari ayat lain yang bermakna setara.")
    
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        kitab_target = st.selectbox("Pilih Kitab:", list(df_alkitab['kitab'].unique()), key="sel_kitab")
    with col_k2:
        df_target_kitab = df_alkitab[df_alkitab['kitab'] == kitab_target]
        pasal_unik = sorted(list(df_target_kitab['pasal'].unique()))
        pasal_target = st.selectbox("Pilih Pasal:", pasal_unik, key="sel_pasal")
    with col_k3:
        df_target_pasal = df_target_kitab[df_target_kitab['pasal'] == pasal_target]
        ayat_unik = sorted(list(df_target_pasal['ayat'].unique()))
        ayat_target = st.selectbox("Pilih Ayat:", ayat_unik, key="sel_ayat")
        
    # Ambil data ayat target secara penuh
    row_target = df_target_pasal[df_target_pasal['ayat'] == ayat_target].iloc[0]
    st.info(f"**Ayat Terpilih:** {row_target['kitab']} {row_target['pasal']}:{row_target['ayat']}\n\n> *\"{row_target['teks_tb']}\"*")
    
    if st.button("Cari Ayat Serupa", key="btn_serupa"):
        with st.spinner("Kecerdasan buatan sedang mencocokkan kemiripan antar ayat Alkitab..."):
            # Ambil indeks baris asli ayat target dari dataframe utama
            idx_asli_target = row_target.name
            
            # Ambil vektor ayat target
            vektor_target = vektor_seluruh_ayat[idx_asli_target].reshape(1, -1)
            
            # Hitung similarity vektor target dengan seluruh ayat di database tersaring
            skor_kemiripan_v2 = cosine_similarity(vektor_target, vektor_tersaring)[0]
            indeks_teratas_v2 = np.argsort(skor_kemiripan_v2)[::-1]
            
            # Saring agar tidak merekomendasikan ayat terpilih itu sendiri
            rekomendasi_akhir = []
            for idx in indeks_teratas_v2:
                baris_recom = df_tersaring.iloc[idx]
                if not (baris_recom['kitab'] == kitab_target and 
                        baris_recom['pasal'] == pasal_target and 
                        baris_recom['ayat'] == ayat_target):
                    rekomendasi_akhir.append((idx, skor_kemiripan_v2[idx]))
                if len(rekomendasi_akhir) == 3:
                    break
            
            if len(rekomendasi_akhir) == 0:
                st.warning("Tidak ditemukan ayat serupa dengan filter aktif.")
            else:
                st.success("Ditemukan 3 ayat dengan keselarasan makna teologis paling serupa!")
                for idx, skor in rekomendasi_akhir:
                    baris = df_tersaring.iloc[idx]
                    with st.expander(f"📍 {baris['kitab']} {baris['pasal']}:{baris['ayat']} (Kesamaan Makna: {skor:.2f})", expanded=True):
                        st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                        st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                        st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
