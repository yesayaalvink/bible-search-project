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

# IMPORT LIBRARY RESMI GOOGLE GENAI & TIPE KONFIGURASI
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
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        with requests.get(url_database, headers=headers, stream=True) as r:
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
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                with requests.get(url, headers=headers, stream=True) as r:
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
# 4. FUNGSI RAG GENERATOR (GEMINI 3.5 FLASH)
# ==========================================
def panggil_gemini_rag(prompt):
    try:
        interaction = client_gemini.interactions.create(
            model="gemini-3.5-flash",
            input=prompt,
            generation_config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"  # Langsung menjawab tanpa lama berpikir
                )
            )
        )
        return interaction.output_text
    except Exception as e:
        return f"ERROR_DETAIL: {repr(e)}"


# ==========================================
# 5. GENERATOR RAG CADANGAN (QWEN -> LLAMA)
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
# 6. FUNGSI GENERATOR UTAMA UNTUK STREAMING RAG
# ==========================================
def generate_rag_stream(prompt):
    # --- 1. COBA AI UTAMA: GEMINI 3.5 FLASH (STREAMING) ---
    try:
        response_stream = client_gemini.models.generate_content_stream(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0  # Matikan thinking agar instan & langsung streaming
                )
            )
        )
        yield "### 🤖 Analisis Teologis AI Utama (Google Gemini 3.5 Flash):\n\n"
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
        return
    except Exception as e:
        yield f"\n\n*(⚠️ AI Utama Gemini gagal/limit. Detail: {repr(e)}. Menghubungi AI Cadangan 1...)*\n\n"
        
    # --- 2. COBA AI CADANGAN 1: LLAMA 3.2 3B (STABIL) ---
    model_id_1 = "meta-llama/Llama-3.2-3B-Instruct"
    try:
        res_text = panggil_hf_model_rag(prompt, model_id_1)
        if res_text and not res_text.startswith("ERROR_DETAIL"):
            yield f"### 🤖 Analisis Teologis AI Cadangan 1 (Meta Llama 3.2):\n\n{res_text}"
            return
        else:
            yield f"\n\n*(⚠️ AI Cadangan 1 gagal. Detail: {res_text}. Menghubungi AI Cadangan 2...)*\n\n"
    except Exception as e:
        yield f"\n\n*(⚠️ AI Cadangan 1 gagal. Detail: {repr(e)}. Menghubungi AI Cadangan 2...)*\n\n"
        
    # --- 3. COBA AI CADANGAN 2: MISTRAL 7B ---
    model_id_2 = "mistralai/Mistral-7B-Instruct-v0.3"
    try:
        res_text = panggil_hf_model_rag(prompt, model_id_2)
        if res_text and not res_text.startswith("ERROR_DETAIL"):
            yield f"### 🤖 Analisis Teologis AI Cadangan 2 (Mistral 7B):\n\n{res_text}"
            return
        else:
            yield f"\n\n*(⚠️ Semua sistem AI cadangan gagal. Detail: {res_text})*\n\n"
    except Exception as e:
        yield f"\n\n*(⚠️ Semua sistem AI cadangan gagal. Detail: {repr(e)})*\n\n"


# ==========================================
# 7. PROSES MEMINTA VEKTOR DARI MODEL AI
# ==========================================
def get_vektor_pertanyaan(pertanyaan):
    try:
        vektor = model.encode([pertanyaan])
        return np.array(vektor)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses kalimat: {repr(e)}")
        return None


# ==========================================
# 8. TAMPILAN USER INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="Pencarian Semantik Alkitab", layout="wide")

# ==========================================
# SUNTIKKAN DEKORASI CSS & ORNAMEN VEKTOR POJOK KANAN ATAS (VERSI BERSIH TANPA SPASI DEPAN)
# ==========================================
st.markdown("""<style>
.streamlit-expanderHeader {
    border-left: 5px solid #d4af37 !important;
    background-color: #fcfaf2 !important;
    border-radius: 4px;
    font-weight: bold !important;
}
</style>
<div style="position: absolute; top: -65px; right: 15px; z-index: 999; opacity: 0.95; pointer-events: none;">
<svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
<line x1="50" y1="12" x2="50" y2="4" stroke="#f6e05e" stroke-width="1.5" stroke-dasharray="1 1" />
<line x1="33" y1="18" x2="23" y2="10" stroke="#f6e05e" stroke-width="1.5" stroke-dasharray="1 1" />
<line x1="67" y1="18" x2="77" y2="10" stroke="#f6e05e" stroke-width="1.5" stroke-dasharray="1 1" />
<line x1="28" y1="35" x2="16" y2="35" stroke="#f6e05e" stroke-width="1.5" stroke-dasharray="1 1" />
<line x1="72" y1="35" x2="84" y2="35" stroke="#f6e05e" stroke-width="1.5" stroke-dasharray="1 1" />
<path d="M15 75C32 75 50 80 50 80C50 80 68 75 85 75V40C68 40 50 45 50 45C50 45 32 40 15 40V75Z" fill="#ffffff" stroke="#d4af37" stroke-width="2.2" stroke-linejoin="round"/>
<path d="M50 45V80" stroke="#d4af37" stroke-width="2"/>
<path d="M15 78C32 78 50 83 50 80" stroke="#d4af37" stroke-width="1"/>
<path d="M85 78C68 78 50 83 50 80" stroke="#d4af37" stroke-width="1"/>
<path d="M50 25V58" stroke="#d4af37" stroke-width="3.5" stroke-linecap="round"/>
<path d="M41 35H59" stroke="#d4af37" stroke-width="3.5" stroke-linecap="round"/>
<path d="M48 20 C42 16, 32 18, 28 24 C32 28, 42 26, 48 20" fill="#ffffff" stroke="#718096" stroke-width="1.2" stroke-linejoin="round"/>
<path d="M48 20 C54 16, 64 18, 68 24 C64 28, 54 26, 48 20" fill="#ffffff" stroke="#718096" stroke-width="1.2" stroke-linejoin="round"/>
<path d="M48 20 C46 32, 50 38, 50 44" stroke="#718096" stroke-width="1.5" stroke-linecap="round"/>
<path d="M45 23 C43 24, 40 23, 38 25" stroke="#48bb78" stroke-width="1.5" stroke-linecap="round"/>
<circle cx="38" cy="25" r="1.5" fill="#48bb78" />
</svg>
</div>""", unsafe_allow_html=True)

# --- POP-UP / KARTU SAMBUTAN SHALOM DI AWAL ---
if "tutup_panduan" not in st.session_state:
    st.session_state["tutup_panduan"] = False

if not st.session_state["tutup_panduan"]:
    with st.container(border=True):
        st.markdown("""
        ### 👋 Shalom! Selamat Datang di Aplikasi Pencarian Alkitab
        Aplikasi ini didukung oleh Kecerdasan Buatan (IndoBERT) untuk membantu Anda menjelajahi firman Tuhan secara mendalam berdasarkan topik teologis.
        
        📖 **Fitur Utama yang Tersedia:**
        * **Pencarian Semantik:** Cari ayat Alkitab berdasarkan topik atau makna cerita.
        * **Cari Ayat Serupa:** Pilih satu ayat spesifik, dan AI akan mencari ayat lain di seluruh Alkitab yang memiliki makna paling setara.
        * **Filter Alkitab Pintar:** Anda bisa membatasi pencarian hanya pada Perjanjian Lama, Perjanjian Baru, atau Kitab tertentu saja.
        * **Analisis RAG AI:** Menghasilkan kesimpulan penjelasan teologis otomatis yang mengalir langsung secara real-time!
        """)
        if st.button("Mulai Menjelajahi 🚀"):
            st.session_state["tutup_panduan"] = True
            st.rerun()

st.title("📖 Sistem Pencarian Semantik Alkitab (IndoBERT)")
st.write("Silakan gunakan tab di bawah ini untuk mengakses fitur yang berbeda.")

# --- TAB UTAMA (TABS) ---
tab1, tab2 = st.tabs(["🔍 Pencarian Semantik (Teks)", "🔄 Cari Ayat Serupa (Verse-to-Verse)"])

# Persiapan data pembagian PL dan PB untuk filtering
kitab_unik = list(df_alkitab['kitab'].unique())
kitab_pl = kitab_unik[:39]
kitab_pb = kitab_unik[39:]

# ------------------------------------------
# TAB 1: PENCARIAN SEMANTIK
# ------------------------------------------
with tab1:
    st.subheader("Cari Ayat Alkitab Berdasarkan Topik Makna Kalimat")
    
    # --- MENARUH FILTER DI BAWAH TAB (TAB 1) ---
    st.write("**⚙️ Penyaringan Alkitab (Opsional):**")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_p1 = st.selectbox(
            "Pilih Perjanjian:", 
            ["Seluruh Alkitab", "Perjanjian Lama (PL)", "Perjanjian Baru (PB)"],
            key="filter_p1"
        )
    with col_f2:
        if filter_p1 == "Perjanjian Lama (PL)":
            opsi_kitab1 = ["Semua Kitab PL"] + kitab_pl
        elif filter_p1 == "Perjanjian Baru (PB)":
            opsi_kitab1 = ["Semua Kitab PB"] + kitab_pb
        else:
            opsi_kitab1 = ["Semua Kitab"] + kitab_unik
        filter_k1 = st.selectbox("Pilih Kitab Spesifik:", opsi_kitab1, key="filter_k1")

    # Saring database untuk Tab 1
    if filter_p1 == "Perjanjian Lama (PL)":
        mask_t1 = df_alkitab['kitab'].isin(kitab_pl) if filter_k1 == "Semua Kitab PL" else df_alkitab['kitab'] == filter_k1
    elif filter_p1 == "Perjanjian Baru (PB)":
        mask_t1 = df_alkitab['kitab'].isin(kitab_pb) if filter_k1 == "Semua Kitab PB" else df_alkitab['kitab'] == filter_k1
    else:
        mask_t1 = pd.Series([True] * len(df_alkitab)) if filter_k1 == "Semua Kitab" else df_alkitab['kitab'] == filter_k1

    df_t1 = df_alkitab[mask_t1].reset_index(drop=True)
    vektor_t1 = vektor_seluruh_ayat[mask_t1.values]

    st.write("---")

    # --- INPUT DAN TOMBOL CARI DI DALAM FORM (ENTER BERFUNGSI) ---
    with st.form("pencarian_form"):
        pertanyaan = st.text_input(
            "Masukkan pencarian/pertanyaan mengenai Alkitab :", 
            placeholder="Contoh: Kehendak Tuhan bagi manusia"
        )
        submit_button = st.form_submit_button("Mulai Cari 🚀")

    if submit_button:
        if pertanyaan:
            with st.spinner("AI sedang mencocokkan makna ayat teologis..."):
                vektor_tanya = get_vektor_pertanyaan(pertanyaan)
                
                if vektor_tanya is not None:
                    if len(vektor_tanya.shape) == 1:
                        vektor_tanya = vektor_tanya.reshape(1, -1)
                        
                    skor_kemiripan = cosine_similarity(vektor_tanya, vektor_t1)[0]
                    
                    top_k = min(3, len(df_t1))
                    if top_k == 0:
                        st.warning("Tidak ada ayat yang cocok dengan filter aktif di atas.")
                    else:
                        indeks_teratas = np.argsort(skor_kemiripan)[::-1][:top_k]
                        
                        st.success(f"Ditemukan {top_k} ayat paling relevan!")
                        
                        # Tampilkan hasil ayat
                        daftar_ayat_terpilih = []
                        konteks_ayat = ""
                        for idx in indeks_teratas:
                            baris = df_t1.iloc[idx]
                            skor = skor_kemiripan[idx]
                            daftar_ayat_terpilih.append(baris)
                            konteks_ayat += f"\n- {baris['kitab']} {baris['pasal']}:{baris['ayat']} -> {baris['teks_tb']}"
                            
                            with st.expander(f"📍 {baris['kitab']} {baris['pasal']}:{baris['ayat']} (Tingkat Kemiripan: {skor:.2f})", expanded=True):
                                st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                                st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                                st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")
                        
                        # Jalankan RAG menggunakan Hierarki 3 AI (Streaming!)
                        st.markdown("---")
                        
                        # Susun prompt untuk RAG (Dalam bahasa Indonesia yang menarik & mudah dipahami)
                        prompt_rag = (
                            f"Anda adalah seorang asisten Teologi Kristen yang ahli dalam penafsiran Alkitab. "
                            f"Pengguna sedang mencari topik atau bertanya: '{pertanyaan}'.\n\n"
                            f"Berikut adalah 3 ayat relevan yang ditemukan dari Alkitab:\n{konteks_ayat}\n\n"
                            f"Berikan penjelasan teologis singkat (maksimal 3-4 kalimat) dalam bahasa Indonesia yang menarik dan mudah dipahami, "
                            f"yang menjelaskan korelasi makna teologis antara pertanyaan tersebut dengan ayat-ayat di atas."
                        )
                        
                        # Jalankan proses streaming secara langsung di layar dengan label sumber AI
                        with st.spinner("Mempersiapkan saluran AI streaming..."):
                            st.write_stream(generate_rag_stream(prompt_rag))

# ------------------------------------------
# TAB 2: CARI AYAT SERUPA
# ------------------------------------------
with tab2:
    st.subheader("Cari Ayat yang Memiliki Kedekatan Makna Paling Serupa")
    
    # --- MENARUH FILTER DI BAWAH TAB (TAB 2) ---
    st.write("**⚙️ Penyaringan Alkitab (Opsional):**")
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        filter_p2 = st.selectbox(
            "Pilih Perjanjian:", 
            ["Seluruh Alkitab", "Perjanjian Lama (PL)", "Perjanjian Baru (PB)"],
            key="filter_p2"
        )
    with col_f4:
        if filter_p2 == "Perjanjian Lama (PL)":
            opsi_kitab2 = ["Semua Kitab PL"] + kitab_pl
        elif filter_p2 == "Perjanjian Baru (PB)":
            opsi_kitab2 = ["Semua Kitab PB"] + kitab_pb
        else:
            opsi_kitab2 = ["Semua Kitab"] + kitab_unik
        filter_k2 = st.selectbox("Pilih Kitab Spesifik:", opsi_kitab2, key="filter_k2")

    # Saring database untuk Tab 2
    if filter_p2 == "Perjanjian Lama (PL)":
        mask_t2 = df_alkitab['kitab'].isin(kitab_pl) if filter_k2 == "Semua Kitab PL" else df_alkitab['kitab'] == filter_k2
    elif filter_p2 == "Perjanjian Baru (PB)":
        mask_t2 = df_alkitab['kitab'].isin(kitab_pb) if filter_k2 == "Semua Kitab PB" else df_alkitab['kitab'] == filter_k2
    else:
        mask_t2 = pd.Series([True] * len(df_alkitab)) if filter_k2 == "Semua Kitab" else df_alkitab['kitab'] == filter_k2

    df_t2 = df_alkitab[mask_t2].reset_index(drop=True)
    vektor_t2 = vektor_seluruh_ayat[mask_t2.values]

    st.write("---")
    st.write("Pilih salah satu ayat Alkitab di bawah ini, AI akan mencari ayat lain yang bermakna setara.")
    
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        kitab_target = st.selectbox("Pilih Kitab Target:", list(df_alkitab['kitab'].unique()), key="sel_kitab")
    with col_k2:
        df_target_kitab = df_alkitab[df_alkitab['kitab'] == kitab_target]
        pasal_unik = sorted(list(df_target_kitab['pasal'].unique()))
        pasal_target = st.selectbox("Pilih Pasal Target:", pasal_unik, key="sel_pasal")
    with col_k3:
        df_target_pasal = df_target_kitab[df_target_kitab['pasal'] == pasal_target]
        ayat_unik = sorted(list(df_target_pasal['ayat'].unique()))
        ayat_target = st.selectbox("Pilih Ayat Target:", ayat_unik, key="sel_ayat")
        
    # Ambil data ayat target secara penuh
    row_target = df_target_pasal[df_target_pasal['ayat'] == ayat_target].iloc[0]
    st.info(f"**Ayat Terpilih:** {row_target['kitab']} {row_target['pasal']}:{row_target['ayat']}\n\n> *\"{row_target['teks_tb']}\"*")
    
    if st.button("Cari Ayat Serupa 🔄", key="btn_serupa"):
        with st.spinner("Kecerdasan buatan sedang mencocokkan kemiripan antar ayat Alkitab..."):
            idx_asli_target = row_target.name
            vektor_target = vektor_seluruh_ayat[idx_asli_target].reshape(1, -1)
            
            skor_kemiripan_v2 = cosine_similarity(vektor_target, vektor_t2)[0]
            indeks_teratas_v2 = np.argsort(skor_kemiripan_v2)[::-1]
            
            # Saring agar tidak merekomendasikan ayat terpilih itu sendiri
            rekomendasi_akhir = []
            for idx in indeks_teratas_v2:
                baris_recom = df_t2.iloc[idx]
                if not (baris_recom['kitab'] == kitab_target and 
                        baris_recom['pasal'] == pasal_target and 
                        baris_recom['ayat'] == ayat_target):
                    rekomendasi_akhir.append((idx, skor_kemiripan_v2[idx]))
                if len(rekomendasi_akhir) == 3:
                    break
            
            if len(rekomendasi_akhir) == 0:
                st.warning("Tidak ditemukan ayat serupa dengan filter aktif di atas.")
            else:
                st.success("Ditemukan 3 ayat dengan keselarasan makna teologis paling serupa!")
                for idx, skor in rekomendasi_akhir:
                    baris = df_t2.iloc[idx]
                    with st.expander(f"📍 {baris['kitab']} {baris['pasal']}:{baris['ayat']} (Kesamaan Makna: {skor:.2f})", expanded=True):
                        st.markdown(f"**Terjemahan Baru (TB):**\n> {baris['teks_tb']}")
                        st.markdown(f"**Versi Mudah Dibaca (VMD):**\n> {baris['teks_vmd']}")
                        st.markdown(f"**Alkitab Yang Terbuka (AYT):**\n> {baris['teks_ayt']}")

# --- DEKORASI VISUAL PREMIUM (FOOTER TEOLOGIS) ---
st.markdown("""
<div style="text-align: center; margin-top: 80px; padding: 25px; border-top: 1px solid #e2e8f0; color: #888888;">
    <p style="font-size: 28px; margin-bottom: 5px;">🕊️ &nbsp; 📖 &nbsp; ✝️</p>
    <p style="font-size: 13px; font-style: italic; font-family: 'Georgia', serif; color: #718096; margin-top: 10px;">
        "Sebab firman Allah hidup dan kuat dan lebih tajam dari pada pedang bermata dua mana pun..." — Ibrani 4:12
    </p>
</div>
""", unsafe_allow_html=True)
