import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Library Sastrawi untuk Indo
import re

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AnaText - AI Text Analysis",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATE MANAGEMENT ---
if 'stop_words' not in st.session_state:
    # Default Stop Words
    st.session_state.stop_words = [
        "yang", "di", "dan", "itu", "dengan", "untuk", "tidak", "ini", "dari", 
        "dalam", "akan", "pada", "juga", "saya", "adalah", "ke", "karena", 
        "bisa", "ada", "mereka", "kita", "kamu", "the", "and", "is", "of", "to", "in"
    ]
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# --- CSS CUSTOM (UI SEPERTI SCREENSHOT) ---
# Fitur Light/Dark Mode via CSS Injection
def inject_custom_css(mode):
    if mode == 'Dark':
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
    else:
        bg_color = "#ffffff"
        text_color = "#31333F"
        card_bg = "#f0f2f6"

    st.markdown(f"""
    <style>
        /* Global Styling */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        
        /* Custom Header Styling */
        h1, h2, h3 {{
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }}
        
        /* File Uploader Styling - Mimic "Drag & Drop Area" */
        [data-testid='stFileUploader'] {{
            background-color: {card_bg};
            border: 2px dashed #4c7bf4;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        [data-testid='stFileUploader'] section {{
            padding: 0;
        }}
        
        /* Button Styling */
        .stButton button {{
            background-color: #2563eb;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            width: 100%;
        }}
        .stButton button:hover {{
            background-color: #1d4ed8;
            color: white;
        }}
        
        /* Modal/Expander Styling */
        .streamlit-expanderHeader {{
            background-color: {card_bg};
            border-radius: 5px;
        }}
        
        /* Highlight Sentimen Tabel */
        .positive-bg {{background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px;}}
        .negative-bg {{background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px;}}
        .neutral-bg {{background-color: #f3f4f6; color: #1f2937; padding: 4px 8px; border-radius: 4px;}}

    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI PREPROCESSING ---
def clean_text(text, remove_sw, use_lemma, case_folding, stopwords_list, stemmer):
    if not isinstance(text, str):
        return ""
    
    # 1. Case Folding
    if case_folding:
        text = text.lower()
    
    # Bersihkan karakter non-alphanumeric dasar
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = text.split()
    
    # 2. Hapus Stop Words
    if remove_sw:
        tokens = [word for word in tokens if word not in stopwords_list]
    
    # 3. Lemmatization (Stemming Sastrawi untuk Indo)
    if use_lemma and stemmer:
        # Note: Stemming per kata agak lambat, idealnya batch, tapi untuk keakuratan kita loop
        tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

# --- FUNGSI OPENAI ---
def get_sentiment_ai(client, model, text_list):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(text_list)
    
    for i, text in enumerate(text_list):
        # Skip teks kosong
        if not text.strip():
            results.append("Netral")
            continue
            
        try:
            # Menggunakan pesan sistem yang efisien
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Klasifikasikan sentimen: Positif, Negatif, atau Netral. Jawab 1 kata saja."},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=10
            )
            sentiment = response.choices[0].message.content.strip().replace(".", "")
            results.append(sentiment)
        except Exception as e:
            results.append("Error")
            
        # Update progress
        perc = (i + 1) / total
        progress_bar.progress(perc)
        status_text.text(f"Menganalisis sentimen... {i+1}/{total}")
    
    progress_bar.empty()
    status_text.empty()
    return results

def get_topic_name_ai(client, model, keywords):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (2-4 kata) berdasarkan kata kunci ini."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content.replace('"', '')
    except:
        return "Topik Tak Teridentifikasi"

# --- MODAL STOPWORDS (Menggunakan st.dialog di versi baru atau expander) ---
# Kita gunakan @st.experimental_dialog jika versi streamlit support, 
# jika error update streamlit anda: pip install streamlit --upgrade
@st.experimental_dialog("Kelola Stop Words")
def manage_stopwords_dialog():
    st.write("Tambahkan atau hapus kata yang tidak ingin dianalisis.")
    
    # Input kata baru
    new_word = st.text_input("Tambah kata baru (tekan Enter):")
    if new_word:
        if new_word.lower() not in st.session_state.stop_words:
            st.session_state.stop_words.append(new_word.lower())
            st.rerun() # Refresh agar masuk list

    # Tampilan Multiselect sebagai "Tags"
    current_words = st.multiselect(
        "Daftar Stop Words:",
        options=st.session_state.stop_words,
        default=st.session_state.stop_words
    )
    
    # Update state jika ada yang dihapus via 'x' di multiselect
    if len(current_words) != len(st.session_state.stop_words):
        st.session_state.stop_words = current_words
        st.rerun()

    if st.button("Simpan & Tutup", type="primary"):
        st.rerun()

# --- MAIN APP ---

# 1. Sidebar Config
with st.sidebar:
    st.title("⚙️ Pengaturan")
    
    # Mode Tampilan
    theme_mode = st.segmented_control("Tema Tampilan", ["Light", "Dark"], default="Light")
    inject_custom_css(theme_mode) # Terapkan CSS
    
    st.divider()
    
    st.subheader("Bahasa & Teks")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    text_type = st.selectbox("Tipe Teks", ["Umum", "Ulasan Produk", "Media Sosial", "Berita"])
    
    st.divider()
    
    st.subheader("🤖 Model AI")
    granularity = st.selectbox("Granularitas Sentimen", ["Dasar (Positif/Netral/Negatif)", "Lanjut"])
    num_clusters_input = st.slider("Jumlah Topik (Klaster)", 2, 10, 5)
    
    st.divider()
    
    st.subheader("🔧 Preprocessing")
    check_sw = st.checkbox("Hapus Stop Words", value=True)
    check_lemma = st.checkbox("Aktifkan Lemmatization", value=True)
    check_lower = st.checkbox("Case Folding (lowercase)", value=True)
    
    if st.button("Kelola Stop Words", use_container_width=True):
        manage_stopwords_dialog()

# 2. Halaman Utama
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.write("## 📝") # Bisa ganti logo gambar
with col_title:
    st.title("AnaText")
    st.write("Analisis Teks Berbasis AI")

# Setup API Key (Silent, no box warning)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = "" # User harus setup secrets agar jalan

client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o" 

# --- INPUT AREA (THE CANVAS) ---
container_input = st.container()
with container_input:
    # Dua tab: Upload atau Teks Langsung
    tab_upload, tab_text = st.tabs(["📂 Unggah Dokumen", "✍️ Teks Langsung"])
    
    input_text_list = []
    
    with tab_upload:
        st.info("Mendukung format .txt, .csv, .xlsx")
        uploaded_file = st.file_uploader("Klik atau Seret File ke Sini", type=['csv', 'xlsx', 'txt'])
        
        if uploaded_file:
            # ERROR HANDLING ENCODING (REQ NO. 2)
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df_upload = pd.read_csv(uploaded_file, encoding='latin-1')
            elif uploaded_file.name.endswith('.xlsx'):
                df_upload = pd.read_excel(uploaded_file)
            else:
                # Text file handling logic
                bytes_data = uploaded_file.read()
                try:
                    raw_text = bytes_data.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = bytes_data.decode("latin-1") # Fallback ke Windows-1252/Latin-1
                
                df_upload = pd.DataFrame(raw_text.splitlines(), columns=['Teks'])
            
            # Auto detect kolom teks jika bukan 'Teks'
            possible_cols = [c for c in df_upload.columns if df_upload[c].dtype == 'object']
            if possible_cols:
                text_col = st.selectbox("Konfirmasi Kolom Teks:", possible_cols)
                input_text_list = df_upload[text_col].dropna().astype(str).tolist()
            else:
                st.error("File tidak memiliki kolom teks yang valid.")

    with tab_text:
        direct_text = st.text_area("Tempelkan teks di sini...", height=200)
        if direct_text:
            input_text_list = [t for t in direct_text.split('\n') if t.strip()]

# --- TOMBOL ANALISIS ---
if st.button("🚀 Lakukan Analisis", type="primary"):
    if not input_text_list:
        st.warning("Mohon masukkan data teks terlebih dahulu.")
    elif not client:
        st.error("API Key belum dikonfigurasi di Secrets.")
    else:
        with st.spinner("Sedang memproses..."):
            # 1. Init Data Frame (FULL DATA, NO LIMIT)
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
            
            # 2. Preprocessing
            factory = StemmerFactory()
            stemmer = factory.create_stemmer() if (language == "Indonesia" and check_lemma) else None
            
            # Progress Preprocessing
            progress_text = "Preprocessing data..."
            bar = st.progress(0, text=progress_text)
            
            # Terapkan cleaning row by row
            clean_results = []
            for idx, text in enumerate(df['Teks_Asli']):
                cleaned = clean_text(
                    text, 
                    check_sw, 
                    check_lemma, 
                    check_lower, 
                    st.session_state.stop_words,
                    stemmer
                )
                clean_results.append(cleaned)
                bar.progress((idx+1)/len(df), text=f"Preprocessing {idx+1}/{len(df)}")
            
            df['Teks_Clean'] = clean_results
            bar.empty()

            # 3. TF-IDF & Clustering
            # Hapus baris kosong hasil cleaning
            df = df[df['Teks_Clean'].str.strip() != ""]
            
            if len(df) < num_clusters_input:
                st.warning(f"Jumlah data ({len(df)}) kurang dari jumlah klaster yang diminta ({num_clusters_input}). Klaster disesuaikan menjadi {len(df)}.")
                actual_clusters = len(df)
            else:
                actual_clusters = num_clusters_input

            vectorizer = TfidfVectorizer(max_features=2000)
            tfidf_matrix = vectorizer.fit_transform(df['Teks_Clean'])
            feature_names = vectorizer.get_feature_names_out()
            
            # KMeans Fix: Ensure n_clusters is strictly respected
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            kmeans.fit(tfidf_matrix)
            df['Cluster_ID'] = kmeans.labels_

            # 4. AI Labeling untuk Klaster
            cluster_names = {}
            for i in range(actual_clusters):
                # Cari centroid
                centroid = kmeans.cluster_centers_[i]
                # Cari kata top
                top_indices = centroid.argsort()[-5:][::-1]
                top_words = [feature_names[ind] for ind in top_indices]
                # Labeling
                if top_words:
                    label = get_topic_name_ai(client, MODEL_NAME, top_words)
                    cluster_names[i] = label
                else:
                    cluster_names[i] = f"Topik {i+1}"
            
            df['Topik'] = df['Cluster_ID'].map(cluster_names)

            # 5. Sentimen Analysis (AI) - FULL DATA
            df['Sentimen'] = get_sentiment_ai(client, MODEL_NAME, df['Teks_Asli'].tolist())

            # Simpan ke Session
            st.session_state.data = df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.analysis_done = True
            st.rerun()

# --- HASIL ANALISIS (DASHBOARD) ---
if st.session_state.analysis_done and st.session_state.data is not None:
    df = st.session_state.data
    
    st.write("---")
    st.subheader("📊 Insight Dashboard")
    
    # Overview Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dokumen", len(df))
    sentiment_counts = df['Sentimen'].value_counts()
    top_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "-"
    m2.metric("Dominasi Sentimen", top_sentiment)
    m3.metric("Jumlah Topik", df['Topik'].nunique())

    # Tabs
    tab_sum, tab_sent, tab_topic, tab_word = st.tabs([
        "Ringkasan Eksekutif", "Analisis Sentimen", "Klaster Topik", "Keyword Analysis"
    ])

    # 1. Ringkasan Eksekutif
    with tab_sum:
        st.info("Insight dihasilkan oleh AI berdasarkan keseluruhan data.")
        # Generate summary on the fly
        if st.button("Generate AI Summary"):
            with st.spinner("Menulis laporan..."):
                try:
                    summary_prompt = f"Data: {len(df)} teks. Sentimen: {sentiment_counts.to_dict()}. Topik Utama: {df['Topik'].value_counts().head(3).index.tolist()}. Buat ringkasan eksekutif paragraf pendek."
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role":"user", "content": summary_prompt}]
                    )
                    st.markdown(res.choices[0].message.content)
                except:
                    st.error("Gagal generate summary.")

    # 2. Sentimen Detail
    with tab_sent:
        col_chart, col_table = st.columns([1, 2])
        with col_chart:
            fig_sent = px.pie(
                names=sentiment_counts.index, 
                values=sentiment_counts.values, 
                hole=0.5,
                color=sentiment_counts.index,
                color_discrete_map={'Positif':'#34d399', 'Negatif':'#f87171', 'Netral':'#9ca3af'}
            )
            st.plotly_chart(fig_sent, use_container_width=True)
        
        with col_table:
            st.write("**Tabel Detail Sentimen**")
            filter_s = st.multiselect("Filter Sentimen", df['Sentimen'].unique(), default=df['Sentimen'].unique())
            df_view = df[df['Sentimen'].isin(filter_s)][['Teks_Asli', 'Sentimen']]
            
            # Styling pandas styler
            def color_sentiment(val):
                color = ''
                if val == 'Positif': color = 'background-color: #d1fae5; color: #065f46'
                elif val == 'Negatif': color = 'background-color: #fee2e2; color: #991b1b'
                elif val == 'Netral': color = 'background-color: #f3f4f6; color: #1f2937'
                return color

            st.dataframe(
                df_view.style.map(color_sentiment, subset=['Sentimen']), 
                use_container_width=True,
                height=400
            )

    # 3. Clustering
    with tab_topic:
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            topic_count = df['Topik'].value_counts().reset_index()
            topic_count.columns = ['Topik', 'Jumlah']
            fig_bar = px.bar(topic_count, x='Jumlah', y='Topik', orientation='h', color='Jumlah')
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_t2:
            st.markdown("### Daftar Topik")
            st.table(topic_count)

    # 4. Word Cloud & TF-IDF
    with tab_word:
        all_text = " ".join(df['Teks_Clean'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        
        # TF-IDF Table
        st.write("**Top Kata Unik (TF-IDF)**")
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        words_freq = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]
        st.table(pd.DataFrame(words_freq, columns=["Kata", "Skor Penting"]))
