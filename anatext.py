import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="TextInsight AI - Analisis Teks Cerdas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM UNTUK UI (CARD STYLE & COLORING) ---
st.markdown("""
<style>
    .stTextArea textarea {font-size: 16px !important;}
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 10px;}
    .reportview-container .main .block-container{padding-top: 2rem;}
    /* Highlight Sentimen */
    .positive-bg {background-color: #d4edda; color: #155724; padding: 5px; border-radius: 5px;}
    .negative-bg {background-color: #f8d7da; color: #721c24; padding: 5px; border-radius: 5px;}
    .neutral-bg {background-color: #fff3cd; color: #856404; padding: 5px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- SETUP OPENAI CLIENT ---
# Mengambil API Key dari st.secrets (untuk keamanan saat deploy)
# Jika dijalankan lokal tanpa secrets.toml, bisa fallback ke input manual atau hardcode (tidak disarankan)
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except:
    api_key = st.sidebar.text_input("Masukkan OpenAI API Key Anda:", type="password")

client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = "gpt-4o" # Menggunakan gpt-4o (setara/lebih baik dari 4.1 untuk saat ini)

# --- FUNGSI UTILITAS & AI ---

def get_sentiment_ai(text_list):
    """Mengirim batch teks ke OpenAI untuk analisis sentimen."""
    results = []
    # Progress bar
    progress_bar = st.progress(0)
    
    # Batching untuk efisiensi (mengirim 1 per 1 agar akurat, atau batching json untuk hemat)
    # Di sini kita loop sederhana untuk demonstrasi kejelasan
    for i, text in enumerate(text_list):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Anda adalah analis sentimen. Tentukan sentimen teks berikut. Jawab HANYA dengan satu kata: 'Positif', 'Negatif', atau 'Netral'."},
                    {"role": "user", "content": f"Teks: {text}"}
                ],
                temperature=0
            )
            sentiment = response.choices[0].message.content.strip()
            results.append(sentiment)
        except Exception as e:
            results.append("Error")
        progress_bar.progress((i + 1) / len(text_list))
    
    progress_bar.empty()
    return results

def generate_ai_summary(df_summary, top_keywords):
    """Meminta AI membuat Executive Summary."""
    context = f"""
    Data Statistik:
    - Total Data: {df_summary['total']}
    - Sentimen: {df_summary['sentiment_dist']}
    - Topik Terbanyak mencakup {df_summary['top_topic_perc']}% data.
    - Kata kunci TF-IDF teratas: {', '.join(top_keywords)}
    
    Buatlah 'Ringkasan Eksekutif' profesional dalam bahasa Indonesia. Jelaskan sentimen mayoritas, apa yang sedang hangat dibicarakan (berdasarkan kata kunci), dan saran singkat.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Anda adalah konsultan data expert."},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Gagal menghasilkan ringkasan AI. Periksa API Key."

def get_topic_label(keywords):
    """Meminta AI memberi nama topik berdasarkan kata kunci klaster."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Berikan nama topik singkat (maks 4 kata) yang merepresentasikan kata-kata kunci berikut."},
                {"role": "user", "content": f"Keywords: {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content.replace('"', '')
    except:
        return "Topik Umum"

def interpret_tfidf(keywords):
    """AI menjelaskan mengapa kata-kata ini penting dalam TF-IDF."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Jelaskan secara singkat insight bisnis dari kata-kata kunci TF-IDF ini."},
                {"role": "user", "content": f"Kata unik (TF-IDF High Score): {', '.join(keywords)}"}
            ]
        )
        return response.choices[0].message.content
    except:
        return "Interpretasi tidak tersedia."

# --- SIDEBAR: PENGATURAN ---
with st.sidebar:
    st.header("⚙️ Pengaturan Analisis")
    
    st.subheader("Preprocessing")
    language = st.selectbox("Bahasa Teks", ["Indonesia", "Inggris"])
    
    default_stopwords = "yang, di, dan, itu, dengan, untuk, tidak, ini, dari, dalam, akan, pada, juga, saya, adalah, ke, karena, bisa, ada, mereka, kita, kamu"
    stopwords_input = st.text_area("Stop Words (pisahkan koma)", value=default_stopwords, height=100)
    stop_words_list = [w.strip() for w in stopwords_input.split(',')]
    
    st.subheader("Kontrol Model")
    num_clusters = st.slider("Jumlah Topik (Klaster)", min_value=2, max_value=10, value=3)
    
    st.info("Pastikan API Key OpenAI sudah terpasang di Secrets atau Input Field.")

# --- HALAMAN 1: THE CANVAS (INPUT) ---
st.title("🧠 TextInsight AI")
st.markdown("### Platform Analisis Teks Terintegrasi")

# State Management
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("Tempelkan Teks (Satu baris per dokumen/kalimat)", height=200, placeholder="Contoh:\nProduk ini sangat bagus.\nPelayanan lambat sekali.\nHarga terlalu mahal untuk kualitas ini.")

with col2:
    uploaded_file = st.file_uploader("Atau Unggah File (.csv, .xlsx, .txt)", type=['csv', 'xlsx', 'txt'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df_upload = pd.read_excel(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")
            df_upload = pd.DataFrame(raw_text.splitlines(), columns=['Teks'])
        
        # Deteksi kolom teks
        text_col = st.selectbox("Pilih Kolom Teks", df_upload.columns)
        input_text_list = df_upload[text_col].astype(str).tolist()
    else:
        input_text_list = []

# Tombol Aksi Utama
if st.button("🚀 Lakukan Analisis Komprehensif", type="primary", use_container_width=True):
    if not input_text and not input_text_list:
        st.error("Mohon masukkan teks atau unggah file terlebih dahulu.")
    elif not client:
        st.error("API Key belum dimasukkan.")
    else:
        with st.spinner("Sedang memproses... (TF-IDF, Clustering, Sentiment AI Analysis)"):
            # 1. Persiapan Data
            if not input_text_list:
                input_text_list = [line for line in input_text.split('\n') if line.strip()]
            
            df = pd.DataFrame(input_text_list, columns=['Teks_Asli'])
            
            # Limitasi data untuk demo (agar tidak boros token OpenAI user saat testing)
            # Anda bisa menghapus limit ini untuk produksi
            if len(df) > 50:
                st.warning("Menampilkan analisis untuk 50 baris pertama demi efisiensi API Token.")
                df = df.head(50)

            # 2. Sentimen Analisis (Via OpenAI)
            df['Sentimen'] = get_sentiment_ai(df['Teks_Asli'].tolist())

            # 3. Preprocessing Lokal (TF-IDF)
            tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_features=1000)
            try:
                tfidf_matrix = tfidf_vectorizer.fit_transform(df['Teks_Asli'])
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                # 4. Klasterisasi (K-Means)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                kmeans.fit(tfidf_matrix)
                df['Cluster_ID'] = kmeans.labels_
                
                # 5. Labeling Topik per Klaster (AI & TF-IDF keywords)
                cluster_names = {}
                for i in range(num_clusters):
                    # Ambil centroid
                    centroid = kmeans.cluster_centers_[i]
                    # Ambil top 5 kata dari centroid
                    top_indices = centroid.argsort()[-5:][::-1]
                    top_words = [feature_names[ind] for ind in top_indices]
                    # Minta AI namai topiknya
                    topic_label = get_topic_label(top_words)
                    cluster_names[i] = topic_label
                
                df['Topik'] = df['Cluster_ID'].map(cluster_names)
                
                # Simpan ke session state
                st.session_state.data = df
                st.session_state.vectorizer = tfidf_vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.feature_names = feature_names
                st.session_state.analysis_done = True
                
            except ValueError:
                st.error("Teks terlalu sedikit atau stop words menghilangkan semua kata. Coba kurangi stop words.")

# --- HALAMAN 2: THE INSIGHT DASHBOARD ---
if st.session_state.analysis_done:
    df = st.session_state.data
    
    st.divider()
    
    # TABS NAVIGASI
    tab1, tab2, tab3, tab4 = st.tabs(["📑 Ringkasan Eksekutif", "🎭 Analisis Sentimen", "Topic Clustering", "🔠 Kata Kunci & TF-IDF"])
    
    # --- TAB 1: RINGKASAN EKSEKUTIF ---
    with tab1:
        st.subheader("💡 Interpretasi AI: Ringkasan Eksekutif")
        
        # Persiapan data ringkasan
        sentiment_counts = df['Sentimen'].value_counts(normalize=True).to_dict()
        top_topic = df['Topik'].value_counts(normalize=True).idxmax()
        top_topic_perc = round(df['Topik'].value_counts(normalize=True).max() * 100, 1)
        
        # Ambil top TF-IDF words global
        sum_tfidf = st.session_state.tfidf_matrix.sum(axis=0)
        tfidf_scores = [(word, sum_tfidf[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
        sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:5]
        top_keywords = [w[0] for w in sorted_tfidf]

        summary_data = {
            'total': len(df),
            'sentiment_dist': str(sentiment_counts),
            'top_topic_perc': top_topic_perc
        }
        
        with st.spinner("AI sedang menulis ringkasan..."):
            ai_summary = generate_ai_summary(summary_data, top_keywords)
            st.info(ai_summary)
            
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Dokumen", len(df))
        col_m2.metric("Dominasi Sentimen", df['Sentimen'].mode()[0])
        col_m3.metric("Topik Utama", top_topic)

    # --- TAB 2: SENTIMEN ---
    with tab2:
        st.subheader("Visualisasi Sentimen")
        
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            sent_counts = df['Sentimen'].value_counts().reset_index()
            sent_counts.columns = ['Sentimen', 'Jumlah']
            fig_pie = px.pie(sent_counts, values='Jumlah', names='Sentimen', 
                             color='Sentimen',
                             color_discrete_map={'Positif':'#28a745', 'Negatif':'#dc3545', 'Netral':'#ffc107', 'Error':'#6c757d'},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_s2:
            st.markdown("##### Filter Data Sentimen")
            filter_sent = st.multiselect("Pilih Sentimen:", options=df['Sentimen'].unique(), default=df['Sentimen'].unique())
            df_filtered = df[df['Sentimen'].isin(filter_sent)]
            
            # Styling Tabel Sentimen
            def highlight_sentiment(val):
                if val == 'Positif': return 'background-color: #d4edda; color: #155724'
                elif val == 'Negatif': return 'background-color: #f8d7da; color: #721c24'
                elif val == 'Netral': return 'background-color: #fff3cd; color: #856404'
                return ''

            st.dataframe(
                df_filtered[['Teks_Asli', 'Sentimen']].style.map(highlight_sentiment, subset=['Sentimen']),
                use_container_width=True,
                height=400
            )

    # --- TAB 3: KLUSTER TOPIK ---
    with tab3:
        st.subheader("Struktur & Distribusi Topik")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            topic_counts = df['Topik'].value_counts().reset_index()
            topic_counts.columns = ['Topik', 'Jumlah']
            fig_bar = px.bar(topic_counts, x='Jumlah', y='Topik', orientation='h', title="Distribusi Topik", color='Jumlah')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_t2:
            st.markdown("##### Detail Klaster")
            for topic in df['Topik'].unique():
                with st.expander(f"📂 {topic}"):
                    sample_texts = df[df['Topik'] == topic]['Teks_Asli'].head(3).tolist()
                    st.write("**Contoh Teks:**")
                    for t in sample_texts:
                        st.text(f"- {t}")

    # --- TAB 4: KATA KUNCI & TF-IDF ---
    with tab4:
        col_k1, col_k2 = st.columns(2)
        
        with col_k1:
            st.subheader("☁️ Word Cloud")
            text_combined = " ".join(df['Teks_Asli'])
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words_list).generate(text_combined)
            
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
        with col_k2:
            st.subheader("📊 Analisis TF-IDF (Kata Unik)")
            # Visualisasi TF-IDF Scores tertinggi
            top_n = 10
            tfidf_sum = st.session_state.tfidf_matrix.sum(axis=0)
            words_freq = [(word, tfidf_sum[0, idx]) for word, idx in st.session_state.vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
            
            df_tfidf = pd.DataFrame(words_freq, columns=['Kata', 'Skor TF-IDF'])
            fig_tfidf = px.bar(df_tfidf, x='Skor TF-IDF', y='Kata', orientation='h', color='Skor TF-IDF')
            fig_tfidf.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_tfidf, use_container_width=True)
            
            st.markdown("**Interpretasi Kata Kunci:**")
            interpretation = interpret_tfidf([w[0] for w in words_freq])
            st.info(interpretation)

    # --- FITUR EKSPOR ---
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Hasil Analisis (CSV)",
        data=csv,
        file_name='hasil_analisis_ai.csv',
        mime='text/csv',
    )