import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from src.text_preprocessor import clean_text

st.set_page_config(
    page_title="E-Commerce Sentiment Classifier",
    page_icon="💬",
    layout="wide"
)

# Header
st.title("💬 Indonesian E-Commerce Sentiment Analysis Web App")
st.markdown("Developed by **Irene Margareth** | [GitHub](https://github.com/irene120304)")
st.markdown("---")

# Load Model & Vectorizer
@st.cache_resource
def load_artifacts():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    vectorizer, model = load_artifacts()
except Exception as e:
    st.error("Model file belum ditemukan. Silakan jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

# Layout 2 Kolom
col_left, col_right = st.columns([1.2, 0.8])

with col_left:
    st.subheader("🔍 Live Sentiment Tester")
    st.write("Ketik ulasan produk e-commerce apa saja untuk menguji sentimen secara instan:")
    
    example_texts = [
        "Ketik teks sendiri...",
        "Bahan bajunya sangat nyaman, pengiriman kilat dan seller ramah banget!",
        "Kualitas sangat buruk, baru pakai sehari sudah rusak dan seller tidak mau refund.",
        "Pengemasan rapih dan barang sampai dengan selamat sesuai deskripsi."
    ]
    selected_example = st.selectbox("Contoh Ulasan Siap Pakai:", example_texts)
    
    default_text = "" if selected_example == "Ketik teks sendiri..." else selected_example
    user_input = st.text_area("Masukkan Ulasan Produk:", value=default_text, height=120, placeholder="Tulis ulasan di sini...")
    
    if st.button("🚀 Analisis Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            # Preprocessing & Predict
            cleaned = clean_text(user_input)
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            probabilities = model.predict_proba(transformed)[0]
            prob_dict = dict(zip(model.classes_, probabilities))
            
            # Tampilkan Hasil
            st.markdown("### Hasil Prediksi:")
            if prediction == "Positive":
                st.success(f"🟢 **SENTIMEN POSITIF** (Keyakinan Model: {prob_dict['Positive']:.1%})")
            else:
                st.error(f"🔴 **SENTIMEN NEGATIF** (Keyakinan Model: {prob_dict['Negative']:.1%})")
            
            # Probability Gauge/Bar
            prob_df = pd.DataFrame({
                'Sentimen': ['Positive', 'Negative'],
                'Probabilitas': [prob_dict['Positive'], prob_dict['Negative']]
            })
            fig_prob = px.bar(
                prob_df, x='Probabilitas', y='Sentimen', orientation='h',
                color='Sentimen',
                color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                text=prob_df['Probabilitas'].apply(lambda x: f"{x:.1%}")
            )
            fig_prob.update_layout(showlegend=False, height=220, xaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_prob, use_container_width=True)
            
            with st.expander("🛠️ Detail Text Preprocessing"):
                st.write("**Teks Asli:**", user_input)
                st.write("**Setelah Cleaning & Stopwords Removal:**", cleaned if cleaned else "(Teks kosong)")

with col_right:
    st.subheader("📊 Dataset & Model Architecture")
    st.info("""
    **Spesifikasi Pipeline NLP:**
    - **Language:** Bahasa Indonesia
    - **Preprocessing:** Case Folding, Punctuation/URL Cleaning, Indonesian Stopwords Removal
    - **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency)
    - **Algorithm:** Logistic Regression
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
    """)
    
    # Visualisasi kata penting (Model Coefficients)
    if hasattr(model, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        top_positive_idx = np.argsort(coefs)[-5:]
        top_negative_idx = np.argsort(coefs)[:5]
        
        top_words = [feature_names[i] for i in top_positive_idx] + [feature_names[i] for i in top_negative_idx]
        top_weights = [coefs[i] for i in top_positive_idx] + [coefs[i] for i in top_negative_idx]
        top_types = ['Positive Cue'] * 5 + ['Negative Cue'] * 5
        
        df_weights = pd.DataFrame({
            'Word': top_words,
            'Weight': top_weights,
            'Sentiment Influenced': top_types
        }).sort_values('Weight', ascending=True)
        
        fig_weights = px.bar(
            df_weights, x='Weight', y='Word', color='Sentiment Influenced',
            title="Top Influential Words in Classification",
            color_discrete_map={'Positive Cue': '#2ecc71', 'Negative Cue': '#e74c3c'},
            orientation='h'
        )
        fig_weights.update_layout(height=320)
        st.plotly_chart(fig_weights, use_container_width=True)