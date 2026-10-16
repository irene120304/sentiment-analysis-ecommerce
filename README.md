# 💬 Indonesian E-Commerce Sentiment Analysis Web App

> **End-to-End NLP Sentiment Classification & Analytics Web App** built with Python, Scikit-Learn (TF-IDF + Logistic Regression), and Streamlit to analyze and predict customer sentiment from Indonesian e-commerce product reviews.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview
Dalam bisnis e-commerce, ulasan pelanggan (customer reviews) mengandung wawasan penting tentang kualitas produk dan layanan. Aplikasi ini menyederhanakan pemantauan reputasi produk dengan:
1. Membersihkan teks ulasan bahasa Indonesia dari noise, tanda baca, dan stopwords.
2. Mengekstrak representasi numerik menggunakan **TF-IDF Vectorization (Unigram & Bigram)**.
3. Mengklasifikasikan sentimen secara akurat ke dalam **Positif** atau **Negatif** menggunakan **Logistic Regression**.
4. Menyediakan antarmuka web interaktif berbasis **Streamlit** untuk pengujian real-time dan visualisasi kata-kata penentu sentimen.

---

## 🛠️ Tech Stack & Workflow
- **Programming:** Python
- **Natural Language Processing (NLP):** Regex, Stopwords Filtering, TF-IDF Vectorizer
- **Machine Learning:** Scikit-Learn (Logistic Regression, Train-Test Split, Metrics)
- **Interactive UI:** Streamlit, Plotly Express
- **Model Persistence:** Pickle

---

## 👩‍💻 Author
**Irene**
- **GitHub:** [@irene120304](https://github.com/irene120304)
- **LinkedIn:** [linkedin.com/in/username-irene](https://linkedin.com) *(ganti dengan URL profil LinkedIn kamu)*
- **Target Role:** Data Scientist | Machine Learning Engineer | Data Analyst

## 🚀 How to Run Locally

```bash
# Clone this repository
git clone https://github.com/irene120304/sentiment-analysis-ecommerce.git
cd sentiment-analysis-ecommerce

# Install dependencies
pip install -r requirements.txt

# Generate dataset and train the model
python train_model.py

# Launch Streamlit web app
streamlit run app.py
