import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from src.text_preprocessor import clean_text

print("🔄 Generating dataset ulasan e-commerce...")

positive_reviews = [
    "Barang bagus banget pengiriman cepat dan kurir ramah",
    "Kualitas produk original sesuai dengan deskripsi penjual recommended",
    "Sangat puas belanja di toko ini packing rapih dan bubble wrap tebal",
    "Harga terjangkau tapi kualitas mewah banget terima kasih seller",
    "Bahan adem nyaman dipakai ukuran pas sekali warna sesuai foto",
    "Respon penjual sangat cepat dan membantu produk berfungsi normal",
    "Sudah langganan beli di sini tidak pernah mengecewakan mantap",
    "Pengemasan super aman dan pengiriman kilat barang mulus tanpa cacat",
    "Suara jernih bass mantap baterai awet banget untuk harga segini",
    "Baju sangat cantik jahitan rapi bahan tebal tidak terawang bintang lima",
    "Produk original segel resmi garansi aman recommended seller top",
    "Suka banget sama warnanya real pict dan pengiriman super cepat",
    "Pelayanan sangat memuaskan barang sampai sebelum estimasi keren",
    "Barang sesuai ekspektasi bahan halus jahitan rapih terima kasih",
    "Kualitas juara harga murah meriah packing tebal mantap"
]

negative_reviews = [
    "Barang jelek tidak sesuai gambar bahan tipis dan kasar kecewa",
    "Pengiriman lambat sekali penjual tidak merespon chat parah",
    "Barang rusak saat sampai packing hanya plastik tipis tanpa bubble wrap",
    "Ukuran sangat kecil tidak muat padahal pesan ukuran besar rugi beli",
    "Warna beda jauh dengan foto produk kualitas sangat mengecewakan",
    "Produk kw tiruan tidak original jangan beli di toko ini penipu",
    "Barang cacat ada sobekan di bagian jahitan tolong kembalikan uang",
    "Baterai cepat habis dan speaker pecah tidak worth it sama sekali",
    "Seller sangat lambat balas chat komplain tidak ditanggapi buruk sekali",
    "Pesanan tidak lengkap barang yang dikirim kurang satu minta refund susah",
    "Baju bau apek dan tipis banget nyesel beli di sini kapok",
    "Barang tidak berfungsi sama sekali mati total kualitas sangat buruk",
    "Toko tidak amanah barang tidak dikirim sesuai pesanan mengecewakan",
    "Packaging hancur pas sampai dan barang pecah sangat tidak profesional",
    "Kualitas murahan gampang rusak jangan mau beli di sini zonk"
]

# Perbanyak data sintetis dengan variasi
np.random.seed(42)
reviews_data = []

for _ in range(500):
    text = np.random.choice(positive_reviews)
    reviews_data.append((text, "Positive"))
    
for _ in range(500):
    text = np.random.choice(negative_reviews)
    reviews_data.append((text, "Negative"))

df = pd.DataFrame(reviews_data, columns=['review_text', 'sentiment'])
os.makedirs('data', exist_ok=True)
df.to_csv('data/indonesian_product_reviews.csv', index=False)
print(f"✅ Dataset berhasil dibuat: data/indonesian_product_reviews.csv ({len(df)} data)")

# Preprocessing
df['cleaned_text'] = df['review_text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train_vec, y_train)

# Evaluasi
y_pred = model.predict(X_test_vec)
print(f"\n📊 Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Simpan model
os.makedirs('models', exist_ok=True)
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model & Vectorizer berhasil disimpan ke folder models/!")