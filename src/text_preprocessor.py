import re
import string

# Stopwords bahasa Indonesia umum yang tidak membawa sentimen
INDONESIAN_STOPWORDS = set([
    'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
    'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'oleh',
    'saat', 'harus', 'itu', 'adalah', 'akan', 'bisa', 'ada', 'dari', 'dengan', 'di'
])

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari URL, mention, angka, tanda baca, dan whitespace berlebih.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    words = text.split()
    words = [w for w in words if w not in INDONESIAN_STOPWORDS and len(w) > 1]
    
    return ' '.join(words)