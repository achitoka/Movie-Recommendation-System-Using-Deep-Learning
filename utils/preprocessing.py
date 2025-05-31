import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

def load_and_prepare_data():
    # 1. Load raw CSV
    ratings = pd.read_csv("data/ratings.csv")
    movies  = pd.read_csv("data/movies.csv")

    # 2. Buat LabelEncoder untuk userId dan movieId
    user_enc  = LabelEncoder().fit(ratings['userId'])
    movie_enc = LabelEncoder().fit(ratings['movieId'])

    # 3. Filter movies agar hanya yang ada di ratings
    valid_movie_ids = set(movie_enc.classes_)
    movies = movies[movies['movieId'].isin(valid_movie_ids)].copy()

    # 4. Simpan encoding movieId
    movies['movieId_enc'] = movie_enc.transform(movies['movieId'])

    # 5. Ubah kolom 'genres' (string) jadi list of labels
    #    Contoh: "Action|Adventure|Sci-Fi" → ["Action","Adventure","Sci-Fi"]
    movies['genres_list'] = movies['genres'].apply(lambda s: s.split('|') if isinstance(s, str) else [])

    # 6. Multi‐hot encode genres dengan CountVectorizer
    count = CountVectorizer(tokenizer=lambda x: x, lowercase=False)
    genre_matrix = count.fit_transform(movies['genres_list']).toarray().astype(np.float32)
    #    → bentuk (n_movies, n_unique_genres)

    # 7. Merge average rating (actual_avg_rating)
    average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.rename(columns={'rating': 'actual_avg_rating'}, inplace=True)
    movies = movies.merge(average_ratings, on='movieId', how='left')

    # 8. ============================
    #    Tambahkan tokenisasi judul:
    #    - Fit Tokenizer di seluruh judul film
    #    - Ubah jadi sequences, kemudian pad hingga MAX_TITLE_LEN
    #    - Simpan ke kolom baru 'title_seq'
    # 9. ============================

    # 8a. Tentukan MAX_TITLE_LEN sesuai yang dipakai saat training NCF+CNN
    MAX_TITLE_LEN = 50

    # 8b. Buat direktori untuk menyimpan tokenizer (jika belum ada)
    os.makedirs("models", exist_ok=True)

    # 8c. Inisialisasi tokenizer (jumlah kata maksimal, OOV token, dll)
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(movies['title'])  # Fit di seluruh judul

    # 8d. Convert tiap judul ke list of integers, lalu pad
    sequences   = tokenizer.texts_to_sequences(movies['title'].astype(str).values)
    title_seq_matrix = pad_sequences(
        sequences,
        maxlen=MAX_TITLE_LEN,
        padding='post',
        truncating='post'
    ).astype(np.int32)
    #    → shape: (n_movies, MAX_TITLE_LEN)

    # 8e. Simpan setiap row vector ke kolom baru
    movies['title_seq'] = list(title_seq_matrix)

    # 8f. Simpan tokenizer ke disk agar konsisten saat inference
    with open("models/title_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # 10. Siapkan daftar semua genre unik (ordered)
    all_genres = sorted({ g for sublist in movies['genres_list'] for g in sublist })

    return ratings, movies, movie_enc, user_enc, genre_matrix, all_genres