import numpy as np
import pandas as pd
import streamlit as st

def recommend_top_n(user_id, model, movies, user_enc, genre_matrix, top_n=10, genre_filter=None):
    """
    Menghasilkan Top-N rekomendasi untuk user_id berdasar model (DLRM atau NCF+CNN).
    - Model DLRM hanya mengharapkan input: 'user_idx', 'movie_idx', 'genre', 'year'.
    - Model NCF+CNN mengharapkan keempat input di atas, plus 'title_seq'.
    """
    try:
        user_id = int(user_id)
    except ValueError:
        st.error("User ID harus berupa angka.")
        return pd.DataFrame()

    if user_id not in user_enc.classes_:
        st.warning("User ID tidak ditemukan di data latih.")
        return pd.DataFrame()

    # Ambil encoded index user
    u = user_enc.transform([user_id])[0]

    # Filter film berdasarkan genre jika diperlukan
    if genre_filter and genre_filter != "Semua Genre":
        genre_mask     = movies['genres_list'].apply(lambda g: genre_filter in g)
        filtered_movies = movies[genre_mask].reset_index(drop=True)
        genre_input    = genre_matrix[genre_mask.values]
    else:
        filtered_movies = movies.copy().reset_index(drop=True)
        genre_input     = genre_matrix

    if len(filtered_movies) == 0:
        st.warning("Tidak ada film dengan genre tersebut.")
        return pd.DataFrame()

    # 1) Ambil array movie_idxs dan user_vec
    movie_idxs = filtered_movies['movieId_enc'].values.astype(np.int32)
    user_vec   = np.full((len(filtered_movies),), u, dtype=np.int32)

    # 2) Ambil array tahun
    year_input = filtered_movies['year'].values.astype(np.float32)

    # 3) Siapkan dictionary input dasar
    input_data = {
        'user_idx':  user_vec,       # (num_films,)
        'movie_idx': movie_idxs,     # (num_films,)
        'genre':     genre_input,    # (num_films, n_unique_genres)
        'year':      year_input      # (num_films,)
    }

    # 4) Ambil daftar nama input layer model dari model.inputs
    input_layer_names = [inp.name.split(':')[0] for inp in model.inputs]

    # 5) Jika model menuntut 'title_seq', isi juga dari movies['title_seq']
    if 'title_seq' in input_layer_names:
        # filtered_movies['title_seq'] adalah list/ndarray panjang MAX_TITLE_LEN
        title_seq_list  = filtered_movies['title_seq'].tolist()  
        title_seq_batch = np.stack(title_seq_list).astype(np.int32)
        input_data['title_seq'] = title_seq_batch

    # 6) Panggil model.predict dan flatten prediksi
    preds = model.predict(input_data, batch_size=512).flatten()

    # 7) Ambil index top-N berdasarkan skor tertinggi
    topn_idx = np.argsort(preds)[::-1][:top_n]
    result   = filtered_movies.iloc[topn_idx].copy()

    # 8) Tambahkan kolom skor prediksi dan rata-rata rating asli
    result['pred_rating']       = preds[topn_idx]
    result['actual_avg_rating'] = result['actual_avg_rating'].fillna(0.0)

    return result[['title', 'genres', 'year', 'pred_rating', 'actual_avg_rating']]
