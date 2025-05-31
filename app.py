import streamlit as st
import tensorflow as tf
from utils.preprocessing import load_and_prepare_data
from utils.recommender import recommend_top_n

st.title("🎬 Sistem Rekomendasi Film Menggunakan Deep Learning")

# 1. Load data & preprocessing
ratings, movies, movie_enc, user_enc, genre_matrix, genre_list = load_and_prepare_data()

# 2. Load model tanpa Lambda layer
try:
    model_dlrm      = tf.keras.models.load_model("models/dlrm_model.h5", compile=False)
    model_ncf_cnn   = tf.keras.models.load_model("models/ncf_cnn_model.h5", compile=False)
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan model sudah disimpan ulang tanpa Lambda layer.")
    st.stop()

# 3. Pilihan model (UI)
model_option = st.selectbox("📌 Pilih model:", ["DLRM", "NCF+CNN"])

# 4. Input User ID
user_id = st.number_input(
    "🧑 Masukkan User ID:",
    min_value=int(ratings['userId'].min()),
    max_value=int(ratings['userId'].max()),
    value=int(ratings['userId'].min())
)

# 5. Filter genre
selected_genre = st.selectbox("🎞️ Filter Genre (Opsional)", ["Semua Genre"] + genre_list)

# 6. Top-N
top_n = st.slider("🔢 Jumlah rekomendasi (Top-N):", 5, 20, 10)

# 7. Tombol rekomendasi
if st.button("📽️ Tampilkan Rekomendasi"):
    # Pilih model yang dipakai
    model = model_dlrm if model_option == "DLRM" else model_ncf_cnn

    # Panggil rekomendasi
    result = recommend_top_n(
        user_id,
        model,
        movies,
        user_enc,
        genre_matrix,
        top_n,
        selected_genre
    )

    if not result.empty:
        st.subheader(f"🎯 Rekomendasi untuk User {user_id} – Model: {model_option}")
        if selected_genre != "Semua Genre":
            st.caption(f"Hanya menampilkan film dengan genre: **{selected_genre}**")
        for _, row in result.iterrows():
            st.write(f"- 🎬 **{row['title']}**")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📅 Tahun: `{int(row['year'])}`")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📊 Prediksi: `{row['pred_rating']:.2f}`")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;⭐ Rating Aktual: `{row['actual_avg_rating']:.2f}`")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🎞️ Genre: `{row['genres']}`")
    else:
        st.warning("Tidak ada rekomendasi ditemukan untuk user ini.")
