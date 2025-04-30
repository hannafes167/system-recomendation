# Laporan Proyek Machine Learning - Hanna Febriani Sutarman

## Project Overview

Rekomendasi buku adalah fitur penting dalam dunia literasi digital, membantu pengguna menemukan buku baru yang sesuai dengan preferensi mereka. Dalam proyek ini, dibangun sistem rekomendasi buku menggunakan dua pendekatan utama, yaitu Content Based Filtering (CBF) dan Collaborative Filtering (CF), berdasarkan dataset dari Goodreads.

**Proyek ini penting karena:**
  - Membantu pengguna memilih buku dengan lebih mudah dan personal.
  - Memberikan pengalaman pengguna yang lebih baik di platform berbasis literasi
  - Mengimplementasikan metode machine learning populer dalam dunia nyata.
    
**Referensi:**
  - Seperti dijelaskan oleh Towards Data Science (n.d.), sistem rekomendasi dapat dibangun dengan pendekatan berbasis konten dan kolaboratif, dengan menggunakan metrik
    evaluasi seperti Precision@K untuk mengukur relevansi hasil rekomendasi.
  - Menurut Ricci et al. (2011), pendekatan berbasis konten memanfaatkan fitur dari item untuk mencari kemiripan, sedangkan pendekatan kolaboratif mencari pola perilaku
    antar pengguna.



## Business Understanding

### Problem Statements
 - Bagaimana merekomendasikan buku berdasarkan kesamaan konten.
 - Bagaimana merekomendasikan buku berdasarkan perilaku pengguna lain yang mirip?

### Goals
  - Membuat model Content Based Filtering untuk merekomendasikan buku berdasarkan kesamaan konten.
  - Membuat model Collaborative Filtering untuk merekomendasikan buku berdasarkan perilaku pengguna lain yang mirip.
    
**Solution Approach**
### Solution Statements:
  - Menggunakan Content Based Filtering dengan teknik TF-IDF pada judul buku untuk menghitung kemiripan antar buku.
  - Menggunakan Collaborative Filtering berbasis user similarity dengan cosine similarity untuk menemukan pengguna serupa.


## Data Understanding
**Dataset yang digunakan adalah:**
  - books.csv: berisi informasi tentang buku (judul, penulis, tahun terbit, jumlah rating, dll).
  - ratings.csv: berisi informasi rating yang diberikan pengguna ke buku.
  - Dataset bersumber dari: Kaggle - Goodreads Books (https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)

**Jumlah data:**
  - Books: RangeIndex: 10000 entries, 0 to 9999 Data columns (total 23 columns)
  - Ratings: RangeIndex: 981756 entries, 0 to 981755 Data columns (total 3 columns)

**Fitur-fitur:**

File books
- id:	ID unik dalam dataset ini (mungkin hanya untuk indexing internal).
- book_id:	ID unik buku di platform Goodreads.
- best_book_id:	ID untuk versi terbaik dari buku tersebut (mungkin hardcover/edisi populer).
- work_id:	ID untuk karya secara keseluruhan (semua edisi buku yang sama).
- books_count:	Jumlah total edisi buku (paperback, hardcover, dll).
- isbn:	Kode ISBN (International Standard Book Number) versi 10 digit.
- isbn13:	ISBN versi 13 digit (lebih baru dan standar saat ini).
- authors:	Nama penulis buku.
- original_publication_year:	Tahun pertama kali buku diterbitkan.
- original_title:	Judul asli buku (mungkin berbeda dengan versi terjemahan/edisi baru).
- title:	Judul buku pada edisi ini.language_code	Kode bahasa buku (contoh: "eng" untuk English).
- average_rating:	Rata-rata rating buku dari semua pengguna.
- ratings_count:	Total rating yang diberikan untuk buku ini (semua versi edisi).
- work_ratings_count:	Total rating untuk work_id-nya (gabungan semua edisi).
- work_text_reviews_count:	Jumlah ulasan tertulis (bukan hanya bintang).
- ratings_1: - ratings_5	Jumlah pengguna yang memberi rating 1 sampai 5 bintang.
- image_url:	URL ke gambar sampul buku (ukuran besar).
- small_image_url:	URL ke gambar sampul versi kecil.

File ratings
- book_id:	ID buku yang diberi rating (terkait dengan book_id di dataset books).
- user_id:	ID pengguna yang memberi rating.
- rating:	Nilai rating yang diberikan oleh pengguna (biasanya dari 1–5).
  
**Kondisi Data:**
  - Terdapat missing values pada kolom: (isbn: 700, isbn13: 585, original_publication_year: 21, original_title: 585, language_code: 1.084)
  - Jumlah pengguna unik: 53.424
  - Jumlah buku unik yang diberi rating: 10.000
  - Visualisasi:
     - Distribusi nilai rating yang diberikan pengguna menunjukkan kecenderungan rating tinggi atau rendah. Yang terbanyak adalah rating 4 dan terendah rating 1.
     - Buku dengan jumlah rating terbanyak, yang bisa menjadi kandidat populer untuk rekomendasi. Top 10 buku dengan jumlah rating terbanyak adalah The End of Poverty, Harry Potter and the Half-Blood Prince (Harry Potter, #6), Harry Potter and the Order of the Phoenix (Harry Potter, #5), Burmese Days, Galapagos, The Lover, The Potrait of a Lady, Tropic of Cancer, I am Chariotte Simmons.
     - Penguna yang paling aktif memberi rating, penting untuk analisis perilaku dan filtering strategi. Pengguna paling aktif dengan memberikan rating sebanyak 199.

## Data Preparation
1. Pengecekan dan penanganan missing value:
   - Kolom seperti isbn, isbn13, original_publication_year, original_title, dan language_code memiliki missing value. Namun, kolom-kolom tersebut tidak digunakan dalam
   pemodelan, sehingga dibiarkan kosong tanpa mempengaruhi hasil akhir.
   - Alasan: Fokus hanya pada kolom yang relevan untuk sistem rekomendasi, sehingga tidak perlu membersihkan kolom yang tidak dipakai.
2. Pemilihan kolom:
   - Dari dataset books: diambil kolom book_id, title, authors, dan average_rating.
   - Dari dataset ratings: digunakan seluruh kolom (user_id, book_id, rating).
   - Alasan: Kolom-kolom ini cukup untuk membangun sistem CBF (berdasarkan konten buku) dan CF (berdasarkan pola rating pengguna).
3. Filtering data:
   - Hanya dipertahankan buku yang memiliki minimal 50 rating, dan pengguna yang memberikan minimal 50 rating.
   - Jumlah data setelah filtering: 421.012 baris.
   - Alasan: Untuk mengurangi sparsity (kekosongan) dalam matriks user-item dan meningkatkan kualitas rekomendasi.
4. Penggabungan data (merge):
   - Dataset books dan ratings digabungkan berdasarkan book_id, menghasilkan data baru bernama ratings_final yang berisi rating beserta informasi buku.
   - Alasan: Untuk mempermudah pemodelan dan pengambilan fitur konten seperti judul dan penulis.
5. Persiapan untuk Content-Based Filtering (CBF):
   - Menggabungkan kolom title dan authors sebagai deskripsi konten buku.
   - Menggunakan TF-IDF Vectorizer untuk mengubah teks menjadi vektor numerik.
   - Mengisi nilai kosong pada fitur gabungan jika ada.
   - Alasan: TF-IDF digunakan untuk menangkap kemiripan antar buku berdasarkan kata-kata penting dalam judul dan nama penulis.
6. Persiapan untuk Collaborative Filtering (CF):
   - Membuat user-item matrix dari data ratings_final, di mana baris merepresentasikan pengguna dan kolom merepresentasikan buku.
   - Matriks ini digunakan untuk menghitung kemiripan antar pengguna menggunakan pendekatan K-Nearest Neighbors (KNN).
   - Alasan: CF merekomendasikan buku berdasarkan preferensi pengguna yang mirip, sehingga memerlukan pola interaksi user-item.
   

## Modeling
**1. Content Based Filtering (CBF)**
  - Metode: TF-IDF Vectorization + Cosine Similarity.
  - Output: Top-5 buku yang mirip berdasarkan input judul buku.
  - Kelebihan:
     - Tidak memerlukan data pengguna lain.
     - Bagus untuk buku-buku baru (cold start problem di user side).
  - Kekurangan:
     - Rekomendasi terbatas hanya pada fitur konten yang digunakan (judul).

**2. Collaborative Filtering (CF)**
  - Metode: User-Based Collaborative Filtering menggunakan Cosine Similarity.
  - Output: Top-5 buku berdasarkan pengguna serupa.
  - Kelebihan:
     - Dapat merekomendasikan buku yang sangat beragam.
     - Mengandalkan pengalaman komunitas pengguna.
  - Kekurangan:
     - Membutuhkan data interaksi pengguna dalam jumlah besar.
     - Rentan terhadap cold start problem (untuk pengguna baru).

**Top 5 recommendation**
- Rekomendasi model CBF untuk buku The Women in the Dunes: ['Me and Earl and the Dying Girl', 'The Jungle', 'The Fault in Our Stars', 'Paper Towns', 'Looking for Alaska'] Buku-buku ini memiliki tema emosional, psikologis, atau eksistensial yang relevan.
- Rekomendasi model CF untuk pengguna ID 23637: ['Islands in the Stream', 'Boy: Tales of Childhood', 'The Path Between the Seas', 'A Briefer History of Time', 'Play It as It Lays'] Rekomendasi ini mencerminkan minat pengguna terhadap sejarah, biografi, dan sains populer.

## Evaluation

Pada tahap evaluasi, dilakukan pengukuran performa model rekomendasi untuk dua pendekatan: Collaborative Filtering (CF) dan Content-Based Filtering (CBF). Masing-masing pendekatan dievaluasi dengan metode yang sesuai dengan karakteristik model dan data.

1. Collaborative Filtering (CF)
   Model CF dievaluasi menggunakan metrik Precision@5 karena cocok untuk menilai kualitas rekomendasi berdasarkan perilaku pengguna. Metrik ini mengukur proporsi item yang
   relevan di antara 5 item teratas yang direkomendasikan.
   
   Rumus:
   
   Precision@K = (Jumlah item relevan dalam K rekomendasi teratas) / K
   
   Dalam evaluasi menggunakan data pengguna ID 23637, diperoleh hasil:
   
   Precision@5 = 0.40
   
   Artinya, dari 5 buku yang direkomendasikan, 2 buku memiliki rating ≥ 4, yang menunjukkan relevansi tinggi terhadap preferensi pengguna tersebut.
   Hasil ini dapat dianggap cukup baik mengingat keterbatasan model sederhana dan data yang sparse. Ke depan, performa model CF dapat ditingkatkan dengan metode matrix
   factorization atau pendekatan hybrid filtering.

3. Content-Based Filtering (CBF)
   Berbeda dengan CF, model CBF tidak dapat dievaluasi secara kuantitatif karena tidak ada label eksplisit (rating) yang menunjukkan relevansi item. Oleh karena itu,
   evaluasi dilakukan secara kualitatif dengan membandingkan kesamaan konten antar buku.

   Model CBF memberikan rekomendasi berikut untuk buku “The Women in the Dunes”:
   - Me and Earl and the Dying Girl
   - The Jungle
   - The Fault in Our Stars
   - Paper Towns
   - Looking for Alaska

     Buku-buku tersebut memiliki tema yang sejalan, yaitu psikologis, emosional, dan eksistensial. Ini menunjukkan bahwa model mampu menangkap fitur tematik dan narati
     dari buku input.
     
     **Analisis Kualitatif:**
     - “The Fault in Our Stars” dan “Looking for Alaska” membahas topik kehilangan, emosi, dan makna hidup.
     - “The Jungle” serta “Me and Earl and the Dying Girl” membawa tema perjuangan hidup dan refleksi diri, sesuai dengan nuansa eksistensial dalam “The Women in the Dunes”.

## References
Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2011). Introduction to recommender systems handbook. Springer.

Towards Data Science. (n.d.). Building a recommendation system. Retrieved April 30, 2025, from https://towardsdatascience.com/building-a-recommendation-system
