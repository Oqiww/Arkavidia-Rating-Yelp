# Yelp Review Rating Prediction - Datavidia 10.0 (1st Place Solution)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-88c249)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D33A)

Repository ini berisi *source code* dan dokumentasi untuk solusi **Juara 1** di kompetisi data science **Datavidia Arkavidia 10.0** (Yelp Review Rating Prediction). 

**Tim NeedNamaTim:**
* Zaky Muhammad Fauzi
* Syauqi Gathan Setyapratama
* Ghifary Wibisono

---

## Challenge Overview
Tujuan utama dari tugas ini adalah membangun model *end-to-end* yang mampu mengintegrasikan data ulasan mentah pelanggan (sekitar 5,5 juta baris) dengan berbagai metadata pendukung untuk memprediksi kepuasan pelanggan berupa rating ordinal (1-5 bintang). 

Tantangan utama kompetisi ini meliputi:
* **Rich Text Data**: Menggali kompleksitas semantik dari teks ulasan yang bervariasi.
* **Scattered Metadata**: Atribut penting (seperti status *elite* pengguna atau metrik bisnis) tersebar di berbagai file JSON (`business`, `user`, `tip`, `checkin`).
* **Class Imbalance**: Proporsi target tidak seimbang, di mana 39% dari seluruh ulasan merupakan rating 5 bintang.
* **Strict Evaluation**: Metrik **Quadratic Weighted Kappa (QWK)** yang memberikan penalti eksponensial terhadap tebakan yang jarak antar kelasnya meleset jauh.

---

## Architecture & Methodology
Solusi kami menggunakan pendekatan **Hybrid Intelligence**, menggabungkan stabilitas fitur tabular tradisional dengan pemahaman semantik dari model *Deep Learning*. Pipeline ini terdiri dari 4 pilar utama:

### 1. Strict OOF Target Encoding
Mencegah *target leakage* yang sering terjadi dengan menerapkan teknik *leave-one-out*. Saat menghitung historis rata-rata bintang untuk suatu restoran/pengguna pada baris data latih tertentu, rating dari baris tersebut dikurangkan dari agregat populasi.

### 2. Metadata Enrichment & Feature Engineering
* **Atribut Bisnis & Pengguna**: Mengekstrak fitur prediktif langsung dari file JSON seperti `RestaurantsPriceRange2`, ketersediaan `WiFi`, kategori bisnis (Top 20), durasi tahun profil elite, dan ukuran jejaring (`friends_count`).
* **Lexicon-Based Sentiment**: Menggunakan `vaderSentiment` untuk mengekstrak skor *Compound*. Skor ini sangat terkalibrasi untuk menangkap intensi dan polaritas sentimen (positif/negatif) dari struktur teks ulasan pelanggan.
* **Sinyal Popularitas & Interaksi**: Mengakumulasi metrik seperti jumlah jempol (`useful`, `funny`, `cool`), jumlah pujian (tips), dan frekuensi check-in bisnis.

### 3. NLP Injection (Fast DistilBERT)
* Memanfaatkan arsitektur `distilbert-base-uncased` dengan membekukan (*freeze*) *embedding layer* dan 4 lapisan pertama agar komputasi menjadi sangat ringan.
* Model bahasa ini di-*fine-tune* selama 1 Epoch menggunakan *Mixed Precision* (FP16).
* *Output logit* regresi dari DistilBERT diekstrak dan disimpan sebagai satu kolom meta-fitur numerik yang memiliki daya prediktif sangat tinggi (`nlp_prediction`).

### 4. The Grand Ensemble (LightGBM)
* Menggabungkan matriks *dense* tabular (25 fitur numerik) dengan matriks teks *TF-IDF N-grams* (10.000 fitur) ke dalam format **SciPy CSR Sparse Matrix** yang hemat memori.
* Meta-model **LightGBM Regressor** dilatih di atas representasi *sparse* ini menggunakan **5-Fold Stratified Cross Validation** dengan objektif *MSE Loss*.
* **Threshold Optimization**: Alih-alih melakukan pembulatan standar (`round()`), kami memformulasikan penentuan kelas bintang sebagai optimasi matematis menggunakan algoritma **Nelder-Mead** untuk mencari batasan potong (*optimal thresholds*) yang paling memaksimalkan metrik QWK.

---

## Performance
* **Local CV (5-Fold OOF QWK)**: `0.91262`
* **Leaderboard Rank**: 1st Place 

---


