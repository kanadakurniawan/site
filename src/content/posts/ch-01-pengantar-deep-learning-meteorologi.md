---
title: "Pengantar Deep Learning untuk Meteorologi"
description: "Bab pertama — memahami posisi deep learning dalam machine learning dan kebumian, peta aplikasi meteo yang dibahas di buku ini, kapan DL layak dipakai, serta panduan menyiapkan lingkungan Google Colab + TensorFlow."
pubDate: 2026-09-01
categories: ["Deep Learning", "Meteorologi"]
tags: ["deep learning", "meteorologi", "tensorflow", "colab", "pengantar"]
draft: true
---


# Pengantar Deep Learning untuk Meteorologi

Buku ini ditujukan sebagai **materi pengenalan**, bukan hasil riset baru. Seluruh isi
merupakan ringkasan dan penyusunan ulang dari literatur klasik machine learning, dengan
contoh-contoh yang dekat dengan dunia meteorologi Indonesia.

> **Prasyarat bab ini:** tidak ada — ini titik awal. Bab berikutnya mengasumsikan
> Bab 1 dikuasai.

## 1.1 Machine Learning dan Deep Learning

*Machine learning* (pembelajaran mesin) adalah cara membuat komputer belajar pola dari
data tanpa diprogram secara eksplisit untuk setiap aturan. Alih-alih menulis
"jika hujan kemarin dan kelembapan tinggi, maka ...", kita memberi model ribuan contoh
dan membiarkannya **menemukan sendiri** pola yang berguna — misalnya, bobot mana yang
paling menjelaskan data — melalui proses belajar yang dijelaskan di Bab 4.

*Deep learning* adalah salah satu cabang machine learning yang menggunakan **jaringan
saraf berlapis (neural network)** — model matematis yang terinspirasi dari susunan neuron
biologis. "Dalam" (deep) merujuk pada banyaknya lapisan; setiap lapisan mengekstrak
representasi yang semakin abstrak dari data [1].

![Keterkaitan artificial intelligence, machine learning, dan deep learning](figures/fig-1-1-hierarki-ai-by-chatgpt.png)

Singkatnya, *artificial intelligence* (AI) adalah payung terluas, *machine learning*
adalah cabangnya yang belajar dari data, dan *deep learning* adalah bagian dari machine
learning yang menggunakan jaringan saraf berlapis. Model *generative* modern seperti
chatbot juga termasuk deep learning — tetapi definisi dan keterbatasannya dibahas lebih
lanjut di Bab 10.

Perbedaan praktisnya: model machine learning klasik (seperti regresi, pohon keputusan,
atau SVM) umumnya butuh fitur yang dirancang manual oleh manusia, sedangkan deep learning
dapat belajar representasi langsung dari data mentah — asalkan data cukup banyak. Bab 2
akan membahas komponen dasar jaringan ini: **neuron, perceptron, dan fungsi aktivasi**.

## 1.2 Mengapa Deep Learning Relevan Sekarang

Deep learning bukan teknologi baru dalam konsep (perceptron pertama lahir 1958 [2]),
tetapi baru praktis digunakan secara luas dalam dekade terakhir karena:

1. **Data besar** — sensor otomatis, reanalysis seperti ERA5, dan arsip klimatologi
   menyediakan data meteorologi dalam jumlah besar (Bab 6 membahas sumber datanya).
2. **Komputasi murah** — GPU dan *cloud notebook* gratis seperti **Google Colab** membuat
   pelatihan jaringan saraf dapat dilakukan tanpa server mahal.
3. **Tooling matang** — TensorFlow/Keras dan PyTorch menyediakan API yang relatif mudah
   dipelajari [3].

Kombinasi ini membuat deep learning dapat diadopsi oleh mahasiswa dan praktisi kebumian —
bukan hanya peneliti ilmu komputer [4].

## 1.3 Peta Aplikasi Deep Learning dalam Meteorologi

Deep learning telah diterapkan di berbagai permasalahan meteorologi. Agar pembaca tidak
tersesat, buku ini memetakan aplikasi tersebut dan menandai mana yang **dibahas di sini**
secara mendalam dan mana yang hanya diarahkan ke literatur lanjut:

| Aplikasi | Contoh pertanyaan | Dibahas di buku ini |
|---|---|---|
| Prediksi deret waktu | Berapa tinggi pasang surut besok? Berapa hujan minggu depan? | **Bab 6–9** |
| Klasifikasi kejadian | Hujan lebat atau tidak? Level bahaya apa? | **Bab 3, 5, 9** |
| Imputasi data hilang | Bagaimana mengisi gap data stasiun? | Bab 6 (dasar), Bab 10 (generative) |
| *Nowcasting* | Apa yang terjadi 0–6 jam ke depan (radar/satelit)? | Bab 10 (arah riset) |
| *Downscaling* / data spasial | Dari skala reanalysis ke skala lokal | Bab 10 (arah riset) |

Fokus buku ini sengaja dibatasi pada **dua aplikasi inti**: prediksi besaran (regresi)
dan klasifikasi kejadian, pada data deret waktu meteorologi Indonesia. Pembaca yang ingin
melanjutkan ke CNN, NLP, atau model spasial diarahkan ke Bab 10.

## 1.4 Kapan Deep Learning Layak, Kapan Tidak

Deep learning bukan solusi untuk semua masalah. Aturan praktis yang akan dipakai di
seluruh buku:

- **Gunakan model statistik klasik dulu sebagai pembanding (baseline).** Regresi linear,
  ARIMA, atau *persistence* ("keadaan besok = keadaan hari ini") sering kali lebih dari
  cukup untuk data pendek atau pola sederhana. Deep learning hanya layak jika **mengalahkan
  baseline** dengan data yang cukup — prinsip ini menjadi tulang punggung Bab 7–9.
- **Perhatikan ukuran data.** Jaringan saraf besar membutuhkan banyak data; untuk deret
  waktu stasiun dengan puluhan ribu pengamatan, model *sequence* seperti LSTM (Bab 7)
  adalah pilihan yang masuk akal, bukan model raksasa.
- **Utamakan keterbacaan dan kepercayaan di konteks operasional.** Di lingkungan seperti
  BMKG, model yang sederhana dan dapat dijelaskan kadang lebih diterima daripada model
  "kotak hitam" — Bab 10 membahas interpretasi dan keterbatasan.

Mental model ini menjaga pembaca dari *overhype*: deep learning adalah satu alat di dalam
kotak peralatan, bukan pengganti semua metode.

## 1.5 Menyiapkan Lingkungan Kerja

Buku ini menggunakan **Google Colab** (gratis) dengan **TensorFlow/Keras** [3]. Colab
berjalan di browser tanpa instalasi lokal dan menyediakan GPU untuk latihan.

Untuk memulai, buka colab.research.google.com, buat notebook baru, lalu jalankan:

```python
import tensorflow as tf

print(tf.__version__)
print("GPU tersedia:", tf.config.list_physical_devices("GPU"))
```

Notebook pendamping bab ini (`00_fondasi_tensorflow`) berisi langkah verifikasi dan latihan
pertama. Pastikan versi TensorFlow yang terinstal sesuai dengan versi yang tercantum pada
metadata bab, agar hasil dapat direproduksi.

## 1.6 Tensor Pertama dengan Contoh Data Cuaca Mini

Semua data yang masuk ke jaringan saraf direpresentasikan sebagai **tensor** — generalisasi
matriks ke dimensi berapa pun. Dengan contoh data cuaca:

| Contoh | Bentuk | Dimensi (rank) |
|---|---|---|
| Suhu satu pengamatan, `26.5` | skalar (tensor 0D) | 0 |
| Suhu min & max satu hari, `[26.5, 31.0]` | vektor (tensor 1D) | 1 |
| Suhu min & max 3 hari, `[[26.5, 31.0], [26.8, 30.5], [27.2, 32.1]]` | matriks (tensor 2D) | 2 |
| Suhu min & max di 3 stasiun selama 3 hari | tensor 3D | 3 |

Dalam TensorFlow, tensor dibuat dengan `tf.constant` atau `tf.Variable`:

```python
import tensorflow as tf

suhu_hari = tf.constant([26.5, 26.8, 27.2])      # 1D: 3 pengamatan
print(suhu_hari.shape)                            # (3,)
```

Bentuk (`shape`) tensor inilah yang nantinya menentukan *input shape* arsitektur model di
Bab 2 dan Bab 7. Pembaca tidak perlu menguasai aljabar tensor secara mendalam — yang penting
adalah memahami (a) urutan dimensi, dan (b) bahwa data temporal akan disusun menjadi
*timesteps* saat masuk ke model sekuensial.

## Ringkasan

- Deep learning adalah cabang machine learning berbasis jaringan saraf berlapis; relevan
  untuk meteorologi karena data besar, komputasi murah, dan tooling matang.
- Fokus buku: prediksi deret waktu dan klasifikasi kejadian untuk data Indonesia; aplikasi
  lain (nowcasting, downscaling, generative) diarahkan ke Bab 10.
- Deep learning layak digunakan jika mengalahkan baseline yang sederhana; jangan pernah
  memulai tanpa pembanding.
- Lingkungan kerja: Google Colab + TensorFlow/Keras; semua data dibawa sebagai tensor.

## References

1. Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553,
   pp. 436–444, May 2015.
2. F. Rosenblatt, "The perceptron: A probabilistic model for information storage and
   organization in the brain," *Psychological Review*, vol. 65, no. 6, pp. 386–408, 1958.
3. M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems,"
   2015. [Online]. Available: https://www.tensorflow.org/
4. M. Reichstein et al., "Deep learning and process understanding for data-driven Earth
   system science," *Nature*, vol. 566, no. 7743, pp. 195–204, Feb. 2019.