---
title: "Pengantar Neuron Buatan dan Jaringan Neural"
description: "Bab pertama — dari inspirasi biologis neuron hingga model matematis perceptron dan konsep fungsi aktivasi yang menjadi dasar deep learning."
pubDate: 2026-09-01
categories: ["Deep Learning"]
tags: ["neuron", "perceptron", "fungsi aktivasi", "pengantar"]
draft: true
---


# Pengantar Neuron Buatan dan Jaringan Neural

Buku ini ditujukan sebagai **materi pengenalan**, bukan hasil riset baru. Seluruh isi
merupakan ringkasan dan penyusunan ulang dari literatur klasik machine learning, dengan
contoh-contoh yang dekat dengan dunia meteorologi Indonesia.

## 1.1 Dari Neuron Biologis ke Model Matematis

Deep learning terinspirasi dari cara neuron biologis bekerja: sebuah sel saraf menerima
banyak sinyal masukan, menjumlahkannya, lalu "menembak" (fire) jika totalnya melewati
ambang tertentu.

Model matematis paling sederhana, **perceptron** [1], merepresentasikan proses itu sebagai:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

dengan $x_i$ adalah fitur input, $w_i$ bobot, $b$ bias, dan $f$ fungsi aktivasi. Pada
perceptron klasik, $f$ adalah fungsi tangga (step function). Pada neural network modern,
$f$ adalah fungsi non-linear yang bisa diturunkan, seperti ReLU atau sigmoid [2].

## 1.2 Kenapa Fungsi Aktivasi Penting

Tanpa fungsi aktivasi non-linear, tumpukan lapisan linier hanyalah transformasi linier
gabungan — tidak lebih kuat dari satu lapisan. Non-linearitas inilah yang membuat jaringan
mampu memetakan hubungan rumit antara input cuaca dan output prediksi, misalnya ketinggian
pasang surut dari seri waktu sebelumnya.

Fungsi aktivasi yang umum:

| Nama | Rumus | Sifat |
|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | output (0,1) |
| ReLU | $f(x) = \max(0, x)$ | murah, banyak dipakai |
| Tanh | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | output (−1,1) |
| GELU | $f(x) = x\,\Phi(x)$ | smooth, populer di transformer |

## 1.3 Dari Perceptron ke Jaringan Berlapis

Satu perceptron hanya mampu memisahkan kelas yang *linearly separable*. Untuk menangani
pola non-linier — seperti dinamika pasang surut yang dipengaruhi banyak konstituen harmonik
[3] — neuron disusun menjadi lapisan (layer). Jaringan dengan satu lapisan tersembunyi dan
fungsi aktivasi non-linear dikenal sebagai *universal approximator* [4]: mampu mendekati
fungsi kontinu apa pun dengan akurasi sebarang, asal cukup lebar.

Kombinasi bobot, bias, dan fungsi aktivasi yang tersusun berlapis inilah yang disebut
neural network — dan bila lapisannya banyak, disebut *deep* neural network.

## 1.4 Relevansi untuk Meteorologi

Pola data meteorologi (curah hujan, angin, muka air laut) umumnya non-linear dan memiliki
komponen periodik. Neuron buatan memberikan cara untuk belajar pola tersebut langsung dari
data, tanpa harus merumuskan persamaan fisika secara eksplisit [5]. Bab-bab berikutnya akan
membahas bagaimana jaringan dilatih, diatur, dan diterapkan pada data meteorologi
Indonesia.

## References

1. F. Rosenblatt, "The perceptron: A probabilistic model for information storage and
   organization in the brain," *Psychological Review*, vol. 65, no. 6, pp. 386–408, 1958.
2. X. Glorot, A. Bordes, and Y. Bengio, "Deep sparse rectifier neural networks," in
   *Proc. 14th Int. Conf. Artificial Intelligence and Statistics*, 2011, pp. 315–323.
3. R. Pawlowicz, B. Beardsley, and S. Lentz, "Classical tidal harmonic analysis including
   error estimates in MATLAB using T_TIDE," *Computers & Geosciences*, vol. 28, no. 8,
   pp. 929–937, 2002.
4. K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal
   approximators," *Neural Networks*, vol. 2, no. 5, pp. 359–366, 1989.
5. Q. Zhang et al., "Deep learning for improved global precipitation forecasting," in
   *Proc. 31st Conf. Neural Information Processing Systems (NeurIPS)*, 2017, pp. 4872–4881.