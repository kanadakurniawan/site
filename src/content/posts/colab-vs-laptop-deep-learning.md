---
title: "Google Colab vs Komputer Sendiri: Mana yang Cukup untuk Eksperimen Deep Learning?"
description: "Perbandingan jujur Google Colab (gratis/berbayar) vs laptop vs workstation untuk eksperimen deep learning, dari setup, GPU, RAM, storage, sampai kapan Anda butuh upgrade. Ditulis dari pengalaman praktisi, bukan brosur."
pubDatetime: 2026-09-05
tags: ["Deep Learning", "tools", "google colab", "gpu", "praktisi"]
draft: false
---

> Catatan pengalaman pribadi, bukan endorsement vendor. Pilihan tergantung model, data, dan anggaran Anda. Saya pakai semua opsi di bawah untuk hal berbeda.

Pertanyaan "Colab atau laptop?" adalah salah satu yang paling sering muncul di grup belajar *deep learning* Indonesia, sering kali dari mahasiswa yang baru mulai skripsi atau praktisi yang baru dapat anggaran riset kecil. Jawaban di internet biasanya bias ke salah satu kutub: ada yang memuja Colab gratis, ada yang langsung menyuruh beli GPU Rp 30 juta. Keduanya berlebihan. Tulisan ini mencoba menjawab dengan kerangka yang lebih jujur: **pertanyaan Anda bukan tentang alat, tapi tentang model, data, dan ritme kerja Anda**. Sisanya baru bicara Colab, laptop, atau cloud VM.

## Pertanyaan yang Sebenarnya Anda Tanyakan

Jangan tanya "Colab bagus tidak?". Tanya: **model sebesar apa, data sebesar apa, seberapa sering Anda melatih**. Sebelum membandingkan fitur, lebih jujur kalau kita pegang tiga profil pengguna yang sering saya temui:

1. **Mahasiswa skripsi / baru mulai**: baru kenal TensorFlow atau PyTorch, dataset relatif kecil, target lulus dan presentasi.
2. **Peneliti aktif / praktisi**: eksperimen harian, perlu *reproducibility*, sesekali eksplorasi arsitektur besar.
3. **Tim produksi / konsultan**: dataset besar (puluhan hingga ratusan GB), *pipeline* berjalan mingguan, latensi bukan masalah tapi *throughput* adalah.

Ketiganya punya jawaban berbeda untuk pertanyaan "Colab cukup tidak?". Sisanya tulisan ini adalah menjawabnya.

## Google Colab: Gratisan yang Ajaib, Tapi Punya Batas

Colab adalah *sweet spot* untuk yang baru mulai: tidak perlu install, GPU gratis, *notebook* langsung jalan. Tapi "gratis" di sini punya arti yang sempit. Tabel di bawah pakai harga rilis publik Google saat artikel ini ditulis; angka rupiah saya pakai kurs tengah sekitar Rp 16.000 per USD sebagai pembanding saja.

| Paket | Harga (kira-kira) | GPU yang mungkin didapat | RAM | Durasi sesi | Disk | Prioritas antrian |
|---|---|---|---|---|---|---|
| **Free** | Rp 0 | Tidak dijamin (sering Tesla T4, kadang lebih rendah) | sekitar 12 GB | Maks 12 jam, idle timeout 90 menit | sekitar 78 GB, **reset tiap sesi** | Rendah (kadang dialokasikan ke TPU/CPU saja) |
| **Colab Pro** | sekitar Rp 120.000/bulan | Lebih konsisten (T4, kadang V100/A100) | sekitar 25 GB (GPU tinggi) hingga sekitar 50 GB | Lebih lama, lebih stabil | sekitar 150 GB, tetap reset | Lebih tinggi |
| **Colab Pro+** | sekitar Rp 480.000/bulan | A100 lebih diutamakan (saat tersedia) | sekitar 50 GB atau lebih | Maks 24 jam | sekitar 150 GB, tetap reset | Tertinggi |

> Sumber: <https://colab.research.google.com/signup> dan <https://colab.research.google.com/plans> (diakses 5 September 2026). Harga dan alokasi GPU bisa berubah sewaktu-waktu.

**Batasan nyata yang jarang dibahas di tutorial YouTube:**

- **Idle timeout 90 menit**. Tutup laptop, sesi putus. Kalau lagi rendering *figure* atau *preprocessing* panjang, simpan *checkpoint* sering-sering.
- **Maksimum 12 jam di Free**, bahkan Pro+ punya batas harian. Training 3 hari? Bukan di Colab.
- **Disk di-reset tiap sesi**, bukan tempat menyimpan dataset besar permanen. Drive mount cepat, tapi Drive punya kuota dan IOPS terbatas.
- **tipe GPU tidak dijamin**. Colab berhak menurunkan Anda ke CPU "kapan saja mereka mau". Sudah menjadi rahasia umum praktisi bahwa sesi pagi lebih mudah dapat T4/A100 daripada sesi sore.
- **Tidak ada SLA**. Kalau besok Google menaikkan harga atau menutup tier gratis, Anda tidak punya kompensasi.

**Kapan Colab cukup:** eksperimen kecil-menengah, mengikuti tutorial, model di bawah sekitar 50 juta parameter, dataset di bawah 10 GB, runtime total di bawah 6 jam.

## Laptop Anda Sekarang: Lebih dari Cukup untuk 80% Kasus

Ini bagian yang paling sering di-*underestimate*. CPU modern plus RAM 16 GB sudah lebih dari cukup untuk MLP, CNN kecil, dan LSTM pendek di bawah satu juta parameter dengan dataset tabular atau *time series* pendek. Banyak eksperimen **Bab 7 dan Bab 8** buku ini saya jalankan di laptop tanpa GPU sama sekali. Yang penting bukan akselerasi, tapi **kesabaran menunggu**.

Kalau punya GPU laptop, lebih baik lagi. Dua ekosistem yang perlu Anda tahu:

- **Apple Silicon (M1/M2/M3/M4)**: pakai *Metal Performance Shaders* (MPS) backend. PyTorch sudah mendukung MPS secara stabil sejak 1.13; TensorFlow mendukung lewat *PluggableDevice*. Untuk training skala kecil-menengah, performa sering setara atau lebih cepat dari laptop Windows dengan GPU RTX kelas bawah.
- **NVIDIA RTX (Windows/Linux)**: CUDA + cuDNN adalah standar *de facto*. Instalasi lebih dulu (driver, CUDA *toolkit*, *library*), tetapi hampir semua tutorial dan *paper* mengasumsikan Anda di sini.

**Keunggulan laptop yang sering dilupakan:**

- **Tidak ada batas sesi**. Tutup laptop, buka besok, training lanjut (kalau Anda simpan *checkpoint*).
- **Data lokal tetap pribadi**, tidak ada yang lewat server pihak ketiga. Penting untuk data BMKG internal atau dataset riset yang belum dipublikasi.
- **Tidak ada biaya bulanan**, modal satu kali.
- **Reproducibility lebih mudah**. File `requirements.txt`, dataset, dan model *checkpoint* semuanya di folder lokal yang sama. Anda tahu *environment* persis yang dipakai.

**Kapan laptop cukup:** *prototyping*, belajar, dataset kecil-menengah (di bawah 10 GB), model sekuensial pendek, eksperimen harian yang harus *reproducible*.

## Cloud VM / Workstation: Saat Anda Butuh Tenaga Lebih

Ketika Colab dan laptop keduanya mentok, Anda butuh VM khusus. Opsi yang umum (urut dari paling sederhana ke paling fleksibel):

- **Paperspace / RunPod / Vast.ai / Lambda Labs**: GPU *on-demand* per jam. Cocok untuk training satu *job* besar. Vast.ai sering paling murah, Paperspace paling mudah untuk pemula.
- **AWS / GCP / Azure**: *overkill* untuk eksperimen, tapi masuk akal kalau sudah produksi (ada tim DevOps, IAM, VPC, dsb.). Di luar cakupan artikel ini.

**Kapan masuk akal:** dataset di atas 50 GB, model besar (LLM, *Vision Transformer* besar), training berhari-hari, *fine-tuning* LLM, eksperimen paralel (beberapa *job* sekaligus).

**Kapan tidak masuk akal:** *budget* terbatas, eksperimen yang *intermittent*, *reproducibility* yang harus sering diulang. Jangan sewa H100 Rp 200.000/jam untuk melatih MLP 100 ribu parameter selama 10 menit. Itu biaya yang bisa Anda pakai beli GPU bekas.

## Tabel Ringkasan

| Aspek | Colab Free | Laptop (dengan GPU) | Cloud VM |
|---|---|---|---|
| **Biaya** | Rp 0 | Rp 0 (sudah dimiliki) | Rp 50.000 sampai 500.000/jam tergantung GPU |
| **GPU** | Tidak dijamin | Konsisten (tipe tetap) | Pilih tipe (A100, H100, dst.) |
| **Batas sesi** | 12 jam (Free), 24 jam (Pro+) | Tidak ada | Tidak ada |
| **Storage** | Reset tiap sesi | Lokal, persisten | Persisten (selama VM jalan) |
| **Cocok untuk** | Belajar, eksperimen kecil | Riset harian, *prototyping* | Training besar, paralel |
| **Risiko utama** | Sesi diputus tiba-tiba | Listrik atau kegagalan hardware | Tagihan membengkak |

## Kasus Nyata dari Pekerjaan Saya

Cerita pertama: eksperimen **LSTM pasang surut Kapuas** (yang juga jadi dasar Bab 8 buku ini) berjalan mulus di Colab Free. Dataset hanya sekitar 5 MB, model kecil, total runtime 30 menit. Colab jelas pilihan tepat di sini, karena saya bahkan tutup browser saat training.

Cerita kedua: percobaan **downscaling curah hujan ERA5 ke grid BMKG** dengan dataset 60 GB (10 tahun data *hourly*, seluruh domain Indonesia). Saya mulai di Colab Pro. Hasilnya: disk reset dua kali sebelum saya sempat *mount* dataset dari Drive, sesi *idle-timeout* di tengah *preprocessing* 4 jam, dan akhirnya saya pindah ke laptop lokal (Apple M2, RAM 24 GB, GPU 19-core). Training 18 jam, saya tidur, buka pagi, hasil sudah ada. **Biaya: Rp 0. Biaya Colab Pro bulan itu: sekitar Rp 120.000 untuk hasil yang lebih buruk.**

Pengalaman ini yang membentuk rekomendasi di bawah.

## Cara Memilih (3 Pertanyaan Praktis)

Sebelum memutuskan, jawab tiga pertanyaan ini dengan kacamata Anda, bukan *hype*:

1. **Seberapa sering Anda melatih?**
   Jarang (1 sampai 2 kali sebulan, eksperimen kecil) → Colab.
   Sering (harian, perlu iterasi cepat) → lokal atau *cloud*.
2. **Seberapa besar data dan model Anda?**
   Kecil (di bawah 10 GB, di bawah 50 juta parameter) → mana saja.
   Besar (di atas 50 GB, atau model besar) → *cloud* VM atau *workstation* lokal khusus.
3. **Butuh reproducibility jangka panjang?**
   Ya (*paper*, skripsi, pengiriman ke tim) → lokal atau *cloud*, jangan Colab Free.
   Tidak (eksplorasi sekali pakai) → Colab cukup.

## Rekomendasi Jujur untuk 3 Profil

- **Mahasiswa skripsi / baru mulai** → **Colab Free** sebagai *default*, plus laptop sebagai cadangan. Hemat biaya, mulai cepat, tidak perlu belajar Linux atau Docker dulu.
- **Peneliti / praktisi aktif**: Pilihan tergantung tipe data. Kalau data Anda tabular atau teks dan paralelisme eksperimen penting (beberapa *job* bersamaan di tab berbeda, atau beberapa komputer dengan akun yang sama), **Colab** (Free untuk ringan, Pro kalau butuh prioritas GPU dan sesi panjang) sudah lebih dari cukup. Beberapa *script* kecil berjalan paralel di banyak tab sering lebih produktif dari satu *workstation* lokal yang dipakai antri. Kalau data Anda citra, audio, atau video besar, atau Anda sangat peduli privasi data, **laptop GPU** (Apple Silicon atau RTX, RAM 16 GB ke atas) sebagai *workhorse* harian dan **Colab Pro** untuk beban puncak.
- **Tim produksi / dataset besar** → **Cloud VM** dengan *scheduler* (mis. Airflow, Prefect) agar tagihan tidak membengkak. Beli *reserved instance* kalau beban sudah stabil.

## Catatan Subjektif Penulis

Untuk kebanyakan eksperimen *deep learning* yang saya jalankan di BMKG, Colab adalah pilihan pertama, bukan laptop. Bukan karena laptop saya tidak mampu, tapi karena satu akun Colab memungkinkan saya membuka empat atau lima tab sekaligus, masing-masing menjalankan *script* berbeda, kadang bahkan di beberapa komputer di kantor dan di rumah, selama dana bulanan cukup. Untuk eksperimen kecil-menengah yang berulang (coba arsitektur A, coba arsitektur B, ubah *learning rate*, ulangi), paralelisme itu yang sering jadi pembeda, bukan *clock speed*. Pengalaman pribadi ini bukan aturan umum; kalau data Anda teks atau tabular seperti kebanyakan kasus meteorologi dasar (*time series* curah hujan, klasifikasi tipe cuaca, indeks ENSO atau MJO), barangkali Anda akan merasakan hal yang sama.

## Penutup: Alat Bantu, Bukan Identitas

Ada satu prinsip yang lebih penting dari pilihan Colab versus laptop: **orang yang punya GPU paling mahal bukan jaminan model bagus**. Yang menentukan hasil adalah data yang bersih, *baseline* yang jujur, dan evaluasi yang terukur. Algoritma *linear regression* pada variabel yang tepat akan mengalahkan arsitektur Transformer raksasa pada fitur yang salah.

Kalau Anda masih bingung mulai dari mana, jawab tiga pertanyaan di atas dulu, lalu pilih opsi paling sederhana yang memenuhi jawaban itu. Jangan pilih berdasarkan tutorial viral atau komparasi panjang di YouTube yang membuat Anda berdiskusi sendiri di kolom komentar selama berminggu-minggu. Pilih berdasarkan **apa yang sebenarnya Anda yang akan Anda lakukan**.

## Referensi dan Bacaan Lanjut

- Google. "Google Colab, Plans and Pricing." https://colab.research.google.com/signup, diakses 5 September 2026.
- Google. "Google Colab, FAQ (resource limits, GPU availability)." https://research.google.com/colaboratory/faq.html, diakses 5 September 2026.
- PyTorch. "Introduction to PyTorch on Apple Silicon (MPS backend)." https://pytorch.org/docs/stable/notes/mps.html, diakses 5 September 2026.
- TensorFlow. "TensorFlow PluggableDevice, Metal (Apple Silicon) GPU support." https://www.tensorflow.org/guide/pluggable_device, diakses 5 September 2026.
- Paszke, A., dkk. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019, https://pytorch.org/, diakses 5 September 2026.
- Chollet, F. *Deep Learning with Python*, edisi ke-2. Manning, 2021. Bab 1 membahas secara jujur kapan *deep learning* *overkill*.
