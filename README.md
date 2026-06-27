# Home Credit Scorecard Model

Membangun model *credit scoring* untuk Home Credit Indonesia: memprediksi probabilitas gagal bayar (*Probability of Default*), mengubahnya menjadi **credit score**, lalu menerjemahkannya ke **kebijakan persetujuan 3-tier** beserta estimasi **Net Profit**.

Proyek ini dikerjakan sebagai *final project* Rakamin Data Science (Task 5 — Home Credit Scorecard Model).

---

## Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Struktur Repo](#struktur-repo)
- [Alur Pengerjaan](#alur-pengerjaan)
- [Hasil Utama](#hasil-utama)
- [Insight & Rekomendasi Bisnis](#insight--rekomendasi-bisnis)
- [Catatan Tata Kelola & Fair-Lending](#catatan-tata-kelola--fair-lending)
- [Cara Menjalankan](#cara-menjalankan)
- [Keterbatasan & Pengembangan Lanjutan](#keterbatasan--pengembangan-lanjutan)
- [Tech Stack](#tech-stack)

---

## Latar Belakang

Home Credit ingin memaksimalkan potensi data untuk memastikan **pelanggan yang mampu melunasi tidak ditolak**, sekaligus memberikan pinjaman dengan *principal*, *maturity*, dan *repayment calendar* yang tepat. Tantangannya adalah menyeimbangkan dua kerugian:

- **Kerugian kredit** — menyetujui pemohon yang akhirnya gagal bayar.
- **Opportunity loss** — menolak pemohon yang sebenarnya baik.

**Goal:** model yang memprediksi apakah pemohon akan membayar tepat waktu atau bermasalah.
**Objective:** ubah prediksi menjadi *scorecard* + kebijakan persetujuan yang menguntungkan.
**Metric:** ROC-AUC, diukur **out-of-fold (held-out)**, bukan pada data training.

---

## Dataset

Sumber: [Home Credit Default Risk](https://rakamin-lms.s3.ap-southeast-1.amazonaws.com/vix-assets/home-credit-indonesia/home-credit-default-risk.zip) (Kaggle / Rakamin).

Untuk menjaga model tetap **sederhana dan dapat dijelaskan**, proyek ini fokus pada tabel utama **`application_train.csv`** (1 baris = 1 pinjaman).

| Item | Nilai |
|---|---|
| Jumlah baris (setelah cleaning) | 307.510 |
| Jumlah fitur (setelah engineering) | 124 |
| Target | `TARGET` (0 = lunas, 1 = gagal/telat bayar) |
| Tingkat gagal bayar | **8,07%** (data sangat *imbalance*) |

Kolom kunci: `EXT_SOURCE_1/2/3` (skor risiko eksternal), `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, `DAYS_EMPLOYED`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`.

---

## Struktur Repo

```
.
├── notebook.ipynb              # Pipeline lengkap (Fase 0–7)
├── outputs/
│   ├── credit_scores.csv       # PD, credit score, & risk tier per nasabah
│   ├── feature_importance.csv  # Importance (gain) tiap fitur
│   ├── tier_summary.csv        # Komposisi & default rate per tier
│   └── net_profit_simulation.csv  # Perbandingan Net Profit antar kebijakan
└── README.md
```

---

## Alur Pengerjaan

Pipeline dibagi menjadi 7 fase:

1. **Setup** — import library & konfigurasi (termasuk asumsi bisnis).
2. **Load & Cleaning** — perbaikan anomali `DAYS_EMPLOYED` (placeholder pensiunan `365243`), pembuangan outlier income 117 juta, encoding kategorikal, dan **pembuangan `CODE_GENDER`** (fair-lending).
3. **EDA & Business Insight** — distribusi target, default rate per jenis penghasilan & usia, kekuatan `EXT_SOURCE`.
4. **Modeling** — **Logistic Regression** (baseline interpretable) vs **LightGBM** (champion), dievaluasi dengan **cross-validation out-of-fold** agar AUC jujur (bukan *in-sample*).
5. **Kalibrasi & Scorecard** — kalibrasi probabilitas (isotonic) lalu konversi PD → credit score (log-odds-to-points).
6. **Tier Risiko & Net Profit** — pembagian 3 tier **berbasis ekonomi** + simulasi Net Profit.
7. **Output** — ekspor hasil ke CSV.

> **Catatan metodologis.** Seluruh AUC dilaporkan secara *out-of-fold*: setiap prediksi dibuat oleh model yang **tidak** dilatih pada baris tersebut. Mengukur AUC pada data training (yang sudah dilihat model) menghasilkan angka yang menggelembung dan tidak mencerminkan performa pada pemohon baru.

---

## Hasil Utama

### Perbandingan Model (Out-of-Fold)

| Model | ROC-AUC | Gini |
|---|---|---|
| Logistic Regression | 0,7415 | 0,483 |
| **LightGBM (terkalibrasi)** | **0,7641** | **0,528** |

LightGBM unggul tipis; Logistic Regression tetap kompetitif dan lebih mudah dijelaskan. KS = **0,394**.

### Kalibrasi

Setelah kalibrasi isotonic, selisih (*gap*) antara PD prediksi dan default rate aktual di setiap desil **< 0,01** — probabilitas dapat dipercaya untuk perhitungan finansial.

### Credit Score

PD diubah menjadi credit score (skor tinggi = risiko rendah): rentang **269–886**, median **570**.

### Kebijakan 3-Tier (berbasis ekonomi)

Titik impas: **PD = 18,7%** (di bawahnya, pemohon masih untung secara ekspektasi). Tier dipotong berdasarkan ekonomi, bukan kuantil:

| Tier | Kebijakan | Populasi | Default Rate |
|---|---|---|---|
| **A — Auto-Approve** | PD ≤ 5% | 47,5% | 2,6% |
| **B — Review** | 5% < PD < 18,7% | 43,2% | 9,7% |
| **C — Reject** | PD ≥ 18,7% | 9,4% | 28,2% |

Default rate naik tegas A→B→C (pemisahan ~11x).

### Simulasi Net Profit

*Asumsi: margin bunga 15%, LGD 65%. Satuan mengikuti `AMT_CREDIT` (Rupiah). Angka bersifat ilustratif.*

| Kebijakan | Disetujui | Net Profit | Uplift vs Approve-All |
|---|---|---|---|
| Approve All | 100% | ≈ 16,64 miliar | — |
| Reject All | 0% | 0 | −16,64 miliar |
| **Three-Tier Policy** | 90,6% | **≈ 17,62 miliar** | **+0,99 miliar (+6%)** |

Cutoff skor optimal ≈ **529**. Kebijakan tier hanya menolak ~9% pemohon paling berisiko (zona rugi), bukan sepertiga buku, sehingga menambah profit alih-alih membuang nasabah menguntungkan.

---

## Insight & Rekomendasi Bisnis

**1 · `EXT_SOURCE` adalah pendorong risiko terkuat** (importance *gain* tertinggi, ~4x fitur berikutnya).
→ Jadikan tulang punggung scorecard. Untuk pemohon **tanpa** `EXT_SOURCE` (*thin-file*), jangan auto-reject — arahkan ke *manual review* atau minta data alternatif. *Missing ≠ buruk.*

**2 · Segmen risiko-rendah justru kecil porsinya.** PNS (*State servant*) dan pensiunan memiliki default rate jauh di bawah rata-rata 8%, sementara segmen *Working* (mayoritas) tertinggi.
→ Lakukan **campaign akuisisi tertarget** ke PNS & pensiunan: menambah komposisi mereka menurunkan kerugian agregat **tanpa** memperketat kebijakan persetujuan.

**3 · Faktor kapasitas-bayar & demografi** (`CREDIT_TERM`, `AMT_ANNUITY`, usia/`DAYS_BIRTH`, lama kerja) melengkapi sinyal.
→ Terapkan aturan *affordability* (batasi rasio cicilan terhadap penghasilan) dan gunakan usia untuk **limit & pricing**, bukan untuk penolakan.

---

## Catatan Tata Kelola & Fair-Lending

- **`CODE_GENDER` dibuang dari fitur** — penggunaan jenis kelamin dalam keputusan kredit melanggar prinsip *fair-lending*. Biaya AUC minimal, model jadi dapat dipertanggungjawabkan.
- **Usia hanya untuk pricing/limit, bukan denial.** Risiko menurun seiring usia, tetapi menolak berdasarkan usia bersifat diskriminatif.
- **Angka Net Profit bergantung pada asumsi** (margin/LGD) — disajikan transparan; yang valid diklaim adalah *uplift relatif* antar kebijakan, bukan proyeksi rupiah pasti.
- Probabilitas **dikalibrasi** sebelum dipakai untuk keputusan finansial.

---

## Cara Menjalankan

1. Unduh dataset dan letakkan `application_train.csv` pada folder yang ditunjuk oleh `DATA_DIR` di notebook.
2. Pasang dependensi:
   ```bash
   pip install numpy pandas matplotlib scikit-learn lightgbm
   ```
3. Jalankan `notebook.ipynb` dari atas ke bawah (Fase 0 → 7). Output tersimpan otomatis di folder `outputs/`.

---

## Keterbatasan & Pengembangan Lanjutan

- **Satu tabel.** Menggunakan hanya `application_train` demi kesederhanaan. Menambahkan agregasi tabel `bureau` (riwayat biro kredit) diperkirakan menaikkan AUC ~1–2 poin.
- **Reject-inference.** Data hanya berisi pemohon yang dulu diterima; model belum "melihat" pemohon yang ditolak.
- **Asumsi ekonomi** perlu divalidasi dengan tim finance sebelum dipakai sebagai proyeksi.
- **Monitoring drift** dan kalibrasi ulang berkala diperlukan untuk produksi.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `matplotlib`

---

## Penulis

**[Yusuf Ahmad]** — Rakamin Data Science Final Project (Task 5)