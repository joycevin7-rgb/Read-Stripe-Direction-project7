# Classical Methods Lab — Group 14: Read Stripe Direction

**Kelompok 14 | Mata Kuliah: Computer Vision | Metode Klasik**

---

## 1. Tujuan Project

### Masalah
Diberikan tiga gambar abu-abu (*grayscale*) berisi pola garis (*stripe swatch*), sistem harus memutuskan apakah setiap gambar memiliki garis **VERTIKAL** atau **HORIZONTAL** — tanpa menggunakan *deep learning*.

### Cara Kerja Sobel Gx dan Gy
- **Sobel Gx** (kernel horizontal) mendeteksi perubahan intensitas piksel **ke arah kolom** (kiri ↔ kanan). Jika garis vertikal, tepi garis menyebabkan perubahan besar ke arah horizontal → **Gx besar**.
- **Sobel Gy** (kernel vertikal) mendeteksi perubahan intensitas piksel **ke arah baris** (atas ↕ bawah). Jika garis horizontal, tepi garis menyebabkan perubahan besar ke arah vertikal → **Gy besar**.
- Kami menggunakan `CV_64F` agar nilai Sobel *negatif* (tepi terang→gelap) tidak terpotong menjadi nol.
- Keputusan dibuat oleh **Tahap C** berdasarkan perbandingan rata-rata absolut |Gx| vs |Gy|.

---

## 2. Struktur Folder

```
stripe-direction/              ← root project
├── README.md
├── requirements.txt
├── config.py                  ← semua parameter & konstanta
├── stage_a.py                 ← Tahap A (Owner A): Sobel Gx
├── stage_b.py                 ← Tahap B (Owner B): Sobel Gy
├── stage_c.py                 ← Tahap C (Owner C): Keputusan arah
├── run_pipeline.py            ← jalankan A→B→C, simpan semua output
├── make_test_swatches.py      ← buat swatch sintetis jika images/ kosong
├── images/
│   ├── swatch_1.png
│   ├── swatch_2.png
│   └── swatch_3.png
├── outputs/
│   ├── labeled_strip.png
│   ├── results_table.csv
│   ├── results_table.md
│   ├── handoffs.json
│   ├── trial_A_signed_vs_abs.png / .csv
│   ├── trial_B_border.png / .csv
│   ├── trial_C_margins.png / .csv
│   └── gradient_maps.png
└── notebook.ipynb             ← pipeline yang sama dalam format notebook
```

---

## 3. Cara Instalasi

**Versi Python yang diperlukan:** Python 3.9 atau lebih baru (diuji pada 3.14.0)

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
numpy>=1.24.0
opencv-python>=4.8.0
matplotlib>=3.7.0
```

---

## 4. Cara Menjalankan

### Langkah 1 — Siapkan gambar swatch
Letakkan file `swatch_1.png`, `swatch_2.png`, `swatch_3.png` yang nyata ke dalam folder `images/`.

Jika file nyata belum tersedia, jalankan pembuat swatch sintetis:
```bash
python make_test_swatches.py
```
> Skrip ini **tidak akan menimpa** file yang sudah ada di `images/`.

### Langkah 2 — Jalankan pipeline penuh
```bash
python run_pipeline.py
```
Semua output akan disimpan ke folder `outputs/` dan ringkasan cetak ke terminal.

### Menjalankan via Notebook
```bash
jupyter notebook notebook.ipynb
```
Jalankan setiap sel dari atas ke bawah. Tahap A, B, C sudah dilabeli dengan ownernya.

---

## 5. Penjelasan Tiap Tahap

### Tahap A — Owner A: Sobel Gx (Perubahan Antar Kolom)

| Aspek | Detail |
|---|---|
| **Input** | Dict `{nama_file: array_grayscale}`, parameter `KSIZE` |
| **Proses** | Hitung Sobel Gx (`cv2.Sobel(..., 1, 0, ksize=KSIZE)`) dengan `CV_64F`; hitung mean Gx *bertanda* dan mean \|Gx\| *absolut* per swatch |
| **Output internal** | Perbandingan signed vs absolute (untuk Trial A) |
| **Handoff ke C** | `mean_abs_Gx` per swatch (rata-rata \|Gx\| seluruh gambar) |

**Mengapa mean absolut, bukan mean bertanda?**  
Tepi gelap→terang dan terang→gelap pada garis yang sama memberi nilai Sobel berlawanan tanda. Jika dijumlah, nilainya saling menghilangkan mendekati nol — padahal tepian nyata tetap ada. Nilai absolut memastikan semua tepi dihitung.

---

### Tahap B — Owner B: Sobel Gy (Perubahan Antar Baris)

| Aspek | Detail |
|---|---|
| **Input** | Dict `{nama_file: array_grayscale}`, `KSIZE` (sama dengan A), `BORDER` |
| **Proses** | Hitung Sobel Gy (`cv2.Sobel(..., 0, 1, ksize=KSIZE)`) dengan `CV_64F`; bandingkan mean \|Gy\| gambar penuh vs setelah membuang tepi 5 piksel |
| **Output internal** | Perbandingan full vs no-border (untuk Trial B) |
| **Handoff ke C** | `mean_abs_Gy` per swatch (gambar *penuh*, tanpa crop) |

**Mengapa membandingkan dengan/tanpa border?**  
`warpAffine` untuk rotasi kecil dapat meninggalkan artefak di tepi gambar yang sedikit meningkatkan gradien palsu. Perbandingan ini menunjukkan seberapa besar pengaruh border tersebut.

---

### Tahap C — Owner C: Keputusan Arah Garis

| Aspek | Detail |
|---|---|
| **Input** | Hanya `handoff_gx` dan `handoff_gy` dari Tahap A dan B |
| **Proses** | Bandingkan `mean_abs_Gx` vs `mean_abs_Gy` dengan aturan margin; uji dua nilai margin |
| **Output** | Keputusan per swatch: `"vertical"`, `"horizontal"`, atau `"uncertain"` |
| **Handoff keluar** | Tabel hasil (`results_table.csv/.md`) |

**Aturan keputusan:**
```
jika  Gx ≥ margin × Gy  →  "vertical"
jika  Gy ≥ margin × Gx  →  "horizontal"
selain itu               →  "uncertain"
```

---

## 6. Daftar Parameter dan Nilainya

| Parameter | Nilai Default | Lokasi | Keterangan |
|---|---|---|---|
| `KSIZE` | `3` | `config.py` | Ukuran kernel Sobel (digunakan A & B) |
| `BORDER` | `5` | `config.py` | Lebar tepi yang di-crop untuk Trial B |
| `MARGINS` | `[1.0, 2.5]` | `config.py` | Nilai margin yang diuji di Tahap C |
| `MAIN_MARGIN` | `2.5` | `config.py` | Margin untuk keputusan utama yang dilaporkan |
| `COMPARISON_MARGIN` | `1.0` | `config.py` | Margin pembanding |
| `SWATCH_FILES` | `["swatch_1.png", ...]` | `config.py` | Urutan file input |

---

## 7. Penjelasan Setiap File Output

| File | Deskripsi |
|---|---|
| `labeled_strip.png` | Tiga swatch berdampingan dengan label keputusan, skala tampilan seragam (vmin=0, vmax=255) |
| `results_table.csv` | Tabel tiga baris: swatch, mean\|Gx\|, mean\|Gy\|, keputusan margin 2.5, keputusan margin 1.0 |
| `results_table.md` | Tabel yang sama dalam format Markdown |
| `handoffs.json` | Nilai persis yang dioper dari A ke C dan dari B ke C |
| `trial_A_signed_vs_abs.png/.csv` | Bukti perbandingan Tahap A: mean Gx bertanda vs absolut |
| `trial_B_border.png/.csv` | Bukti perbandingan Tahap B: \|Gy\| dengan/tanpa border 5 px |
| `trial_C_margins.png/.csv` | Bukti perbandingan Tahap C: keputusan di margin 1.0 vs 2.5, kasus *uncertain* ditandai |
| `gradient_maps.png` | Peta \|Gx\| dan \|Gy\| semua swatch dengan skala tampilan bersama |

---

## 8. Hasil dan Interpretasi

> **Isi tabel ini setelah menjalankan pipeline pada gambar nyata.**

| Swatch | mean \|Gx\| | mean \|Gy\| | Keputusan (m=2.5) | Keputusan (m=1.0) |
|---|---|---|---|---|
| swatch_1.png | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |
| swatch_2.png | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |
| swatch_3.png | _(isi)_ | _(isi)_ | _(isi)_ | _(isi)_ |

**Pada swatch sintetis (make_test_swatches.py):**
- `swatch_1.png` (vertikal, contrast tinggi) → **vertical** ✓
- `swatch_2.png` (horizontal, contrast tinggi) → **horizontal** ✓
- `swatch_3.png` (vertikal, contrast rendah) → lihat output terminal (kemungkinan *uncertain* di margin 2.5 atau *vertical* jika noise cukup rendah)

---

## 9. Catatan untuk Presentasi (per Role)

### Owner A — Stage A (Sobel Gx)

**What to show:**  
Grafik `trial_A_signed_vs_abs.png`: dua bar per swatch — mean Gx bertanda (biru) vs mean \|Gx\| (oranye). Tunjukkan bahwa bar biru bisa mendekati nol sementara bar oranye tetap besar.

**Lines of code to trace:**  
`stage_a.py` baris `gx = cv2.Sobel(...)` → `mean_signed_gx = float(np.mean(gx))` → `abs_gx = np.abs(gx)` → `mean_abs_gx = float(np.mean(abs_gx))`

**Comparison to explain:**  
Signed mean vs absolute mean: tanpa `np.abs()`, mean akan mendekati nol karena tepi positif dan negatif saling menghilangkan. Absolute mean mengukur kekuatan tepi sebenarnya.

**Pertanyaan TA yang mungkin:**

> *Q1: Mengapa kamu pakai `cv2.CV_64F` bukan `cv2.CV_8U`?*  
> A: `CV_8U` mengkliping nilai negatif menjadi 0, sehingga tepi terang→gelap hilang. Dengan `CV_64F`, nilai negatif dipertahankan dan `np.abs()` kemudian menghitung kekuatan penuh kedua jenis tepi.

> *Q2: Apa efeknya jika `KSIZE=5`?*  
> A: Kernel lebih besar mengaburkan lebih banyak sebelum derivatif, mengurangi respons terhadap noise tapi juga bisa melemahkan tepi tajam. Mean \|Gx\| akan sedikit lebih kecil; keputusan bisa berubah untuk swatch low-contrast.

---

### Owner B — Stage B (Sobel Gy)

**What to show:**  
Grafik `trial_B_border.png`: dua bar per swatch — full image (hijau) vs no-border (merah). Tunjukkan bahwa perbedaannya kecil, artinya border effect tidak dominan.

**Lines of code to trace:**  
`stage_b.py` baris `gy = cv2.Sobel(...)` → `gy_cropped = abs_gy[border:-border, border:-border]` → perbandingan `mean_abs_gy_full` vs `mean_abs_gy_no_border`.

**Comparison to explain:**  
Menyertakan tepi gambar sedikit meningkatkan mean karena piksel tepi yang disisipkan saat rotasi (`warpAffine`) membentuk gradien palsu. Namun kita tetap menggunakan gambar penuh untuk handoff karena pengaruhnya kecil dan konsisten.

**Pertanyaan TA yang mungkin:**

> *Q1: Mengapa `KSIZE` di Stage B harus sama dengan Stage A?*  
> A: Karena kita membandingkan mean \|Gx\| (dari A) dengan mean \|Gy\| (dari B). Jika kernel berbeda, besar responsnya tidak sebanding — perbandingannya jadi tidak adil.

> *Q2: Mengapa membuang border 5 piksel, bukan lebih besar seperti 20 piksel?*  
> A: `KSIZE=3` hanya membutuhkan 1 piksel padding untuk Sobel. Rotasi kecil (< 5°) hanya mempengaruhi sudut gambar secara minimal. Border 5 px sudah mencakup artefak rotasi tanpa membuang terlalu banyak data valid.

---

### Owner C — Stage C (Keputusan)

**What to show:**  
Grafik `trial_C_margins.png`: batang menunjukkan rasio dominan/lemah per swatch, dengan garis horizontal di margin 1.0 dan 2.5. Tunjukkan swatch mana yang di bawah 2.5 (uncertain).

**Lines of code to trace:**  
`stage_c.py` fungsi `classify_direction()`: tiga kondisi `if/if/return "uncertain"`. Telusuri per swatch dengan angka nyata dari handoffs.json.

**Comparison to explain:**  
Margin 1.0 memberi keputusan bahkan saat Gx dan Gy hampir sama kekuatannya (tidak andal). Margin 2.5 mengharuskan satu gradien dominan ≥2.5× lipat yang lain — lebih konservatif, kasus uncertain dimunculkan secara jujur.

**Pertanyaan TA yang mungkin:**

> *Q1: Mengapa Stage C hanya boleh menerima nilai handoff, bukan gambar aslinya?*  
> A: Pemisahan tugas yang jelas: C bertugas memutuskan berdasarkan angka ringkasan; jika C bisa akses gambar asli, batas antara tahap hilang dan error lebih sulit diisolasi.

> *Q2: Jika margin diubah menjadi 1.5, apa yang berubah?*  
> A: Swatch yang sebelumnya uncertain (rasio antara 1.0–2.5) akan dapat keputusan. Swatch low-contrast dengan rasio kecil tetap uncertain sampai margin cukup rendah.

---

## 10. Aturan Penting

> [!IMPORTANT]
> **Gambar referensi (ground truth) TIDAK BOLEH masuk ke Tahap A, B, atau C sebagai input.**
> Ground truth hanya boleh digunakan *di luar pipeline* untuk memverifikasi kebenaran keputusan akhir.

> [!WARNING]
> **Jika menggunakan AI untuk menghasilkan kode:** kamu wajib menjalankan dan menguji sendiri setiap baris yang kamu presentasikan. TA dapat meminta kamu mengubah parameter (misalnya `KSIZE`, `BORDER`, margin) secara langsung dan menjelaskan hasilnya.

> [!TIP]
> Sebelum presentasi, jalankan `python run_pipeline.py` sekali lagi dan pastikan semua output di `outputs/` sesuai dengan tabel yang kamu siapkan.
#   R e a d - S t r i p e - D i r e c t i o n - p r o j e c t 7  
 