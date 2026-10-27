# Eksperimen Paper A (E1–E5)

Kode eksperimen untuk naskah *Leakage-Safe Explainable Intrusion Detection for Data Breach Triage in Higher Education*. Setiap eksperimen menghasilkan tabel CSV, gambar PNG, dan ringkasan Markdown yang langsung mengisi bagian Results. Angka kunci juga dikumpulkan di `results/paper_numbers.json` untuk mengganti penanda [X] di naskah.

| Eksp. | Isi | RQ | Bagian naskah |
| --- | --- | --- | --- |
| E1 | Protokol konvensional vs leakage-safe (Random Forest) | RQ1 | 5.1 |
| E2 | LR, DT, RF, XGBoost, Isolation Forest, Autoencoder; bootstrap CI, McNemar, DeLong | RQ1 | 5.2 |
| E3 | Leave-one-day-out: serangan yang belum pernah dilihat | RQ1, RQ2 | 5.2 |
| E4 | Replikasi UNSW-NB15; transfer CIC → UNSW-NB15 dan CIC → PTMA | RQ2 | 5.3 |
| E5 | Kesepakatan TreeSHAP–LIME–importance, kurva deletion, stabilitas LIME, transfer atribusi | RQ2, RQ3 | 5.4, 5.6 |

## 1. Instalasi

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Uji coba dengan data sintetis (±4 menit)

```bash
python tests/make_synthetic.py tests/synthetic
python -m xai_ids.run --config configs/smoke_test.yaml --exp all
```

Data sintetis hanya meniru *format* file. Angkanya tidak bermakna dan **tidak boleh dilaporkan**.

## 3. Menyiapkan data asli

Letakkan paket ini di folder proyek Anda (`Tugas paper\xai-ids-experiments`). Config `configs/cic_unsw_ptma.yaml` sudah mengikuti susunan folder di laptop:

| Dataset | Sumber | Dipakai di |
| --- | --- | --- |
| CIC-IDS2017 versi koreksi | `raw/dataset.zip` (dibaca langsung dari ZIP) | E1–E5 |
| CIC-IDS2017 asli | `raw/original/CIC-IDS2017.zip` | baris pembanding E1 |
| UNSW-NB15 resmi | `raw/Training and Testing Sets UNSW-NB15/` | replikasi E4, E5 |
| UNSW-NB15 flow berlabel | dibuat dari `raw/pcaps 17-2-2015/` + `raw/NUSW-NB15_GT.csv` (3b) | transfer E4 |
| PTMA flow berlabel | dibuat dari `raw/PCAP.zip` + alert Suricata (3c) | transfer E4 |

**3a. Pasang alat (sekali saja).**

1. **Docker Desktop** untuk Windows. CICFlowMeter berjalan di dalam Docker, jadi Java tidak perlu dipasang.
2. **Wireshark** (biasanya sudah ada). `editcap` dipakai untuk mengubah `.pcapng` menjadi `.pcap`.
3. **Suricata** (`raw/Scrtadst/Suricata-8.0.6-1-64bit.msi`) beserta **Npcap**. Rules memakai `raw/Scrtadst/emerging.rules.tar.gz`.

Periksa dengan:

```bash
python scripts/prepare_data.py check
```

**3b. Build CICFlowMeter yang dipatch.**

```bash
python scripts/prepare_data.py build
```

Perintah ini mengambil CICFlowMeter versi Engelen et al. (commit `4dd5319`, dipakai untuk CIC-IDS2017 versi koreksi), lalu mem-build image Docker `cicflowmeter-utc`. Ada dua patch kecil (lihat `tools/cicflowmeter/Dockerfile`):

* Timestamp diberi penanda AM/PM. Versi aslinya mencetak jam 12 tanpa AM/PM, sehingga pukul 01:53 dan 13:53 tidak bisa dibedakan.
* Timestamp selalu dalam UTC. Dengan patch ini, format dan zona waktunya identik dengan `dataset.zip`, karena flow pertama hari Selasa (epoch 1499176424) tercetak `04/07/2017 01:53:44 PM`, sama dengan file koreksi.

**3c. UNSW-NB15: PCAP → flow → label ground truth.**

```bash
python scripts/prepare_data.py unsw --mem-gb 6
```

CICFlowMeter memproses 17 PCAP berukuran 2 GB (`data/unsw_flows_raw/`), dan prosesnya bisa memakan beberapa jam. Setiap flow lalu dicocokkan ke `NUSW-NB15_GT.csv` lewat 5-tuple dan waktu, dalam dua arah. Kategori diseragamkan (spasi dihapus, `Backdoors` → `Backdoor`). Hasilnya ada di `data/unsw_flows_labelled/`. Untuk mengulang pelabelan tanpa ekstraksi ulang, tambahkan `--skip-extract`.

Catatan: folder hanya berisi `1.pcap`–`17.pcap` dari hari 17-2-2015. Rilis resmi hari itu berisi 27 file, dan PCAP hari 22-1-2015 tidak ada. Sebutkan cakupan ini di naskah.

**3d. PTMA: PCAP → flow → label Suricata → pseudonim.**

```bash
python scripts/prepare_data.py ptma --mem-gb 4
```

Urutannya:

1. `today.pcapng` diekstrak dari `raw/PCAP.zip` lalu dikonversi ke `.pcap`.
2. CICFlowMeter mengekstrak flow.
3. Suricata dijalankan dengan ET Open, menghasilkan `data/suricata_ptma/eve.json`.
4. Alert diubah menjadi jadwal label oleh `scripts/suricata_to_schedule.py`. Event mesin (`SURICATA ...`), alert dengan severity > 2, dan kategori benign dibuang. Ringkasan per signature ada di `data/ptma_suricata_signatures.csv`; periksa file ini sebelum menerima label.
5. Flow dilabeli dalam dua arah.
6. IP dipseudonimkan dengan HMAC (`scripts/pseudonymize.py`). Kuncinya disimpan di `%USERPROFILE%\.xai_ids\ptma_hmac.key`; jangan bagikan kunci ini.

Setelah hasil diperiksa, **hapus `data/ptma_flows_labelled_raw/`** karena masih berisi IP asli.

Label Suricata berasal dari signature isi paket, bukan dari statistik flow, sehingga independen dari fitur model. Untuk naskah, validasi sampel acak label secara manual (dua analis, laporkan Cohen's κ).

**3e. Periksa sebelum menjalankan eksperimen.**

```bash
python scripts/check_flows.py data/unsw_flows_labelled
python scripts/check_flows.py data/ptma_flows_labelled
python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --inspect
```

`check_flows.py` memastikan header sama dengan CIC-IDS2017 dan timestamp terbaca 100%. Daftar fitur dari `--inspect` tidak boleh memuat IP, port, timestamp, Flow ID, atau `*_internal`.

## 4. Menjalankan

```bash
python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E1 E2 E3
python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E4 E5
```

Menjalankan E5 otomatis menjalankan E4, karena transfer atribusi butuh model dari E4. Untuk percobaan cepat, aktifkan `sample_frac: 0.2` pada dataset `cic`.

**Perkiraan beban.** CIC-IDS2017 versi koreksi berisi 2,1 juta flow; pemuatan memakai sekitar 2 GB RAM. E2 dengan 6 model × 5 seed adalah langkah terberat (beberapa jam di CPU 8 core, terutama Autoencoder dan XGBoost). LIME (500 alert × 2.000 sampel) butuh puluhan menit.

## 5. Aturan leakage yang diterapkan

1. Duplikat dihapus sebelum partisi. Vektor dengan label bertentangan dibuang. Laporan `overlap_*` harus 0.
2. IP, port, Flow ID, dan timestamp tidak masuk X; disimpan sebagai metadata.
3. Partisi dulu, lalu imputer, encoder, dan scaler di-fit hanya di training.
4. Resampling (opsional) hanya di training.
5. Hiperparameter dan θ* dipilih di validasi; test dipakai sekali.

**Partisi kronologis per (hari, kelas).** Serangan CIC-IDS2017 berlangsung dalam rentang waktu tertentu. Jika satu hari dipotong 60/20/20 berdasarkan waktu saja, serangan sore hari bisa seluruhnya jatuh ke test. Kode memotong setiap aliran (hari, kelas) secara kronologis, sehingga urutan waktu di dalam setiap episode serangan terjaga dan setiap kelas ada di ketiga partisi. Set `split.by_class: false` untuk versi per hari utuh sebagai uji ketahanan.

## 6. Keluaran

```
results/
  paper_numbers.json            angka kunci untuk Abstract dan Results
  E1/E1.md, e1_summary.csv      Tabel 5.1
  E2/E2.md, fig_pr_curves.png   Tabel dan gambar 5.2, uji signifikansi, recall per kelas
  E3/E3.md                      recall untuk serangan yang belum pernah dilihat
  E4/E4.md, fig_transfer_f1.png Tabel 5.3, daftar fitur bersama
  E5/E5.md, fig_*_deletion.png, fig_*_global_agreement.png, fig_*_shap_beeswarm.png
```

## 7. Struktur kode

| File | Isi |
| --- | --- |
| `xai_ids/data.py` | Pemuat dataset generik, normalisasi nama kolom, label biner, metadata |
| `xai_ids/split.py` | Deduplikasi, partisi kronologis/acak/leave-one-day-out, cek leakage, preprocessor |
| `xai_ids/models.py` | Detektor supervised dan unsupervised |
| `xai_ids/metrics.py` | θ*, metrik deteksi, bootstrap, McNemar, DeLong |
| `xai_ids/xai.py` | TreeSHAP, LIME, importance, kesepakatan, deletion, stabilitas |
| `xai_ids/experiments.py` | E1–E5 |
| `scripts/prepare_data.py` | Orkestrasi: cek alat, build CICFlowMeter, siapkan UNSW dan PTMA |
| `scripts/label_flows.py` | Pelabelan flow dari ground truth UNSW atau jadwal alert |
| `scripts/suricata_to_schedule.py` | Alert Suricata (`eve.json`) → jadwal label |
| `scripts/pseudonymize.py` | Pseudonimisasi IP dengan HMAC |
| `scripts/check_flows.py` | Pemeriksaan header, timestamp, dan label |
| `tools/cicflowmeter/Dockerfile` | CICFlowMeter Engelen et al. + patch timestamp UTC/AM-PM |
| `tests/make_synthetic.py` | Data sintetis untuk uji coba |
