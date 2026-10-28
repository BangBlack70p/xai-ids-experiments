# Leakage-safe explainable intrusion detection for data breach triage in higher education

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23716633.svg)](https://doi.org/10.5281/zenodo.23716633)

Experiment code for the paper *Consistent Explanations, Failing Detectors: Leakage-Safe Evaluation of Explainable Intrusion Detection for Data Breach Triage in Higher Education* (S. Gunanto, Universitas Sriwijaya). Indonesian notes: [README.id.md](README.id.md).

| Experiment | Content | Paper section |
| --- | --- | --- |
| E1 | Conventional vs leakage-safe protocol on label-corrected CIC-IDS2017 (Random Forest) | 5.1 |
| E2 | RF, XGBoost, LR, DT, Isolation Forest, autoencoder; bootstrap CI, McNemar, DeLong | 5.2 |
| E3 | Leave-one-day-out: attack types never seen in training | 5.2 |
| E4 | UNSW-NB15 replication; transfer CIC-IDS2017 to UNSW-NB15 and to a campus network (5 seeds) | 5.3 |
| E5 | TreeSHAP vs LIME vs importance, deletion faithfulness, LIME stability, attribution transfer | 5.4 |
| CASE | Explained campus alerts and the effect of analyst label confirmation | 5.5 |

Figures 1 to 7 of the framework are in `figures/` (HTML sources in `figure_sources.tar.gz`).

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                  # exact versions used for the paper (Python 3.14.7)
```

## Quick test on synthetic data

```bash
python tests/make_synthetic.py tests/synthetic
python -m xai_ids.run --config configs/smoke_test.yaml --exp all
```

Synthetic data only mimics the file format; its numbers are meaningless.

## Data

Place the public datasets under `raw/` (see `configs/cic_unsw_ptma.yaml`):

* `raw/dataset.zip`: label-corrected CIC-IDS2017 (Engelen et al., 2021).
* `raw/original/CIC-IDS2017.zip`: original CIC-IDS2017 CSVs (only for the E1 baseline row).
* `raw/Training and Testing Sets UNSW-NB15/`, `raw/pcaps 17-2-2015/`, `raw/NUSW-NB15_GT.csv`: UNSW-NB15.

Flows are re-extracted with CICFlowMeter commit `4dd5319` in Docker (`tools/cicflowmeter/Dockerfile`):

```bash
python scripts/prepare_data.py check
python scripts/prepare_data.py build
python scripts/prepare_data.py unsw
python scripts/prepare_data.py ptma --home-net "[<campus-prefix>,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16]" --internal <campus-prefix>
```

The campus (PTMA) data is not public; see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The `ptma` step labels flows from Suricata ET Open alerts, excludes sources confirmed legitimate by IT staff (`--exclude-src`), and pseudonymises IP addresses with a keyed HMAC. Keep the key outside the repository.

## Run

```bash
python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E1 E2 E3
python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E4 E5 CASE
```

Results are written to `results/` as Markdown, CSV and PNG; key numbers are collected in `results/paper_numbers.json`.

## Leakage rules enforced by the code

1. Duplicates and vectors with conflicting labels are removed before partitioning; overlap between partitions is reported and must be 0.
2. IP addresses, ports, flow IDs and timestamps are kept as metadata, never as features.
3. Partitioning comes first; imputers, encoders and scalers are fitted on the training partition only.
4. Hyperparameters and the threshold θ* are chosen on validation data; the test partition is used once.
5. CIC-IDS2017 is split chronologically within each (day, class) stream, 60/20/20.

## Licence and citation

MIT licence. Please cite the paper and this software release: S. Gunanto, version 1.0.0, Zenodo, doi:10.5281/zenodo.23716633 (see `CITATION.cff`).
