# Dataset register and licensing status

All dataset files are local working data. `data/raw/` and `data/processed/` are
ignored by Git, and no dataset is redistributed with this repository.

## License policy

A dataset is marked **verified** only when the downloaded files include a
license document that can be read locally. If no license file was included, the
status is **license not verified**, even when a hosting page displays a license.
Do not infer or invent a license from a dataset name.

## Local datasets

| Dataset | Local location | Role | License status |
| --- | --- | --- | --- |
| CIC-MalMem-2022 | `data/raw/CIC-MalMem-2022.csv` | Memory-feature detection and family models | license not verified in the downloaded CSV; the Hugging Face dataset card lists Apache-2.0 |
| MalDroid-2020 syscall features | `data/raw/feature_vectors_syscalls_frequency_5_Cat.csv`, `data/raw/feature_vectors_syscallsbinders_frequency_5_Cat.csv` | Five-class syscall and syscall/Binder models | license not verified |
| MalDroid-2020 static features | `data/raw/feature_vectors_static.csv` | Unlabelled; not used for training | license not verified |
| Drebin-215 APK features | `data/raw/drebin-215-dataset-5560malware-9476-benign.csv` | Binary APK static-feature model (`B` = benign, `S` = malware) | license not verified |
| Drebin feature categories | `data/raw/drebin-feature-categories.csv` | Feature-category metadata for the model card | license not verified |
| CIC-Evasive-PDFMal2022 features | `data/raw/PDFMalware2022.csv` (mirror) | PDF binary static-feature model (5,557 malicious / 4,468 benign) | license not verified; no license file ships with the CSV mirror |
| PE static-feature extras | `data/raw/extras/malware_static_pe_features.csv`, `data/raw/extras/pe_section_headers.csv` | Optional only; not used by any training script | license not verified |
| EMBER 2018 | `ember_dataset_2018_2.tar.bz2` in the user's Downloads folder (left untouched); Colab copy in Google Drive | Phase 2 benchmark experiment, run in `notebooks/ember_colab.ipynb` | license not verified in the archive; the `elastic/ember` repository states its data files are MIT-licensed and its source code is AGPL-3.0 |

## Drebin-215 cleaning

`scripts/import_apk_dataset.py` applies this order so the result is measurable
and reproducible:

1. Replace every `?` feature cell with `0` **before** deduplication.
2. Drop exact duplicate rows across all source columns, keeping the first.
3. Drop every row whose 215-feature vector occurs with both labels.
4. Coerce the remaining features to numeric and write the cleaned CSV.

Run on the downloaded source archive, the measured result was:

| Measure | Count |
| --- | --- |
| Source rows | 15,036 |
| `?` cells replaced | 5 (all in `TelephonyManager.getSimCountryIso`) |
| Exact duplicate rows removed | 7,775 |
| Rows after deduplication | 7,261 |
| Conflicting feature patterns removed | 3 |
| Rows removed for conflicting labels | 6 |
| Rows after cleaning | 7,255 |
| Benign / Malware rows | 5,537 / 1,718 |

The importer prints these counts and stores them in
`data/processed/apk_drebin_metadata.json`; nothing is hard-coded. Coercing the
table to numeric before deduplication would merge a representation-only pair
and produce a different total, which is why the order above is fixed.

## CIC-Evasive-PDFMal2022 cleaning and PDF model

`scripts/train_pdf_models.py` normalises the mirrored pdfid feature table before
training:

1. Drop rows without a `Class` label.
2. Convert pdfid counts such as `1(1)` to their leading count.
3. Treat `-1` as unknown and map `text` (`Yes`/`No`/`unclear`) to `1`/`0`/unknown.
4. Parse `header` strings such as `%PDF-1.3` to a version number.
5. Exclude columns that a byte-level extractor cannot reproduce faithfully:
   `metadata size`, `xref Length`, `Colors`, `startxref`, `pageno`,
   `title characters`, and `header`. The remaining 24 features are re-extracted
   from the uploaded PDF bytes, avoiding train/serve skew.

Measured run on the downloaded mirror:

| Measure | Value |
| --- | --- |
| Rows used | 10,024 |
| Benign / malicious | 4,468 / 5,556 |
| Train / validation / test | 8,019 / 1,002 / 1,003 |
| Held-out accuracy | 0.9890 |
| Held-out macro F1 | 0.9889 |
| Held-out ROC-AUC | 0.9996 |

Artifacts: `models/pdf_lightgbm.txt`,
`models/pdf_model_metadata.json`, `models/pdf_training_metrics.json`.
Per-sample PDF family labels are not published with this dataset, so the
dashboard reports only a binary verdict and static indicators, never a guessed
family.



## EMBER 2018 Colab experiment

`notebooks/ember_colab.ipynb` runs the Phase 2 benchmark in Google Colab. It
obtains the official archive, vectorizes the raw PE features with
`elastic/ember`, trains LightGBM and a PyTorch MLP, and evaluates both models
once on the official time-based test split.

| Item | Value |
| --- | --- |
| Archive | `ember_dataset_2018_2.tar.bz2` (about 1.7 GB compressed) |
| SHA-256 | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |
| Official source | `https://ember.elastic.co/ember_dataset_2018_2.tar.bz2` |
| Features | 2,381 pre-extracted PE features, version 2 |
| Split | Labeled train rows ordered by `appeared`; latest 10% used as validation |
| Test policy | Official test split evaluated exactly once, at thresholds chosen on validation |
| Selection | Deployment model chosen on validation macro F1 only |

The archive remains untouched in `Downloads`; extraction and vectorization happen
inside Colab, and `/content` is ephemeral. Full vectorization needs about 25 GB of
free disk and takes one to four hours. Set `SMOKE_TEST = True` to check the
pipeline on a few thousand rows before a full run.

The notebook writes `ember_lightgbm.txt`, `ember_mlp.pt`, `ember_metrics.json`,
`ember_model_card.md`, `ember_model_card.pdf`, and a zip of all five artifacts.
The MLP checkpoint follows the project checkpoint schema, so
`malware_classifier.inference.load_checkpoint` can instantiate `MalwareMLP` from
it, but the dashboard does not expose EMBER features.

The downloaded archive contains no license document, so its status is **license
not verified**. The `elastic/ember` repository states that the data files
referenced in its README are MIT-licensed and that the source code is AGPL-3.0.
Treat the dataset as research data: do not redistribute it and do not treat the
experiment as a production security control.

## Handling rules

- Raw malware binaries are never executed or uploaded by the application.
- Model inputs are pre-extracted numeric feature tables.
- The Drebin file's source archive contained two CSVs and no license document.
- The two PE CSVs under `data/raw/extras/` are optional reference material and
  are deliberately excluded from the training pipelines.
- Keep the EMBER `tar.bz2` in `Downloads`; Phase 2 uploads it to Google Drive
  (or downloads it) and extracts it inside Google Colab instead of locally.
