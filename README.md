# Deep Learning–Based Malware Detection and Classification Framework

An academic framework for detecting and classifying malware with machine-learning and deep-learning techniques. The project is designed around safe, feature-based analysis: it never executes suspicious files.

## Objectives

- Build reproducible malware detection experiments from safe, extracted features.
- Compare baseline models with neural-network approaches.
- Evaluate performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Document methodology, limitations, and ethical handling of datasets.

## Application features

- Detection workflow: benign versus malware.
- Classification workflow: malware category prediction.
- Upload an `.apk` or `.pdf` in the dashboard and get a verdict, confidence, and static indicators.
- Local browser dashboard for analysis, model readiness, and safe static inspection.
- API-first backend ready for raw APK/PDF uploads and pre-extracted CSV features.

## Project layout

```text
malware-ai/
├── configs/                 # Experiment and model configuration
├── data/                    # Local datasets (not committed)
│   ├── raw/
│   └── processed/
├── docs/                    # Project report, diagrams, and references
├── models/                  # Saved model files (not committed)
├── notebooks/               # Exploratory analysis and experiments
├── scripts/                 # Reproducible command-line workflows
├── src/malware_classifier/  # Backend and reusable application source
│   ├── static/              # Browser frontend (HTML/CSS/JavaScript)
│   ├── app.py               # FastAPI application
│   └── config.py            # Application paths and settings
└── tests/                   # Automated tests
```

## Safe workflow

Use only legitimate, safely handled datasets and pre-extracted/static features (for example, PE-header metadata, byte n-grams, or API-call features). Do not execute unknown samples. Work in an isolated research environment when handling malware-related data.

## Initial milestones

1. Select a safe dataset and document its license and label definitions.
2. Create a data-preparation pipeline for extracted features.
3. Implement baseline classifiers.
4. Train and evaluate a deep-learning model.
5. Compare results and write the academic report.

## Run the local application (Windows)

### Easiest method

1. Install Python 3.12 from [python.org](https://www.python.org/downloads/). During installation, select **Add Python to PATH**. Other versions may work, but 3.12 is the tested environment.
2. Extract or copy this project folder anywhere on the computer.
3. Double-click `scripts\\start_project.bat`.
4. Open `http://127.0.0.1:8001` in a browser.

The start script automatically creates a local `.venv` folder and installs only the packages required to run the web application. The first run may take a few minutes while Python and packages are installed. Press `Ctrl+C` in the open terminal window to stop it.

### Using VS Code

Open the project folder in VS Code, then press `Ctrl+Shift+B` and choose **Start Malware AI**. Alternatively, open a terminal in the project folder and run:

Create a virtual environment, install the dashboard dependencies, then start the app:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m uvicorn malware_classifier.app:app --app-dir src --port 8001 --reload
```

Open `http://127.0.0.1:8001` in a browser. The dashboard reports dataset readiness, trained-model status, and the held-out evaluation metrics. For model predictions, Analyze accepts exactly one CSV record containing the selected model's numeric feature columns (in any column order) — 55 features for CIC-MalMem, 215 for Drebin-215 APK, or the MalDroid syscall schemas. Label and identifier columns such as `Class`, `class`, `Category`, `Filename`, and an exported pandas index are ignored when present, so a single row copied from the dataset CSV can be analyzed directly. Raw binaries are never used as model inputs or executed.

The Analyze form also supports non-executing static inspection of `.exe`,
`.doc`, `.pdf`, `.apk`, and `.msi` files. These uploads are analyzed in memory
with a 25 MB limit and are not passed to the CIC-MalMem models; the app returns
hashes and format-specific structural metadata without a malware verdict.

## Import CIC-MalMem-2022

The project includes a reproducible importer for the public
[CIC-MalMem-2022 dataset](https://huggingface.co/datasets/bvk/CIC-MalMem-2022).
It contains 55 extracted memory-forensics features, `Class` and `Category`
labels, and a `Filename` sample identifier. The importer saves the training
split as `data/raw/CIC-MalMem-2022.csv`:

```powershell
python -m pip install -r requirements-app.txt
python scripts/import_dataset.py
```

`Class` is the benign/malware detection label. `Category` is `Benign` for
benign records and one of 15 malware family labels (such as `Trojan-Zeus`) for
malware records; the broad ransomware/spyware/trojan type is its prefix. Keep
`Filename` only for grouping related samples when creating train/test splits;
do not use it as a model feature. The dataset contains repeated memory dumps
per sample, so splitting by individual rows can leak related samples across
training and evaluation sets. Downloaded data is ignored by Git and stays in
`data/raw/`.

The dataset card lists the license as Apache-2.0 and cites Carrier et al.,
"Detecting Obfuscated Malware using Memory Feature Engineering," ICISSP 2022.

## Train the deep-learning models

The PyTorch training script fits a binary detector and a 15-family malware
classifier. It uses an MLP over the 55 numeric features, scales using training
data only, groups the numbered malware memory dumps by shared sample hash,
and reports held-out test metrics. Benign rows use a shared placeholder
filename in this dataset, so the script keeps each benign row separate; see
the [deep-learning guide](docs/deep-learning-guide.md) for the resulting
evaluation limitation and model concepts.

Install training packages and run:

```powershell
python -m pip install -r requirements-training.txt
python scripts/train_models.py
```

Pass `--dataset path\to\features.csv` to train from a different compatible CSV that has `Class`, `Category`, and `Filename` columns. The checkpoints and `training_metrics.json` are written to `models/`, which
is ignored by Git. For the beginner guide PDF, install
`requirements-docs.txt` and run `python scripts/build_guide_pdf.py`; the
generated PDF is placed beside its Markdown source in `docs/`.

## Train the MalDroid-2020 syscall models

The MalDroid CSVs you downloaded can train two separate five-class models:
one for syscall frequencies and one for syscall-plus-Binder frequencies.
They each contain 11,598 records and use numeric `Class` values 1–5. The
provided CSVs do not document human-readable names for those class IDs, so
the models report the numeric IDs. The large `feature_vectors_static.csv`
does not include a `Class` target or sample IDs; it is not used for training,
and its rows must not be labeled by assuming they align with either syscall
CSV.

Install training dependencies and run this command from the project folder:

```powershell
python -m pip install -r requirements-training.txt
python scripts/train_maldroid_models.py `
  --syscalls "data\raw\feature_vectors_syscalls_frequency_5_Cat.csv" `
  --syscalls-binders "data\raw\feature_vectors_syscallsbinders_frequency_5_Cat.csv"
```

The pipeline creates a grouped, stratified 80/10/10 train/validation/test
split. Identical feature rows stay in one split; identical vectors with
conflicting labels are excluded and counted in `models/maldroid_training_metrics.json`.
The model checkpoints and metrics are saved under `models/`.

These syscall models need **pre-extracted syscall/Binder frequency rows**.
They do not inspect an `.apk` directly: the static feature CSV is unlabeled,
and the syscall features normally come from running an app in a controlled
dynamic-analysis environment. The dashboard can classify one feature row
from either MalDroid schema by selecting its model before uploading a CSV;
it still only performs metadata inspection for uploaded APK binaries. Do not
run suspicious APKs on a normal computer or phone to generate the features.

## Train and use the Drebin-215 APK model

The Drebin-215 dataset supplies 215 pre-extracted static APK features. Place
`drebin-215-dataset-5560malware-9476-benign.csv` and
`drebin-feature-categories.csv` in `data/raw/`, then install the training
packages and run the importer and trainer from the project folder:

```powershell
python -m pip install -r requirements-training.txt
python scripts/import_apk_dataset.py
python scripts/train_apk_models.py
```

The importer replaces every `?` cell with `0` before deduplication, removes
exact duplicate rows, and then removes feature vectors that appear under both
labels. It prints the measured counts and writes
`data/processed/apk_drebin_metadata.json`; the expected cleaned total is not
hard-coded. See the [dataset register](docs/datasets.md) for the measured
figures and licensing status.

The trainer compares a LightGBM classifier with a PyTorch MLP on a grouped
80/10/10 split. Identical feature vectors stay in one split, models are selected
on held-out macro F1, and the winning model is recorded in
`models/apk_model_metadata.json`. The dashboard then offers **Drebin-215 APK
static features (215 features)** in the CSV model picker and calls
`POST /api/analyze/apk` with one feature row. Build the model card with:

```powershell
python -m pip install -r requirements-docs.txt
python scripts/build_apk_model_card_pdf.py
```

The model card is written to `docs/apk-model-card.md` and
`docs/apk-model-card.pdf`. The dashboard can also score a raw `.apk` upload:
`POST /api/analyze/file` unpacks it in memory, reconstructs Drebin-compatible
features with androguard, and runs the deployed LightGBM model. The mapping is
approximate, so the response reports how many of the 215 features matched.
No family/category is reported until a category model is trained.
The source archive included no license document, so the dataset and model card
are marked **license not verified**.

## Analyze an APK or PDF

Start the dashboard, open `http://127.0.0.1:8001`, and drop one `.apk` or
`.pdf` file (maximum 25 MB). The result shows:

- malware or benign verdict and model confidence;
- family/category only when a matching model exists;
- static indicators and file/model details;
- an explicit caveat when a model or feature mapping is approximate.

Files stay in memory and are never executed, installed, or opened by a PDF
renderer. The endpoint is `POST /api/analyze/file` with multipart field `file`.
Train the PDF model with:

```powershell
python scripts/train_pdf_models.py
```

It reads `data/raw/PDFMalware2022.csv`, writes `models/pdf_lightgbm.txt`, and
records held-out metrics in `models/pdf_training_metrics.json`. PDF family
labels are not published with the source data, so the PDF result intentionally
reports no family.

## EMBER 2018 benchmark (Colab)

Phase 2 is a standalone Colab experiment in
[`notebooks/ember_colab.ipynb`](notebooks/ember_colab.ipynb). It vectorizes the
official EMBER 2018 archive with `elastic/ember`, trains LightGBM and a PyTorch
MLP, and evaluates both models once on the official time-based test split.
Validation is the latest 10% of labeled training rows ordered by `appeared`, and
the deployment model is selected on validation macro F1 only.

The archive is **not** extracted locally. Keep
`ember_dataset_2018_2.tar.bz2` in `Downloads`, upload it to
`MyDrive/ember/` for Drive mode (the default), or set `ARCHIVE_MODE = "download"`
inside the notebook. The expected archive SHA-256 is
`b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812`.

Open the notebook in Colab, choose a GPU runtime, and run all cells. Full
vectorization needs roughly 25 GB of free disk and one to four hours; set
`SMOKE_TEST = True` at the top to verify the pipeline on a few thousand rows
first. The run writes `ember_lightgbm.txt`, `ember_mlp.pt`, `ember_metrics.json`,
and Markdown/PDF model cards, then downloads them as a zip. The MLP checkpoint
matches the project checkpoint schema but the dashboard does not expose EMBER
features.

The archive contains no license document, so the dataset is recorded as
**license not verified**; the `elastic/ember` repository states its data files
are MIT-licensed and its tooling is AGPL-3.0. See the
[dataset register](docs/datasets.md) for details.
