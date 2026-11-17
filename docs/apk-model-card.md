# Drebin-215 APK static-feature model card

## Model details

- Dataset: Drebin-215
- Raw source file: `drebin-215-dataset-5560malware-9476-benign.csv`
- Cleaned training file: `apk_drebin_clean.csv`
- Task: binary benign/malware classification from 215 pre-extracted APK features
- Deployment model: LightGBM
- Decision threshold: 0.5
- Generated: 2026-10-25T15:32:48.714734+00:00
- License: license not verified

## Intended use

Local academic experimentation and dashboard demonstrations only. The model
classifies one row of pre-extracted numeric APK features. It does not inspect
or execute APK binaries and must not be used as a production security control.

## Data preparation

- Source rows: 15,036
- Source labels: B: 9,476, S: 5,560
- `?` cells replaced with 0 before deduplication: 5
- Exact duplicate rows removed: 7,775
- Conflicting feature patterns removed: 3 (6 rows)
- Rows after cleaning: 7,255
- Cleaned class counts: {'Benign': 5537, 'Malware': 1718}

The `?` replacement happens before deduplication so the measured row counts
stay reproducible. Identical numeric feature vectors are kept in one split.

## Evaluation

- Split sizes: {'train': 5803, 'validation': 726, 'test': 726}
- Seed: 42
- Model selection metric: f1_macro

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| LightGBM | 0.9780 | 0.9698 | 0.9780 | 0.9973 |
| PyTorch MLP | 0.9628 | 0.9499 | 0.9633 | 0.9914 |

### Confusion matrices

LightGBM (rows = actual Benign/Malware, columns = predicted):

- Benign: ['544', '10']
- Malware: ['6', '166']

PyTorch MLP (rows = actual Benign/Malware, columns = predicted):

- Benign: ['534', '20']
- Malware: ['7', '165']

## Top LightGBM gain features

- `transact`: 17759.03
- `onServiceConnected`: 8539.77
- `READ_PHONE_STATE`: 6677.21
- `SEND_SMS`: 4742.25
- `TelephonyManager.getDeviceId`: 3576.25
- `Ljava.lang.Class.getCanonicalName`: 2800.32
- `chmod`: 2375.49
- `android.telephony.gsm.SmsManager`: 1323.08
- `WRITE_HISTORY_BOOKMARKS`: 1319.89
- `TelephonyManager.getSubscriberId`: 1299.38

## SHAP summary

- Sample rows explained: 200

- `WRITE_HISTORY_BOOKMARKS`: 0.8034 mean |SHAP|
- `transact`: 0.7162 mean |SHAP|
- `TelephonyManager.getDeviceId`: 0.6362 mean |SHAP|
- `TelephonyManager.getLine1Number`: 0.6045 mean |SHAP|
- `onServiceConnected`: 0.6027 mean |SHAP|

## Limitations

- Only 215 pre-extracted static features; no raw APK inspection.
- The source archive contained no license document, so the dataset is
  marked "license not verified".
- Static features can be evaded by packing, obfuscation, and dynamic code loading.
- Evaluation is a single grouped hold-out split, not a temporal benchmark.
- Class imbalance remains (benign rows outnumber malware rows).

## Ethical and safety notes

The application never executes uploaded binaries. Raw malware samples are
not redistributed with the repository, and dataset files remain local.
