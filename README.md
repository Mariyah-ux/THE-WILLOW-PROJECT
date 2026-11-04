# Elaria

Educational NLP classification of text into the dataset labels `suicide` and
`non-suicide`. These labels describe content, not a person's diagnosis or safety.
The deployed classifier is the existing XGBoost model, with 5,000 TF-IDF features
reduced to 873 by the saved GA selection.

## Setup and run

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m nltk.downloader punkt_tab stopwords wordnet
python -m BACKEND.app
```

For the exact tested Linux/Python 3.12 environment, install
`requirements-lock-linux-py312.txt` instead of requirements-dev.txt.
The platform-specific lock includes the Linux XGBoost CUDA dependency even
though inference uses CPU. Use the portable requirements files on Windows.

Open http://127.0.0.1:5000/health. Direct execution with
`python BACKEND/app.py` also works. Startup checks the artifact hashes, exact
scikit-learn/XGBoost versions, feature alignment, labels and NLP resources.
Missing resources fail explicitly; the server never downloads data on import.

## API

| Route | Input | Output |
|---|---|---|
| GET / | None | Elaria service identity |
| GET /health | None | Healthy after successful model/resource initialization |
| POST /predict | JSON: `{"text": "I enjoyed reading a book."}` | Dataset label, model_score, metadata record ID and disclaimer |
| POST /predict-csv | Multipart field `file`, UTF-8 .csv with `text` column | Original columns plus prediction, model_score and disclaimer |
| GET /history | None | Latest 100 prediction metadata records |
| GET /statistics | None | Total stored prediction count |

Uploads are limited to 10 MB. Empty/malformed input returns JSON 400; oversized
uploads return 413; internal/database failures return sanitized JSON 500.
CSV processing completes all predictions and output serialization before one
database transaction. A failed input row or transaction does not leave partial
batch records. Existing routes and label names are preserved.

`model_score` is the model's probability for the dataset's `suicide` class.
It is not calibrated as individual suicide risk. The project is not a clinical
diagnostic, screening, emergency-response or personal-safety system.

## Data and privacy

The default database is `BACKEND/database/elaria.db`. Set
`ELARIA_DATABASE_PATH` before startup to choose another location. Only label,
score, source, batch ID and timestamp are retained; raw/cleaned posts are not
stored or logged. CSV input is returned to the requester in the download and
processed in memory. Protect downloaded CSVs and the local database.

No automatic expiration is implemented. For an academic demonstration, use a
temporary database path and remove that database after the session, with the
server stopped. Back up metadata only if retention is intended. An existing
`willow.db` is neither deleted nor migrated automatically; explicitly point
ELARIA_DATABASE_PATH at a reviewed existing database if its history is required.

The server binds to localhost with debug disabled. History and statistics have
no authentication. This remains a local research backend; public/multi-user
deployment needs an access-control and retention design.

## Verification

```bash
python -m pip check
python -m pytest -q tests
python verify_artifacts.py
python evaluate_final_model.py
python final_evaluation/evaluate_final_model.py --output-dir validation_output/legacy_entrypoint
```

Tests use the committed models and temporary SQLite databases. No six-model
retraining or genetic search occurs. `verify_artifacts.py` compares the three
matrix pairs, checks all models and recreates the saved mask's SGD fitness
evaluation. It writes `validation_output/compatibility.json` and exits 1 when
historical metrics differ. **Known finding:** KNN validation accuracy is
56.1333%, versus historical 56.3000%; the other five model reports reproduce.
That nonzero exit must not be treated as a fully passing historical audit.

Evaluation writes new reports under `validation_output/final_evaluation`,
leaving historical reports intact. The legacy nested evaluation entry delegates
to the same implementation.

## Source workflow and artifact preservation

1. `PREPEROCESSING/nlp_preprocessing.py` reads
   `Dataset/Dataset/Suicide_Detection.csv` and writes
   `Processed_Data/processed_dataset.csv`.
2. `PREPEROCESSING/feature_extraction.py` performs the stratified 80:10:10
   split and fits TF-IDF only on training data.
3. `PREPEROCESSING/genetic_algorithm.py` creates the GA selection and matrices.
4. `train_models.py` trains six models and writes `saved_models/` and `reports/`.
5. `evaluate_final_model.py` evaluates the existing XGBoost on the saved test set.

These offline scripts now use repository-relative paths and main guards.
**Do not rerun steps 1–4 during stabilization:** they overwrite training outputs,
the raw/processed datasets are absent, and a new run requires a separately
reviewed artifact manifest. Do not edit hashes simply to bypass bundle checks.
The manifest locks the audited inference files; it is not proof of original
dataset provenance. Pickle/joblib files and object arrays must be trusted.

Training and inference share `BACKEND/text_preprocessing.py`. The original
training script computed stems but discarded them; the effective lemmatization
behavior is preserved. No split, label, model architecture, or trained model
was changed.

`BACKEND/model_handler_v1_backup.py` is an inactive historical backup with old
paths; it is not imported by the app. The duplicate historical reports under
`final_evaluation/final_evaluation/` are preserved.

See [baseline audit](PHASE1_AUDIT.md) and [repair and test results](PHASE1_RESULTS.md).
