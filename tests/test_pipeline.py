"""Integration tests use the committed models, never substitute model mocks."""
import importlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(tmp_path, monkeypatch):
    from BACKEND.database import db_handler as db
    monkeypatch.setattr(db, "DATABASE_PATH", tmp_path / "elaria.db")
    db.initialize_database()
    module = importlib.import_module("BACKEND.app")
    module.app.config.update(TESTING=True)
    return module, module.app.test_client(), db


@pytest.mark.parametrize("text,expected", [
    ("<b>The cats</b> are running, not dogs! https://example.com", "cat running not dog"),
    ("No, never nor not", "no never nor not"),
    ("the and is 123 !!!", ""),
    (None, ""),
    ("", ""),
])
def test_preprocessing(text, expected):
    from BACKEND.text_preprocessing import clean_text
    from PREPEROCESSING.nlp_preprocessing import clean_text as training_clean
    assert clean_text is training_clean
    assert clean_text(text) == expected


def test_bundle_and_order():
    from BACKEND import model_handler as mh
    from BACKEND.artifacts import verify_bundle
    verify_bundle(ROOT)
    assert len(mh.vectorizer.get_feature_names_out()) == 5000
    assert mh.vectorizer.ngram_range == (1, 2)
    assert mh.vectorizer.dtype == np.float32
    assert mh.model.n_features_in_ == 873
    assert mh.label_to_number == {"non-suicide": 0, "suicide": 1}
    assert np.array_equal(np.flatnonzero(mh.mask), mh.selected_feature_indices)
    assert np.array_equal(mh.names, mh.feature_names[mh.selected_feature_indices])


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_saved_feature_alignment(split):
    from BACKEND.model_handler import selected_feature_indices
    x = load_npz(ROOT / f"Feature_Artifacts/X_{split}.npz")
    selected = load_npz(ROOT / f"GA_Artifacts/X_{split}_selected.npz")
    labels = np.load(ROOT / f"Feature_Artifacts/y_{split}.npy", allow_pickle=True)
    assert x.shape[0] == selected.shape[0] == len(labels)
    assert selected.shape[1] == 873
    assert (x[:, selected_feature_indices] != selected).nnz == 0
    assert set(labels) == {"non-suicide", "suicide"}


def test_bundle_rejects_missing_tampered_and_wrong_version(tmp_path, monkeypatch):
    from BACKEND import artifacts
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({
        "versions": {}, "sha256": {"model.pkl": "wrong_hash"},
    }))
    with pytest.raises(FileNotFoundError):
        artifacts.verify_bundle(tmp_path)
    (tmp_path / "model.pkl").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="integrity"):
        artifacts.verify_bundle(tmp_path)
    monkeypatch.setattr(artifacts, "version", lambda _: "0.0")
    with pytest.raises(RuntimeError, match="install requirements"):
        artifacts.verify_bundle(ROOT)


def test_real_prediction_score_and_decoding():
    from BACKEND import model_handler as mh
    text = "I enjoyed a pleasant afternoon reading a book."
    result = mh.predict_text(text)
    x = mh.vectorizer.transform([mh.clean_text(text)])[:, mh.selected_feature_indices]
    predicted = int(mh.model.predict(x)[0])
    assert result["prediction"] == mh.number_to_label[predicted]
    suicide_column = list(mh.model.classes_).index(mh.label_to_number["suicide"])
    assert result["model_score"] == pytest.approx(float(mh.model.predict_proba(x)[0, suicide_column]))


def test_historical_full_test_reproduced():
    from BACKEND import model_handler as mh
    x = load_npz(ROOT / "GA_Artifacts/X_test_selected.npz")
    y = np.load(ROOT / "Feature_Artifacts/y_test.npy", allow_pickle=True).astype(str)
    p = np.array([mh.number_to_label[int(n)] for n in mh.model.predict(x)])
    assert len(y) == 23198
    assert accuracy_score(y, p) == pytest.approx(0.8272695922062246, abs=1e-12)
    assert f1_score(y, p, average="macro") == pytest.approx(0.8263665162645757, abs=1e-12)
    assert confusion_matrix(y, p).tolist() == [[10432, 1164], [2843, 8759]]


@pytest.mark.parametrize("name", ["Decision_Tree", "Random_Forest", "KNN", "Gradient_Boosting", "Voting_Classifier", "XGBoost"])
def test_all_saved_models_load_and_predict(name):
    model = joblib.load(ROOT / f"saved_models/{name}.pkl")
    x = load_npz(ROOT / "GA_Artifacts/X_val_selected.npz")[:2]
    assert model.n_features_in_ == 873
    assert len(model.predict(x)) == 2


def test_api_single_history_statistics(api):
    module, client, db = api
    assert client.get("/").json["project"] == "Elaria"
    assert client.get("/health").json["status"] == "healthy"
    response = client.post("/predict", json={"text": "I enjoyed reading a book."})
    assert response.status_code == 200
    assert response.json["prediction"] in {"suicide", "non-suicide"}
    assert 0 <= response.json["model_score"] <= 1
    assert "not an individual suicide-risk estimate" in response.json["disclaimer"]
    assert "cleaned_text" not in response.json
    assert client.get("/statistics").json["total_predictions"] == 1
    record = client.get("/history").json["records"][0]
    assert record["id"] == response.json["record_id"]
    assert set(record) == {"id", "batch_id", "source", "prediction", "model_score", "created_at"}
    assert b"enjoyed reading" not in db.DATABASE_PATH.read_bytes()


@pytest.mark.parametrize("payload", [[], {}, {"text": None}, {"text": 42}, {"text": " "}, {"text": "!!!123"}])
def test_invalid_json_input(api, payload):
    _, client, db = api
    assert client.post("/predict", json=payload).status_code == 400
    assert db.get_prediction_count() == 0


def test_malformed_json(api):
    _, client, _ = api
    assert client.post("/predict", data="{", content_type="application/json").status_code == 400


def upload(client, content, filename="input.csv"):
    return client.post("/predict-csv", data={"file": (BytesIO(content), filename)})


def test_csv_real_batch(api):
    _, client, db = api
    response = upload(client, b"text,tag\nI enjoyed a good book.,one\nI feel sad and alone.,two\n")
    assert response.status_code == 200
    assert "elaria_predictions.csv" in response.headers["Content-Disposition"]
    result = pd.read_csv(BytesIO(response.data))
    assert len(result) == 2
    assert set(result.columns) == {"text", "tag", "prediction", "model_score", "disclaimer"}
    assert list(result.tag) == ["one", "two"]
    assert result.model_score.between(0, 1).all()
    records = db.get_prediction_history()
    assert len(records) == 2 and len({r["batch_id"] for r in records}) == 1


@pytest.mark.parametrize("content", [b"", b"text\n", b"wrong\nhello\n", b"text\n\"\"\n", b"text\n123\n", b'text\n"unterminated\n', b"text\n\xff\n"])
def test_invalid_csv(api, content):
    _, client, db = api
    response = upload(client, content)
    assert response.status_code == 400 and response.is_json
    assert db.get_prediction_count() == 0


def test_invalid_later_csv_row_does_not_partially_persist(api):
    _, client, db = api
    response = upload(client, b"text\nA pleasant afternoon reading.\n!!!123\n")
    assert response.status_code == 400
    assert db.get_prediction_count() == 0


def test_csv_missing_file_wrong_extension_and_size(api):
    _, client, _ = api
    assert client.post("/predict-csv").status_code == 400
    assert upload(client, b"text\nhello\n", "input.txt").status_code == 400
    assert upload(client, b"text\n" + b"x" * (10 * 1024 * 1024)).status_code == 413


def test_database_transaction_rolls_back(api):
    _, _, db = api
    valid = {"source": "csv", "prediction": "non-suicide", "model_score": 0.2}
    invalid = {**valid, "source": None}
    with pytest.raises(sqlite3.IntegrityError):
        db.save_predictions([valid, invalid])
    assert db.get_prediction_count() == 0


def test_database_errors_are_json_and_do_not_log_content(api, monkeypatch, caplog):
    module, client, _ = api
    def failure(*args, **kwargs):
        raise sqlite3.OperationalError("private synthetic marker")
    monkeypatch.setattr(module, "get_prediction_history", failure)
    monkeypatch.setattr(module, "get_prediction_count", failure)
    monkeypatch.setattr(module, "save_prediction", failure)
    monkeypatch.setattr(module, "save_predictions", failure)
    responses = [client.get("/history"), client.get("/statistics"),
                 client.post("/predict", json={"text": "A pleasant afternoon."}),
                 upload(client, b"text\nA pleasant afternoon.\n")]
    assert all(r.status_code == 500 and r.is_json for r in responses)
    assert all(b"private synthetic marker" not in r.data for r in responses)
    assert "private synthetic marker" not in caplog.text


def test_offline_modules_import_without_processing(tmp_path):
    modules = ["PREPEROCESSING.nlp_preprocessing", "PREPEROCESSING.dataset_analysis",
               "PREPEROCESSING.feature_extraction", "PREPEROCESSING.genetic_algorithm",
               "train_models", "evaluate_final_model", "final_evaluation.evaluate_final_model"]
    code = "import importlib; " + "; ".join(f"importlib.import_module({m!r})" for m in modules)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_missing_nltk_resources_fail_without_downloading(tmp_path):
    code = '''
import nltk
nltk.data.path = []
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected network download")
nltk.download = forbidden
from BACKEND.text_preprocessing import clean_text
try:
    clean_text("resource check")
except RuntimeError as error:
    assert "python -m nltk.downloader" in str(error)
else:
    raise AssertionError("Missing resources should fail explicitly")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
