"""Verify existing matrices/models and reproduce reports; never overwrite artifacts."""
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from BACKEND.artifacts import verify_bundle

ROOT = Path(__file__).resolve().parent


def verify():
    verify_bundle(ROOT)
    indices = np.load(ROOT / "GA_Artifacts/selected_feature_indices.npy")
    data = {}
    report = {"versions": {p: version(p) for p in (
        "numpy", "scipy", "pandas", "scikit-learn", "xgboost", "nltk", "Flask", "joblib"
    )}, "splits": {}, "models": {}, "sha256": {}}
    for split in ("train", "val", "test"):
        original = load_npz(ROOT / f"Feature_Artifacts/X_{split}.npz")
        selected = load_npz(ROOT / f"GA_Artifacts/X_{split}_selected.npz")
        labels = np.load(ROOT / f"Feature_Artifacts/y_{split}.npy", allow_pickle=True).astype(str)
        assert original.shape[0] == selected.shape[0] == len(labels)
        assert (original[:, indices] != selected).nnz == 0
        data[split] = (selected, labels)
        report["splits"][split] = {"original_shape": original.shape,
                                     "selected_shape": selected.shape,
                                     "exact_column_match": True,
                                     "labels": dict(zip(*[v.tolist() for v in np.unique(labels, return_counts=True)]))}
    # Reproduce the original shared RNG's sampling sequence exactly.
    rng = np.random.default_rng(42)
    sampled = {}
    for name, training_size, validation_size in (
        ("KNN", 8000, 3000), ("Gradient_Boosting", 8000, 3000), ("Voting_Classifier", 20000, 5000)
    ):
        rng.choice(len(data["train"][1]), training_size, replace=False)
        sampled[name] = rng.choice(len(data["val"][1]), validation_size, replace=False)
    historical = pd.read_csv(ROOT / "reports/model_comparison.csv").set_index("Model")
    mapping = joblib.load(ROOT / "saved_models/label_mapping.pkl")
    inverse = {int(n): str(label) for label, n in mapping.items()}
    for path in sorted((ROOT / "saved_models").glob("*.pkl")):
        if path.stem == "label_mapping":
            continue
        model = joblib.load(path)
        assert model.n_features_in_ == 873
        x, y = data["val"]
        if path.stem in sampled:
            positions = sampled[path.stem]
            x, y = x[positions], y[positions]
        pred = model.predict(x)
        if path.stem in ("XGBoost", "Voting_Classifier"):
            pred = np.array([inverse[int(n)] for n in pred])
        metrics = {"Accuracy": accuracy_score(y, pred),
                   "Precision": precision_score(y, pred, average="macro", zero_division=0),
                   "Recall": recall_score(y, pred, average="macro", zero_division=0),
                   "F1_Score": f1_score(y, pred, average="macro", zero_division=0)}
        metrics_match = all(np.isclose(value, historical.loc[path.stem, key], atol=1e-12, rtol=0)
                            for key, value in metrics.items())
        report["models"][path.stem] = {"features": model.n_features_in_,
            "classes": model.classes_.tolist(), "validation_rows": len(y),
            "metrics": metrics, "historical_metrics_match": metrics_match}
        print(path.stem, "historical metrics match:", metrics_match, metrics, flush=True)
    # Re-evaluate only the saved mask, not the genetic search or six trained models.
    x, y = data["train"]
    v, z = data["val"]
    a, _ = train_test_split(np.arange(len(y)), train_size=12000, random_state=42, stratify=y)
    b, _ = train_test_split(np.arange(len(z)), train_size=3000, random_state=42, stratify=z)
    fitness_model = SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=5, tol=None, random_state=42)
    fitness_model.fit(x[a], y[a])
    fitness = f1_score(z[b], fitness_model.predict(v[b]), average="macro", zero_division=0)
    report["saved_mask_fitness"] = fitness
    report["feature_reduction_percent"] = 100 * (1 - len(indices) / 5000)
    assert round(fitness * 100, 2) == 82.69
    for directory in ("Feature_Artifacts", "GA_Artifacts", "saved_models"):
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file():
                report["sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["historical_reproduction_pass"] = all(m["historical_metrics_match"] for m in report["models"].values())
    output = ROOT / "validation_output/compatibility.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved compatibility report:", output)
    return report


if __name__ == "__main__":
    result = verify()
    sys.exit(0 if result["historical_reproduction_pass"] else 1)
