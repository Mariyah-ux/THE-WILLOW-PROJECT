
from pathlib import Path

import joblib
import numpy as np

from BACKEND.text_preprocessing import clean_text
from BACKEND.artifacts import verify_bundle
from BACKEND.context_review import review_reason, REVIEW


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "saved_models" / "XGBoost.pkl"
MAPPING_PATH = PROJECT_ROOT / "saved_models" / "label_mapping.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "Feature_Artifacts" / "tfidf_vectorizer.pkl"
FEATURE_INDICES_PATH = PROJECT_ROOT / "GA_Artifacts" / "selected_feature_indices.npy"

manifest = verify_bundle(PROJECT_ROOT)

# --------------------------------------------------
# VERIFY ARTIFACTS
# --------------------------------------------------

required_files = [
    MODEL_PATH,
    MAPPING_PATH,
    VECTORIZER_PATH,
    FEATURE_INDICES_PATH
]

for file_path in required_files:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required model artifact not found: {file_path}"
        )


# --------------------------------------------------
# LOAD VERIFIED ELARIA MODEL ARTIFACTS
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

label_to_number = joblib.load(MAPPING_PATH)
if label_to_number != {"non-suicide": 0, "suicide": 1}:
    raise ValueError("Unexpected Elaria label mapping.")

number_to_label = {
    int(number): str(label)
    for label, number in label_to_number.items()
}

vectorizer = joblib.load(VECTORIZER_PATH)

selected_feature_indices = np.load(
    FEATURE_INDICES_PATH
)

# Validate feature indices
if not np.issubdtype(selected_feature_indices.dtype, np.integer):
    raise ValueError("Feature indices must contain integers.")
if selected_feature_indices.ndim != 1:
    raise ValueError("Feature indices must be a one-dimensional array.")

if len(selected_feature_indices) == 0:
    raise ValueError("No GA-selected features were loaded.")

if len(np.unique(selected_feature_indices)) != len(selected_feature_indices):
    raise ValueError("Duplicate feature indices detected.")

if selected_feature_indices.min() < 0:
    raise ValueError("Negative feature index detected.")

if selected_feature_indices.max() >= len(
    vectorizer.get_feature_names_out()
):
    raise ValueError(
        "Selected feature index exceeds vectorizer vocabulary."
    )

if len(selected_feature_indices) != model.n_features_in_:
    raise ValueError(
        "GA-selected feature count does not match "
        "the XGBoost model input dimensions."
    )

if set(model.classes_) != set(number_to_label.keys()):
    raise ValueError(
        "Model classes do not match the label mapping."
    )

mask = np.load(PROJECT_ROOT / "GA_Artifacts" / "best_feature_mask.npy", allow_pickle=False)
# Trusted, hash-verified historical file contains a numpy object array.
names = np.load(PROJECT_ROOT / "GA_Artifacts" / "selected_feature_names.npy", allow_pickle=True)
feature_names = vectorizer.get_feature_names_out()
if mask.dtype != np.bool_ or mask.shape != (len(feature_names),):
    raise ValueError("GA mask does not match the vectorizer.")
if not np.array_equal(np.flatnonzero(mask), selected_feature_indices):
    raise ValueError("GA mask and feature indices disagree.")
if not np.array_equal(feature_names[selected_feature_indices], names):
    raise ValueError("GA feature names and vocabulary order disagree.")
if len(feature_names) != manifest["features"] or len(names) != manifest["selected_features"]:
    raise ValueError("Feature dimensions disagree with the audited bundle.")
# Check NLP resources during startup without network access or user text.
clean_text("resource check")


# --------------------------------------------------
# TEXT PREDICTION FUNCTION
# --------------------------------------------------

def predict_text(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")

    # 1. Preprocess input text
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        raise ValueError(
            "No usable text remains after preprocessing."
        )

    # 2. Convert text into TF-IDF features
    tfidf_features = vectorizer.transform(
        [cleaned_text]
    )

    # 3. Apply GA-selected features
    selected_features = tfidf_features[
        :, selected_feature_indices
    ]

    # 4. Generate numerical prediction
    prediction = int(
        model.predict(selected_features)[0]
    )

    # 5. Convert numerical prediction into class name
    if prediction not in number_to_label:
        raise ValueError(
            f"Unknown prediction label: {prediction}"
        )

    label = number_to_label[prediction]

    # 6. Obtain probability for the dataset's suicide class
    model_score = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            selected_features
        )[0]

        suicide_label = label_to_number.get("suicide")

        if suicide_label in model.classes_:

            class_index = list(model.classes_).index(
                suicide_label
            )

            model_score = float(
                probabilities[class_index]
            )

    reason = review_reason(text)
    result = {
        "prediction": REVIEW if reason else label,
        "model_score": None if reason else model_score,
        "cleaned_text": cleaned_text
    }

    if reason:
        # In-memory diagnostics only; API explicitly selects public fields.
        result["_model_diagnostics"] = {"prediction": label, "model_score": model_score}
        result["review_reason"] = reason
    return result
