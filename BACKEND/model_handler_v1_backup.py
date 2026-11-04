
from pathlib import Path

import joblib
import numpy as np

from BACKEND.text_preprocessing import clean_text


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "MODEL_TRAINING"
    / "saved_models"
    / "XGBoost.pkl"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "MODEL_TRAINING"
    / "saved_models"
    / "label_mapping.pkl"
)

VECTORIZER_PATH = (
    PROJECT_ROOT
    / "Feature_Artifacts"
    / "tfidf_vectorizer.pkl"
)

FEATURE_INDICES_PATH = (
    PROJECT_ROOT
    / "GA_Artifacts"
    / "selected_feature_indices.npy"
)


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
# LOAD GROUP TRAINED ARTIFACTS
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

label_to_number = joblib.load(MAPPING_PATH)

number_to_label = {
    number: label
    for label, number in label_to_number.items()
}

vectorizer = joblib.load(VECTORIZER_PATH)

selected_feature_indices = np.load(
    FEATURE_INDICES_PATH
)

# Verify feature compatibility
if len(selected_feature_indices) != model.n_features_in_:
    raise ValueError(
        "GA-selected feature count does not match "
        "the XGBoost model input dimensions."
    )

if len(selected_feature_indices) == 0:
    raise ValueError("No GA-selected features were loaded.")

if selected_feature_indices.max() >= len(
    vectorizer.get_feature_names_out()
):
    raise ValueError(
        "Selected feature index exceeds vectorizer vocabulary."
    )

print("Group XGBoost model loaded successfully.")
print("Group TF-IDF vectorizer loaded successfully.")
print("Group label mapping loaded:", label_to_number)
print(
    "Group GA-selected features loaded:",
    len(selected_feature_indices)
)


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

    # 6. Obtain probability for the suicide class
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

    return {
        "prediction": label,
        "model_score": model_score,
        "cleaned_text": cleaned_text
    }
