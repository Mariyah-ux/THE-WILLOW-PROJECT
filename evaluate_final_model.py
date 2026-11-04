
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate(output_dir=None):
    # 1. PATHS

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = BASE_DIR

    MODEL_PATH = os.path.join(
        BASE_DIR, "saved_models", "XGBoost.pkl"
    )

    MAPPING_PATH = os.path.join(
        BASE_DIR, "saved_models", "label_mapping.pkl"
    )

    FEATURE_PATH = os.path.join(
        PROJECT_DIR, "GA_Artifacts", "X_test_selected.npz"
    )

    LABEL_PATH = os.path.join(
        PROJECT_DIR, "Feature_Artifacts", "y_test.npy"
    )

    OUTPUT_DIR = output_dir or os.path.join(BASE_DIR, "validation_output", "final_evaluation")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. LOAD MODEL, MAPPING AND TEST DATA

    print("\nLoading model and test data...")

    model = joblib.load(MODEL_PATH)
    label_to_number = joblib.load(MAPPING_PATH)

    number_to_label = {
        number: label
        for label, number in label_to_number.items()
    }

    classes = [
        label for label, number
        in sorted(label_to_number.items(), key=lambda item: item[1])
    ]

    X_test = load_npz(FEATURE_PATH)
    y_test = np.load(LABEL_PATH, allow_pickle=True).astype(str)

    print("Model loaded.")
    print("Label mapping:", label_to_number)
    print("Test feature shape:", X_test.shape)
    print("Test label count:", len(y_test))

    if X_test.shape[0] != len(y_test):
        raise ValueError("Test features and labels have different row counts.")

    if X_test.shape[1] != model.n_features_in_:
        raise ValueError("Test feature count does not match model input.")

    if not set(np.unique(y_test)).issubset(set(classes)):
        raise ValueError("Test data contains unknown labels.")

    # 3. GENERATE PREDICTIONS

    print("\nGenerating predictions...")

    predictions_numeric = model.predict(X_test)

    y_pred = np.array([
        number_to_label[int(value)]
        for value in predictions_numeric
    ])

    # 4. CALCULATE METRICS

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # 5. CLASSIFICATION REPORT

    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    # 6. CONFUSION MATRIX

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes]
    )

    matrix_df.to_csv(
        os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    )

    # 7. SAVE FINAL METRICS

    results = pd.DataFrame([{
        "Model": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Evaluation_Type": "Full test set",
        "Evaluation_Rows": len(y_test)
    }])

    results.to_csv(
        os.path.join(OUTPUT_DIR, "final_test_results.csv"),
        index=False
    )

    with open(
        os.path.join(OUTPUT_DIR, "classification_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print("\nResults saved in:", OUTPUT_DIR)
    print("\nFINAL MODEL EVALUATION FINISHED!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the existing Elaria XGBoost model without retraining.")
    parser.add_argument("--output-dir", help="Destination for newly computed reports.")
    args = parser.parse_args()
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
