
# ============================================================
# ELARIA
# MEMBER 3: MACHINE LEARNING MODEL TRAINING
# ============================================================

import os
import time
import gc
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS AND FOLDER SETUP
# ============================================================

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    GA_DIR = os.path.join(BASE_DIR, "GA_Artifacts")
    DATA_DIR = os.path.join(BASE_DIR, "Feature_Artifacts")

    OUTPUT_DIR = os.path.join(
        BASE_DIR, "saved_models"
    )

    REPORT_DIR = os.path.join(
        BASE_DIR, "reports"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    RANDOM_STATE = 42

    rng = np.random.default_rng(RANDOM_STATE)


    # ============================================================
    # 2. LOAD TRAINING AND VALIDATION DATA
    # ============================================================

    print("\nLoading selected feature datasets...")

    X_train = load_npz(
        os.path.join(GA_DIR, "X_train_selected.npz")
    ).tocsr()

    X_val = load_npz(
        os.path.join(GA_DIR, "X_val_selected.npz")
    ).tocsr()

    y_train = np.load(
        os.path.join(DATA_DIR, "y_train.npy"),
        allow_pickle=True
    ).astype(str)

    y_val = np.load(
        os.path.join(DATA_DIR, "y_val.npy"),
        allow_pickle=True
    ).astype(str)


    # Validate dataset dimensions

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "Training features and labels have different row counts."
        )

    if X_val.shape[0] != len(y_val):
        raise ValueError(
            "Validation features and labels have different row counts."
        )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Training and validation feature counts do not match."
        )


    # Create consistent numeric label mapping

    classes = np.unique(y_train)

    if len(classes) != 2:
        raise ValueError("This project expects exactly two classes.")

    if not set(np.unique(y_val)).issubset(set(classes)):
        raise ValueError("Validation data contains unknown classes.")

    label_to_number = {
        label: index
        for index, label in enumerate(classes)
    }

    number_to_label = {
        index: label
        for label, index in label_to_number.items()
    }

    y_train_num = np.array(
        [label_to_number[label] for label in y_train],
        dtype=np.int32
    )

    y_val_num = np.array(
        [label_to_number[label] for label in y_val],
        dtype=np.int32
    )

    print("\nTraining data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Classes:", classes)
    print("Label mapping:", label_to_number)

    joblib.dump(
        label_to_number,
        os.path.join(OUTPUT_DIR, "label_mapping.pkl")
    )


    # ============================================================
    # 3. HELPER FUNCTIONS
    # ============================================================

    results = []


    def get_sample_indices(total_rows, sample_size):

        sample_size = min(sample_size, total_rows)

        return rng.choice(
            total_rows,
            size=sample_size,
            replace=False
        )


    def save_model(name, model):

        model_path = os.path.join(
            OUTPUT_DIR,
            name + ".pkl"
        )

        joblib.dump(model, model_path)

        print("\nModel saved successfully:", model_path)


    def evaluate_model(
        name,
        predictions,
        actual_labels,
        evaluation_type
    ):

        predictions = np.asarray(predictions).astype(str)
        actual_labels = np.asarray(actual_labels).astype(str)

        metrics = {
            "Model": name,

            "Accuracy": accuracy_score(
                actual_labels,
                predictions
            ),

            "Precision": precision_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0
            ),

            "Recall": recall_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0
            ),

            "F1_Score": f1_score(
                actual_labels,
                predictions,
                average="macro",
                zero_division=0
            ),

            "Evaluation_Type": evaluation_type,

            "Evaluation_Rows": len(actual_labels)
        }

        results.append(metrics)

        print("\nClassification Report:")

        print(
            classification_report(
                actual_labels,
                predictions,
                labels=classes,
                zero_division=0
            )
        )

        matrix = confusion_matrix(
            actual_labels,
            predictions,
            labels=classes
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=classes,
            columns=classes
        )

        matrix_df.to_csv(
            os.path.join(
                REPORT_DIR,
                name + "_confusion_matrix.csv"
            )
        )

        print("Confusion matrix saved.")

        return metrics


    def train_and_evaluate(
        name,
        model,
        X_fit,
        y_fit,
        X_evaluate,
        y_actual,
        evaluation_type
    ):

        print("\n==========", name.upper(), "==========")

        start_time = time.time()

        model.fit(X_fit, y_fit)

        training_time = (
            time.time() - start_time
        ) / 60

        print(
            "Training time:",
            round(training_time, 2),
            "minutes"
        )

        # Save immediately after training

        save_model(name, model)

        predictions = model.predict(X_evaluate)

        evaluate_model(
            name,
            predictions,
            y_actual,
            evaluation_type
        )

        return model


    # ============================================================
    # 4. RANDOM FOREST
    # ============================================================

    try:

        rf = RandomForestClassifier(
            n_estimators=60,
            max_depth=25,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=2,
            random_state=RANDOM_STATE
        )

        rf = train_and_evaluate(
            "Random_Forest",
            rf,
            X_train,
            y_train,
            X_val,
            y_val,
            "Full validation set"
        )

    except Exception as error:

        print("\nRandom Forest failed:", error)


    gc.collect()


    # ============================================================
    # 5. DECISION TREE
    # ============================================================

    try:

        dt = DecisionTreeClassifier(
            max_depth=30,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        dt = train_and_evaluate(
            "Decision_Tree",
            dt,
            X_train,
            y_train,
            X_val,
            y_val,
            "Full validation set"
        )

    except Exception as error:

        print("\nDecision Tree failed:", error)


    gc.collect()


    # ============================================================
    # 6. K-NEAREST NEIGHBORS
    # ============================================================

    try:

        print("\n========== K-NEAREST NEIGHBORS ==========")

        # Use a smaller sample because KNN is expensive
        # on large datasets.

        knn_train_indices = get_sample_indices(
            X_train.shape[0],
            8000
        )

        knn_val_indices = get_sample_indices(
            X_val.shape[0],
            3000
        )

        X_knn_train = X_train[knn_train_indices]
        y_knn_train = y_train[knn_train_indices]

        X_knn_val = X_val[knn_val_indices]
        y_knn_val = y_val[knn_val_indices]

        knn = KNeighborsClassifier(
            n_neighbors=5,
            algorithm="brute",
            n_jobs=2
        )

        knn = train_and_evaluate(
            "KNN",
            knn,
            X_knn_train,
            y_knn_train,
            X_knn_val,
            y_knn_val,
            "Sampled validation set"
        )

        del X_knn_train, X_knn_val

    except Exception as error:

        print("\nKNN failed:", error)


    gc.collect()


    # ============================================================
    # 7. GRADIENT BOOSTING
    # ============================================================

    try:

        print("\n========== GRADIENT BOOSTING ==========")

        # GradientBoostingClassifier requires dense input.
        # Convert only small samples to protect RAM.

        gb_train_indices = get_sample_indices(
            X_train.shape[0],
            8000
        )

        gb_val_indices = get_sample_indices(
            X_val.shape[0],
            3000
        )

        X_gb_train = X_train[
            gb_train_indices
        ].toarray().astype(np.float32)

        y_gb_train = y_train[gb_train_indices]

        X_gb_val = X_val[
            gb_val_indices
        ].toarray().astype(np.float32)

        y_gb_val = y_val[gb_val_indices]

        gb = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=2,
            random_state=RANDOM_STATE
        )

        gb = train_and_evaluate(
            "Gradient_Boosting",
            gb,
            X_gb_train,
            y_gb_train,
            X_gb_val,
            y_gb_val,
            "Sampled validation set"
        )

        del X_gb_train, X_gb_val

    except Exception as error:

        print("\nGradient Boosting failed:", error)


    gc.collect()


    # ============================================================
    # 8. XGBOOST
    # ============================================================

    try:

        print("\n========== XGBOOST ==========")

        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=RANDOM_STATE
        )

        start_time = time.time()

        # Train with numeric labels

        xgb.fit(
            X_train,
            y_train_num
        )

        training_time = (
            time.time() - start_time
        ) / 60

        print(
            "XGBoost training time:",
            round(training_time, 2),
            "minutes"
        )

        save_model("XGBoost", xgb)

        # Predict numeric labels

        xgb_predictions_num = xgb.predict(X_val)

        # Convert predictions to original class names

        xgb_predictions = np.array([
            number_to_label[int(value)]
            for value in xgb_predictions_num
        ])

        evaluate_model(
            "XGBoost",
            xgb_predictions,
            y_val,
            "Full validation set"
        )

    except Exception as error:

        print("\nXGBoost failed:", error)


    gc.collect()


    # ============================================================
    # 9. VOTING CLASSIFIER
    # ============================================================

    try:

        print("\n========== VOTING CLASSIFIER ==========")

        # Voting combines Random Forest, Decision Tree,
        # and XGBoost.

        # Use a sample to control training time and RAM.

        voting_train_indices = get_sample_indices(
            X_train.shape[0],
            20000
        )

        voting_val_indices = get_sample_indices(
            X_val.shape[0],
            5000
        )

        X_voting_train = X_train[
            voting_train_indices
        ]

        y_voting_train = y_train_num[
            voting_train_indices
        ]

        X_voting_val = X_val[
            voting_val_indices
        ]

        y_voting_val = y_val[
            voting_val_indices
        ]

        voting = VotingClassifier(

            estimators=[

                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=40,
                        max_depth=20,
                        min_samples_leaf=5,
                        n_jobs=2,
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "dt",
                    DecisionTreeClassifier(
                        max_depth=25,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE
                    )
                ),

                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=80,
                        max_depth=4,
                        learning_rate=0.1,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=2,
                        random_state=RANDOM_STATE
                    )
                )

            ],

            voting="soft",
            n_jobs=1
        )

        start_time = time.time()

        voting.fit(
            X_voting_train,
            y_voting_train
        )

        training_time = (
            time.time() - start_time
        ) / 60

        print(
            "Voting Classifier training time:",
            round(training_time, 2),
            "minutes"
        )

        save_model(
            "Voting_Classifier",
            voting
        )

        voting_predictions_num = voting.predict(
            X_voting_val
        )

        voting_predictions = np.array([
            number_to_label[int(value)]
            for value in voting_predictions_num
        ])

        evaluate_model(
            "Voting_Classifier",
            voting_predictions,
            y_voting_val,
            "Sampled validation set"
        )

        del X_voting_train, X_voting_val

    except Exception as error:

        print("\nVoting Classifier failed:", error)


    gc.collect()


    # ============================================================
    # 10. SAVE MODEL COMPARISON REPORT
    # ============================================================

    print("\n========== MODEL COMPARISON ==========")

    if results:

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="F1_Score",
            ascending=False
        )

        results_path = os.path.join(
            REPORT_DIR,
            "model_comparison.csv"
        )

        results_df.to_csv(
            results_path,
            index=False
        )

        print(
            results_df.to_string(index=False)
        )

        print("\nComparison report saved:", results_path)

    else:

        print("No models completed successfully.")

    print("\n================================================")
    print("MEMBER 3 TRAINING SCRIPT FINISHED")
    print("================================================")

    print("\nSaved models folder:", OUTPUT_DIR)
    print("Saved reports folder:", REPORT_DIR)




if __name__ == "__main__":
    main()
