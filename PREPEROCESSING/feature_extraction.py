from pathlib import Path

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    start_time = time.time()

    # ==========================================
    # 1. CREATE OUTPUT FOLDER
    # ==========================================

    os.makedirs((PROJECT_ROOT / 'Feature_Artifacts'), exist_ok=True)

    # ==========================================
    # 2. LOAD PROCESSED DATASET
    # ==========================================

    print("\nLoading processed dataset...")

    data = pd.read_csv(
        (PROJECT_ROOT / 'Processed_Data/processed_dataset.csv')
    )

    data = data.dropna(subset=["cleaned_text", "class"])

    X = data["cleaned_text"].astype(str)
    y = data["class"].astype(str)

    print("Dataset loaded:", len(data), "records")

    # ==========================================
    # 3. SPLIT DATASET
    # ==========================================

    print("\nSplitting dataset...")

    # 80% training, 20% temporary
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Divide remaining 20% equally:
    # 10% validation and 10% testing
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print("Training records:", len(X_train_text))
    print("Validation records:", len(X_val_text))
    print("Testing records:", len(X_test_text))

    # ==========================================
    # 4. INITIALIZE TF-IDF VECTORIZER
    # ==========================================

    print("\nInitializing TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32
    )

    # ==========================================
    # 5. EXTRACT FEATURES
    # ==========================================

    print("\nFitting TF-IDF on training data...")

    X_train = vectorizer.fit_transform(X_train_text)

    print("Transforming validation data...")
    X_val = vectorizer.transform(X_val_text)

    print("Transforming testing data...")
    X_test = vectorizer.transform(X_test_text)

    print("\nFeature extraction completed!")

    # ==========================================
    # 6. DISPLAY FEATURE INFORMATION
    # ==========================================

    print("\n========== FEATURE INFORMATION ==========")

    print("Number of features:", len(vectorizer.get_feature_names_out()))

    print("Training matrix shape:", X_train.shape)
    print("Validation matrix shape:", X_val.shape)
    print("Testing matrix shape:", X_test.shape)

    # ==========================================
    # 7. SAVE FEATURE MATRICES
    # ==========================================

    print("\nSaving feature matrices...")

    save_npz((PROJECT_ROOT / 'Feature_Artifacts/X_train.npz'), X_train)
    save_npz((PROJECT_ROOT / 'Feature_Artifacts/X_val.npz'), X_val)
    save_npz((PROJECT_ROOT / 'Feature_Artifacts/X_test.npz'), X_test)

    # ==========================================
    # 8. SAVE LABELS
    # ==========================================

    np.save((PROJECT_ROOT / 'Feature_Artifacts/y_train.npy'), y_train.to_numpy())
    np.save((PROJECT_ROOT / 'Feature_Artifacts/y_val.npy'), y_val.to_numpy())
    np.save((PROJECT_ROOT / 'Feature_Artifacts/y_test.npy'), y_test.to_numpy())

    # ==========================================
    # 9. SAVE TF-IDF VECTORIZER
    # ==========================================

    joblib.dump(
        vectorizer,
        (PROJECT_ROOT / 'Feature_Artifacts/tfidf_vectorizer.pkl')
    )

    # ==========================================
    # 10. FINAL REPORT
    # ==========================================

    elapsed_time = (time.time() - start_time) / 60

    print("\n========== FINAL REPORT ==========")
    print("Training features saved!")
    print("Validation features saved!")
    print("Testing features saved!")
    print("TF-IDF vectorizer saved!")

    print("\nOutput folder: Feature_Artifacts")
    print("Time taken:", round(elapsed_time, 2), "minutes")

    print("\nFEATURE EXTRACTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
