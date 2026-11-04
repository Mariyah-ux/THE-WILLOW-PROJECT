from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Load the dataset
    file_path = (PROJECT_ROOT / 'Dataset/Dataset/Suicide_Detection.csv')

    data = pd.read_csv(file_path)

    # Display basic information
    print("\n========== DATASET OVERVIEW ==========")

    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])

    print("\nColumn names:")
    print(data.columns.tolist())



    # Check missing values
    print("\n========== MISSING VALUES ==========")
    print(data.isnull().sum())

    # Check duplicate records
    print("\n========== DUPLICATE RECORDS ==========")
    print("Duplicates:", data.duplicated().sum())

    # Class distribution
    print("\n========== CLASS DISTRIBUTION ==========")
    print(data["class"].value_counts())

    # Generate class distribution graph
    os.makedirs((PROJECT_ROOT / 'reports'), exist_ok=True)

    data["class"].value_counts().plot(
        kind="bar",
        title="Class Distribution"
    )

    plt.xlabel("Class")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    plt.savefig((PROJECT_ROOT / 'reports/class_distribution.png'))
    plt.show()

    # TEXT LENGTH ANALYSIS

    data["text_length"] = data["text"].astype(str).apply(len)

    print("\n========== TEXT LENGTH ANALYSIS ==========")

    print("Average text length:", data["text_length"].mean())
    print("Minimum text length:", data["text_length"].min())
    print("Maximum text length:", data["text_length"].max())

    # Text length graph

    plt.figure(figsize=(10, 5))

    plt.hist(data["text_length"], bins=50)

    plt.title("Distribution of Text Lengths")
    plt.xlabel("Number of Characters")
    plt.ylabel("Number of Posts")

    plt.tight_layout()
    plt.savefig((PROJECT_ROOT / 'reports/text_length_distribution.png'))
    plt.show()

    print("\nText analysis completed successfully!")


if __name__ == "__main__":
    main()
