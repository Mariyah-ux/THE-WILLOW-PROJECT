"""Prepare the original dataset without changing the learned text transform."""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
from BACKEND.text_preprocessing import clean_text


def main():
    data = pd.read_csv(PROJECT_ROOT / "Dataset/Dataset/Suicide_Detection.csv")
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    data = data.dropna(subset=["text", "class"])
    data = data.drop_duplicates(subset=["text"])
    data["cleaned_text"] = data["text"].apply(clean_text)
    output = PROJECT_ROOT / "Processed_Data/processed_dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Processed {len(data)} records; saved to {output}")


if __name__ == "__main__":
    main()
