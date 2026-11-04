# Elaria corrected candidate — new reproduction

The original model/vectorizer were irrecoverable after workspace restoration.
This package contains a newly fitted candidate, not recovered original bytes.
The cleaner was reconstructed from recorded source and matches the historical
cleaner SHA-256 exactly: 015f35dbcfd0787b03f902e0f96cba88d039999e8632d2a322409e18484fe6e9.

One XGBoost fit uses parameters read directly from the historical production
XGBoost (120 trees, depth 5, learning rate .1, subsample/colsample .8, hist,
seed 42, two threads), with full 5,000-feature TF-IDF, ngrams 1–2, min_df 3,
max_df .95, sublinear_tf and float32. No GA or parameter search occurred.

The verified CSV reconstructs the historical 80:10:10 split. Training/validation
matrix alignment and label sequence checks passed. The vocabulary and model
were fitted only on training data. The candidate was frozen after validation,
then test data was evaluated once. Split row numbers are zero-based CSV data
positions, not fabricated source identifiers. No duplicate regrouping was done;
historical split/data-label limitations remain.

Labels describe content categories; scores are not personal risk estimates.
Negation and contrast terms are preserved, but synthetic probes show remaining
context limitations. Probe outputs are qualitative checks without invented
sentence-level dataset labels. See smoke_results.json for every example.

## Actual reproduction metrics

| Metric | Validation | Final test |
|---|---:|---:|
| accuracy | 0.8944260033625038 | 0.8973618415380636 |
| macro_precision | 0.896072012158714 | 0.8985901113752874 |
| macro_recall | 0.8944357372771597 | 0.8973690260133451 |
| macro_f1 | 0.8943178734854822 | 0.8972839053559756 |
| suicide_precision | 0.9216806412973372 | 0.9207812357397098 |
| suicide_recall | 0.8621789346664368 | 0.8695914497500431 |
| suicide_f1 | 0.8909374304163883 | 0.8944545414247086 |
| confusion_matrix | [[10745, 850], [1599, 10003]] | [[10728, 868], [1513, 10089]] |
| false_negatives | 1599 | 1513 |
| false_positives | 850 | 868 |

Fit time: 110.7396106860001 seconds.

Python/dependencies: {"numpy": "2.3.5", "scipy": "1.17.0", "pandas": "2.2.3", "scikit-learn": "1.9.1", "xgboost": "3.4.1", "joblib": "1.5.3", "nltk": "3.9.2", "python": "3.12.14"}

Historical reference (not a target): test accuracy 89.74%, macro F1 89.73%.
