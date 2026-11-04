# Recovery and reproduction report

Original binary recovery: FAILED. Searches of accessible workspaces, temporary
archives, six repository snapshots, local Git objects/reflogs, remote branch
history, and saved Library files found no corrected binaries.

Every delivered candidate file is newly reproduced or reconstructed. No file
is claimed to have been recovered from the original candidate directory.
The preprocessing source is a reconstruction whose bytes match its previously
recorded hash. Config/metrics/reports are newly generated; historical values
are reference evidence only.

## Integrity

- model.pkl: 366dc47c145480500a450a1b73a6766156d0255a1870bd799c0d242e1011ccc5; matches historical binary hash: True
- vectorizer.pkl: d6982dcc36a05a28e32f7baec594f9e21c89230e370791b2080ea7b1741617d2; matches historical binary hash: True

Original corrected bytes remain unavailable; an equal hash on a newly generated
file indicates byte identity, not successful recovery from storage.
20 valid historical model/artifact files remain unchanged. The invalid temporary
file is excluded from the preservation inventory and package. Production code,
branches, and historical reports were not modified. No Git publication occurred.

## Missing evidence and deviations

Original saved split indices and training script were lost. Their documented
logic was reconstructed; the regenerated split matches all saved label orders
and historical training/validation matrices within float32 rounding.
Original annotation criteria are unavailable. Synthetic denials, quotations,
hypotheticals, and third-person statements have no fabricated gold labels.
The new standalone loader uses the packaged preprocessing file; Flask integration
is not changed in this reproduction. Existing production continues unchanged.
The preprocessing policy still removes punctuation/sentence boundaries and some
stopwords, so context understanding and negation are not solved.

## Local verification

Use Python 3.12 (tested version recorded in configuration.json). Extract the ZIP,
open a terminal inside ELARIA_FINAL_CANDIDATE, and run:

```text
python -m venv .venv
```

Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1`.
Activate on Linux/macOS: `source .venv/bin/activate`.

```text
python -m pip install -r requirements.txt
python -m nltk.downloader punkt_tab stopwords wordnet
python smoke.py
```

Expected result: status PASS, both binary labels possible, saved bundle hash and
version checks pass, batch/single agreement, invalid input rejection, and identical
predictions from a fresh Python process. Do not run training for inference.
`smoke.load_bundle()` and `smoke.predict(list_of_texts, bundle)` provide standalone
inference; no routes or database writes are involved.

Reproduction (optional, never required to load): run.py is included with saved split
indices. The script refuses to overwrite model.pkl. It requires the verified CSV
and original project artifact directory, neither bundled here. In a separate output
copy without existing model binaries, use `python run.py train --dataset PATH
--legacy-root REPOSITORY`, inspect validation, then `python run.py finalize
--dataset PATH --legacy-root REPOSITORY`. Finalize refuses a repeated test evaluation.
Do not use the held-out test data for selecting or modifying the pipeline.

## Saved-artifact verification

{
  "status": "PASS",
  "checks": [
    "saved bundle hashes and versions",
    "model/vectorizer load and 5000 feature dimensions",
    "binary label mapping",
    "new input preprocessing",
    "both output labels",
    "batch matches single predictions",
    "six invalid input cases",
    "repeat determinism",
    "clean subprocess determinism"
  ]
}
