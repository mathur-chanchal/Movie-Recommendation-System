#!/usr/bin/env bash
# Reproduce everything from the supplied dataset (≈2 minutes). Run from this folder.
set -e
python3 01_generate_data.py     # 500 -> 5,000 rows + labelled faults
python3 02_preprocess.py        # clean -> data/dataset_clean.csv
python3 03_eda.py               # figures/01-04
python3 04_evaluate.py          # tuning, test metrics, diversity, audit -> results/, figures/05-07
python3 -m pytest -q            # 21 tests
echo "Dashboard:  streamlit run app_ui.py   |   API:  python3 app.py"
