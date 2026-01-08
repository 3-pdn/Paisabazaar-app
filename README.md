# Paisabazar — Loan Eligibility Prediction (Advanced Project)

This is an advanced end-to-end ML project for predicting loan eligibility (Paisabazar project).
It includes:
- Synthetic sample dataset (`data/paisabazar_sample.csv`)
- EDA script (`eda.py`)
- Training script with preprocessing + hyperparameter tuning (`train.py`)
- Saved model & pipeline (`models/pipeline.joblib`) produced when `train.py` is run
- Streamlit app to serve the model (`app_streamlit.py`)
- Utilities for preprocessing (`utils.py`)
- Requirements in `requirements.txt`
- Example usage instructions below

## Folder structure
```
Paisabazar_Advanced/
├─ data/
│  └─ paisabazar_sample.csv
├─ models/
│  └─ pipeline.joblib   (created after running train.py)
├─ README.md
├─ requirements.txt
├─ eda.py
├─ train.py
├─ app_streamlit.py
├─ utils.py
└─ example_run.txt
```

## How to run (example)
1. Create a virtual environment
   ```
   python -m venv .venv
   source .venv/bin/activate   # mac/linux
   .venv\Scripts\activate    # windows
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run EDA
   ```
   python eda.py
   ```

4. Train model (this will save `models/pipeline.joblib`)
   ```
   python train.py
   ```

5. Run Streamlit app (after training)
   ```
   streamlit run app_streamlit.py
   ```

## Notes
- The dataset included is synthetic and for demonstration. Replace `data/paisabazar_sample.csv` with your real dataset if available.
- `train.py` performs preprocessing, uses `RandomizedSearchCV` with XGBoost classifier and saves a full pipeline.
- The Streamlit app expects `models/pipeline.joblib` to exist.
