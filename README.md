# Online Fraud Detection System — beginner project

This project includes your located CSV dataset, a Random Forest training script, a Streamlit website, and an optional FastAPI backend.

## Dataset schema checked
The CSV contains 2,500 rows and these seven columns:
- `transaction_amount`
- `transaction_hour`
- `new_device`
- `international_transaction`
- `failed_attempts`
- `account_age_days`
- `is_fraud` (target label: 0 or 1)

The inspected file has no missing values in those columns. It contains 2,371 rows labelled 0 and 129 labelled 1. This appears to be a synthetic educational dataset; do not present its metrics as real-world bank performance.

## Run on a Mac (VS Code terminal)
1. Install Python 3 if needed.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Train and evaluate the model:
   ```bash
   python train_model.py
   ```
   This creates `fraud_model.joblib` and prints metrics from a held-out test split. Use the actual printed values in your report.
6. Start the website:
   ```bash
   streamlit run app.py
   ```
   Streamlit will print a local URL, usually `http://localhost:8501`.

## Optional separate API
After completing steps 1–5, run:
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` to test `POST /predict` interactively.

Example JSON request:
```json
{
  "transaction_amount": 50.0,
  "transaction_hour": 14,
  "new_device": 0,
  "international_transaction": 0,
  "failed_attempts": 0,
  "account_age_days": 365
}
```

## Beginner definitions
- **Frontend:** the page the user interacts with.
- **Backend:** Python code that validates input and calls the model.
- **API:** a defined way for another program to send data to and receive data from the backend.
- **Features (X):** the six transaction inputs.
- **Target (y):** `is_fraud`, the known label used during training.
- **Random Forest:** a group of decision trees whose combined predictions produce a classification.

## Important
This is a classroom prototype. A synthetic dataset and its model cannot establish real fraud-detection accuracy. A prediction is not proof of wrongdoing and must not be used to approve, decline, or block real transactions. Only load model files you trust, because joblib/pickle-based files can execute code when loaded.
