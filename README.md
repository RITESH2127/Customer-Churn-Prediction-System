# 🚀 Customer Churn Prediction System
Predict customer churn for telecom providers using an end-to-end ML pipeline + a lightweight web UI — trained models, preprocessing artifacts, and a FastAPI backend to serve real-time predictions.

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Top Language](https://img.shields.io/github/languages/top/RITESH2127/Customer-Churn-Prediction-System)](https://github.com/RITESH2127/Customer-Churn-Prediction-System)
[![Repo Size](https://img.shields.io/github/repo-size/RITESH2127/Customer-Churn-Prediction-System)](https://github.com/RITESH2127/Customer-Churn-Prediction-System)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

---

## 📌 Project Overview
Customer Churn Prediction System is a production-oriented, end-to-end machine learning application designed to identify telecom customers at high risk of churning. It demonstrates the full ML lifecycle: data preparation → model training & selection → artifact saving → REST API serving → responsive frontend for real-time predictions. The project was built to showcase practical ML engineering, explainability (feature importance), and an accessible demo UI for stakeholders.

---

## ✨ Key Features
- 🔁 End-to-end ML pipeline: data collection (download or synthesize), preprocessing, training, evaluation.
- 🤖 Multiple model training and automatic best-model selection (Logistic Regression, Random Forest, Gradient Boosting).
- 📦 Artifacts persisted as joblib files: preprocessor + best model + feature names.
- ⚡ FastAPI backend with a /predict endpoint for real-time predictions.
- 🎨 Responsive frontend (Glassmorphism-style) to collect customer attributes and display churn probability.
- 📊 Feature importance plot saved to models/feature_importance.png.
- 🧪 Minimal test dependency (pytest) for adding tests.
- 🧩 Clear modular code — easy to extend and productionize.

---

## 📁 Repo Layout (annotated)

```text
.
├── LICENSE                              # MIT License
├── README.md                            # (this file)
├── requirements.txt                     # Python dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset (large CSV)
├── best_model.joblib                    # Serialized trained model (optional)
├── preprocessor.joblib                  # Serialized preprocessing pipeline
├── feature_names.joblib                 # Saved training feature set
├── feature_importance.png               # Saved feature importance plot
├── index.html                           # Frontend HTML (served at /)
├── script.js                            # Frontend JS (POSTs to /predict)
├── style.css                            # Frontend CSS
├── main.py                              # FastAPI app (serves UI & /predict)
├── prepare_data.py                      # Downloads or synthesizes dataset
└── train.py                             # Training pipeline (EDA, train & save)
```

---

## 🧠 System Architecture (how it works)
1. prepare_data.py fetches the Telco churn dataset from a remote source. If unavailable, it synthesizes a realistic dataset (same schema).
2. train.py:
   - Loads CSV from data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   - Performs EDA & preprocessing (handles TotalCharges, drops customerID)
   - Builds ColumnTransformer pipeline (StandardScaler for numeric, OneHotEncoder for categorical)
   - Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting)
   - Evaluates on test set (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Selects the best model by F1 score and saves:
     - models/best_model.joblib
     - models/preprocessor.joblib
     - models/feature_names.joblib
   - Generates feature_importance.png (top 20 features)
3. main.py (FastAPI) loads the saved model + preprocessor on startup, exposes:
   - GET / → serves frontend index.html
   - POST /predict → accepts customer JSON (see schema below), preprocesses, returns prediction + probability
4. Frontend (index.html, script.js, style.css) presents a form, sends POST to /predict, and displays results.

Diagram (conceptual):
Frontend (browser) ↔ FastAPI (main.py)
FastAPI loads -> preprocessor.joblib + best_model.joblib
Training pipeline (train.py) → models/* artifacts
Data ingestion (prepare_data.py) → data/*.csv

---

## ⚙️ Tech Stack / Built With
- Language: Python 3.8+
- Web framework: FastAPI
- ML: scikit-learn
- Data: pandas, numpy
- Serialization: joblib
- Visuals: matplotlib, seaborn
- Frontend: HTML5, CSS3, JavaScript
- Dev server: uvicorn
- Testing: pytest

Notable libraries:
- scikit-learn (modeling, pipeline)
- FastAPI (REST API & static file serving)
- joblib (save/load model & preprocessing artifacts)

---

## 🚀 Getting Started — Run Locally

Prerequisites
- Python 3.8+
- pip (or pipx/venv)
- Git (optional)

Quick setup (recommended in a virtual environment)

1) Clone repository
```bash
git clone https://github.com/RITESH2127/Customer-Churn-Prediction-System.git
cd Customer-Churn-Prediction-System
```

2) Create & activate virtual environment (optional but recommended)
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Prepare dataset
- The project tries to download the Telco dataset. If that fails it synthesizes a dataset with realistic distributions.

```bash
python prepare_data.py
# Output: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

5) Train models (generates models/ directory)
```bash
python train.py
# Produces models/best_model.joblib, models/preprocessor.joblib, models/feature_names.joblib
# Also saves models/feature_importance.png (top 20 features)
```

6) Start the API + serve the frontend
Option A — run via uvicorn (recommended during development)
```bash
uvicorn main:app --reload
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Option B — run the module directly
```bash
python main.py
```

7) Open the app in a browser
```
http://127.0.0.1:8000
```

---

## 🔎 API — /predict

Endpoint
- POST /predict
- Content-Type: application/json
- Body: JSON object with customer fields (see schema below)
- Response: JSON { "prediction": "Churn" | "Retained", "probability": 0.0–1.0 }

Customer JSON schema (fields accepted by the API)
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.00,
  "TotalCharges": "840.0"
}
```

Note: In the code, TotalCharges is expected as a string (to match original dataset format) and is converted to numeric on the server.

Example curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "gender":"Female",
        "SeniorCitizen":0,
        "Partner":"No",
        "Dependents":"No",
        "tenure":2,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"Fiber optic",
        "OnlineSecurity":"No",
        "OnlineBackup":"No",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"No",
        "StreamingMovies":"No",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges":99.50,
        "TotalCharges":"199.00"
      }'
```

Example response
```json
{
  "prediction": "Churn",
  "probability": 0.7423459281921387
}
```

---

## 🖼️ Frontend Screenshots / Demo
- The frontend is located at root index.html with static assets in the repository:
  - index.html
  - style.css
  - script.js
- After starting the server, visit http://127.0.0.1:8000 to view a polished form for entering customer attributes and receiving churn predictions.

Feature importance (example):
![Feature Importance](/feature_importance.png)
> (If running locally, view the generated image at models/feature_importance.png after training.)

---

## 🛠️ Troubleshooting & Tips
- Model not loaded? Ensure models/best_model.joblib and models/preprocessor.joblib exist. If missing: run python train.py and restart the server.
- If prepare_data.py fails to download the IBM dataset it will synthesize data automatically — this is intentional for offline use.
- Uvicorn import error? Use the uvicorn command from your virtualenv (pip install -r requirements.txt).
- If frontend JS fails to POST, check CORS or ensure you are making requests to the same host/port as the server.

---

## 📈 Roadmap / Future Scope
Planned / suggested improvements:
- 🔐 Add authentication (API key / OAuth) and user sessions for auditability.
- 🐳 Dockerize the app and add a docker-compose for a one-command start.
- ☁️ Add CI/CD and deployment recipes for platforms like Render / Heroku / AWS.
- 🔄 Implement continuous retraining / scheduled model updates (via Airflow or GitHub Actions).
- 📊 Add a dashboard (Streamlit / Dash / React + Chart.js) for cohort analysis and retention strategies.
- 🧪 Add unit & integration tests for training and API endpoints (pytest coverage).
- ♻�� Add configuration via environment variables and a proper settings file.

---

## 🤝 Contributing
Contributions are very welcome — thank you for helping improve this project!

Suggested workflow:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feat/awesome-feature
   ```
3. Make changes, add tests, and run them:
   ```bash
   pip install -r requirements.txt
   pytest
   ```
4. Keep commits small & focused; follow conventional commit messages when possible.
5. Open a Pull Request with a clear description and why the change is needed.

Code quality guidelines:
- Use black/flake8 for formatting & linting (not included in repo by default).
- Add unit tests for new logic (train/test pipeline, API handlers).
- Document breaking changes in the PR description.

---

## 🧾 License
This project is licensed under the MIT License — see the LICENSE file for details.

---

## 👨‍💻 Author & Contact
Ritesh Kumar — https://github.com/RITESH2127

For issues, improvements, or collaboration, please open an issue in this repository or reach out via GitHub.

---

## ✅ Quick Reference / Checklist
- [ ] Clone repo
- [ ] Create virtualenv
- [ ] pip install -r requirements.txt
- [ ] python prepare_data.py
- [ ] python train.py
- [ ] uvicorn main:app --reload
- [ ] Open http://127.0.0.1:8000 and test the UI

---

Thank you for checking out the Customer Churn Prediction System — built to be readable, reproducible, and production-ready. If you'd like, I can also:
- produce a Dockerfile and docker-compose for one-command deployment,
- add a small pytest test suite scaffold,
- or create a GitHub Actions workflow to run tests and train on cron.
