# Customer Churn Prediction System

This is an end-to-end Machine Learning web application designed to predict whether a telecom customer is likely to churn. It includes a complete ML pipeline, a FastAPI REST backend, and a modern, responsive frontend.

## 📁 Project Structure

```text
d:\churn
├── app
│   └── main.py              # FastAPI application
├── data                     # Raw CSV data
├── frontend
│   ├── index.html           # UI Layout
│   ├── script.js            # UI Logic
│   └── style.css            # Premium Tech Styling
├── models                   # Serialized Models and Preprocessors
│   ├── best_model.joblib
│   ├── feature_importance.png
│   └── preprocessor.joblib
├── notebooks                # Reserved for EDA and experimental notebooks
├── src
│   ├── prepare_data.py      # Data downloading/synthesis logic
│   └── train.py             # Main ML pipeline (Preprocessing, Modeling, Evaluation)
├── README.md
└── requirements.txt
```

## 🚀 How to Run Locally

### 1. Prerequisites
Make sure you have Python 3.8+ installed.

### 2. Install Dependencies
Navigate to the project directory and install the required Python packages:
```bash
cd d:/churn
pip install -r requirements.txt
```

### 3. Data Preparation
To download the Telco Customer Churn dataset (or synthesize it if the URL fails), run:
```bash
python src/prepare_data.py
```

### 4. Train the Model
Run the machine learning pipeline to preprocess data, train multiple classifier models (Logistic Regression, Random Forest, Gradient Boosting), and save the best one automatically in the `models/` directory:
```bash
python src/train.py
```
> Note: This will also generate a `feature_importance.png` plot in the `models/` folder.

### 5. Start the Application
Run the FastAPI backend server using Uvicorn:
```bash
uvicorn app.main:app --reload
```

### 6. Use the Web App
Open your browser and navigate to:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Fill in the customer details and click **Predict Churn** to get real-time AI-driven probability.

## 🧠 Technical Details
* **Machine Learning:** Scikit-Learn
* **Models Evaluated:** Logistic Regression, Random Forest, Gradient Boosting
* **Backend:** FastAPI (Python)
* **Frontend:** HTML5, CSS3 (Glassmorphism, Gradient animations), Vanilla JS
* **Data Processing:** Pandas, NumPy
