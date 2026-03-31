# 🚀 Customer Churn Prediction System

An end-to-end **Machine Learning Web Application** that predicts whether a telecom customer is likely to churn. This project demonstrates a complete ML pipeline integrated with a modern web interface, making it production-ready and ideal for real-world deployment.

---

## 📌 Overview

Customer churn is a critical problem in subscription-based businesses. This system leverages machine learning to analyze customer behavior and predict churn probability, enabling businesses to take proactive retention actions.

---

## ✨ Features

* 🔍 End-to-end ML pipeline (Data → Training → Prediction)
* 🤖 Multiple model training & automatic best model selection
* 📊 Feature importance visualization
* ⚡ FastAPI-based high-performance backend
* 🎨 Modern responsive frontend (Glassmorphism UI)
* 🔄 Real-time churn prediction with probability score
* 📁 Clean, modular, and scalable project structure
* 🚀 Ready for GitHub and deployment

---

## 📁 Project Structure

```text
churn/
├── app/
│   └── main.py                # FastAPI backend application
├── data/                      # Dataset storage
├── frontend/
│   ├── index.html             # User Interface
│   ├── script.js              # Frontend logic
│   └── style.css              # Styling (Glassmorphism UI)
├── models/                    # Trained models & artifacts
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   └── feature_importance.png
├── notebooks/                 # EDA & experimentation (optional)
├── src/
│   ├── prepare_data.py        # Data fetching / generation
│   └── train.py               # Training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Backend:** FastAPI
* **Frontend:** HTML5, CSS3, JavaScript
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 🧠 Machine Learning Workflow

1. Data Collection & Preparation
2. Data Cleaning & Preprocessing
3. Feature Encoding & Scaling
4. Model Training:

   * Logistic Regression
   * Random Forest
   * Gradient Boosting
5. Model Evaluation:

   * Accuracy
   * Precision, Recall, F1-score
   * ROC-AUC
6. Best Model Selection & Saving
7. Feature Importance Visualization

---

## 🚀 Getting Started (Run Locally)

### 1️⃣ Prerequisites

* Python 3.8 or higher installed
* pip package manager

---

### 2️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd churn
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Prepare the Dataset

Download or generate the dataset:

```bash
python src/prepare_data.py
```

---

### 5️⃣ Train the Model

Run the ML pipeline:

```bash
python src/train.py
```

> This step trains multiple models, selects the best one, and saves it in the `models/` directory along with preprocessing artifacts and feature importance visualization.

---

### 6️⃣ Start the Backend Server

```bash
uvicorn app.main:app --reload
```

---

### 7️⃣ Launch the Application

Open your browser and go to:

```
http://127.0.0.1:8000
```

Enter customer details and click **Predict Churn** to get real-time predictions.

---

## 📊 Output

* Churn Prediction (Yes / No)
* Probability Score
* Feature Importance Graph

---

## 📌 Use Cases

* Telecom companies
* Subscription-based businesses
* Customer retention strategies
* Data science portfolio projects

---

## 📈 Future Improvements

* 🔐 User authentication system
* ☁️ Cloud deployment (AWS / Render / Docker)
* 📊 Advanced dashboards (Streamlit / React)
* 🔄 Continuous model retraining
* 📡 API integration with real-time databases

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 💡 Author

Developed as a complete end-to-end Machine Learning project for real-world application and portfolio showcase.

---
