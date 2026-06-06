# 📉 Telecom Customer Churn Prediction System

An end-to-end Machine Learning system for predicting telecom customer churn using **XGBoost**, **MLflow**, **FastAPI**, **Gradio**, and **Great Expectations**.
This project demonstrates a production-oriented ML workflow including data validation, feature engineering, experiment tracking, model serving, and interactive deployment.

---

# 🚀 Project Highlights

* ✅ Built a complete ML pipeline from raw CSV ingestion to deployment
* ✅ Implemented production-grade data validation with Great Expectations
* ✅ Trained an optimized XGBoost churn classifier with ROC-AUC of **0.837**
* ✅ Achieved **82.1% recall** for churn detection
* ✅ Integrated MLflow for experiment tracking and model artifact management
* ✅ Developed real-time prediction APIs using FastAPI
* ✅ Created an interactive Gradio frontend for business users
* ✅ Structured codebase using modular MLOps-style architecture

---

# 🧠 Problem Statement

Customer churn is one of the most critical business problems in the telecom industry.
The objective of this project is to predict whether a customer is likely to churn based on demographic, billing, and service usage data.

This enables telecom companies to:

* Identify high-risk customers
* Improve customer retention strategies
* Reduce revenue loss
* Launch targeted interventions

---

# 🏗️ System Architecture

```text
Raw Data
   │
   ▼
Data Validation (Great Expectations)
   │
   ▼
Preprocessing & Feature Engineering
   │
   ▼
XGBoost Model Training
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Model Serialization & Artifacts
   │
   ▼
FastAPI Inference API
   │
   ▼
Gradio Web Interface
```

---

# 📂 Project Structure

```text
Telcom-Customer-Churn-Predictor/
│
├── app/                         # FastAPI + Gradio application
├── configs/                     # Configuration files
├── data/
│   ├── raw_data/                # Raw telecom dataset
│   └── processed/               # Processed datasets
│
├── mlruns/                      # MLflow tracking artifacts
├── model/                       # Saved production model
├── notebooks/                   # Exploratory notebooks
├── scripts/                     # Pipeline & testing scripts
├── src/
│   ├── app/                     # FastAPI entrypoint
│   ├── data/                    # Data processing logic
│   ├── features/                # Feature engineering
│   ├── models/                  # Model training
│   ├── serving/                 # Inference pipeline
│   └── utils/                   # Validation utilities
│
├── tests/                       # Unit tests
├── requirements.txt
└── README.md
```

---

# ⚙️ Tech Stack

| Category            | Technologies           |
| ------------------- | ---------------------- |
| Programming         | Python                 |
| ML Framework        | XGBoost                |
| Data Processing     | Pandas, NumPy          |
| API Framework       | FastAPI                |
| Frontend UI         | Gradio                 |
| Experiment Tracking | MLflow                 |
| Validation          | Great Expectations     |
| Model Serialization | Pickle + MLflow PyFunc |
| Deployment Ready    | Docker Compatible      |

---

# 📊 Dataset Information

* Dataset Size: **7,043 customers**
* Features: **21 original features**
* Engineered Features: **31 final features**
* Target Variable: `Churn`

### Feature Categories

#### Customer Demographics

* Gender
* Partner
* Dependents

#### Service Information

* InternetService
* OnlineSecurity
* StreamingTV
* TechSupport

#### Billing & Contract

* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

---

# ✅ Data Validation Pipeline

Implemented automated validation checks using Great Expectations:

### Schema Validation

* Required column existence
* Null value checks
* Type consistency

### Business Rule Validation

* Valid contract types
* Valid internet service categories
* Binary field validation

### Statistical Validation

* Monthly charge range checks
* Tenure constraints
* Total charges consistency

### Validation Results

```text
Data validation PASSED: 23/23 checks successful
```

---

# 🧹 Feature Engineering

Implemented production-consistent preprocessing pipeline:

### Binary Encoding

Applied deterministic binary mappings:

```python
{
    "Yes": 1,
    "No": 0
}
```

### One-Hot Encoding

Generated dummy variables for multi-category features using:

```python
pd.get_dummies(drop_first=True)
```

### Feature Alignment

Ensured inference-time schema consistency by:

* Saving feature column order
* Reindexing missing columns
* Preventing train/serve skew

---

# 🤖 Model Training

### Model Used

* XGBoost Classifier

### Training Optimizations

* Class imbalance handling using `scale_pos_weight`
* Efficient preprocessing pipeline
* Fast inference optimization

### Performance Metrics

| Metric    | Score |
| --------- | ----- |
| ROC-AUC   | 0.837 |
| Recall    | 0.821 |
| Precision | 0.490 |
| F1 Score  | 0.614 |
| Accuracy  | 72.6% |

### Training Statistics

```text
Training Time: 0.40s
Inference Time: 0.0038s
Throughput: 370,171 samples/sec
```

---

# 📈 Classification Report

```text
              precision    recall  f1-score   support

           0      0.914     0.692     0.788      1035
           1      0.490     0.821     0.614       374

    accuracy                          0.726      1409
```

---

# 🧪 MLflow Integration

Integrated MLflow for:

* Experiment tracking
* Metric logging
* Artifact storage
* Model versioning
* Reproducibility

Tracked:

* Precision
* Recall
* ROC-AUC
* Training time
* Prediction latency
* Validation results

---

# 🌐 FastAPI Inference Service

Developed production-ready REST APIs using FastAPI.

### Endpoints

#### Health Check

```http
GET /
```

#### Prediction Endpoint

```http
POST /predict
```

### Example Request

```json
{
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "tenure": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

### Example Response

```json
{
  "prediction": "Likely to churn"
}
```

---

# 🎨 Gradio Web Interface

Built an interactive UI for non-technical users:

* Real-time predictions
* Dropdown-based feature selection
* Professional dark-themed dashboard
* Example customer profiles
* Integrated directly into FastAPI

---

# 🛠️ Installation

## Clone Repository

```bash
git clone <repository-url>
cd Telcom-Customer-Churn-Predictor
```

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training Pipeline

```bash
python3 scripts/run_pipeline.py \
--input data/raw_data/og.csv \
--target Churn
```

---

# ▶️ Run FastAPI Server

```bash
python3 -m uvicorn src.app.main:app --reload
```

---

# 🌍 Access Application

### FastAPI Swagger Docs

```text
http://127.0.0.1:8000/docs
```

### Gradio UI

```text
http://127.0.0.1:8000/ui
```

---

# 🔥 Key Engineering Learnings

* Built production-grade ML architecture
* Solved train/serve skew using feature schema persistence
* Implemented robust data quality validation
* Integrated ML experiment management workflows
* Developed deployable inference APIs
* Structured scalable ML project directories

---

# 📌 Future Improvements

* Docker containerization
* CI/CD integration using GitHub Actions
* Kubernetes deployment
* SHAP explainability integration
* Real-time streaming predictions
* Drift monitoring and retraining pipelines

---

# 👩💻 Author

**Samiksha Solanke**

Aspiring Machine Learning Engineer focused on:

* Applied Machine Learning
* MLOps
* Backend AI Systems
* Production ML Deployment