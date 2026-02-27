# Loan Default Prediction Pipeline - Airflow Lab

**Author:** Balaji Sundar Anand Babu  
**Course:** MLOps - Northeastern University  
**Date:** February 2026

---

## Overview

This project implements an end-to-end **Loan Default Prediction** machine learning pipeline using **Apache Airflow** orchestrated within **Docker** containers. The pipeline uses a **Random Forest Classifier** to predict whether a loan applicant will default based on various financial and demographic features.

### Pipeline Tasks

| Task | Description |
|------|-------------|
| `load_data_task` | Load loan applicant data from CSV file |
| `preprocess_task` | Encode categorical variables, scale features, split data |
| `build_model_task` | Train Random Forest classifier and save model |
| `evaluate_task` | Evaluate model performance and display metrics |

---

## ML Model

This pipeline performs **binary classification** to predict loan defaults using the following approach:

- **Algorithm:** Random Forest Classifier
- **Features:** Age, Income, Loan Amount, Credit Score, Employment Years, Debt-to-Income Ratio, Previous Defaults, Loan Purpose
- **Target:** Default (1 = Default, 0 = No Default)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Dataset Features

| Feature | Description |
|---------|-------------|
| `age` | Applicant's age |
| `income` | Annual income |
| `loan_amount` | Requested loan amount |
| `credit_score` | Credit score (300-850) |
| `employment_years` | Years at current job |
| `debt_to_income` | Debt-to-income ratio |
| `previous_defaults` | Number of previous defaults |
| `loan_purpose` | Purpose of loan (personal/home/auto) |
| `default` | Target variable (0 or 1) |

---

## Project Structure

```
airflow_loan_prediction/
├── dags/
│   ├── data/
│   │   └── loan_data.csv          # Dataset
│   ├── model/
│   │   └── loan_model.pkl         # Saved model (generated)
│   ├── src/
│   │   ├── __init__.py
│   │   └── pipeline.py            # ML pipeline functions
│   └── loan_dag.py                # Airflow DAG definition
├── logs/                          # Airflow logs
├── plugins/                       # Airflow plugins
├── config/                        # Airflow config
├── .env                           # Environment variables
├── docker-compose.yaml            # Docker configuration
└── README.md                      # This file
```

---

## Prerequisites

- **Docker Desktop** installed and running
  - [Mac Installation](https://docs.docker.com/desktop/install/mac-install/)
  - [Windows Installation](https://docs.docker.com/desktop/install/windows-install/)
  - [Linux Installation](https://docs.docker.com/desktop/install/linux-install/)
- Minimum **4GB RAM** allocated to Docker (8GB recommended)

---

## Installation & Setup

### Step 1: Clone/Create Project Directory

```bash
mkdir -p ~/mlops_labs/airflow_loan_prediction
cd ~/mlops_labs/airflow_loan_prediction
```

### Step 2: Create Directory Structure

```bash
mkdir -p dags/data dags/model dags/src logs plugins config
touch dags/src/__init__.py
```

### Step 3: Download Docker Compose File

```bash
# Mac/Linux
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'

# Windows
curl -o docker-compose.yaml https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml
```

### Step 4: Create Environment File

```bash
echo "AIRFLOW_UID=50000" > .env
```

### Step 5: Configure docker-compose.yaml

Make the following edits in `docker-compose.yaml`:

```yaml
# 1. Disable example DAGs (around line 61)
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

# 2. Add Python packages (around line 70)
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn}

# 3. Update credentials (around line 230)
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
```

### Step 6: Add Project Files

Place the following files in their respective locations:
- `dags/data/loan_data.csv` - Dataset
- `dags/src/pipeline.py` - Pipeline functions
- `dags/loan_dag.py` - DAG definition

---

## Running the Pipeline

### Step 1: Initialize Airflow Database

```bash
docker compose up airflow-init
```

Wait for the message: `airflow-init-1 exited with code 0`

### Step 2: Start Airflow Services

```bash
docker compose up
```

Wait until you see:
```
airflow-webserver-1  | 127.0.0.1 - - [...] "GET /health HTTP/1.1" 200 ...
```

### Step 3: Access Airflow UI

1. Open browser: **http://localhost:8080**
2. Login credentials:
   - Username: `airflow2`
   - Password: `airflow2`

### Step 4: Run the DAG

1. Find **"Loan_Default_Prediction"** in the DAGs list
2. Toggle the switch to **ON** (enable the DAG)
3. Click the **▶️ Play button** to trigger manually
4. Click on the DAG name → **Graph** tab to monitor progress
5. Once complete, click **evaluate_task** → **Logs** to view results

### Step 5: Stop Airflow

```bash
docker compose down
```

---

## Pipeline Functions

### `load_data()`
Loads loan data from CSV file and serializes it for XCom transfer.

### `preprocess_data(data)`
- Encodes categorical variables (loan_purpose) using LabelEncoder
- Scales numerical features using StandardScaler
- Splits data into training (80%) and test (20%) sets

### `build_save_model(data, filename)`
- Trains a Random Forest Classifier with 100 estimators
- Handles class imbalance with `class_weight='balanced'`
- Saves model to pickle file
- Returns training metrics and feature importance

### `evaluate_model(filename, metrics_data, test_data)`
- Loads saved model
- Generates predictions on test set
- Outputs accuracy, confusion matrix, and classification report
- Displays feature importance ranking

---

## Expected Output

After running the pipeline, the `evaluate_task` logs will display:

```
📊 RESULTS:
  Training Accuracy: XX.XX%
  Test Accuracy: XX.XX%

📋 Confusion Matrix:
  [[TN  FP]
   [FN  TP]]

📈 Classification Report:
              precision    recall  f1-score   support
  No Default       X.XX      X.XX      X.XX         X
     Default       X.XX      X.XX      X.XX         X

🎯 Feature Importance Ranking:
  1. credit_score: 0.XXXX
  2. debt_to_income: 0.XXXX
  3. income: 0.XXXX
  ...
```

---

## DAG Visualization

```
load_data_task → preprocess_task → build_model_task → evaluate_task
```

---

## Technologies Used

- **Apache Airflow 2.9.2** - Workflow orchestration
- **Docker & Docker Compose** - Containerization
- **Python 3.x** - Programming language
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **Random Forest** - Classification algorithm

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Docker Setup](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [MLOps Course - Northeastern University](https://github.com/raminmohammadi/MLOps)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `docker: command not found` | Launch Docker Desktop app first |
| YAML syntax errors | Check indentation (use spaces, not tabs) |
| DAG not appearing | Check for Python syntax errors in `loan_dag.py` |
| Import errors in logs | Verify `_PIP_ADDITIONAL_REQUIREMENTS` in docker-compose.yaml |

---

## License

This project is for educational purposes as part of the MLOps course at Northeastern University.