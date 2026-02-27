"""
Loan Default Prediction Pipeline Functions
==========================================
This module contains functions for a classification pipeline that predicts
loan defaults using Random Forest classifier.
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths
DATA_PATH = "/opt/airflow/dags/data/loan_data.csv"
MODEL_DIR = "/opt/airflow/dags/model"


def load_data():
    """
    Load loan data from CSV file and serialize it.

    Returns:
        bytes: Serialized pandas DataFrame
    """
    print("=" * 50)
    print("TASK 1: Loading Data")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH)

    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Default rate: {df['default'].mean() * 100:.1f}%")

    # Serialize and return
    return pickle.dumps(df)


def preprocess_data(data):
    """
    Preprocess the loan data: encode categoricals, scale features, split data.

    Args:
        data: Serialized DataFrame from load_data task

    Returns:
        bytes: Serialized dictionary containing train/test splits
    """
    print("=" * 50)
    print("TASK 2: Preprocessing Data")
    print("=" * 50)

    # Deserialize
    df = pickle.loads(data)

    # Encode categorical variable (loan_purpose)
    le = LabelEncoder()
    df['loan_purpose_encoded'] = le.fit_transform(df['loan_purpose'])

    # Define features and target
    feature_cols = ['age', 'income', 'loan_amount', 'credit_score',
                    'employment_years', 'debt_to_income', 'previous_defaults',
                    'loan_purpose_encoded']

    X = df[feature_cols]
    y = df['default']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"✓ Encoded categorical features")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Features scaled using StandardScaler")

    # Package everything
    processed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_cols,
        'scaler': scaler,
        'label_encoder': le
    }

    return pickle.dumps(processed_data)


def build_save_model(data, filename):
    """
    Build Random Forest classifier and save it to file.

    Args:
        data: Serialized preprocessed data dictionary
        filename: Name of file to save model

    Returns:
        bytes: Serialized training metrics
    """
    print("=" * 50)
    print("TASK 3: Building and Saving Model")
    print("=" * 50)

    # Deserialize
    processed_data = pickle.loads(data)

    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    feature_names = processed_data['feature_names']

    # Build Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle imbalanced classes
    )

    model.fit(X_train, y_train)

    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)

    # Get feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))

    print(f"✓ Model trained successfully")
    print(f"✓ Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"✓ Top 3 important features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:3]:
        print(f"  - {feat}: {imp:.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': processed_data['scaler'],
            'label_encoder': processed_data['label_encoder'],
            'feature_names': feature_names
        }, f)

    print(f"✓ Model saved to {model_path}")

    # Return metrics for next task
    metrics = {
        'train_accuracy': train_accuracy,
        'feature_importance': feature_importance
    }

    return pickle.dumps(metrics)


def evaluate_model(filename, metrics_data, test_data):
    """
    Load the saved model and evaluate on test set.

    Args:
        filename: Name of saved model file
        metrics_data: Serialized training metrics
        test_data: Serialized preprocessed data with test set

    Returns:
        str: Evaluation results summary
    """
    print("=" * 50)
    print("TASK 4: Evaluating Model")
    print("=" * 50)

    # Load model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)

    model = model_package['model']

    # Get test data
    processed_data = pickle.loads(test_data)
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])

    # Get training metrics
    train_metrics = pickle.loads(metrics_data)

    print(f"✓ Model loaded from {model_path}")
    print(f"\n📊 RESULTS:")
    print(f"  Training Accuracy: {train_metrics['train_accuracy'] * 100:.2f}%")
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"\n📋 Confusion Matrix:")
    print(f"  {conf_matrix}")
    print(f"\n📈 Classification Report:")
    print(class_report)

    # Feature importance summary
    print(f"\n🎯 Feature Importance Ranking:")
    sorted_features = sorted(
        train_metrics['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, imp) in enumerate(sorted_features, 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    return f"Test Accuracy: {test_accuracy * 100:.2f}%"