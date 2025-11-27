import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import joblib
import os


def train_fraud_detection_model():
    """Train XGBoost model for fraud detection"""
    
    print("=" * 60)
    print("Training Fraud Detection Model")
    print("=" * 60)
    
    # Check if data exists
    data_file = 'ml/fraud_data.csv'
    if not os.path.exists(data_file):
        print(f"\n⚠ Training data not found at {data_file}")
        print("Generating synthetic data...")
        from generate_data import generate_synthetic_fraud_data
        df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.15)
        df.to_csv(data_file, index=False)
        print(f"✓ Generated and saved training data to {data_file}")
    else:
        print(f"\nLoading training data from {data_file}...")
        df = pd.read_csv(data_file)
    
    print(f"✓ Loaded {len(df)} transactions")
    print(f"  - Legitimate: {(df['is_fraud']==0).sum()}")
    print(f"  - Fraudulent: {(df['is_fraud']==1).sum()}")
    
    # Prepare features and target
    feature_cols = [
        'amount', 'hour', 'day_of_week', 'is_weekend',
        'merchant_risk', 'velocity_5min', 'amount_ratio',
        'unusual_hour', 'high_risk_category'
    ]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Split data:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Testing: {len(X_test)} samples")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5,  # Handle class imbalance
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print("✓ Model trained successfully")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"                 Legit  Fraud")
    print(f"Actual Legit     {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Fraud     {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n✓ ROC AUC Score: {roc_auc:.4f}")
    
    # Average Precision Score
    avg_precision = average_precision_score(y_test, y_pred_proba)
    print(f"✓ Average Precision Score: {avg_precision:.4f}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='roc_auc', n_jobs=-1
    )
    print(f"✓ Cross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("Feature Importance")
    print("=" * 60)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Save model
    model_path = 'ml/fraud_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Test predictions at different thresholds
    print("\n" + "=" * 60)
    print("Fraud Detection at Different Thresholds")
    print("=" * 60)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        tp = ((y_pred_threshold == 1) & (y_test == 1)).sum()
        fp = ((y_pred_threshold == 1) & (y_test == 0)).sum()
        tn = ((y_pred_threshold == 0) & (y_test == 0)).sum()
        fn = ((y_pred_threshold == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nThreshold: {threshold:.1f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  Flagged:   {(y_pred_proba >= threshold).sum()} / {len(y_test)} ({(y_pred_proba >= threshold).mean():.1%})")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel ready for deployment at: {model_path}")
    print("You can now run the API with this trained model.")
    
    return model


if __name__ == "__main__":
    model = train_fraud_detection_model()