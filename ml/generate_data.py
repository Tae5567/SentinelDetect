import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.15):
    """
    Generate synthetic transaction data for fraud detection training
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Proportion of fraudulent transactions
    
    Returns:
        DataFrame with transaction features and labels
    """
    
    np.random.seed(42)
    random.seed(42)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    data = []
    
    # Generate legitimate transactions
    for i in range(n_legit):
        # Normal business hours (9 AM - 9 PM)
        hour = np.random.choice(range(9, 22), p=[0.05]*3 + [0.15]*6 + [0.1]*4)
        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Normal transaction amounts (log-normal distribution)
        amount = np.random.lognormal(4.5, 1.2)  # Mean ~$100
        amount = min(amount, 2000)  # Cap at $2000 for legitimate
        
        # Low merchant risk
        merchant_risk = np.random.beta(2, 8)  # Skewed towards low risk
        
        # Low velocity
        velocity_5min = np.random.poisson(0.5)  # Average 0.5 transactions per 5 min
        
        # Amount ratio close to 1 (normal spending pattern)
        amount_ratio = np.random.normal(1.0, 0.3)
        amount_ratio = max(0.5, min(amount_ratio, 2.0))
        
        # Rarely unusual hours
        unusual_hour = 1 if hour < 6 or hour > 23 else 0
        
        # Rarely high-risk categories
        high_risk_category = 1 if random.random() < 0.05 else 0
        
        data.append({
            'amount': round(amount, 2),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'merchant_risk': round(merchant_risk, 4),
            'velocity_5min': velocity_5min,
            'amount_ratio': round(amount_ratio, 4),
            'unusual_hour': unusual_hour,
            'high_risk_category': high_risk_category,
            'is_fraud': 0
        })
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        # Unusual hours more common
        hour = np.random.choice(range(24), p=[0.08]*6 + [0.02]*13 + [0.08]*5)
        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Higher amounts
        amount = np.random.lognormal(6.0, 1.5)  # Mean ~$400
        amount = max(amount, 500)  # Minimum $500 for fraud
        
        # Higher merchant risk
        merchant_risk = np.random.beta(8, 2)  # Skewed towards high risk
        
        # Higher velocity
        velocity_5min = np.random.poisson(3.0)  # Average 3 transactions per 5 min
        
        # Unusual amount ratio
        amount_ratio = np.random.normal(3.5, 1.5)
        amount_ratio = max(2.0, min(amount_ratio, 10.0))
        
        # More unusual hours
        unusual_hour = 1 if hour < 6 or hour > 23 else 0
        
        # More high-risk categories
        high_risk_category = 1 if random.random() < 0.4 else 0
        
        data.append({
            'amount': round(amount, 2),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'merchant_risk': round(merchant_risk, 4),
            'velocity_5min': velocity_5min,
            'amount_ratio': round(amount_ratio, 4),
            'unusual_hour': unusual_hour,
            'high_risk_category': high_risk_category,
            'is_fraud': 1
        })
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add transaction IDs
    df['transaction_id'] = [f'txn_{i:06d}' for i in range(len(df))]
    
    return df


if __name__ == "__main__":
    print("Generating synthetic fraud detection dataset...")
    
    # Generate training data
    df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.15)
    
    # Save to CSV
    output_file = 'ml/fraud_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {len(df)} transactions")
    print(f"✓ Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"✓ Saved to {output_file}")
    print("\nDataset statistics:")
    print(df.describe())
    print("\nClass distribution:")
    print(df['is_fraud'].value_counts())