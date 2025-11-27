from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List
import asyncio

from .config import settings
from .models import (
    TransactionRequest, TransactionResponse, FraudScore,
    HealthCheck, Statistics, RiskLevel
)
from .database import get_db, Transaction, init_db
from .redis_client import redis_client
from .kafka_producer import kafka_producer
from .ml_model import fraud_model

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time payment fraud detection API with ML and streaming"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("=" * 60)
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 60)
    
    # Initialize database
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
    
    # Check Redis connection
    if redis_client.ping():
        print("✓ Redis connected")
    else:
        print("✗ Redis connection failed")
    
    # Check Kafka connection
    if kafka_producer.is_connected():
        print("✓ Kafka producer connected")
    else:
        print("✗ Kafka producer connection failed")
    
    # Load ML model
    if fraud_model.model is not None:
        print("✓ ML model loaded")
    else:
        print("⚠ ML model not loaded")
    
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    kafka_producer.close()
    print("Services shut down gracefully")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    services = {
        "redis": "connected" if redis_client.ping() else "disconnected",
        "kafka": "connected" if kafka_producer.is_connected() else "disconnected",
        "ml_model": "loaded" if fraud_model.model is not None else "not_loaded"
    }
    
    status = "healthy" if all(v in ["connected", "loaded"] for v in services.values()) else "degraded"
    
    return HealthCheck(
        status=status,
        timestamp=datetime.utcnow(),
        services=services
    )


@app.post(
    f"{settings.API_V1_PREFIX}/transactions",
    response_model=TransactionResponse,
    tags=["Transactions"]
)
async def submit_transaction(
    transaction: TransactionRequest,
    db: Session = Depends(get_db)
):
    """
    Submit a transaction for fraud detection
    
    - **transaction_id**: Unique transaction identifier
    - **amount**: Transaction amount
    - **merchant_id**: Merchant identifier
    - **card_number**: Card number (will be masked)
    - **timestamp**: Transaction timestamp
    - **merchant_category**: Merchant category
    - **location**: Transaction location
    """
    try:
        # Check if transaction already exists
        existing = db.query(Transaction).filter(
            Transaction.transaction_id == transaction.transaction_id
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Transaction {transaction.transaction_id} already exists"
            )
        
        # Update velocity counter in Redis
        redis_client.increment_velocity(transaction.card_number)
        
        # Predict fraud
        fraud_score, reasons = fraud_model.predict(transaction.dict())
        risk_level = fraud_model.determine_risk_level(fraud_score)
        is_fraud = fraud_model.is_fraud(fraud_score)
        
        # Create transaction record
        db_transaction = Transaction(
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            merchant_id=transaction.merchant_id,
            card_number=transaction.card_number,
            timestamp=transaction.timestamp,
            merchant_category=transaction.merchant_category,
            location=transaction.location,
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            risk_level=risk_level
        )
        db.add(db_transaction)
        db.commit()
        
        # Send to Kafka for streaming
        transaction_data = transaction.dict()
        transaction_data['fraud_score'] = fraud_score
        transaction_data['risk_level'] = risk_level
        transaction_data['is_fraud'] = is_fraud
        transaction_data['timestamp'] = transaction.timestamp.isoformat()
        
        kafka_producer.send_transaction(transaction_data)
        
        # Send fraud alert if detected
        if is_fraud:
            alert = {
                'transaction_id': transaction.transaction_id,
                'alert_type': 'fraud_detected',
                'severity': risk_level,
                'message': f"Fraud detected: {', '.join(reasons)}",
                'timestamp': datetime.utcnow().isoformat()
            }
            kafka_producer.send_fraud_alert(alert)
        
        # Cache the result
        redis_client.set(
            f"fraud_score:{transaction.transaction_id}",
            {
                'fraud_score': fraud_score,
                'risk_level': risk_level,
                'is_fraud': is_fraud,
                'reasons': reasons
            },
            ttl=settings.REDIS_CACHE_TTL
        )
        
        return TransactionResponse(
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            merchant_id=transaction.merchant_id,
            fraud_score=fraud_score,
            risk_level=RiskLevel(risk_level),
            is_fraud=is_fraud,
            processed_at=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get(
    f"{settings.API_V1_PREFIX}/transactions/{{transaction_id}}/score",
    response_model=FraudScore,
    tags=["Transactions"]
)
async def get_fraud_score(transaction_id: str, db: Session = Depends(get_db)):
    """Get fraud score for a specific transaction"""
    
    # Check cache first
    cached = redis_client.get(f"fraud_score:{transaction_id}")
    if cached:
        return FraudScore(
            transaction_id=transaction_id,
            fraud_score=cached['fraud_score'],
            risk_level=RiskLevel(cached['risk_level']),
            is_fraud=cached['is_fraud'],
            reasons=cached.get('reasons', []),
            timestamp=datetime.utcnow()
        )
    
    # Query database
    transaction = db.query(Transaction).filter(
        Transaction.transaction_id == transaction_id
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return FraudScore(
        transaction_id=transaction.transaction_id,
        fraud_score=float(transaction.fraud_score) if transaction.fraud_score else 0.0,
        risk_level=RiskLevel(transaction.risk_level) if transaction.risk_level else RiskLevel.LOW,
        is_fraud=transaction.is_fraud if transaction.is_fraud is not None else False,
        reasons=[],
        timestamp=transaction.updated_at or transaction.created_at
    )


@app.get(
    f"{settings.API_V1_PREFIX}/transactions",
    response_model=List[TransactionResponse],
    tags=["Transactions"]
)
async def list_transactions(
    limit: int = 100,
    fraud_only: bool = False,
    db: Session = Depends(get_db)
):
    """List recent transactions"""
    query = db.query(Transaction).order_by(Transaction.created_at.desc())
    
    if fraud_only:
        query = query.filter(Transaction.is_fraud == True)
    
    transactions = query.limit(limit).all()
    
    return [
        TransactionResponse(
            transaction_id=t.transaction_id,
            amount=float(t.amount),
            merchant_id=t.merchant_id,
            fraud_score=float(t.fraud_score) if t.fraud_score else 0.0,
            risk_level=RiskLevel(t.risk_level) if t.risk_level else RiskLevel.LOW,
            is_fraud=t.is_fraud if t.is_fraud is not None else False,
            processed_at=t.created_at
        )
        for t in transactions
    ]


@app.get(
    f"{settings.API_V1_PREFIX}/statistics",
    response_model=Statistics,
    tags=["Statistics"]
)
async def get_statistics(db: Session = Depends(get_db)):
    """Get fraud detection statistics"""
    total = db.query(Transaction).count()
    fraud_count = db.query(Transaction).filter(Transaction.is_fraud == True).count()
    high_risk = db.query(Transaction).filter(Transaction.risk_level == "high").count()
    
    avg_score_result = db.query(Transaction.fraud_score).filter(
        Transaction.fraud_score.isnot(None)
    ).all()
    
    avg_score = sum(float(s[0]) for s in avg_score_result) / len(avg_score_result) if avg_score_result else 0.0
    
    return Statistics(
        total_transactions=total,
        fraud_detected=fraud_count,
        fraud_rate=fraud_count / total if total > 0 else 0.0,
        average_fraud_score=avg_score,
        high_risk_count=high_risk
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)