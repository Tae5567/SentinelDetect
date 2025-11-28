# Real-Time Payment Fraud Detection API

A production-grade fraud detection system that processes payment transactions in real-time using machine learning and event-driven architecture. Built to demonstrate enterprise-level fintech engineering capabilities.

## What It Does

Analyzes payment transactions as they occur, calculating fraud risk scores in under 100ms using a hybrid approach:
- **Machine Learning**: XGBoost model trained on transaction patterns 
- **Rule-Based Detection**: Business rules for high-risk scenarios (velocity checks, amount thresholds, merchant profiling)
- **Real-Time Streaming**: Kafka event pipeline for scalable async processing
- **Pattern Analysis**: Redis-cached velocity tracking and historical behavior analysis

Transactions are scored on a 0-1 scale and classified into risk levels (low/medium/high/critical), with automatic fraud alerts triggered for suspicious activity.

## Tech Stack

**Backend & API**
- FastAPI 
- Pydantic
- SQLAlchemy

**Data Layer**
- PostgreSQL - Transaction persistence
- Redis - Caching & velocity tracking
- Kafka - Event streaming

**Machine Learning**
- XGBoost - Fraud classification model
- scikit-learn - Model training pipeline
- pandas/numpy - Feature engineering

**Infrastructure**
- Docker & Docker Compose - Containerization
- pytest - Testing framework

## Architecture

```
Transaction → FastAPI → ML Model → Fraud Score
                 ↓           ↓
              Kafka    →   Consumer → PostgreSQL
                 ↓
              Redis (Velocity/Cache)
```

Event-driven microservices architecture with async processing, enabling horizontal scalability and sub-100ms latency at 1000+ transactions per second.