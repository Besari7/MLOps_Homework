# High Cardinality Prediction Service

A Flask-based ML prediction service demonstrating MLOps Level 1 & 2 practices with CI/CD pipeline.

## Features
- Feature hashing for high cardinality categorical variables
- Health check endpoint
- Prediction endpoint
- Comprehensive testing suite

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/ --max-line-length=120

# Build Docker image
docker build -t prediction-service .

# Run container
docker run -p 5000:5000 prediction-service
```

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Make prediction

## CI/CD Pipeline

The GitHub Actions workflow includes:
1. Build - Install dependencies
2. Unit Test - Fast, isolated tests
3. Lint - Code quality check
4. Package - Docker build
5. Smoke Test - Deployment verification
