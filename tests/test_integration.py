"""
Component/Integration Tests for Model Serving.

Unlike unit tests, these tests verify the interaction between components:
- Model serving logic with feature engineering
- API endpoints with data processing
- End-to-end prediction flow

These tests use the Flask test client to simulate HTTP requests.
"""

import sys
import os
import json
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_serving import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Integration tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Verify health endpoint returns 200 OK."""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Verify health endpoint returns valid JSON."""
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'

    def test_health_includes_version(self, client):
        """Verify health response includes model version."""
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'model_version' in data


class TestPredictEndpoint:
    """Integration tests for prediction endpoint."""

    def test_predict_with_categorical_features(self, client):
        """Verify prediction works with categorical features."""
        payload = {
            "features": {
                "categorical": {
                    "user_id": "user_12345",
                    "product_id": "prod_67890"
                }
            }
        }
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'prediction' in data

    def test_predict_with_numerical_features(self, client):
        """Verify prediction works with numerical features."""
        payload = {
            "features": {
                "categorical": {"category": "electronics"},
                "numerical": {"price": 29.99, "quantity": 2}
            }
        }
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'features_used' in data

    def test_predict_returns_model_version(self, client):
        """Verify prediction response includes model version."""
        payload = {"features": {"categorical": {"test": "value"}}}
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert 'model_version' in data

    def test_predict_invalid_input(self, client):
        """Verify 400 error for invalid input."""
        payload = {"invalid": "data"}
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_predict_empty_features(self, client):
        """Verify prediction works with empty features."""
        payload = {"features": {"categorical": {}, "numerical": {}}}
        response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200


class TestRootEndpoint:
    """Integration tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Verify root endpoint returns API information."""
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'service' in data
        assert 'endpoints' in data


class TestEndToEndFlow:
    """End-to-end integration tests simulating real usage."""

    def test_full_prediction_flow(self, client):
        """
        Test complete prediction flow:
        1. Check health
        2. Make prediction
        3. Verify response format
        """
        # Step 1: Health check
        health_response = client.get('/health')
        assert health_response.status_code == 200

        # Step 2: Make prediction
        payload = {
            "features": {
                "categorical": {
                    "user_id": "user_abc123",
                    "item_id": "item_xyz789",
                    "category": "electronics"
                },
                "numerical": {
                    "price": 149.99,
                    "quantity": 1,
                    "user_age": 25
                }
            }
        }
        predict_response = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Step 3: Verify response
        assert predict_response.status_code == 200
        data = json.loads(predict_response.data)
        assert 0 <= data['prediction'] <= 1  # Score should be normalized
        assert len(data['features_used']) == 6  # 3 categorical + 3 numerical

    def test_deterministic_predictions(self, client):
        """Verify same input produces same prediction (determinism)."""
        payload = {
            "features": {
                "categorical": {"user": "test_user"},
                "numerical": {"value": 100}
            }
        }

        response1 = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        response2 = client.post(
            '/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )

        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)

        assert data1['prediction'] == data2['prediction']
