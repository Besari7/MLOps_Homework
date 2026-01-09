"""
Model Serving Module for High Cardinality Prediction Service.

This module provides a Flask-based REST API for serving ML predictions.
"""

from flask import Flask, request, jsonify
from feature_engineering import create_feature_vector, validate_input

app = Flask(__name__)

# Simulated model weights (in production, load from file/registry)
MODEL_VERSION = "1.0.0"


def mock_predict(features: dict) -> float:
    """
    Mock prediction function.

    In a real scenario, this would load a trained model and make predictions.
    For demonstration, we use a simple formula based on hashed features.

    Args:
        features: Processed feature dictionary.

    Returns:
        Prediction score between 0 and 1.
    """
    # Simple mock: combine hashed values with some randomness
    hash_sum = sum(v for v in features.values() if isinstance(v, (int, float)))
    # Normalize to 0-1 range
    score = (hash_sum % 100) / 100.0
    return round(score, 4)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_version": MODEL_VERSION
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expects JSON body with 'features' key containing categorical
    and/or numerical features.

    Example request:
    {
        "features": {
            "categorical": {"user_id": "user_12345", "product_id": "prod_67890"},
            "numerical": {"price": 29.99, "quantity": 2}
        }
    }
    """
    try:
        data = request.get_json()

        if not validate_input(data):
            return jsonify({
                "error": "Invalid input. Expected 'features' key in request body."
            }), 400

        features = data["features"]
        categorical = features.get("categorical", {})
        numerical = features.get("numerical", {})

        # Process features
        processed_features = create_feature_vector(
            categorical_features=categorical,
            numerical_features=numerical,
            num_buckets=1000
        )

        # Make prediction
        prediction = mock_predict(processed_features)

        return jsonify({
            "prediction": prediction,
            "model_version": MODEL_VERSION,
            "features_used": list(processed_features.keys())
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def root():
    """Root endpoint with API info."""
    return jsonify({
        "service": "High Cardinality Prediction Service",
        "version": MODEL_VERSION,
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Make prediction"
        }
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
