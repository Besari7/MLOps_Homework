"""
Feature Engineering Module for High Cardinality Prediction Service.

This module implements feature hashing (the "hashing trick") to handle
high cardinality categorical features efficiently.
"""

import mmh3
from typing import List, Dict, Any, Optional


def hash_feature(value: str, num_buckets: int = 1000) -> int:
    """
    Hash a categorical feature value to a bucket index using MurmurHash3.

    This technique is used to handle high cardinality categorical features
    (e.g., user IDs, product IDs, IP addresses) by mapping them to a fixed
    number of buckets.

    Args:
        value: The categorical feature value to hash.
        num_buckets: The number of buckets to hash into (default: 1000).

    Returns:
        An integer bucket index in range [0, num_buckets).

    Raises:
        ValueError: If num_buckets is less than 1.
    """
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")

    # Use MurmurHash3 for fast, uniform hashing
    hash_value = mmh3.hash(value, seed=42, signed=False)
    return hash_value % num_buckets


def hash_features_batch(values: List[str], num_buckets: int = 1000) -> List[int]:
    """
    Hash multiple categorical feature values to bucket indices.

    Args:
        values: List of categorical feature values to hash.
        num_buckets: The number of buckets to hash into.

    Returns:
        List of bucket indices corresponding to each input value.
    """
    return [hash_feature(v, num_buckets) for v in values]


def create_feature_vector(
    categorical_features: Dict[str, str],
    numerical_features: Optional[Dict[str, float]] = None,
    num_buckets: int = 1000
) -> Dict[str, Any]:
    """
    Create a feature vector from raw input features.

    Combines hashed categorical features with numerical features
    into a single feature dictionary.

    Args:
        categorical_features: Dictionary of categorical feature name-value pairs.
        numerical_features: Optional dictionary of numerical feature name-value pairs.
        num_buckets: Number of buckets for categorical feature hashing.

    Returns:
        Dictionary containing processed features.
    """
    processed = {}

    # Hash categorical features
    for name, value in categorical_features.items():
        processed[f"{name}_hashed"] = hash_feature(str(value), num_buckets)

    # Add numerical features as-is
    if numerical_features:
        for name, value in numerical_features.items():
            processed[name] = float(value)

    return processed


def validate_input(data: Dict[str, Any]) -> bool:
    """
    Validate input data for prediction.

    Args:
        data: Input data dictionary.

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if "features" not in data:
        return False
    return True
