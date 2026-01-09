"""
Smoke Test for Deployment Verification.

This is an END-TO-END test that verifies the deployed service is:
1. Running and accessible
2. Responding to health checks
3. Capable of making predictions

This test is "end-to-end" because it:
- Tests the actual deployed container (not mocks)
- Makes real HTTP requests over the network
- Verifies the complete request/response cycle
- Validates the service from a user's perspective

Usage:
    python smoke_test.py [--url URL] [--retries N] [--delay S]
"""

import argparse
import json
import sys
import time
import requests


def wait_for_service(url: str, max_retries: int = 30, delay: float = 1.0) -> bool:
    """
    Wait for the service to become available.

    Args:
        url: Base URL of the service.
        max_retries: Maximum number of retries.
        delay: Delay between retries in seconds.

    Returns:
        True if service is available, False otherwise.
    """
    print(f"Waiting for service at {url}...")

    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Service is up after {i + 1} attempts!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass

        print(f"Attempt {i + 1}/{max_retries} - Service not ready, retrying...")
        time.sleep(delay)

    print("Service did not become available in time.")
    return False


def test_health_endpoint(url: str) -> bool:
    """
    Test the health check endpoint.

    Args:
        url: Base URL of the service.

    Returns:
        True if health check passes, False otherwise.
    """
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{url}/health", timeout=10)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code != 200:
            print("FAIL: Health check did not return 200 OK")
            return False

        data = response.json()
        if data.get("status") != "healthy":
            print("FAIL: Service status is not 'healthy'")
            return False

        print("PASS: Health check successful!")
        return True

    except Exception as e:
        print(f"FAIL: Health check failed with error: {e}")
        return False


def test_prediction_endpoint(url: str) -> bool:
    """
    Test the prediction endpoint with a sample request.

    Args:
        url: Base URL of the service.

    Returns:
        True if prediction succeeds, False otherwise.
    """
    print("\n=== Testing Prediction Endpoint ===")

    payload = {
        "features": {
            "categorical": {
                "user_id": "smoke_test_user",
                "product_id": "smoke_test_product",
                "category": "test_category"
            },
            "numerical": {
                "price": 99.99,
                "quantity": 1
            }
        }
    }

    try:
        response = requests.post(
            f"{url}/predict",
            json=payload,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code != 200:
            print("FAIL: Prediction did not return 200 OK")
            return False

        data = response.json()

        if "prediction" not in data:
            print("FAIL: Response missing 'prediction' field")
            return False

        if not isinstance(data["prediction"], (int, float)):
            print("FAIL: Prediction is not a number")
            return False

        print("PASS: Prediction endpoint successful!")
        return True

    except Exception as e:
        print(f"FAIL: Prediction failed with error: {e}")
        return False


def test_root_endpoint(url: str) -> bool:
    """
    Test the root endpoint.

    Args:
        url: Base URL of the service.

    Returns:
        True if root endpoint works, False otherwise.
    """
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get(url, timeout=10)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code != 200:
            print("FAIL: Root endpoint did not return 200 OK")
            return False

        print("PASS: Root endpoint successful!")
        return True

    except Exception as e:
        print(f"FAIL: Root endpoint failed with error: {e}")
        return False


def run_smoke_tests(url: str, skip_wait: bool = False) -> bool:
    """
    Run all smoke tests.

    Args:
        url: Base URL of the service.
        skip_wait: Skip waiting for service to be ready.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("=" * 50)
    print("SMOKE TEST - Deployment Verification")
    print("=" * 50)
    print(f"Target URL: {url}")

    # Wait for service if needed
    if not skip_wait:
        if not wait_for_service(url):
            return False

    # Run tests
    results = []
    results.append(("Health Check", test_health_endpoint(url)))
    results.append(("Prediction", test_prediction_endpoint(url)))
    results.append(("Root Endpoint", test_root_endpoint(url)))

    # Summary
    print("\n" + "=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("ALL SMOKE TESTS PASSED!")
    else:
        print("SOME SMOKE TESTS FAILED!")

    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smoke test for prediction service")
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="Base URL of the service (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for service to be ready"
    )

    args = parser.parse_args()

    success = run_smoke_tests(args.url, args.skip_wait)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
