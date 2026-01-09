# Stop the Line Simulation Guide

This document explains how to intentionally break the CI/CD pipeline to demonstrate the "Stop the Line" principle.

## Option 1: Introduce a Bug in Feature Engineering

Edit `src/feature_engineering.py` and change the `hash_feature` function:

```python
# BEFORE (correct)
def hash_feature(value: str, num_buckets: int = 1000) -> int:
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")
    hash_value = mmh3.hash(value, seed=42, signed=False)
    return hash_value % num_buckets

# AFTER (broken - returns wrong type)
def hash_feature(value: str, num_buckets: int = 1000) -> int:
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1")
    hash_value = mmh3.hash(value, seed=42, signed=False)
    return str(hash_value % num_buckets)  # Bug: returns string instead of int
```

This will cause the unit tests to fail because they verify the return type is `int`.

## Option 2: Introduce a Syntax Error

Add a syntax error to any Python file:

```python
# In src/feature_engineering.py, add this line anywhere:
def broken_function(
    # Missing closing parenthesis and body
```

This will cause the build to fail immediately when Python tries to import the module.

## Option 3: Break Linting

Add code that violates flake8 rules:

```python
# Add undefined variable usage
x = undefined_variable_name
```

This will cause the lint stage to fail with F821 error.

## Steps to Demonstrate

1. **Make the change** in your local code
2. **Commit and push** to GitHub:
   ```bash
   git add -A
   git commit -m "Test: intentional bug for Stop the Line demo"
   git push
   ```
3. **Observe the pipeline** in GitHub Actions - it will fail
4. **Take a screenshot** of the failed build
5. **Revert the change**:
   ```bash
   git revert HEAD
   git push
   ```
6. **Observe the pipeline** succeed again

## Expected Result

The GitHub Actions workflow will:
- Stop at the **Unit Test** stage (if bug option)
- Stop at the **Build** stage (if syntax error option)
- Stop at the **Lint** stage (if linting error option)

This demonstrates that bad code is **blocked from reaching production**.
