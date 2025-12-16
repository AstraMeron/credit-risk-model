import pytest
import pandas as pd
import os

# Test 1: Check if the processed data file exists (General Health Check)
def test_data_file_exists():
    path = os.path.join('data', 'processed', 'processed_data.csv')
    assert os.path.exists(path), f"Processed data file not found at {path}"

# Test 2: Verify the target column 'FraudResult' is in the data
def test_target_column_presence():
    path = os.path.join('data', 'processed', 'processed_data.csv')
    df = pd.read_csv(path)
    assert 'FraudResult' in df.columns, "The target column 'FraudResult' is missing!"

# Test 3: Ensure there are no missing values in the final dataset
def test_no_missing_values():
    path = os.path.join('data', 'processed', 'processed_data.csv')
    df = pd.read_csv(path)
    assert df.isnull().sum().sum() == 0, "Data contains missing values which will break the model"