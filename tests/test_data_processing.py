import os
import pytest
import pandas as pd

# Define the path to your processed data
DATA_PATH = os.path.join('data', 'processed', 'processed_data.csv')

def test_data_file_exists():
    """Check if the processed data file exists (Local test only)."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Data file not found; skipping in CI environment")
    assert os.path.exists(DATA_PATH)

def test_no_missing_values():
    """Ensure there are no null values in the final dataset."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Data file not found; skipping in CI environment")
        
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0

def test_required_columns_present():
    """Verify that the necessary features for the model are present."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Data file not found; skipping in CI environment")
        
    df = pd.read_csv(DATA_PATH)
    # Adjust these names based on your actual final columns
    required_cols = ['Amount', 'Value', 'PricingStrategy']
    for col in required_cols:
        assert col in df.columns