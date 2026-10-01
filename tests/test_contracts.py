"""Contract tests for Titanic feature selection and labels."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Import the local module directly for focused tests.
from train import REQUIRED_COLUMNS, load_and_clean  # Validate the production feature contract.


class TitanicDataContractTests(unittest.TestCase):
    """Verify that target-proxy columns are not carried into modelling."""

    def test_excludes_alive_proxy_column(self):
        row = {"survived": 1, "pclass": 1, "sex": "female", "age": 30.5, "sibsp": 0, "parch": 0, "fare": 71.2833, "embarked": "C", "alone": True, "alive": "yes"}  # Include the proxy to confirm it is ignored.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "titanic.csv"  # Isolate test data from user files.
            pd.DataFrame([row]).to_csv(path, index=False)  # Write a minimal valid dataset.
            cleaned = load_and_clean(path)  # Run the real cleaner.
        self.assertEqual(list(cleaned.columns), REQUIRED_COLUMNS)  # Retain only the approved input schema.
        self.assertNotIn("alive", cleaned.columns)  # Confirm direct leakage is excluded.
        self.assertEqual(cleaned.iloc[0]["fare"], 71.2833)  # Keep currency precision.


if __name__ == "__main__":
    unittest.main()  # Support direct execution.
