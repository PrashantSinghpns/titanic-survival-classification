import pandas as pd
import pytest
from train import load_and_clean

@pytest.mark.parametrize("target", [0.5, 1.5, -1, 2])
def test_fractional_and_invalid_labels_are_rejected_before_casting(tmp_path, target):
    row = {"survived": target, "pclass": 1, "sex": "female", "age": 30.5,
           "sibsp": 0, "parch": 0, "fare": 71.2833, "embarked": "C", "alone": True}
    path = tmp_path / "invalid.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_and_clean(path)
