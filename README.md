# Titanic Survival Classification

A binary-classification project that predicts passenger survival from pre-voyage passenger attributes. The model workflow is designed to avoid the common notebook mistakes of target leakage and independently scaling the test set.

## Engineering decisions

- `alive` is excluded because it directly reveals the target.
- Age and fare retain float precision; the project never casts the full dataframe to integers.
- The scaler is fitted on training folds only through a pipeline.
- Logistic regression and random forest are compared with stratified five-fold cross-validation on the training data.
- The final test set is held back until after model selection; accuracy, F1, ROC-AUC, and per-class metrics are saved.

## Results

Using the 891-row seaborn Titanic dataset with an 80/20 stratified split (`random_state=42`), the Random Forest was selected by five-fold cross-validation ROC-AUC. Its final held-out metrics were **accuracy 0.7989**, **survivor F1 0.7353**, and **ROC-AUC 0.8349**. This historical dataset is a learning benchmark, not a real-world survival decision system.

## Run

```powershell
python -m venv .venv  # Create an isolated environment.
.\.venv\Scripts\python -m pip install -r requirements.txt  # Install dependencies.
.\.venv\Scripts\python train.py --data "path\to\titanic.csv"  # Train and evaluate the project.
```
