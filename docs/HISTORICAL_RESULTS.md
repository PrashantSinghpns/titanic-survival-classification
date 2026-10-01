# Historical evaluation record

The text below preserves the prior README result for traceability. Its source CSV and saved training artifacts are not included here, so these numbers were not reproduced in this review.

The revised workflow evaluates only the CV-selected model on the test set and now records dataset and environment provenance.

## Results

Using the 891-row seaborn Titanic dataset with an 80/20 stratified split (`random_state=42`), the Random Forest was selected by five-fold cross-validation ROC-AUC. Its final held-out metrics were **accuracy 0.7989**, **survivor F1 0.7353**, and **ROC-AUC 0.8349**. This historical dataset is a learning benchmark, not a real-world survival decision system.
