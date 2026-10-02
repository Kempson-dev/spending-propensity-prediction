# Spending Propensity Prediction

Predicting how likely online game players are to spend money, using a leakage-safe scikit-learn pipeline and three classification models. Built as university coursework for the Machine Learning module of my BSc in Data Science and Artificial Intelligence at Bournemouth University.

## The problem

Each of roughly 10,000 players is classed as a **NonSpender**, an **Occasional** spender or a **Whale** (a high spender). Whales are the most valuable group to identify but make up only 4.5% of players, so the main challenge is class imbalance.

| Class | Share of players |
| --- | --- |
| Occasional | 49.9% |
| NonSpender | 45.6% |
| Whale | 4.5% |

## Approach

1. **Data cleaning:** removed duplicates and impossible values (such as ages over 120), set missing achievement counts to 0, and filled other gaps with the median or most common value.
2. **Behavioural features only:** predicted spending from how people play (play time, sessions per week, player level, engagement and so on), and deliberately excluded purchase columns such as total spend, which would give the answer away. Categorical columns were encoded.
3. **Stratified train-test split**, made before any resampling so no test data leaks into training.
4. **Pipeline:** `StandardScaler` and `SMOTE` run inside an imbalanced-learn pipeline, so scaling and oversampling happen only on the training folds during cross-validation.
5. **Models:** Logistic Regression (linear and interpretable), SVM with an RBF kernel (non-linear) and Random Forest (ensemble).
6. **Tuning:** `GridSearchCV`, optimising macro F1 so all three classes count equally.
7. **Evaluation:** stratified 5-fold cross-validation, then a held-out test set, using accuracy, precision, recall, F1, confusion matrices and per-class reports.

## Results

| Model | Test accuracy | Weighted F1 | CV weighted F1 |
| --- | --- | --- | --- |
| **Random Forest** | **75.3%** | **0.753** | 0.752 ± 0.005 |
| SVM (RBF) | 73.4% | 0.739 | 0.738 ± 0.007 |
| Logistic Regression | 72.5% | 0.733 | 0.730 ± 0.007 |

- **Random Forest performed best overall.**
- **Strong generalisation:** test scores were within 0.3% of cross-validation scores for every model, so none of them overfit.
- **The Whale class shows a trade-off.** Logistic Regression caught 83% of Whales (recall) but with more false alarms, while Random Forest was more precise but caught only 50%. Which is better depends on the business cost of missing a high spender.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook spending_propensity_prediction.ipynb
```

The notebook expects `online_gaming_v2.csv` in the same folder. The dataset was provided as part of the coursework and isn't included in this repo.

## Tech

Python, pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn, Jupyter
