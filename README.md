# CS-433 Project 1: Predicting coronary heart disease (MICHD)

Binary classification on the BRFSS survey data (328,135 training people, 321 features),
using NumPy only.

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods, plus helpers `mse`, `mse_gradient`, `sigmoid`, `log_loss`, `log_loss_gradient` |
| `run.py` | Full pipeline: preprocessing, threshold selection, final training, writes `submission.csv` |
| `helpers.py` | Course-provided CSV loading and submission writing |

## Reproduce the submission

1. Unzip the course data so that `data/dataset/x_train.csv`, `data/dataset/y_train.csv`
   and `data/dataset/x_test.csv` exist.
2. Run `python run.py` (about 4 minutes, NumPy only). It writes `submission.csv`.

The random split uses `np.random.seed(1)`, so every run produces the same file.

## Pipeline

1. Drop features with no observed value (`MAX_NAN_RATIO = 1.0`).
2. Replace missing values by the training-set median of each feature.
3. Drop constant features, standardize with the training mean and standard deviation,
   and add a bias column.
4. Map labels from {-1, 1} to {0, 1}.
5. Train regularized logistic regression with gradient descent
   (lambda = 1e-4, gamma = 0.5, 300 iterations, initial weights 0).
6. Validation: random 80/20 split. Train on 80% and pick the probability threshold
   that maximizes F1 on the held-out 20%.
7. Retrain on all training data, predict the test set with that threshold,
   and map predictions back to {-1, 1}.

## Validation results (20% hold-out)

| Model | Threshold | F1 | Accuracy |
|---|---|---|---|
| Always predict healthy | - | 0.000 | 0.912 |
| Least squares | 0.22 | 0.404 | 0.867 |
| Ridge regression, lambda = 1e-4 | 0.22 | 0.404 | 0.869 |
| Logistic regression | 0.20 | 0.406 | 0.864 |
| Reg. logistic, threshold fixed at 0.5 | 0.50 | 0.165 | 0.915 |
| Reg. logistic, drop features >50% missing | 0.21 | 0.405 | 0.870 |
| **Reg. logistic, final (`run.py`)** | **0.21** | **0.411** | **0.871** |