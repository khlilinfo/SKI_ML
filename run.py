"""Train the final model and write the AIcrowd submission file."""

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression, sigmoid

DATA_PATH = "data/dataset"
MAX_NAN_RATIO = 1.0  # drop features missing for more than half of the people
LAMBDA = 1e-4
GAMMA = 0.5
MAX_ITERS = 300
SEED = 1


def add_bias(x):
    """Prepend a column of ones so that w[0] acts as the intercept."""
    return np.c_[np.ones(len(x)), x]


def preprocess(x_train, x_test):
    """Clean both sets using statistics computed on the training set only.

    Steps: drop mostly-missing features, fill NaNs with the training median,
    drop constant features, standardize, add the bias column.
    """
    keep = np.mean(np.isnan(x_train), axis=0) <= MAX_NAN_RATIO
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)

    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    keep = std > 0
    x_train = (x_train[:, keep] - mean[keep]) / std[keep]
    x_test = (x_test[:, keep] - mean[keep]) / std[keep]
    return add_bias(x_train), add_bias(x_test)


def f1_score(y_true, y_pred):
    """F1 score for labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn)


def best_threshold(y_true, probs):
    """Return the probability threshold that maximizes F1, and that F1."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (probs >= t) * 1) for t in thresholds]
    i = int(np.argmax(scores))
    return thresholds[i], scores[i]


def main():
    np.random.seed(SEED)
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    y = (y_train == 1) * 1.0  # {-1, 1} -> {0, 1} for logistic regression
    tx_train, tx_test = preprocess(x_train, x_test)
    initial_w = np.zeros(tx_train.shape[1])

    # 1) Hold out 20% of the training data to choose the decision threshold.
    idx = np.random.permutation(len(y))
    split = int(0.8 * len(y))
    tr, va = idx[:split], idx[split:]
    w, _ = reg_logistic_regression(
        y[tr], tx_train[tr], LAMBDA, initial_w, MAX_ITERS, GAMMA
    )
    probs = sigmoid(tx_train[va].dot(w))
    threshold, f1 = best_threshold(y[va], probs)
    accuracy = np.mean((probs >= threshold) == y[va])
    print(f"validation: threshold={threshold:.2f} F1={f1:.4f} acc={accuracy:.4f}")

    # 2) Retrain on all training data and predict the test set.
    w, loss = reg_logistic_regression(y, tx_train, LAMBDA, initial_w, MAX_ITERS, GAMMA)
    y_pred = np.where(sigmoid(tx_test.dot(w)) >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"train loss={loss:.4f}, wrote submission.csv ({len(y_pred)} rows)")


if __name__ == "__main__":
    main()