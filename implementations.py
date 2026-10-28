import numpy as np


def mse(y, tx, w):
    """Calculate the mean squared error.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        w (np.ndarray): Weights.

    Returns:
        float: Mean squared error.
    """
    e = y - tx.dot(w)
    return 0.5 * np.mean(e**2)


def mse_gradient(y, tx, w):
    """Compute the gradient of the mean squared error.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        w (np.ndarray): Weights.

    Returns:
        np.ndarray: Gradient of the mean squared error.
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Perform one step of gradient descent for mean squared error.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        w (np.ndarray): Weights.
        gamma (float): Learning rate.

    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = mse_gradient(y, tx, w)
        w = w - gamma * grad

    loss = mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Perform stochastic gradient descent for mean squared error.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        w (np.ndarray): Weights.
        gamma (float): Learning rate.

    """
    w = initial_w
    for n_iter in range(max_iters):
        n = np.random.randint(0, len(y))
        grad = mse_gradient(y[n : n + 1], tx[n : n + 1], w)
        w = w - gamma * grad

    loss = mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.

    Returns:
        np.ndarray: Weights that minimize the least squares error.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        lambda_ (float): Regularization parameter.

    Returns:
        np.ndarray: Weights that minimize the ridge regression error.
    """
    a = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mse(y, tx, w)
    return w, loss


def sigmoid(t):
    """Compute the sigmoid function.

    Args:
        t (np.ndarray): Input array.

    Returns:
        np.ndarray: Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-t))

def log_loss(y, tx, initial_w,):
    """Compute the logistic loss and perform gradient descent.

    Args:
        y (np.ndarray): True labels.
        tx (np.ndarray): Input features.
        initial_w (np.ndarray): Initial weights.
        max_iters (int): Maximum number of iterations.
        gamma (float): Learning rate."""

    z= tx.dot(initial_w)
    return np.mean(np.logaddexp(0,z) - y*z)