import numpy as np


def nesterov_accelerated_gradient(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Nesterov accelerated gradient method for LASSO regression.

    Args:
        X (ndarray): The feature matrix of shape (n_samples, n_features).
        y (ndarray): The target values of shape (n_samples,).
        lambda_ (float): The regularization parameter.
        lr (float, optional): The learning rate. Defaults to 0.01.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-6.

    Returns:
        Tuple[ndarray, List[float]]: A tuple containing the optimized coefficient vector and the history of loss values during optimization.
    """

    def soft_thresholding(x: np.ndarray, lambda_: float) -> np.ndarray:
        """Compute the soft thresholding function.

        Args:
            x (ndarray): The input array.
            lambda_ (float): The thresholding parameter.

        Returns:
            ndarray: The result of applying soft thresholding to the input array.
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)

    n, p = X.shape
    beta = np.zeros(p)
    beta_prev = np.zeros(p)
    t = 1
    t_prev = 1
    loss_history = []

    for _ in range(max_iter):
        y_tilde = beta + ((t_prev - 1) / t) * (beta - beta_prev)
        gradient = -X.T.dot(y - X.dot(y_tilde)) / n
        beta_prev = beta.copy()
        beta = soft_thresholding(y_tilde - lr * gradient, lr * lambda_)
        t_prev = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        loss = np.linalg.norm(y - X.dot(beta)) ** 2 / (
            2 * n
        ) + lambda_ * np.linalg.norm(beta, ord=1)
        loss_history.append(loss)
        if np.linalg.norm(beta - beta_prev, ord=2) < tol:
            break

    return beta, loss_history
