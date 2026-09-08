"""
linear_regression.py
---------------------
A complete, from-scratch implementation of Linear Regression using only NumPy.

Mathematical background
========================
Given a design matrix X of shape (m, n) — where m is the number of training
examples and n is the number of features (including a bias/intercept column
of ones) — and a parameter vector theta of shape (n, 1), linear regression
models the target y as:

    h(X) = X . theta                                (hypothesis function)

Training theta is done by minimizing the Mean Squared Error (MSE) cost:

    J(theta) = (1 / 2m) * sum( (h(X) - y)^2 )

using batch Gradient Descent, with the gradient of J with respect to theta:

    grad = (1 / m) * X^T . (h(X) - y)

and the parameter update rule:

    theta := theta - alpha * grad

where alpha is the learning rate.

This module exposes a single class, LinearRegressionScratch, whose public
methods (hypothesis, compute_cost, compute_gradient, fit, predict) map
directly onto the mathematics above, so that the code and the theory can be
read side by side.
"""

import numpy as np


class LinearRegressionScratch:
    """
    Linear Regression trained via batch Gradient Descent, implemented
    entirely with NumPy (no scikit-learn is used anywhere inside this class).

    Works for both:
      - Simple Linear Regression (a single feature column)
      - Multiple Linear Regression (many feature columns)

    because all operations are expressed as matrix algebra that generalizes
    to any number of features.

    Parameters
    ----------
    learning_rate : float
        Step size (alpha) used in each Gradient Descent update.
    n_iterations : int
        Number of Gradient Descent iterations to run.
    random_state : int or None
        Seed used to initialize theta reproducibly.

    Attributes
    ----------
    theta : np.ndarray of shape (n_features + 1, 1)
        Learned parameters after calling fit(). theta[0] is the intercept
        (bias) term, theta[1:] are the feature coefficients.
    cost_history : list of float
        MSE cost recorded after every Gradient Descent iteration. Used to
        plot the cost-convergence curve and to verify the model is learning.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 random_state: int | None = 42):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be a positive integer.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.theta = None          # learned parameters (set in fit())
        self.cost_history = []     # MSE recorded at every iteration

    # ------------------------------------------------------------------ #
    # Internal helper: add a bias (intercept) column of ones to X
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_bias_column(X: np.ndarray) -> np.ndarray:
        """Prepend a column of 1s to X so that X . theta includes theta_0."""
        m = X.shape[0]
        ones = np.ones((m, 1))
        return np.hstack([ones, X])

    # ------------------------------------------------------------------ #
    # 1. Hypothesis function:  h(X) = X . theta
    # ------------------------------------------------------------------ #
    def hypothesis(self, X_with_bias: np.ndarray) -> np.ndarray:
        """
        Compute predictions h(X) = X . theta.

        Parameters
        ----------
        X_with_bias : np.ndarray, shape (m, n_features + 1)
            Feature matrix that ALREADY includes the bias column of ones.

        Returns
        -------
        np.ndarray, shape (m, 1)
            Predicted values.
        """
        return X_with_bias @ self.theta

    # ------------------------------------------------------------------ #
    # 2. Mean Squared Error cost function
    # ------------------------------------------------------------------ #
    def compute_cost(self, X_with_bias: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost:

            J(theta) = (1 / 2m) * sum( (h(X) - y)^2 )

        The factor of 1/2 is a mathematical convenience: it cancels out
        the "2" that appears when differentiating the squared term,
        producing a cleaner gradient expression.

        Parameters
        ----------
        X_with_bias : np.ndarray, shape (m, n_features + 1)
        y : np.ndarray, shape (m, 1)

        Returns
        -------
        float
            Scalar cost value.
        """
        m = X_with_bias.shape[0]
        predictions = self.hypothesis(X_with_bias)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        return float(cost)

    # ------------------------------------------------------------------ #
    # 3. Gradient of the cost function w.r.t. theta
    # ------------------------------------------------------------------ #
    def compute_gradient(self, X_with_bias: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of J(theta) with respect to theta:

            grad = (1 / m) * X^T . (h(X) - y)

        Parameters
        ----------
        X_with_bias : np.ndarray, shape (m, n_features + 1)
        y : np.ndarray, shape (m, 1)

        Returns
        -------
        np.ndarray, shape (n_features + 1, 1)
            Gradient vector, one entry per parameter in theta.
        """
        m = X_with_bias.shape[0]
        predictions = self.hypothesis(X_with_bias)
        errors = predictions - y
        gradient = (1 / m) * (X_with_bias.T @ errors)
        return gradient

    # ------------------------------------------------------------------ #
    # 4. Gradient Descent training loop
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """
        Train the model using batch Gradient Descent.

        Steps performed on every iteration:
            1. Compute predictions (hypothesis)
            2. Compute cost (MSE)
            3. Compute gradients
            4. Update parameters (theta)
            5. Store the cost for this iteration

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Feature matrix WITHOUT a bias column (it is added internally).
        y : np.ndarray, shape (m,) or (m, 1)
            Target values.

        Returns
        -------
        self : LinearRegressionScratch
            The fitted model (so calls can be chained).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        X_with_bias = self._add_bias_column(X)
        n_features_plus_bias = X_with_bias.shape[1]

        # --- Initialize parameters reproducibly ---
        rng = np.random.default_rng(self.random_state)
        self.theta = rng.normal(loc=0.0, scale=0.01, size=(n_features_plus_bias, 1))

        self.cost_history = []

        for _ in range(self.n_iterations):
            # 1 & 2: predictions + cost
            cost = self.compute_cost(X_with_bias, y)
            self.cost_history.append(cost)

            # 3: gradients
            gradient = self.compute_gradient(X_with_bias, y)

            # 4: parameter update
            self.theta = self.theta - self.learning_rate * gradient

        return self

    # ------------------------------------------------------------------ #
    # Predict on new data (adds bias column internally)
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new samples.

        Parameters
        ----------
        X : np.ndarray, shape (m, n_features)
            Feature matrix WITHOUT a bias column.

        Returns
        -------
        np.ndarray, shape (m,)
            Predicted values (flattened to 1-D for convenience).
        """
        if self.theta is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_with_bias = self._add_bias_column(X)
        predictions = self.hypothesis(X_with_bias)
        return predictions.flatten()

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def intercept_(self) -> float:
        """The learned bias / intercept term (theta_0)."""
        if self.theta is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return float(self.theta[0, 0])

    @property
    def coef_(self) -> np.ndarray:
        """The learned feature coefficients (theta_1 ... theta_n)."""
        if self.theta is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.theta[1:, 0]


# ---------------------------------------------------------------------- #
# Manual evaluation metrics (NumPy only — no sklearn.metrics allowed)
# ---------------------------------------------------------------------- #
def mean_squared_error_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Mean Squared Error manually:

        MSE = (1 / m) * sum( (y_true - y_pred)^2 )

    Note: this is the *evaluation* MSE (divided by m), which is distinct
    from the (1 / 2m) cost used internally during Gradient Descent.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    m = y_true.shape[0]
    return float((1 / m) * np.sum((y_true - y_pred) ** 2))


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the R-squared (coefficient of determination) manually:

        R^2 = 1 - (SS_res / SS_tot)

    where:
        SS_res = sum( (y_true - y_pred)^2 )         (residual sum of squares)
        SS_tot = sum( (y_true - mean(y_true))^2 )   (total sum of squares)

    R^2 measures the proportion of variance in the target that is
    explained by the model. 1.0 = perfect fit, 0.0 = no better than
    predicting the mean, negative = worse than predicting the mean.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return float(1 - (ss_res / ss_tot))
