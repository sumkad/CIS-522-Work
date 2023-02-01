import numpy as np
import random


"""
    A linear regression model that uses gradient descent to fit the model.
"""


class LinearRegression:
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    w: np.ndarray
    b: float

    def __init__(self) -> None:
        # raise NotImplementedError()
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.
        """
        biasX = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        biasW = np.linalg.pinv((biasX.T @ biasX)) @ biasX.T @ y
        self.b = biasW[-1]
        self.w = biasW[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.
        """
        self.w = np.array([[random.uniform(-0.001, 0.001)] for i in range(X.shape[1])])
        self.b = 0
        for epoch in range(epochs):
            guesses = X @ self.w + self.b
            err = y - guesses
            self.w = self.w + lr * (X.T @ err)
            self.b = self.b + lr * np.sum(err)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        return X @ self.w + self.b
