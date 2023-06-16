import time

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from fuzzylearn.regression.fast.optimum import FLAutoOptunaRegressor

# Define the correlation matrix
correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])

# Generate correlated data
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=correlation_matrix, size=100)
coefficients = np.array([2, -3])  # True coefficients
bias = 5  # True bias term
noise = np.random.randn(100)  # Random noise
y = np.dot(X, coefficients) + bias + noise  # Target variable (regression output)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)


start_time = time.time()
model = FLAutoOptunaRegressor(
    optimizer="auto_optuna",
    error_measurement_metric="mean_absolute_error(y_true, y_pred)",
)
model.fit(X=X_train, y=y_train, X_valid=None, y_valid=None)
print("--- %s seconds for training ---" % (time.time() - start_time))

start_time = time.time()
y_pred = model.predict(X=X_test)
print("--- %s seconds for prediction ---" % (time.time() - start_time))


print("r2 score :")
print(r2_score(y_test, y_pred))
print("mean absolute error : ")
print(mean_absolute_error(y_test, y_pred))
