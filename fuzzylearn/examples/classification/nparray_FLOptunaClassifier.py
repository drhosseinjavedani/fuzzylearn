import time

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fuzzylearn.classification.fast.optimum import FLOptunaClassifier

X = np.array(
    [
        [2.0, 1.0],
        [1.5, 2.0],
        [3.0, 3.0],
        [2.5, 2.5],
        [1.0, 1.5],
        [3.5, 3.5],
        [2.0, 3.0],
        [3.0, 2.0],
        [1.5, 1.0],
        [2.5, 1.5],
        [2.5, 1.0],
        [1.0, 2.5],
        [3.0, 2.0],
        [1.5, 1.0],
        [2.5, 1.5],
        [2.5, 1.0],
        [1.0, 2.5],
    ]
)

y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

start_time = time.time()
model = FLOptunaClassifier(
    optimizer="optuna",
    fuzzy_type_list=["simple", "triangular"],
    fuzzy_cut_range=[0.05, 0.45],
    metrics_list=["cosine", "manhattan"],
    number_of_intervals_range=[5, 14],
    threshold_range=[0.1, 12.0],
    error_measurement_metric='f1_score(y_true, y_pred, average="weighted")',
)
model.fit(X=X_train, y=y_train, X_valid=None, y_valid=None)
print("--- %s seconds for training ---" % (time.time() - start_time))

start_time = time.time()
y_pred = model.predict(X=X_test)
print("--- %s seconds for prediction ---" % (time.time() - start_time))

print("classification_report :")
print(classification_report(y_test, y_pred))
print("confusion_matrix : ")
print(confusion_matrix(y_test, y_pred))
print("roc_auc_score : ")
print(roc_auc_score(y_test, y_pred))
print("f1_score : ")
print(f1_score(y_test, y_pred))
