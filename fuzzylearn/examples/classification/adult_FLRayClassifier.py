import time

import pandas as pd
import ray
from category_encoders import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fuzzylearn.classification.fast.ray import FLRayClassifier

urldata = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# column names
col_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "label",
]
# read data
data = pd.read_csv(urldata, header=None, names=col_names, sep=",")
# use sample of 1000 rows of data only
data = data.sample(20000)
data.head()

data.loc[data["label"] == "<=50K", "label"] = 0
data.loc[data["label"] == " <=50K", "label"] = 0

data.loc[data["label"] == ">50K", "label"] = 1
data.loc[data["label"] == " >50K", "label"] = 1

data["label"] = data["label"].astype(int)

# Train test split

X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y["label"], random_state=42
)


int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
float_cols = X_train.select_dtypes(include=["float"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


print("int_cols")
print(int_cols)
print("float_cols")
print(float_cols)
print("cat_cols")
print(cat_cols)


pipeline = Pipeline(
    [
        # int missing values imputers
        (
            "intimputer",
            MeanMedianImputer(imputation_method="median", variables=int_cols),
        ),
        # category missing values imputers
        ("catimputer", CategoricalImputer(variables=cat_cols)),
        #
        ("catencoder", OrdinalEncoder()),
    ]
)

X_train = pipeline.fit_transform(X_train, y_train)
X_test = pipeline.transform(X_test)


start_time = time.time()
model = FLRayClassifier(
    number_of_intervals=5,
    fuzzy_type="triangular",
    fuzzy_cut=0.3,
    threshold=0.7,
    metric="euclidean",
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
