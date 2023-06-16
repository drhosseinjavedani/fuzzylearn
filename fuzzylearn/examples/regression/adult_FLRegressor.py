import time

import pandas as pd
import ray
from category_encoders import OrdinalEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from fuzzylearn.regression.fast.fast import FLRegressor

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


X = data.drop(["label", "capital-gain"], axis="columns")
y = data.loc[:, data.columns == "capital-gain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
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
model = FLRegressor(
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


print("r2 score :")
print(r2_score(y_test, y_pred))
print("mean absolute error : ")
print(mean_absolute_error(y_test, y_pred))
