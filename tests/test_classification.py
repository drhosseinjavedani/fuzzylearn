import time
import ray
from category_encoders import OrdinalEncoder
import pandas as pd
import optuna
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.metrics import (
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fuzzylearn.classification.fast.fast import FLClassifier
from fuzzylearn.classification.fast.optimum import FLAutoOptunaClassifier
from fuzzylearn.classification.fast.ray import FLRayClassifier
from fuzzylearn.classification.fast.optimum import FLOptunaClassifier
from fuzzylearn.classification.fast.optimum import FLOptunaClassifier



def _replace_numbers_with_blank(string):
    return ''.join(['' if char.isdigit() else char for char in string])

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
data = data.sample(200)


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


# functions for classifications
def run_classifiers(obj, X_train, y_train, X_test, y_test,X_valid=None, y_valid=None):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate f1 score.

    Parameters
    ----------
    obj: Object
        Best estimator for classification
    X_train: pd.DataFrame
        Training dataframe
    y_train : pd.DataFrame
        Training target
    X_test: pd.DataFrame
        Testing dataframe
    y_test : pd.DataFrame
        Testing target
    Return
    ----------
        True

    """
    obj.fit(X=X_train, y=y_train,X_valid=X_valid, y_valid=y_valid)
    y_preds = obj.predict(X=X_test)
    return f1_score(y_test,y_preds, average='weighted')
model = (
) 
models_classifiers = {
    "FLClassifier": {
    'number_of_intervals':5,
    'fuzzy_type':"triangular",
    'fuzzy_cut':0.3,
    'threshold':0.7,
    'metric':"euclidean",
    },

    "FLAutoOptunaClassifier1":{
    'optimizer':"auto_optuna",
    'error_measurement_metric':'f1_score(y_true, y_pred, average="weighted")',
    },
    # "FLAutoOptunaClassifier2":{
    # 'optimizer':"auto_optuna_ray",
    # 'error_measurement_metric':'f1_score(y_true, y_pred, average="weighted")',
    # },
    "FLOptunaClassifier1":{
    'optimizer':"optuna",
    'metrics_list':["cosine", "manhattan"],
    'fuzzy_type_list':["simple", "triangular"],
    'fuzzy_cut_range':[0.05, 0.45],
    'number_of_intervals_range':[5, 14],
    'threshold_range':[0.1, 12.0],
    'error_measurement_metric':'f1_score(y_true, y_pred, average="weighted")',
    'n_trials':10,
    },
    "FLOptunaClassifier2":{
    'optimizer':"optuna_ray",
    'metrics_list':["cosine", "manhattan"],
    'fuzzy_type_list':["simple", "triangular"],
    'fuzzy_cut_range':[0.05, 0.45],
    'number_of_intervals_range':[5, 14],
    'threshold_range':[0.1, 12.0],
    'error_measurement_metric':'f1_score(y_true, y_pred, average="weighted")',
    'n_trials':5,
    },
    "FLRayClassifier":{
    'number_of_intervals':5,
    'fuzzy_type':"triangular",
    'fuzzy_cut': 0.3,
    'threshold':0.7,
    'metric':"euclidean",

    },
    

}

def test_best_estimator():
    """Test feature scally selector add"""
    # functions for classifiers
    def run_all_classifiers(pause_iteration=False):
        """
        Loop trough some of the classifiers that already 
        created 
        Parameters
        ----------
        pause_iteration: boolean
            To pause the running of the function after each iteration.
        Return
        ----------
            None
        """
        for model,params in models_classifiers.items():
            model_name = _replace_numbers_with_blank(f'{model}')
            params = f'{models_classifiers[model]}'
            obj=eval(model_name + "(**"+params+")")
            # run classifiers
            f1 = run_classifiers(obj, X_train, y_train, X_test, y_test)
            assert f1>= 0.0
    run_all_classifiers()

# run all tests in once
test_best_estimator()


