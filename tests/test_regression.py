import time
import ray
from category_encoders import OrdinalEncoder
import pandas as pd
import optuna
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from sklearn.metrics import (
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from fuzzylearn.regression.fast.fast import FLRegressor
from fuzzylearn.regression.fast.optimum import FLAutoOptunaRegressor
from fuzzylearn.regression.fast.ray import FLRayRegressor
from fuzzylearn.regression.fast.optimum import FLOptunaRegressor
from fuzzylearn.regression.fast.optimum import FLOptunaRegressor
import urllib.request
import zipfile


def _replace_numbers_with_blank(string):
    return ''.join(['' if char.isdigit() else char for char in string])


urldata = "https://archive.ics.uci.edu/static/public/2/adult.zip"
adult_data = "fuzzylearn/data/adult.zip"
try:
    urllib.request.urlretrieve(urldata, adult_data)
except Exception as e:
    print("error!")
with zipfile.ZipFile("fuzzylearn/data/adult.zip", "r") as zip_ref:
    zip_ref.extractall("fuzzylearn/data/adult")
folder_path = "fuzzylearn/data/adult/"
dataset_filename = "adult.data"
# df = pd.read_csv(folder_path + dataset_filename)


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
data = pd.read_csv(
    folder_path + dataset_filename, header=None, names=col_names, sep=","
)
# use sample of 1000 rows of data only
data = data.sample(1000)

X = data.drop(["label", "capital-gain"], axis="columns")
y = data.loc[:, data.columns == "capital-gain"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)


int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
float_cols = X_train.select_dtypes(include=["float"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


pipeline_steps = []
if len(int_cols) > 0:
    # append int missing values imputers
    pipeline_steps.append(
        (
            "intimputer",
            MeanMedianImputer(imputation_method="median", variables=int_cols),
        )
    )
if len(float_cols) > 0:
    # append float missing values imputers
    pipeline_steps.append(
        (
            "floatimputer",
            MeanMedianImputer(imputation_method="mean", variables=float_cols),
        )
    )
if len(cat_cols) > 0:
    # append cat missing values imputers
    pipeline_steps.append(("catimputer", CategoricalImputer(variables=cat_cols)))
    # encode categorical variables
    pipeline_steps.append(("catencoder", OrdinalEncoder()))


pipeline = Pipeline(pipeline_steps)

X_train = pipeline.fit_transform(X_train, y_train)
X_test = pipeline.transform(X_test)



# functions for regressions
def run_regressors(obj, X_train, y_train, X_test, y_test,X_valid=None, y_valid=None):
    """
    A function to get best estimator fit it again, calculate predictions
    and calculate f1 score.

    Parameters
    ----------
    obj: Object
        Best estimator for regression
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
    return mean_absolute_error(y_test,y_preds)

 
models_regressors = {
    "FLRegressor": {
    'number_of_intervals':5,
    'fuzzy_type':"triangular",
    'fuzzy_cut':0.3,
    'threshold':0.7,
    'metric':"euclidean",
    },

    "FLAutoOptunaRegressor1":{
    'optimizer':"auto_optuna",
    'error_measurement_metric':'mean_absolute_error(y_true, y_pred )',
    },
    # "FLAutoOptunaRegressor2":{
    # 'optimizer':"auto_optuna_ray",
    # 'error_measurement_metric':'mean_absolute_error(y_true, y_pred)',
    # },
    "FLOptunaRegressor1":{
    'optimizer':"optuna",
    'metrics_list':["cosine", "manhattan"],
    'fuzzy_type_list':["simple", "triangular"],
    'fuzzy_cut_range':[0.05, 0.45],
    'number_of_intervals_range':[5, 14],
    'threshold_range':[0.1, 12.0],
    'error_measurement_metric':'mean_absolute_error(y_true, y_pred)',
    'n_trials':10,
    },
    "FLOptunaRegressor2":{
    'optimizer':"optuna_ray",
    'metrics_list':["cosine", "manhattan"],
    'fuzzy_type_list':["simple", "triangular"],
    'fuzzy_cut_range':[0.05, 0.45],
    'number_of_intervals_range':[5, 14],
    'threshold_range':[0.1, 12.0],
    'error_measurement_metric':'mean_absolute_error(y_true, y_pred)',
    'n_trials':5,
    },
    "FLRayRegressor":{
    'number_of_intervals':5,
    'fuzzy_type':"triangular",
    'fuzzy_cut': 0.3,
    'threshold':0.7,
    'metric':"euclidean",

    },
    

}

def test_best_estimator():
    """Test feature scally selector add"""
    # functions for regressors
    def run_all_regressors(pause_iteration=False):
        """
        Loop trough some of the regressors that already 
        created 
        Parameters
        ----------
        pause_iteration: boolean
            To pause the running of the function after each iteration.
        Return
        ----------
            None
        """
        for model,params in models_regressors.items():
            model_name = _replace_numbers_with_blank(f'{model}')
            params = f'{models_regressors[model]}'
            obj=eval(model_name + "(**"+params+")")
            # run classifiers
            f1 = run_regressors(obj, X_train, y_train, X_test, y_test)
            assert f1>= 0.0
    run_all_regressors()

# run all tests in once
test_best_estimator()



