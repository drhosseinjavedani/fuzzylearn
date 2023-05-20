from fuzzylearn.classification.fast.fast import FLfastClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_auc_score
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropConstantFeatures
import pandas as pd
import time
import xgboost
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
from sklearn.model_selection import train_test_split
import numpy as np

test_size = 0.3
n_features = 20
study_optimize_objective_n_trials = 50

data = pd.read_csv('/Users/hjavedani/Documents/fuzzylearn/fuzzylearn/data/database.csv',sep=',')
data=data.dropna(subset=['Output'])

X= data.iloc[:,data.columns!='Output']
y= data.iloc[:,data.columns=='Output']

col_with_nulls_numeric =['TR_Anterior', 'TR1', 'TR2',
       'TR3', 'TR4', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'TR10',
       'Extremo_Range', 'TR_Rompido', 'Major_Relevante',
       'Max_Min_Barra_Padrao', 'Leg1', 'Leg1_CT', 'Leg1_Anterior',
       'Leg1_CT_Anterior', 'Extensao_Reversao_Ant',
       'Max_Min_Percentual_Barra_Padrao']
col_with_nulls_categorical =['Data', 'Ativo', 'Padrao_AA', 'Padrao_Anterior', 'Padrao_Atual',
       'OOrigemRef', 'OrigemRef', 'AtualRef', 'OOrigem', 'Origem', 'Atual',
       'CorOrigem', 'Direcao', 'Dir_Leg1_Atual', 'Direcao_TF_Maior',
       'Dir_Padrao_Leg1']


X[col_with_nulls_numeric] = X[col_with_nulls_numeric].replace({'0':np.nan, 0:np.nan})
X[col_with_nulls_categorical] = X[col_with_nulls_categorical].replace({'0':np.nan, 0:np.nan})


X[col_with_nulls_categorical] = X[col_with_nulls_categorical].astype(str)


X['Data'] = pd.to_datetime(X['Data'])
X['dayofweek']=X['Data'].dt.dayofweek
X['dayofmont']=X['Data'].dt.day


drop_list = ['Data','TR_Anterior','TR9','TR10']
X = X.drop(drop_list, axis='columns')

y['Output'] = y['Output'].str.replace('Pullback','0')
y['Output'] = y['Output'].str.replace('Reversal','1')


y['Output'] = y['Output'].astype(int)

print(y.head())
print(type(y))
print(y.value_counts(dropna=False))

print(X.head())
print(type(X))

# train-test split 80%-20%  
X_train, X_test, y_train, y_test =train_test_split(X,y,  stratify=y["Output"], test_size=test_size, random_state=42)

shap_feature_selector_factory = (
    ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
        X=X_train,
        y=y_train,
        verbose=10,
        random_state=0,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 20],
            "n_estimators": [100, 1000],
            "gamma": [0.0, 1.0],

        },
        fit_params = {
            "callbacks": None,
        },
        method="optuna",
        n_features=n_features,
        threshold = None,
        list_of_obligatory_features_that_must_be_in_model=[],
        list_of_features_to_drop_before_any_selection=[],
    )
    .set_shap_params(
        model_output="raw",
        feature_perturbation="interventional",
        algorithm="v2",
        shap_n_jobs=-1,
        memory_tolerance=-1,
        feature_names=None,
        approximate=False,
        shortcut=False,
    )
    .set_optuna_params(
            measure_of_accuracy="f1_score(y_true, y_pred)",
            # optuna params
            with_stratified=False,
            test_size=.3,
            n_jobs=-1,
            # optuna params
            # optuna study init params
            study=optuna.create_study(
                storage=None,
                sampler=TPESampler(),
                pruner=HyperbandPruner(),
                study_name="example of optuna optimizer",
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=study_optimize_objective_n_trials,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
            )
)


int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
float_cols = X_train.select_dtypes(include=["float"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


print('int_cols')
print(int_cols)
print('float_cols')
print(float_cols)
print('cat_cols')
print(cat_cols)


pipeline =Pipeline([
            # drop constant features
            #('dropconstantfeatures',DropConstantFeatures(tol=0.8, missing_values='ignore')),
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            #("sfsf", shap_feature_selector_factory),
 ])

X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)


print(X_train.isnull().mean()*100)
print(X_test.isnull().mean()*100)

start_time = time.time()
model = FLfastClassifier(number_of_intervals=7,threshold=0.9,metric = 'manhattan').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
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

