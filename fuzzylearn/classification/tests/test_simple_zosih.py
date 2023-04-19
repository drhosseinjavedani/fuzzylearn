from fuzzylearn.classification.simple.simple import FLClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropConstantFeatures
import pandas as pd
from zoish.feature_selectors.shap_selectors import ShapFeatureSelector
import xgboost
import optuna


data_for_train_test_split = pd.read_csv("fuzzylearn/data/train.csv")
data_for_train_test_split = data_for_train_test_split.loc[:,((data_for_train_test_split.columns!='PassengerId') & (data_for_train_test_split.columns!='Name'))]

train = data_for_train_test_split[0:int(0.5*len(data_for_train_test_split))]
test = data_for_train_test_split[int(0.5*len(data_for_train_test_split)):]


test=test.dropna(subset=['Survived'])
train=train.dropna(subset=['Survived'])

test['Survived'].value_counts(dropna=False)

y_train = train.loc[:,train.columns=='Survived']
X_train = train.loc[:,train.columns!='Survived']
y_test = test.loc[:,test.columns=='Survived']
X_test = test.loc[:,test.columns!='Survived']

shap_feature_selector_factory = (
    ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
        X=X_train,
        y=y_train,
        verbose=10,
        random_state=0,
        estimator=xgboost.XGBClassifier(),
        estimator_params={
            "max_depth": [4, 20],
            "gamma": [0, 10],
            "n_estimators": [100, 500],
            "learning_rate": [0.01, 0.1],


        },
        fit_params = {
            "callbacks": None,
        },
        method="optuna",
        # if n_features=None only the threshold will be considered as a cut-off of features grades.
        # if threshold=None only n_features will be considered to select the top n features.
        # if both of them are set to some values, the threshold has the priority for selecting features.
        n_features=5,
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
                sampler=optuna.samplers.TPESampler(),
                pruner=optuna.pruners.HyperbandPruner(),
                study_name="example of optuna optimizer",
                direction="maximize",
                load_if_exists=False,
                directions=None,
            ),
            # optuna optimization params
            study_optimize_objective=None,
            study_optimize_objective_n_trials=100,
            study_optimize_objective_timeout=600,
            study_optimize_n_jobs=-1,
            study_optimize_catch=(),
            study_optimize_callbacks=None,
            study_optimize_gc_after_trial=False,
            study_optimize_show_progress_bar=False,
            )
)


int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()

pipeline =Pipeline([
            # drop constant features
            ('dropconstantfeatures',DropConstantFeatures(tol=0.8, missing_values='ignore')),
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            ("sfsf", shap_feature_selector_factory),
            # add any regression model from sklearn e.g., LinearRegression
            ('classification', xgboost.XGBClassifier())



 ])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))