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



all_data = pd.read_csv("fuzzylearn/data/tot_merged_ukbb.csv")
data_for_train_test_split=all_data[0:30000]
complete_separate_data=all_data[60000:70000]


# data_for_train_test_split = data_for_train_test_split.loc[:,((data_for_train_test_split.columns!='PassengerId') & (data_for_train_test_split.columns!='Name'))]

train = data_for_train_test_split[0:int(0.5*len(data_for_train_test_split))]
test = data_for_train_test_split[int(0.5*len(data_for_train_test_split)):]
unseen = complete_separate_data.copy()

print(unseen.head())

test=test.dropna(subset=['DX_Coronary_artery_disease'])
train=train.dropna(subset=['DX_Coronary_artery_disease'])
unseen=unseen.dropna(subset=['DX_Coronary_artery_disease'])

drop_features = [
    "ascvd_10yr",
    'DX_Angina', 
    'DX_Revascularization', 
    'DX_Atherosclerotic_cardiovascular_disease', 
    'DX_Myocardial_infarction', 
    'AGE_of_CAD',
    'DX_Coronary_artery_disease'
    ]

y_train = train.loc[:,train.columns=='DX_Coronary_artery_disease']
y_unseen = unseen.loc[:,unseen.columns=='DX_Coronary_artery_disease']
y_test = test.loc[:,test.columns=='DX_Coronary_artery_disease']

X_train = train.drop(drop_features,axis='columns')
X_test = test.drop(drop_features,axis='columns')
X_unseen = unseen.drop(drop_features,axis='columns')

new_selected = [
    "QRISK3_2017",
    'Merged_Current_employment_status.Unemployed', 
    'Merged_Current_employment_status.Doing_unpaid_or_voluntary_work',
    'Merged_Current_employment_status.Retired', 
    'Merged_Current_employment_status.In_paid_employment_or_self_employed', 
    'Merged_Qualifications.Other_professional_qualifications_eg_nursing_teaching', 
    'Merged_Medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones.Cholesterol_lowering_medication',
    'Merged_Current_employment_status.Looking_after_home_and_or_family', 
    'AGE_of_Survival', 
    'Merged_Current_employment_status.Unable_to_work_because_of_sickness_or_disability', 
    'Corrected_illnesses_of_siblings.Diabetes'
      ]

# new_selected_ = [
#     'All_Mean_PEF',
#     'getting_up_in_morning_f1170', 
#     'home_location_at_assessment_north_coordinate_rounded_f20075', 
#     'neutrophill_percentage_f30200', 
#     'AGE_of_10yr_Start', 
#     'lipoprotein_a_f30790', 
#     'AGE_of_Survival', 
#     'home_location_at_assessment_east_coordinate_rounded_f20074', 
#     'AGE_of_Death', 
#     'DX_Atrial_fibrillation', 
#     'creatinine_f30700', 
#     'All_Mean_SBP', 
#     'treatmentmedication_code_f20003.Antiplatelet', 
#     'albumin_f30600', 
#     'monocyte_count_f30130', 
#     'AGE_of_Censored', 
#     'DX_Heart_failure', 
#     'gamma_glutamyltransferase_f30730', 
#     'glucose_f30740', 
#     'apolipoprotein_b_f30640', 
#     'igf1_f30770', 
#     'cholesterol_f30690', 
#     'month_of_birth_f52', 
#     'QRISK3_2017', 
#     'shbg_f30830'
#     ]
X_train = X_train[new_selected]
X_test = X_test[new_selected]
X_unseen = X_unseen[new_selected]


int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()

print('int_cols')
print(int_cols)
print('float_cols')
print(float_cols)
print('cat_cols')
print(cat_cols)

# shap_feature_selector_factory = (
#     ShapFeatureSelector.shap_feature_selector_factory.set_model_params(
#         X=X_train,
#         y=y_train,
#         verbose=10,
#         random_state=0,
#         estimator=xgboost.XGBClassifier(),
#         estimator_params={
#             "max_depth": [4, 100],
#             "n_estimators": [100, 1000],
#             "gamma": [0.0, 1.0],

#         },
#         fit_params = {
#             "callbacks": None,
#         },
#         method="optuna",
#         # if n_features=None only the threshold will be considered as a cut-off of features grades.
#         # if threshold=None only n_features will be considered to select the top n features.
#         # if both of them are set to some values, the threshold has the priority for selecting features.
#         n_features=25,
#         threshold = None,
#         list_of_obligatory_features_that_must_be_in_model=[],
#         list_of_features_to_drop_before_any_selection=[],
#     )
#     .set_shap_params(
#         model_output="raw",
#         feature_perturbation="interventional",
#         algorithm="v2",
#         shap_n_jobs=-1,
#         memory_tolerance=-1,
#         feature_names=None,
#         approximate=False,
#         shortcut=False,
#     )
#     .set_optuna_params(
#             measure_of_accuracy="r2_score(y_true, y_pred)",
#             # optuna params
#             with_stratified=False,
#             test_size=.3,
#             n_jobs=-1,
#             # optuna params
#             # optuna study init params
#             study=optuna.create_study(
#                 storage=None,
#                 sampler=TPESampler(),
#                 pruner=HyperbandPruner(),
#                 study_name="example of optuna optimizer",
#                 direction="maximize",
#                 load_if_exists=False,
#                 directions=None,
#             ),
#             # optuna optimization params
#             study_optimize_objective=None,
#             study_optimize_objective_n_trials=20,
#             study_optimize_objective_timeout=600,
#             study_optimize_n_jobs=-1,
#             study_optimize_catch=(),
#             study_optimize_callbacks=None,
#             study_optimize_gc_after_trial=False,
#             study_optimize_show_progress_bar=False,
#             )
# )

pipeline =Pipeline([
            # drop constant features
            #('dropconstantfeatures',DropConstantFeatures(tol=0.8, missing_values='ignore')),
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),
            # category missing values imputers
            # ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            # ("sfsf", shap_feature_selector_factory),
 ])



X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)


# ShapFeatureSelector.shap_feature_selector_factory.plot_features_all(
#     type_of_plot="summary_plot",
#     path_to_save_plot="../plots/shap_optuna_search_regression_summary_plot"
# )


start_time = time.time()
model = FLfastClassifier(number_of_intervals='freedman',threshold=0.1,metric = 'manhattan').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
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


X_unseen = pipeline.transform(X_unseen)
y_pred = model.predict(X=X_unseen)


print("classification_report for unseen:")
print(classification_report(y_unseen, y_pred))
print("confusion_matrix for unseen: ")
print(confusion_matrix(y_unseen, y_pred))
print("roc_auc_score for unseen: ")
print(roc_auc_score(y_unseen, y_pred))
print("f1_score for unseen: ")
print(f1_score(y_unseen, y_pred))

