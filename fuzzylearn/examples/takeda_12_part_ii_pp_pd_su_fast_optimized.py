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

data = pd.read_csv('fuzzylearn/data/raw_train_all_pp_pd_su_df_after_adjustment_train_data_0.5_label_ii_12_trained_model.csv', sep=",")
print(data.columns)

cols_to_drop =[
    'subject_id',
    'Unnamed: 0',
    'label_ii',

    'mo0nth0',
    'mo0nth06',
    'mo0nth12',
    'mo0nth18',
    'mo0nth24',
    'mo0nth30',
    'mo0nth36',

    'Mo1nth0', 
    'Mo1nth06', 
    'Mo1nth12', 
    'Mo1nth18', 
    'Mo1nth24', 
    'Mo1nth30', 
    'Mo1nth36', 


    'SMo2nth0',
    'SMo2nth06',
    'SMo2nth12',
    'SMo2nth18',
    'SMo2nth24',
    'SMo2nth30',
    'SMo2nth36',
       
]

# drop some columns


data["label"] = data["label_ii"].astype(int)

# # Train test split

data.drop(cols_to_drop, errors='ignore', inplace=True, axis='columns')
print(data.columns)


X = data.loc[:, data.columns != "label"]
y = data.loc[:, data.columns == "label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y["label"], random_state=42
)


print(X_train.head())
print(X_test.head())

print(X_train.columns)
print(X_test.columns)

print(y_train.columns)
print(y_test.columns)

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
            #('intimputer', MeanMedianImputer(
            #    imputation_method='median', variables=int_cols)),
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),

 ])

X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)


start_time = time.time()
model = FLfastClassifier(number_of_intervals=15,threshold=0.7,metric = 'euclidean').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
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


shared_features=set([
                'mds_updrs_part_ii_summary_score',
                'mds_updrs_part_i_summary_score',
                'PGS002249_Lourida_I_PRS_249273_Alzheimer_disease_JAMA_2019_NR_as_GRCh38_246084_',
                'pt_prs',
                'PGS000193_Coleman_JRI_PRS_1138_Major_depression_Mol_Psychiatry_2020_GRCh37_to_GRCh38_1138_',
                'code_upd2302_facial_expression',
                'code_upd2107_pat_quest_sleep_problems',
                'code_upd2202_saliva_and_drooling',
                'PGS001678_Tanigawa_Y_PRS_449_WA_ICVF_in_tract_parahippocampal_part_of_cingulum_L_medRxiv_2021_GRCh37_to_GRCh38_392_',
                'code_upd2212_walking_and_balance',
                'code_upd2213_freezing',
                'PGS001641_Tanigawa_Y_PRS_1005_Volume_of_white_matter_normalised_for_head_size_medRxiv_2021_GRCh37_to_GRCh38_900_',
                'mds_updrs_part_iii_summary_score',
                'on_dopamine_agonist',
                'moca_visuospatial_executive_subscore',
                'code_upd2313_posture',
                'code_upd2301_speech_problems',
                'mds_updrs_part_i_pat_quest_sub_score',
                'code_upd2209_turning_in_bed',
                'code_upd2210_tremor',
                'code_upd2317a_rest_tremor_amplitude_right_upper_extremity',
                'code_upd2318_consistency_of_rest_tremor',
                'moca_abstraction_subscore',

            ])

        
X_train = X_train[shared_features]
X_test = X_test[shared_features]

int_cols = X_train.select_dtypes(include=["int"]).columns.tolist()
float_cols = X_train.select_dtypes(include=["float"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()


print('int_cols')
print(int_cols)
print('float_cols')
print(float_cols)
print('cat_cols')
print(cat_cols)


pipeline_selected_features =Pipeline([
            # drop constant features
            #('dropconstantfeatures',DropConstantFeatures(tol=0.8, missing_values='ignore')),
            # int missing values imputers
            #('intimputer', MeanMedianImputer(
            #    imputation_method='median', variables=int_cols)),
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),

 ])

X_train = pipeline_selected_features.fit_transform(X_train,y_train)
X_test = pipeline_selected_features.transform(X_test)


start_time = time.time()
model = FLfastClassifier(number_of_intervals=15,threshold=0.7,metric = 'euclidean').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
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