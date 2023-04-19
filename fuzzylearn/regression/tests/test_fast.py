from fuzzylearn.regression.fast.fast import FLfastRegressor 
from sklearn.metrics import r2_score, mean_absolute_error
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropConstantFeatures
from sklearn.model_selection import train_test_split
import pandas as pd

test_size =0.33
random_state = 42

months_moca_12 = pd.read_csv("/Users/hjavedani/Documents/moca_prediction/moca_prediction/src/datasets/processed_data/ppmi/df_for_train_12.csv/part-00000-42555a7c-f741-435b-b845-821c80799e95-c000.csv",sep=',')
print(months_moca_12.head())

drop_cols = [
    'participant_id',
    'visit_month',
    'diagnosis_at_baseline',
    'case_control_other_at_baseline',
    'case_control_other_latest',
    'diagnosis_latest',

]
months_moca_12.drop(drop_cols, axis=1, inplace=True)
print(months_moca_12.head())

X = months_moca_12.loc[:, months_moca_12.columns != "target"]
y = months_moca_12.loc[:, months_moca_12.columns == "target"]

print(y)
print(type(y))

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=test_size, random_state=random_state)
print('----------------')
print(y_test)
print(type(y_test))


int_cols =  X_train.select_dtypes(include=['int']).columns.tolist()
float_cols =  X_train.select_dtypes(include=['float']).columns.tolist()
cat_cols =  X_train.select_dtypes(include=['object']).columns.tolist()


pipeline =Pipeline([
            # int missing values imputers
            ('intimputer', MeanMedianImputer(
                imputation_method='median', variables=int_cols)),
            # float missing values imputers
            ('floatimputer', MeanMedianImputer(
                imputation_method='mean', variables=float_cols)),
            # category missing values imputers
            ('catimputer', CategoricalImputer(variables=cat_cols)),
            #
            ('catencoder', OrdinalEncoder()),
            # DropConstantFeatures
            ('DropConstantFeatures',DropConstantFeatures(missing_values='include'))
            # scalling #TODO

 ])



X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)

print('X_train.T.head()')
print(X_train.head(30))




model = FLfastRegressor(number_of_intervals=10,threshold=0.5,metric = 'manhattan').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
y_pred = model.predict(X=X_test)


print(y_pred)
print(y_test)

print('r2_score(y_test, y_pred)')
print(r2_score(y_test, y_pred))
print('mean_absolute_error(y_test, y_pred)')
print(mean_absolute_error(y_test, y_pred))