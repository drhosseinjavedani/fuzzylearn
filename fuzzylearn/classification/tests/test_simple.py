from fuzzylearn.classification.simple.simple import FLClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline
from feature_engine.selection import DropConstantFeatures
import pandas as pd



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


 ])


X_train = pipeline.fit_transform(X_train,y_train)
X_test = pipeline.transform(X_test)


model = FLClassifier(number_of_intervals=6,threshold=0.7,metric = 'manhattan').fit(X=X_train,y=y_train,X_valid=None,y_valid=None)
y_pred = model.predict(X=X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))