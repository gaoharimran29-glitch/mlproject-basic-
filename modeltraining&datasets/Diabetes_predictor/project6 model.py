import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score , f1_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r'cleaned_diabetes.csv')

X = df.drop(columns='Outcome')
y = df['Outcome']

col_trans = make_column_transformer(
    (MinMaxScaler() , ['Pregnancies', 'Glucose_BMI' , 'Glucose_Age' , "Age_BMI", 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']) ,
    remainder='passthrough'
)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

sm = SMOTE(random_state=42)
X_train , y_train = sm.fit_resample(X_train , y_train)

models = {
    "Logistic regressions: " :LogisticRegression(max_iter=1000) ,
    "RandomForest Classifier: " : RandomForestClassifier(random_state=42) ,
    "Adaboost Classifier: " : AdaBoostClassifier(random_state=42) ,
    "Gradient Boost: " : GradientBoostingClassifier(random_state=42) ,
    "Decision tree: " : DecisionTreeClassifier(random_state=42) ,
    "Support vector: " : SVC(random_state=42) ,
    "K Neighbors: " : KNeighborsClassifier() ,
    "XGBoost:" : XGBClassifier() ,
    "Catboost: " : CatBoostClassifier(verbose=0)
}

for name , model in models.items():
    pipe = make_pipeline(col_trans , model)
    pipe.fit(X_train , y_train)
    y_test_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)
    print(name)
    print()
    print("Accuracy Scores: ")
    print('Testing: ' , accuracy_score(y_test , y_test_pred))
    print('Training: ' , accuracy_score(y_train , y_train_pred))
    print("F1 Score:")
    print('Testing: ' , f1_score(y_test , y_test_pred))
    print('Training: ' , f1_score(y_test , y_test_pred))
    print('-'*40)

#adaboost working good
""" Adaboost Classifier: 

Accuracy Scores: 
Testing:  0.7467532467532467
Training:  0.80125
F1 Score:
Testing:  0.6666666666666666
Training:  0.6666666666666666 """