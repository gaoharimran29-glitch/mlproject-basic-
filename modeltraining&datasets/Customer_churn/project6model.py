import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
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
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline

df = pd.read_csv(r'project6/Clean_Customer_Churn.csv')

X = df.drop(columns='Churn')
y = df['Churn']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['gender' , 'InternetService' , 'Contract' , 'PaymentMethod']) ,
    (StandardScaler() , ['tenure' , 'MonthlyCharges' , 'TotalCharges']) ,
    remainder='passthrough'
)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

models = {
    "Logistic regressions: " :LogisticRegression(max_iter=1000) ,
    "RandomForest Classifier: " : RandomForestClassifier(random_state=42) ,
    "Adaboost Classifier: " : AdaBoostClassifier(random_state=42) ,
    "Gradient Boost: " : GradientBoostingClassifier(random_state=42) ,
    "Decision tree: " : DecisionTreeClassifier(random_state=42) ,
    "Support vector: " : SVC(random_state=42) ,
    "K Neighbors: " : KNeighborsClassifier() ,
    "XGBoost:" : XGBClassifier() ,
    "Catboost: " : CatBoostClassifier(verbose=0) ,
    "GaussianNB" : GaussianNB()
}

for name , model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', col_trans),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)])
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

""" Catboost: 

Accuracy Scores:
Testing:  0.7835344215755855
Training:  0.8587149449769258
F1 Score:
Testing:  0.5938748335552596
Training:  0.5938748335552596 """