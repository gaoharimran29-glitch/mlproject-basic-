import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
import joblib

df = pd.read_csv(r'Cleaned_Home_Loan.csv')

X = df.drop(columns='Loan_Status')
y = df['Loan_Status']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Gender' , 'Married' , 'Education' , 'Self_Employed' , 'Property_Area']) , 
    (StandardScaler() , ['Dependents' , 'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']) ,
    remainder='passthrough'
)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

pipe = make_pipeline(col_trans , AdaBoostClassifier(algorithm='SAMME' , learning_rate=0.5 , n_estimators=100))
pipe.fit(X_train , y_train)

joblib.dump(pipe , "Home_Loan_approval_Predcitor.pkl")

# Adaboost Classifier
#Testing Score:  0.8695652173913043
#Training Score:  0.9014492753623189