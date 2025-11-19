import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'Cleaned_Home_Loan.csv')

X = df.drop(columns='Loan_Status')
y = df['Loan_Status']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Gender' , 'Married' , 'Education' , 'Self_Employed' , 'Property_Area']) , 
    (StandardScaler() , ['Dependents' , 'ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']) ,
    remainder='passthrough'
)

models = {
    "Logistic regressions: " :LogisticRegression(max_iter=1000) ,
    "RandomForest Classifier: " : RandomForestClassifier(random_state=42) ,
    "Adaboost Classifier: " : AdaBoostClassifier(random_state=42) ,
    "Gradient Boost: " : GradientBoostingClassifier(random_state=42) ,
    "Decision tree: " : DecisionTreeClassifier(random_state=42) ,
    "Support vector: " : SVC(random_state=42) ,
    "K Neighbors: " : KNeighborsClassifier()
}

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

for name , model in models.items():
    pipe = make_pipeline(col_trans , model)
    pipe.fit(X_train , y_train)
    y_test_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)
    print(name)
    print("Testing Score: " , accuracy_score(y_test_pred , y_test))
    print("Training Score: " , accuracy_score(y_train_pred , y_train))
    print()
    print('-'*40)
    print()

#got 2 best models
#Logistic regressions: 
#Testing Score:  0.7908163265306123
#Training Score:  0.826530612244898

#Adaboost Classifier: 
#Testing Score:  0.8010204081632653
#Training Score:  0.8227040816326531
