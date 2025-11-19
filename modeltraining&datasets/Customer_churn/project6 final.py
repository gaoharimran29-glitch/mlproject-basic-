import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score , f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

df = pd.read_csv(r'Clean_Customer_Churn.csv')

X = df.drop(columns='Churn')
y = df['Churn']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['gender' , 'InternetService' , 'Contract' , 'PaymentMethod']) ,
    (StandardScaler() , ['tenure' , 'MonthlyCharges' , 'TotalCharges']) ,
    remainder='passthrough'
)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

model = CatBoostClassifier(verbose=0 , learning_rate=0.05 , l2_leaf_reg=3 , iterations=200 , depth=4 , border_count=32)

pipe = Pipeline(steps=[
        ('preprocessor', col_trans),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)])
pipe.fit(X_train , y_train)

joblib.dump(pipe , 'Clean_Customer_churn.pkl')