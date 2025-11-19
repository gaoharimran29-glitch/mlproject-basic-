import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

df = pd.read_csv(r"Cleaned Employee dataset.csv")

X = df.drop(columns='Salary')
y = df['Salary']

col_trans = make_column_transformer(
    (StandardScaler() , ['Age' , 'Years of Experience']) ,
    (OneHotEncoder(handle_unknown='ignore') , ['Gender' , 'Job Title']) ,
    (OrdinalEncoder(categories=[["Bachelor's" , "Master's" , "PhD"]]) , ['Education Level']) ,
    remainder='passthrough'
)


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
pipe = make_pipeline(col_trans , ElasticNet(alpha=0.02))
pipe.fit(X_train , y_train)

y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)
Testing_score = r2_score(y_test , y_pred_test)
Training_score = r2_score(y_train , y_pred_train)

print("Testing Score: " , Testing_score) #0.908839
print("Training Score: " , Training_score) #0.943742