import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler , PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.metrics import r2_score

df = pd.read_csv(r"project3/Clean Calroies data.csv")

print(df.head())

X = df.drop(columns='Calories')
y = df['Calories']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Gender']) ,
    (StandardScaler() , ["Age" , "Height" , "Weight" , "Duration" , "Heart_Rate" , "Body_Temp"]) ,
    remainder='passthrough'
)

models = {
    "Linear Regression": LinearRegression() , 
    "Ridge" : Ridge(alpha=0.01) ,
    "Lasso" : Lasso(alpha=0.01) ,
    "Elastic" : ElasticNet(alpha=0.01) ,
    "Random forest" : RandomForestRegressor(n_estimators=200 , random_state=42) ,
    "Decision Tree" : DecisionTreeRegressor() ,
    "Gradient Boost" : GradientBoostingRegressor(n_estimators=200 , random_state=42) ,
    "Polynomial Regression" : make_pipeline(PolynomialFeatures(degree=2) , LinearRegression()) ,
    "Adaboost Regressror" : AdaBoostRegressor(n_estimators=200 , learning_rate=0.1) ,
    "XGBoost regressor" : XGBRegressor() ,
    "CatBoost" : CatBoostRegressor(n_estimators=200 , learning_rate=0.1 , verbose=0)
}

for name , model in models.items():
    X_train, X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.2)
    pipe = make_pipeline(col_trans , model)
    pipe.fit(X_train , y_train)
    y_testpred = pipe.predict(X_test)
    y_trainpred = pipe.predict(X_train)
    score = r2_score(y_test , y_testpred)
    score2 = r2_score(y_train , y_trainpred)
    print("Name: " , name)
    print('Test R2 Score: ' , score)
    print('Train R2 score: ' , score2)
    print('-'*40)

""" Name:  CatBoost
Test R2 Score:  0.9995455516664498
Train R2 score:  0.9996027306369671 """