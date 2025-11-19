import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder , PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.metrics import r2_score

df = pd.read_csv(r"project2/Cleaned Employee dataset.csv")

X = df.drop(columns='Salary')
y = df['Salary']

col_trans = make_column_transformer(
    (StandardScaler() , ['Age' , 'Years of Experience']) ,
    (OneHotEncoder(handle_unknown='ignore') , ['Gender' , 'Job Title']) ,
    (OrdinalEncoder(categories=[["Bachelor's" , "Master's" , "PhD"]]) , ['Education Level']) ,
    remainder='passthrough'
)

models = {
    "Linear Regression": LinearRegression() , 
    "Ridge" : Ridge(alpha=0.01) ,
    "Lasso" : Lasso(alpha=0.01) ,
    "ElasticNet" : ElasticNet(alpha=0.01) ,
    "Polynomial Regression" : make_pipeline(PolynomialFeatures(degree=2) , LinearRegression()) ,
    "Random forest" : RandomForestRegressor(n_estimators=200 , random_state=42) ,
    "Decision Tree" : DecisionTreeRegressor() ,
    "Gradient Boost" : GradientBoostingRegressor(n_estimators=200 , random_state=42) ,
    "Adaboost Regressror" : AdaBoostRegressor(n_estimators=200 , learning_rate=0.1)
}

for name , model in models.items():
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    pipe = make_pipeline(col_trans , model)
    pipe.fit(X_train , y_train)
    y_pred = pipe.predict(X_test)
    score = r2_score(y_test , y_pred)
    print("Model name: " , name)
    print("R2 Score:" , score)

#elastic net is working best and score is 0.90798
""" Model name:  ElasticNet
R2 Score: 0.9079802977591143
Testing Score: 0.90883
Training Score: 0.943742 """