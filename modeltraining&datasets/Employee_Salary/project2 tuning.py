import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder , StandardScaler , OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import ElasticNet

df = pd.read_csv(r"Cleaned Employee Dataset.csv")

X = df.drop(columns="Salary")
y = df["Salary"]

col_trans = make_column_transformer(
    (StandardScaler() , ['Age' , 'Years of Experience']) ,
    (OneHotEncoder(handle_unknown='ignore') , ['Gender' , 'Job Title']) ,
    (OrdinalEncoder(categories=[["Bachelor's" , "Master's" , "PhD"]]) , ['Education Level']) ,
    remainder='passthrough'
)

r = ElasticNet()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

alphas = np.arange(0.0 , 0.06, 0.01)

params = {
    'elasticnet__alpha': alphas
}

pipe = make_pipeline(col_trans , r)

grid = GridSearchCV(estimator=pipe , param_grid=params , scoring='r2' , cv=5 , n_jobs=-1)
grid.fit(X_train , y_train)

print("Best alpha:", grid.best_params_)
print("Best CV R²:", grid.best_score_)

y_pred = grid.best_estimator_.predict(X_test)
print("Test R² score:", r2_score(y_test, y_pred))


""" Best alpha: {'elasticnet__alpha': np.float64(0.02)}
Best CV R²: 0.9031692521619771
Test R² score: 0.9088398241471498 """