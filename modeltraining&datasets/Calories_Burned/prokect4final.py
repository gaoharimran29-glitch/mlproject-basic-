import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv(r"Clean Calroies data.csv")

X = df.drop(columns='Calories')
y = df['Calories']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Gender']) ,
    (StandardScaler() , ["Age" , "Height" , "Weight" , "Duration" , "Heart_Rate" , "Body_Temp"]) ,
    remainder='passthrough'
)

model = CatBoostRegressor(n_estimators=200 , learning_rate=0.05 , depth=6 , random_state=42 , verbose=0)
X_train, X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.2)
pipe = make_pipeline(col_trans , model)
pipe.fit(X_train , y_train)

joblib.dump(pipe , 'Calories Burned Predictor.pkl')