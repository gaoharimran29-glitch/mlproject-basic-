import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score , mean_squared_error
import joblib

df = pd.read_csv(r"StudentPerformance.csv")

X = df.drop(columns='Performance Index')
y = df['Performance Index']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Extracurricular Activities']) ,
    (StandardScaler() , ['Hours Studied' , 'Previous Scores' , 'Sleep Hours' , 'Sample Question Papers Practiced']) ,
    remainder='passthrough'
    )

pipe = make_pipeline(col_trans , LinearRegression())

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=42 , test_size=0.2)
pipe.fit(X_train , y_train)

print("Model Performance Metrics")

y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)

print("MSE: " , mean_squared_error(y_test , y_pred_test))
print("Testing Accuracy : " , r2_score(y_test , y_pred_test))
print("Training Accuracy : " , r2_score(y_train , y_pred_train))

joblib.dump(pipe , "student_performance_model.pkl")
print("Model Saved")