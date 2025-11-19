import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score , mean_squared_error

df = pd.read_csv(r"project1/StudentPerformance.csv")

X = df.drop(columns='Performance Index')
y = df['Performance Index']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore') , ['Extracurricular Activities']) ,
    (StandardScaler() , ['Hours Studied' , 'Previous Scores' , 'Sleep Hours' , 'Sample Question Papers Practiced']) ,
    remainder='passthrough'
    )

models = {
    "Linear Regression": LinearRegression() , 
    "Ridge" : Ridge(alpha=0.01) ,
    "Lasso" : Lasso(alpha=0.01) ,
    "Elastic" : ElasticNet()
}

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)


for name , model in models.items():
    pipe = make_pipeline(col_trans , model)
    pipe.fit(X_train , y_train)
    print("Model name: " , name)
    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    print("MSE: " , mean_squared_error(y_test , y_pred_test))
    print("Testing Accuracy : " , r2_score(y_test , y_pred_test))
    print("Training Accuracy : " , r2_score(y_train , y_pred_train))

#linear regression is working best
""" Model name:  Linear Regression
MSE:  4.082628398521861
Testing Accuracy :  0.9889832909573145
Training Accuracy :  0.9886898790682355 """