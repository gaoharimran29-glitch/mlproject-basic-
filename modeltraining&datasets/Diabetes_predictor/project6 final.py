import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , f1_score
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv(r'cleaned_diabetes.csv')

X = df.drop(columns='Outcome')
y = df['Outcome']

col_trans = make_column_transformer(
    (MinMaxScaler() , ['Pregnancies', 'Glucose' , 'Age' , "BMI", 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction']) ,
    remainder='passthrough'
)

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

sm = SMOTE(random_state=42)
X_train , y_train = sm.fit_resample(X_train , y_train)

model = AdaBoostClassifier( estimator=DecisionTreeClassifier(max_depth=1 , min_samples_leaf=5 , min_samples_split=10) ,
    algorithm='SAMME' , learning_rate=1 , n_estimators=500)

pipe = make_pipeline(col_trans , model)
pipe.fit(X_train , y_train)

#adaboost working good
""" Accuracy Scores: 
Testing:  0.7597402597402597
Training:  0.84125
F1 Score:
Testing:  0.6666666666666666
Training:  0.6666666666666666 """

joblib.dump(pipe , 'diabetes_predictor.pkl')