import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score

df = pd.read_csv(r'Cleaned_Home_Loan.csv')

X = df.drop(columns='Loan_Status')
y = df['Loan_Status']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']),
    (StandardScaler(), ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']),
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = make_pipeline(col_trans, AdaBoostClassifier())

ada_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
    'base_estimator__max_depth': [1, 2, 3, 4]
}

grid_ada = GridSearchCV(pipe, ada_params, cv=5, scoring='f1', n_jobs=-1)
grid_ada.fit(X_train, y_train)

print("🔹 Best Parameters for AdaBoost:", grid_ada.best_params_)
print("🔹 Best CV Accuracy:", grid_ada.best_score_)

y_pred_ada = grid_ada.best_estimator_.predict(X_test)
print("🔹 Test Accuracy:", f1_score(y_test, y_pred_ada))