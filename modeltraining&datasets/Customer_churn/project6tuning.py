import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline

df = pd.read_csv(r'project6/Clean_Customer_Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

X = df.drop(columns='Churn')
y = df['Churn']

col_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['gender', 'InternetService', 'Contract', 'PaymentMethod']),
    (StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(steps=[
    ('preprocessor', col_trans),
    ('smote', SMOTEN(random_state=42)),
    ('classifier', CatBoostClassifier(verbose=0, random_state=42))
])

param_grid = {
    'classifier__depth': [4, 6, 8],
    'classifier__learning_rate': [0.03, 0.05, 0.1],
    'classifier__iterations': [200, 400],
    'classifier__l2_leaf_reg': [1, 3, 5],
    'classifier__border_count': [32, 64]
}

grid = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_grid,
    n_iter=10,        
    scoring='f1',
    cv=3,
    verbose=0,
    n_jobs=-1,
    random_state=42
)

grid.fit(X_train, y_train)

print('Best Parameters:', grid.best_params_)
print('Best F1 Score (CV):', grid.best_score_)

y_pred = grid.predict(X_test)
print('\nTest Accuracy:', accuracy_score(y_test, y_pred))
print('Test F1 Score:', f1_score(y_test, y_pred))
