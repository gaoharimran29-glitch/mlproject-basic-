import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r'cleaned_diabetes.csv')

X = df.drop(columns='Outcome')
y = df['Outcome']

col_trans = make_column_transformer(
    (MinMaxScaler(), [
        'Pregnancies', 'Glucose_BMI', 'Glucose_Age', 'Age_BMI',
        'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction',
    ]),
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(), random_state=42
)

pipe = Pipeline([
    ('scaler', col_trans),
    ('adaboostclassifier', ada)
])

params = {
    'adaboostclassifier__n_estimators': [100, 200, 300, 400],
    'adaboostclassifier__learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
    'adaboostclassifier__estimator__max_depth': [1, 2, 3, 4]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("✅ Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nFinal Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Final F1 Score:", round(f1_score(y_test, y_pred), 4))

""" ✅ Best Parameters: {'adaboostclassifier__algorithm': 'SAMME', 'adaboostclassifier__estimator__max_depth': 4, 'adaboostclassifier__learning_rate': 1, 'adaboostclassifier__n_estimators': 300}

Final Accuracy: 0.7597
Final F1 Score: 0.6667 """