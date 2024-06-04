import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app_usage = pd.read_csv('/path/to/your_dataset.csv')

label_encoder = LabelEncoder()
app_usage['appName'] = label_encoder.fit_transform(app_usage['appName'])
app_usage['category'] = label_encoder.fit_transform(app_usage['category'])

app_usage['hour_of_day'] = app_usage['time'] % 24
app_usage['day_of_week'] = (app_usage['time'] // 24) % 7


X = app_usage[['appName', 'hour_of_day', 'day_of_week', 'duration']]
y = app_usage['category']

scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Baseline Accuracy of the model: {accuracy}")

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5)


grid_search.fit(X_train, y_train)


print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best accuracy score: {grid_search.best_score_}")

best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Random Forest Accuracy after Hyperparameter Tuning: {accuracy_best}")
