import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Read the data
app_usage = pd.read_excel(r'C:\Users\Admin\Desktop\Machine learning\My_mini _project\app_usage.xlsx')

# Define the input and target variables
X = app_usage.iloc[:, :-1]
y = app_usage.drop('category' , axis=1)


# One-hot encode the categorical variables in X
X = pd.get_dummies(X)
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns=X.columns)

from sklearn.preprocessing import LabelEncoder

# Encode the categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(app_usage['category'])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# Initialize the random forest classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = rf_model.predict(X_test)

# Convert the predictions from continuous to binary values
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Random Forest Baseline Accuracy of the model: {accuracy}")

# #Hyper-Tuning of random forest
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# # # # define hyperparameters to tune
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [2, 4, 6, 8],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10]
# }
# # initialize random forest classifier
# rf = RandomForestClassifier()

# # # initialize grid search
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# # fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # print the best hyperparameters and accuracy score
# print(f"Best hyperparameters: {grid_search.best_params_}")
# print(f"Best accuracy score: {grid_search.best_score_}")

# # Define the XGBoost model
# xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# # Train the model on the training set
# xgb_model.fit(X_train, y_train)

# # Predict the usage on the testing set
# y_pred = xgb_model.predict(X_test)

# # Calculate the accuracy of the model
# accuracy = xgb_model.score(X_test, y_test)
# print("\n\nXGBoost Baseline Accuracy of the model:", accuracy)

# # define the hyperparameters to tune for xgboost
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'learning_rate': [0.01, 0.1, 0.5, 1],
#     'max_depth': [3, 5, 7, 9],
#     'subsample': [0.5, 0.7, 0.9],
#     'colsample_bytree': [0.5, 0.7, 0.9],
#     'reg_alpha': [0, 0.1, 1, 10],
#     'reg_lambda': [0, 0.1, 1, 10]
# }

# # perform grid search with cross-validation
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # print the best hyperparameters
# print("Best hyperparameters:", grid_search.best_params_)

# # predict on test set using the best model
# y_pred = grid_search.predict(X_test)

# # evaluate the model on test set
# # accuracy = (y_pred == y_test).mean()
# accuracy = accuracy_score(y_test, y_pred)
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'learning_rate': [0.01, 0.1, 0.5, 1],
#     'max_depth': [3, 5, 7, 9],
#     'subsample': [0.5, 0.7, 0.9],
#     'colsample_bytree': [0.5, 0.7, 0.9],
#     'reg_alpha': [0, 0.1, 1, 10],
#     'reg_lambda': [0, 0.1, 1, 10]
# }
# # {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.9}

print("Accuracy of the model:", accuracy)