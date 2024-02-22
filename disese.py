import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# data loaded into X (features) and y (target values).
disease_x_train = np.loadtxt('disease_X_train.txt')
disease_y_train = np.loadtxt('disease_y_train.txt')
#This shows us that our dataset has 10 features and each feature has 353 value
print(disease_x_train.shape)

disease_x_test = np.loadtxt('disease_X_test.txt')
disease_y_test = np.loadtxt('disease_y_test.txt')

# Compute the regression baseline using the mean of the training data.
baseline_prediction = np.mean(disease_y_train)
baseline_mse = mean_squared_error(disease_y_test, [baseline_prediction] * len(disease_y_test))
print(f"Baseline MSE: {baseline_mse:.2f}")

# Fit a Linear Regression model.
linear_model = LinearRegression()
linear_model.fit(disease_x_train, disease_y_train)
linear_predictions = linear_model.predict(disease_x_test)
linear_mse = mean_squared_error(disease_y_test, linear_predictions)
print(f"Linear Model MSE: {linear_mse:.2f}")

# Fit a Decision Tree Regressor model.
decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(disease_x_train, disease_y_train)
tree_predictions = decision_tree_model.predict(disease_x_test)
tree_mse = mean_squared_error(disease_y_test, tree_predictions)
print(f"Decision Tree Model MSE: {tree_mse:.2f}")

# Fit a Random Forest Regressor model.
random_forest_model = RandomForestRegressor()
random_forest_model.fit(disease_x_train, disease_y_train)
forest_predictions = random_forest_model.predict(disease_x_test)
forest_mse = mean_squared_error(disease_y_test, forest_predictions)
print(f"Random Forest Model MSE: {forest_mse:.2f}")
