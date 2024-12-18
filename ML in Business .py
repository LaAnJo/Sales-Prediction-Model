import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("Amazon Sale Report 2 (2) - Copy new.csv")

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

data['TotalValue'] = data['Qty'] * data['Amount']
data['IsShipped'] = (data['Status'] == 'Shipped').astype(int)

label_encoders = {}
categorical_columns = ['Fulfilment', 'ship-service-level', 'Category', 'Size', 'Courier Status']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

scaler = StandardScaler()
data[['Amount', 'Qty', 'TotalValue']] = scaler.fit_transform(data[['Amount', 'Qty', 'TotalValue']])

X, y_total_value = data[['Qty', 'Amount', 'Category', 'Size', 'Fulfilment', 'Month', 'IsShipped']], data['TotalValue']

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    sns.lineplot(x=y_true, y=y_true, color='red')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.show()

def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance * np.abs(y_true)
    accuracy = np.mean(within_tolerance) * 100  # Convert to percentage
    return accuracy

def random_split(X, y, test_size=0.1):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(test_size * X.shape[0])

    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

X_train1, X_test1, y_train1, y_test1 = random_split(X, y_total_value)
linear_model = LinearRegression()
linear_model.fit(X_train1, y_train1)
y_pred_linear = linear_model.predict(X_test1)
print("Linear Regression:")
print("MSE:", mean_squared_error(y_test1, y_pred_linear))
print("R² Score:", r2_score(y_test1, y_pred_linear))
accuracy_linear = calculate_accuracy(y_test1.values, y_pred_linear)
print("Accuracy:", accuracy_linear, "%")
plot_predictions(y_test1, y_pred_linear, "Linear Regression: Predicted vs Actual TotalValue")

X_train2, X_test2, y_train2, y_test2 = random_split(X, y_total_value)
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train2, y_train2)
y_pred_tree = decision_tree_model.predict(X_test2)
print("Decision Tree Regressor:")
print("MSE:", mean_squared_error(y_test2, y_pred_tree))
print("R² Score:", r2_score(y_test2, y_pred_tree))
accuracy_tree = calculate_accuracy(y_test2.values, y_pred_tree)
print("Accuracy:", accuracy_tree, "%")
plot_predictions(y_test2, y_pred_tree, "Decision Tree: Predicted vs Actual TotalValue")

X_train3, X_test3, y_train3, y_test3 = random_split(X, y_total_value)
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train3, y_train3)
y_pred_lasso = lasso_model.predict(X_test3)
print("Lasso Regression:")
print("MSE:", mean_squared_error(y_test3, y_pred_lasso))
print("R² Score:", r2_score(y_test3, y_pred_lasso))
accuracy_lasso = calculate_accuracy(y_test3.values, y_pred_lasso)
print("Accuracy:", accuracy_lasso, "%")
plot_predictions(y_test3, y_pred_lasso, "Lasso Regression: Predicted vs Actual TotalValue")


categorical_columns = ['Status', 'Fulfilment', 'ship-service-level', 'Category', 'Size', 'Courier Status']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
numerical_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()
