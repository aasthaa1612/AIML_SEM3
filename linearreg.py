import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Display first few rows
print("First 5 rows:\n", data.head())

# --- Correlation Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.show()

# Independent and dependent variables
# Here we try to predict 'Glucose' based on 'BMI'
X = data[['BMI']]
y = data['Glucose']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict
y_pred = reg_model.predict(X_test)

# --- Metrics ---
print("\n--- Regression Metrics ---")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Coefficient:", reg_model.coef_[0])
print("Intercept:", reg_model.intercept_)

# --- Regression Line Plot ---
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label='Actual Data')
sorted_idx = X['BMI'].argsort()
plt.plot(X['BMI'][sorted_idx], reg_model.predict(X.iloc[sorted_idx]), color='red', linewidth=2, label='Regression Line')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('Linear Regression: BMI vs Glucose (Diabetes Dataset)')
plt.legend()
plt.show()
