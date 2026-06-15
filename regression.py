# ==============================
# SUPERVISED LEARNING COMPLETE VISUALIZATION
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Step 1: Load dataset
# ------------------------------
data = pd.read_csv("diabetes.csv")  # 👈 change filename

target_col = 'target'

print("First 5 rows:\n", data.head())

# ------------------------------
# Step 2: Correlation Heatmap
# ------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# Step 3: Feature & Target setup
# ------------------------------
# 👇 Change 'target' to the column name of your dependent variable
target_col = 'target'
X = data.drop(target_col, axis=1)
y = data[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Step 4: LINEAR REGRESSION
# ------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_lin, color='blue')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

print("\nLinear Regression R² Score:", lin_reg.score(X_test_scaled, y_test))

# ------------------------------
# Step 5: LOGISTIC REGRESSION
# ------------------------------
# For demonstration, convert target to binary if it’s not categorical
# (You can skip this if you already have a classification target)
if len(y.unique()) > 2:
    y_class = (y > y.mean()).astype(int)
else:
    y_class = y.copy()

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_class.loc[y_train.index])
y_pred_log = log_reg.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_class.loc[y_test.index], y_pred_log)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("\nClassification Report:\n", classification_report(y_class.loc[y_test.index], y_pred_log))

# ------------------------------
# Step 6: Logistic Decision Boundary (if 2 features)
# ------------------------------
if X.shape[1] >= 2:
    X_plot = X.iloc[:, :2]  # take first 2 features
    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(X_plot)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
    
    log_reg2 = LogisticRegression()
    log_reg2.fit(X_train2, y_train2)
    
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = log_reg2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_class, edgecolors='k', cmap='coolwarm')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Boundary")
    plt.show()
