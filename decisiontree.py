import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Load Dataset ---
data = pd.read_csv("diabetes.csv")

# Display first few rows
print("First 5 rows:\n", data.head())

# --- Correlation Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Diabetes Dataset")
plt.show()

# --- Independent & Dependent Variables ---
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- SVM Model ---
svm_model = SVC(kernel='rbf', random_state=42)  # You can change kernel to 'linear' or 'poly' if you want
svm_model.fit(X_train_scaled, y_train)

# --- Prediction ---
y_pred = svm_model.predict(X_test_scaled)

# --- Evaluation Metrics ---
print("\n--- Support Vector Machine Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (Diabetes Dataset)")
plt.show()
