# ===================== COMMON PREPROCESSING BLOCK =====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------- LOAD DATASET --------------------
df = pd.read_csv("merged_dataset.csv")

# ----------------------- CLEAN DATE ----------------------
df['Released Date'] = pd.to_datetime(df['Released Date'], errors='coerce')
df['Release_Year'] = df['Released Date'].dt.year
df['Release_Month'] = df['Released Date'].dt.month
df['Release_Day'] = df['Released Date'].dt.day
df = df.drop(columns=['Released Date'])

# ----------------------- CLEAN MONEY ----------------------
def clean_money(x):
    if pd.isna(x):
        return np.nan
    return pd.to_numeric(str(x).replace(",", "").replace("₹", ""), errors='coerce')

for col in ['Worldwide','India Gross','Overseas']:
    df[col] = df[col].apply(clean_money)

# ----------------------- DROP USELESS COLUMN --------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# ----------------------- TARGET ---------------------------
# For classification models:
df_class = df.dropna(subset=["Verdict"])
y_class = df_class["Verdict"]
X_class = df_class.drop(columns=["Verdict"])

# For regression model:
df_reg = df.dropna(subset=["Worldwide"])
y_reg = df_reg["Worldwide"]
X_reg = df_reg.drop(columns=["Worldwide"])

# ----------------------- COLUMN TYPES ----------------------
numeric_cols_class = X_class.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols_class = X_class.select_dtypes(include=['object']).columns.tolist()

numeric_cols_reg = X_reg.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()

# ----------------------- PREPROCESSING ---------------------
numeric_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess_class = ColumnTransformer([
    ("num", numeric_transform, numeric_cols_class),
    ("cat", categorical_transform, categorical_cols_class)
])

preprocess_reg = ColumnTransformer([
    ("num", numeric_transform, numeric_cols_reg),
    ("cat", categorical_transform, categorical_cols_reg)
])

# ----------------------- TRAIN/TEST SPLITS -----------------
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("COMMON PREPROCESSING COMPLETED.\n")



# ===================== LOGISTIC REGRESSION =====================

from sklearn.linear_model import LogisticRegression

log_model = Pipeline([
    ("prep", preprocess_class),
    ("clf", LogisticRegression(max_iter=2000))
])

log_model.fit(Xc_train, yc_train)
log_pred = log_model.predict(Xc_test)

print("LOGISTIC REGRESSION RESULTS\n")
print("Accuracy:", accuracy_score(yc_test, log_pred))
print(classification_report(yc_test, log_pred))

cm = confusion_matrix(yc_test, log_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()



# ===================== SVM CLASSIFIER =====================

from sklearn.svm import SVC

svm_model = Pipeline([
    ("prep", preprocess_class),
    ("clf", SVC(kernel="rbf"))
])

svm_model.fit(Xc_train, yc_train)
svm_pred = svm_model.predict(Xc_test)

print("\nSVM CLASSIFIER RESULTS\n")
print("Accuracy:", accuracy_score(yc_test, svm_pred))
print(classification_report(yc_test, svm_pred))

cm = confusion_matrix(yc_test, svm_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("SVM - Confusion Matrix")
plt.show()



# ===================== DECISION TREE CLASSIFIER =====================

from sklearn.tree import DecisionTreeClassifier

tree_model = Pipeline([
    ("prep", preprocess_class),
    ("clf", DecisionTreeClassifier())
])

tree_model.fit(Xc_train, yc_train)
tree_pred = tree_model.predict(Xc_test)

print("\nDECISION TREE CLASSIFIER RESULTS\n")
print("Accuracy:", accuracy_score(yc_test, tree_pred))
print(classification_report(yc_test, tree_pred))

cm = confusion_matrix(yc_test, tree_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree - Confusion Matrix")
plt.show()



# ===================== LINEAR REGRESSION =====================

from sklearn.linear_model import LinearRegression

lin_model = Pipeline([
    ("prep", preprocess_reg),
    ("reg", LinearRegression())
])

lin_model.fit(Xr_train, yr_train)
lin_pred = lin_model.predict(Xr_test)

print("\nLINEAR REGRESSION RESULTS\n")
print("MAE:", mean_absolute_error(yr_test, lin_pred))
print("MSE:", mean_squared_error(yr_test, lin_pred))
print("R²:", r2_score(yr_test, lin_pred))

# True vs Predicted plot
plt.figure(figsize=(7,5))
plt.scatter(yr_test, lin_pred)
plt.xlabel("True Worldwide")
plt.ylabel("Predicted Worldwide")
plt.title("Linear Regression - True vs Predicted")
plt.show()
