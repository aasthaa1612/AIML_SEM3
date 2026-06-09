import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv("t20.csv")
X = data.drop("span", axis=1)
y = data["span"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("Original class distribution:\n", y.value_counts())

ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X_train, y_train)
print("After Oversampling:\n", y_over.value_counts())

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
print("After Undersampling:\n", y_under.value_counts())

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print("After SMOTE:\n", y_smote.value_counts())

clf1 = LogisticRegression(max_iter=1000)
clf1.fit(X_train, y_train)
print("\nWithout Class Weights:\n", classification_report(y_test, clf1.predict(X_test)))

clf2 = LogisticRegression(class_weight="balanced", max_iter=1000)
clf2.fit(X_train, y_train)
print("\nWith Class Weights:\n", classification_report(y_test, clf2.predict(X_test)))
