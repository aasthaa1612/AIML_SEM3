# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from collections import Counter

# # Load file
# # file_path = "/mnt/data/t20.csv"
# df = pd.read_csv("t20.csv")

# # Drop unnecessary columns
# df = df.drop(columns=["Unnamed: 0", "Unnamed: 15", "Player", "Span"])

# df["100"] = pd.to_numeric(df["100"], errors="coerce")

# # Create target column: HasCentury (1 if player scored >=1 century, else 0)
# df["HasCentury"] = (df["100"] > 0).astype(int)

# X = df.drop(columns=["HasCentury"])
# y = df["HasCentury"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42, stratify=y
# )

# # Plot function
# def plot_distribution(y, title):
#     counts = Counter(y)
#     plt.figure(figsize=(6,4))
#     plt.bar(counts.keys(), counts.values())
#     plt.xticks([0,1], ["No Century", "Has Century"])
#     plt.title(title)
#     plt.xlabel("Classes")
#     plt.ylabel("Number of Samples")
#     plt.show()

# # Before resampling
# plot_distribution(y_train, "Class Distribution Before Resampling")

# # Random Oversampling
# def random_oversample(X, y):
#     df_combined = pd.concat([X, y], axis=1)
#     max_size = df_combined[y.name].value_counts().max()
    
#     lst = [df_combined]
#     for class_index, group in df_combined.groupby(y.name):
#         lst.append(group.sample(max_size - len(group), replace=True, random_state=42))
#     df_balanced = pd.concat(lst)
    
#     return df_balanced.drop(columns=[y.name]), df_balanced[y.name]

# X_ros, y_ros = random_oversample(X_train, y_train)
# plot_distribution(y_ros, "Class Distribution After Random Oversampling")

# # Random Undersampling
# def random_undersample(X, y):
#     df_combined = pd.concat([X, y], axis=1)
#     min_size = df_combined[y.name].value_counts().min()
    
#     df_balanced = df_combined.groupby(y.name).apply(
#         lambda x: x.sample(min_size, random_state=42)
#     ).reset_index(drop=True)
    
#     return df_balanced.drop(columns=[y.name]), df_balanced[y.name]

# X_rus, y_rus = random_undersample(X_train, y_train)
# plot_distribution(y_rus, "Class Distribution After Random Undersampling")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve)
import seaborn as sns

# 1️⃣ Load Data
df = pd.read_csv("t20.csv")
df = df.drop(columns=["Unnamed: 0", "Unnamed: 15", "Player", "Span"])
df["100"] = pd.to_numeric(df["100"], errors="coerce")
df["HasCentury"] = (df["100"] > 0).astype(int)

X = df.drop(columns=["HasCentury"])
y = df["HasCentury"]

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3️⃣ Function: Plot Class Distribution
def plot_distribution(y, title):
    counts = Counter(y)
    plt.figure(figsize=(6,4))
    plt.bar(counts.keys(), counts.values())
    plt.xticks([0,1], ["No Century", "Has Century"])
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of Samples")
    plt.show()

plot_distribution(y_train, "Class Distribution Before Resampling")

# 4️⃣ Random Oversampling
def random_oversample(X, y):
    df_combined = pd.concat([X, y], axis=1)
    max_size = df_combined[y.name].value_counts().max()
    lst = [df_combined]
    for class_index, group in df_combined.groupby(y.name):
        lst.append(group.sample(max_size - len(group), replace=True, random_state=42))
    df_balanced = pd.concat(lst)
    return df_balanced.drop(columns=[y.name]), df_balanced[y.name]

X_ros, y_ros = random_oversample(X_train, y_train)
plot_distribution(y_ros, "Class Distribution After Random Oversampling")

# 5️⃣ Random Undersampling
def random_undersample(X, y):
    df_combined = pd.concat([X, y], axis=1)
    min_size = df_combined[y.name].value_counts().min()
    df_balanced = df_combined.groupby(y.name).apply(
        lambda x: x.sample(min_size, random_state=42)
    ).reset_index(drop=True)
    return df_balanced.drop(columns=[y.name]), df_balanced[y.name]

X_rus, y_rus = random_undersample(X_train, y_train)
plot_distribution(y_rus, "Class Distribution After Random Undersampling")

# 6️⃣ Train Logistic Regression models
model_orig = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
model_ros = LogisticRegression(max_iter=1000, random_state=42).fit(X_ros, y_ros)
model_rus = LogisticRegression(max_iter=1000, random_state=42).fit(X_rus, y_rus)

# 7️⃣ Function: ROC Curves
def plot_roc_curves(models, X_tests, y_test, labels):
    plt.figure(figsize=(7,6))
    for model, X_t, label in zip(models, X_tests, labels):
        y_prob = model.predict_proba(X_t)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_roc_curves([model_orig, model_ros, model_rus],
                [X_test, X_test, X_test],
                y_test,
                ["Original", "Oversampled", "Undersampled"])

# 8️⃣ Function: Precision-Recall Curves
def plot_pr_curves(models, X_tests, y_test, labels):
    plt.figure(figsize=(7,6))
    for model, X_t, label in zip(models, X_tests, labels):
        y_prob = model.predict_proba(X_t)[:,1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=label)
    plt.title("Precision-Recall Curves Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_pr_curves([model_orig, model_ros, model_rus],
               [X_test, X_test, X_test],
               y_test,
               ["Original", "Oversampled", "Undersampled"])

# 9️⃣ Function: Threshold vs F1 Graph
def threshold_f1_graph(model, X_test, y_test, label, thresholds=np.arange(0.1,1.0,0.1)):
    y_prob = model.predict_proba(X_test)[:,1]
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    plt.plot(thresholds, f1_scores, marker='o', label=label)

plt.figure(figsize=(8,5))
threshold_f1_graph(model_orig, X_test, y_test, "Original")
threshold_f1_graph(model_ros, X_test, y_test, "Oversampled")
threshold_f1_graph(model_rus, X_test, y_test, "Undersampled")
plt.title("Threshold vs F1 Score Comparison")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.show()
