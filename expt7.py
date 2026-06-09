# EXPERIMENT 7 — DATA PREPROCESSING ON DIABETES DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv(r"C:\Users\ASTHA BAHETI\OneDrive\Documents\AIML\diabetes.csv")

print("Original Data:")
print(data.head())

# Select numeric columns for scaling
numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# If you had a categorical column (example):
# categorical_features = ['Gender']

# Preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        # ('cat', OneHotEncoder(), categorical_features)  # Not required for diabetes.csv
    ]
)

# Create preprocessing pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Apply pipeline
processed_data = pipeline.fit_transform(data[numeric_features])

# Convert result back to DataFrame
processed_df = pd.DataFrame(processed_data, columns=numeric_features)

print("\nProcessed (Scaled) Data:")
print(processed_df.head())

# Plot before and after scaling
plt.figure(figsize=(12, 6))

# Before scaling
plt.subplot(1, 2, 1)
plt.scatter(data['Glucose'], data['BMI'], color='blue')
plt.title("Before Scaling")
plt.xlabel("Glucose")
plt.ylabel("BMI")

# After scaling
plt.subplot(1, 2, 2)
plt.scatter(processed_df['Glucose'], processed_df['BMI'], color='orange')
plt.title("After Scaling")
plt.xlabel("Scaled Glucose")
plt.ylabel("Scaled BMI")

plt.tight_layout()
plt.show()
