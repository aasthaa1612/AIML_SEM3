import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as stats

data = pd.read_csv('diabetes.csv')

data_missing = data.copy()
data_missing.loc[1, 'Glucose'] = np.nan
data_missing.loc[5, 'BMI'] = np.nan

data_missing['Glucose'] = data_missing['Glucose'].fillna(data_missing['Glucose'].median())
data_missing['BMI'] = data_missing['BMI'].fillna(data_missing['BMI'].median())

Q1 = data_missing[['Glucose', 'BMI']].quantile(0.25)
Q3 = data_missing[['Glucose', 'BMI']].quantile(0.75)
IQR = Q3 - Q1

outliers =  ((data_missing[['Glucose', 'BMI']] < (Q1 - 1.5 * IQR)) |
             (data_missing[['Glucose', 'BMI']] > (Q3 + 1.5 * IQR)))

data_no_outliers = data_missing.copy()
for column in ['Glucose', 'BMI']:
    lower_bound = Q1[column] - 1.5 * IQR[column]
    upper_bound = Q3[column] + 1.5 * IQR[column]

    data_no_outliers[column] = np.where(data_no_outliers[column] < lower_bound, lower_bound, data_no_outliers[column])

    data_no_outliers[column] = np.where(data_no_outliers[column] > upper_bound, upper_bound, data_no_outliers[column])

print("Original Data: \n", data.head ())
print("\nAfter Handling Missing Values: \n", data_missing. head())
print("\nAfter Handling Outliers: \n", data_no_outliers.head())


plt.figure(figsize=(12, 6))

#print("\nAfter Handling Outliers:\n", data_no_outliers.head())

plt.figure(figsize=(12, 6))
plt.subplot (1, 2, 1)
sns.boxplot(data=data_missing[['Glucose', 'BMI']])
plt.title('Before Outlier Handling' )

plt.subplot (1, 2, 2)
sns.boxplot(data=data_no_outliers[['Glucose', 'BMI']])
plt.title('After Outlier Handling')

plt.tight_layout ()
plt.show()