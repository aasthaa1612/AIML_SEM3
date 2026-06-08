

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")

sns.countplot(x='Age',data=data)
plt.title('age distribution')
plt.show()

sns.countplot(x='BMI',data=data)
plt.title('BMI')
plt.show()

sns.countplot(x='Glucose',data=data)
plt.title('Glucose')
plt.show()

sns.heatmap(data[['Glucose','Insulin','Age']].corr(),annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('correlation heatmap')
plt.show()