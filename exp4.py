
# import pandas as pd

# data={
#     "name":['alice','bob','charlie','donald','ava'],
#     "age":[20,18,19,21,19],
#     "salary":[20000,23000,90000,85000,10000],
#     "department":["HR","Engineering","HR","Printing","data"]
#     }

# df=pd.DataFrame(data)
# df.to_csv("clean_data.csv",index=False)
# data=pd.read_csv("clean_data.csv")
# print(data)

import pandas as pd

data=pd.read_csv("clean_data.csv")

print("Number of rows:",data.shape[0])
print("Number of columns:",data.shape[0])
print("/n first five rows of the dataset") 
print(data.head())

print("/nNumber of missing values in each column:")
print(data.isnull().sum())

print("/nSum of numerical columns:")
print(data.sum(numeric_only=True))

print("/nMean of numerical columns:")
print(data.mean(numeric_only=True))

print("/n Minimumof numerical columns:")
print(data.min(numeric_only=True))

print("/nSum of numerical columns:")
print(data.max(numeric_only=True))

data.to_csv("exported_data.csv",index=False)