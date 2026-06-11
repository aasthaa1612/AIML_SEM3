import pandas as pd

# Load both CSV files
df1 = pd.read_csv("Bollywood Box Office Collection (2019-2023).csv")
df2 = pd.read_csv("bollywood2023.csv")

# Merge vertically (stack rows)
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save merged dataset
merged_df.to_csv("merged_dataset.csv", index=False)

print("Merged dataset created successfully!")
print("Shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())
