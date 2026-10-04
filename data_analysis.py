import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard_data.csv")

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show dataset information
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Count legitimate and fraud transactions
print("\nTransaction class distribution:")
print(df["Class"].value_counts())