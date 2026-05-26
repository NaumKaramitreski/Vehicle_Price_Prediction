import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Australian_Vehicle_Prices.csv")

df.columns = df.columns.str.strip()

print(df.head())
print(df.info())

print("Dataset shape:", df.shape)
print(df.isnull().sum())
print(df.describe())

df["Price"] = (
    df["Price"].astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])

print(df["Price"].head())
print(df.shape)

plt.figure()
df["Price"].hist(bins=30)
plt.title("Vehicle Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Vehicles")
plt.show()

df["Price"] = np.log1p(df["Price"])

plt.figure()
df["Price"].hist(bins=30)
plt.title("Log Price Distribution")
plt.show()

X = df.drop("Price", axis=1)
y = df["Price"]

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

