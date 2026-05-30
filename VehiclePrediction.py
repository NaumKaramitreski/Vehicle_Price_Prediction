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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
}

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred_log = pipe.predict(X_test)

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print(results_df)

results_df = results_df.sort_values(by="RMSE")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
print("Best model based on RMSE:", best_model_name)

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Model Comparison using MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison using RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Comparison using R²")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.xticks(rotation=30)
plt.show()

rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipe.fit(X_train, y_train)

feature_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()

importances = rf_pipe.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df.head(15))

top_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Most Important Features - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()

top_features = feature_importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Most Important Features - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()import pandas as pd
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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])
