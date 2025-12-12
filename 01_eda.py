import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "survey_results_public.csv")

print("Loading data from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Basic check on the dataset (rows. columsn)")
print(df.shape)

print("\n columns name in the dataset")
print(df.columns.tolist())

print("\n printing 5 rows from the dataset")
print(df.head())

print("\n printing the info and missing values in the dataset")
print(df.info())

cols_interest_in = [
    "Country",
    "YearsCode",
    "EdLevel",
    "ConvertedComp",
    "YearsCodePro",
    "DevType",
    "LanguageWorkedWith",
    "WorkWeekHrs"]

salary_df = df[cols_interest_in.copy()]

print("\n shape before drooping missing salary", salary_df.shape)

print("\n dropping missing values in ConvertedComp column")
salary_df = salary_df.dropna(subset=["ConvertedComp"])

print("\n shape after dropping missing salary", salary_df.shape)

print("\n Quick look at some rows after dropping missing rows")
print(salary_df.head())

def to_numeric_value(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value).strip()
    if value_str.upper() == "NA":
        return np.nan
    try:
        return float(value_str)
    except ValueError:
        return np.nan

# Apply the function to create numeric columns
salary_df["YearsCode_num"] = salary_df["YearsCode"].apply(to_numeric_value)
salary_df["YearsCodePro_num"] = salary_df["YearsCodePro"].apply(to_numeric_value)

print("\nSample of YearsCode and YearsCode_num:")
print(salary_df[["YearsCode", "YearsCode_num"]].head(10))

print("\nSample of YearsCodePro and YearsCodePro_num:")
print(salary_df[["YearsCodePro", "YearsCodePro_num"]].head(10))

exp_salary_df = salary_df.dropna(subset=["YearsCodePro_num", "ConvertedComp"])
print("\n Shape after dropping the columns from YearsCodePro_num and ConvertedComp", exp_salary_df.shape)

print(exp_salary_df.head())
# Step 1: groupby and median
grouped = exp_salary_df.groupby("YearsCodePro_num")["ConvertedComp"].median()
# Step 2: turn Series into DataFrame
median_salary_by_exp_df = grouped.reset_index()
# Step 3: sort values
median_salary_by_exp_df = median_salary_by_exp_df.sort_values("YearsCodePro_num")
print("\n Median salary based on experiance:")
print(median_salary_by_exp_df.head(20))

country_salary_df = salary_df.dropna(subset=["Country", "ConvertedComp"])
print("\n Shape after dropping the columns from Country and ConvertedComp", country_salary_df.shape)

print(country_salary_df.head())

# Median salary by country
median_salary_by_country = (
    country_salary_df
    .groupby("Country")["ConvertedComp"]
    .median()
    .reset_index()
    .sort_values("ConvertedComp", ascending=True)
)

# Sort by median salary descending
median_salary_by_country = median_salary_by_country.sort_values(
    "ConvertedComp", ascending=False
)

print("\nTop 10 countries by median salary:")
print(median_salary_by_country.head(10))

print("\nBottom 10 countries by median salary:")
print(median_salary_by_country.tail(10))

print("\n using linear regression to predict salary based on years of professional coding experience and working hours")

salary_df["WorkWeekHrs_num"] = salary_df["WorkWeekHrs"]
print(salary_df.head())

model_df = salary_df.dropna(subset=["YearsCodePro_num", "WorkWeekHrs_num", "ConvertedComp"])
print("Shape after dropping missing values from YearsCodePro_num, WorkWeekHrs_num, ConvertedComp:", model_df.shape)

X  = model_df[["YearsCodePro_num", "WorkWeekHrs_num"]] # Features
y  = model_df["ConvertedComp"] # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

print("Training data shape:", X_train.shape, y_train.shape)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("\nLinear Regression RMSE:", rmse)

# Show first 5 actual vs predicted
results_df = X_test.copy()
results_df["ActualSalary"] = y_test
results_df["PredictedSalary"] = y_pred

print("\nSample of actual vs predicted salary:")
print(results_df[["ActualSalary", "PredictedSalary", "YearsCodePro_num", "WorkWeekHrs_num"]].head(20))
