# job_salary_ml
Using regression for salary and classification for job type - application that predicts the salary based on skills


# Stack Overflow Developer Salary Analysis

## Project goal

Analyze the Stack Overflow Developer Survey to understand how professional experience, working hours, and location relate to developer compensation, and build a simple baseline model to predict salary using Python and scikit-learn. [web:96][web:116]

## Dataset

- Source: Stack Overflow Annual Developer Survey (Kaggle mirror of the public survey data). [web:96][web:113]
- Size: ~64k responses and 61 columns, including compensation (`ConvertedComp`), experience (`YearsCode`, `YearsCodePro`), country, role (`DevType`), technologies, and working hours (`WorkWeekHrs`). [web:96]
- This project focuses on salary (`ConvertedComp`) plus a small set of interpretable features for a beginner-friendly, end-to-end analysis. [web:96][web:110]

## EDA and data cleaning

Main steps:

- Selected a subset of relevant columns: `ConvertedComp`, `Country`, `YearsCode`, `YearsCodePro`, `DevType`, `LanguageWorkedWith`, and `WorkWeekHrs`. [web:96]
- Filtered to rows with non-missing salary, resulting in ~35k usable responses for compensation analysis. [web:96][web:113]
- Converted `YearsCode` and `YearsCodePro` from strings (e.g., `"7"`, `"NA"`) into numeric columns (`YearsCode_num`, `YearsCodePro_num`), treating `"NA"` and other invalid entries as missing. [web:96][web:110]
- Created filtered DataFrames for:
  - `exp_salary_df`: respondents with both salary and professional experience.
  - `country_salary_df`: respondents with both salary and country.

Key exploratory findings:

- **Salary vs professional experience**: Median compensation increases sharply over the first few years of professional coding, rising from roughly 24k at 1 year to around 41k at 4 years in the survey’s converted currency. [web:96][web:113]
- **Salary by country**: Median salaries vary dramatically between countries in the sample, with the highest-paying countries offering several times the compensation of the lowest-paying ones. [web:96][web:110]

## Baseline modeling

Objective: Predict developer salary as a regression problem using a simple, interpretable baseline model. [web:116][web:119]

Modeling setup:

- **Target**: `ConvertedComp` (annual compensation). [web:96]
- **Features**:
  - `YearsCodePro_num` – numeric years of professional coding experience.
  - `WorkWeekHrs_num` – weekly working hours (numeric copy of `WorkWeekHrs`). [web:96]
- **Data filtering**: Dropped rows with missing values in `ConvertedComp`, `YearsCodePro_num`, or `WorkWeekHrs_num`, leaving ~32k rows for modeling. [web:96][web:113]
- **Train/test split**: 80% training, 20% test using `train_test_split` from scikit-learn with `random_state=42` for reproducibility. [web:121][web:127]
- **Model**: `LinearRegression` from scikit-learn as a baseline. [web:116][web:118]

Evaluation:

- Metric: Root Mean Squared Error (RMSE) on the test set, derived from `mean_squared_error`. [web:119][web:126]
- Result: Linear Regression RMSE ≈ **214,000** in the survey’s converted currency. [web:116][web:119]
- Sample predictions show that the model captures some overall trend but struggles with very high and very low salaries due to the high variance and skew in compensation data when using only two simple numeric features. [web:96][web:113]

## Insights and limitations

Insights:

- Professional experience is strongly associated with higher median compensation, especially in the first few years of coding professionally. [web:96][web:113]
- Compensation differs significantly by country, emphasising the importance of geography when interpreting salary benchmarks from global surveys. [web:96][web:110]
- A simple linear model with only experience and working hours provides a useful baseline but cannot fully explain the large spread in developer salaries. [web:116][web:119]

Limitations and next steps:

- The current model ignores many potentially important factors such as role (`DevType`), technologies (`LanguageWorkedWith`), company size, and education, which are available in the survey data. [web:96]
- No log transformation or outlier treatment has been applied to `ConvertedComp`, contributing to the high RMSE. [web:119][web:126]
- Future improvements could include:
  - Log-transforming salary to reduce skew.
  - Adding categorical features via one-hot encoding (e.g., country, role, technologies).
  - Trying non-linear models such as `RandomForestRegressor` or gradient boosting. [web:125][web:134]

## Tech stack

- Python 3 (Miniconda / VS Code).
- pandas and NumPy for data loading, cleaning, and EDA. [web:63]
- scikit-learn for train/test splitting, linear regression, and evaluation. [web:116][web:121]
- (Optional) matplotlib / seaborn for visualizations. [web:63]

## Repository structure

Example structure (simplified):

