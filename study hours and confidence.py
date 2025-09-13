# Import necessary libraries
import pandas as pd        # to create and manage datasets in table form (like Excel but in Python)
import numpy as np         # for handling math operations and generating random numbers
import statsmodels.api as sm   # for running OLS regression (ordinary least squares)
from linearmodels.iv import IV2SLS   # for running instrumental variables regression (2SLS)

# -----------------------------
# Step 1: Simulate artificial data
# -----------------------------
np.random.seed(42)   # sets a random seed so results are reproducible (same numbers every run)
n = 200              # number of observations (like having 200 students in our dataset)

# Instrument variable (distance from library, random between 0.1 and 5)
library_distance = np.random.uniform(0.1, 5, n)

# Endogenous regressor (study_hours) = affected by instrument + random noise
# Idea: further the library, fewer hours studied (negative relationship)
study_hours = 10 - 1.5 * library_distance + np.random.normal(0, 1, n)

# Outcome variable (confidence in exam) = depends on study_hours + some random variation
confidence = 2 + 0.4 * study_hours + np.random.normal(0, 1, n)

# -----------------------------
# Step 2: Put everything into a DataFrame
# -----------------------------
# A DataFrame is like a spreadsheet: rows = observations, columns = variables
df = pd.DataFrame({
    'confidence': confidence,
    'study_hours': study_hours,
    'library_distance': library_distance
})

# -----------------------------
# Step 3: Run OLS regression
# -----------------------------
# First add a constant (intercept term) to study_hours
X_ols = sm.add_constant(df['study_hours'])
# Fit a simple linear regression: confidence ~ study_hours
ols_model = sm.OLS(df['confidence'], X_ols).fit()

print("\n=== OLS Results ===")
print(ols_model.summary())   # shows coefficients, R-squared, and other stats

# -----------------------------
# Step 4: Run IV regression (2SLS)
# -----------------------------
# Here we use library_distance as an instrument for study_hours
iv_model = IV2SLS.from_formula(
    'confidence ~ 1 + [study_hours ~ library_distance]',
    data=df
).fit()

print("\n=== IV (2SLS) Results ===")
print(iv_model.summary)   # notice: .summary is a property here (not a function call with ())
