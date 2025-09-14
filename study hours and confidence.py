# okay… importing stuff I think I need
import pandas as pd   # kinda like Excel but in Python
import numpy as np    # numbers and randomness basically
import statsmodels.api as sm  # for running regression thingy
from linearmodels.iv import IV2SLS  # for the scary IV regression (2SLS) thing

# -----------------------------
# Step 1: make up some fake data
# -----------------------------
np.random.seed(42)   # makes randomness less random?? reproducible? idk
n = 200  # let’s just pick 200 “people” or whatever

# instrument variable… “distance from library”
library_distance = np.random.uniform(0.1, 5, n)  # random numbers between 0.1 and 5

# study hours depend on library distance + some noise
# (further library = less studying, at least that’s the story I’m telling myself)
study_hours = 10 - 1.5 * library_distance + np.random.normal(0, 1, n)

# outcome: confidence in exam = study hours + randomness
confidence = 2 + 0.4 * study_hours + np.random.normal(0, 1, n)

# -----------------------------
# Step 2: throw it all in a DataFrame
# -----------------------------
# DataFrame = like a spreadsheet with rows/columns
df = pd.DataFrame({
    'confidence': confidence,
    'study_hours': study_hours,
    'library_distance': library_distance
})

# just to see if it worked…
print(df.head())  # shows first 5 rows, hopefully not broken

# -----------------------------
# Step 3: OLS regression
# -----------------------------
# apparently you need a constant (intercept)
X_ols = sm.add_constant(df['study_hours'])
y = df['confidence']

# fit the thing: confidence ~ study_hours
ols_model = sm.OLS(y, X_ols).fit()

print("\n=== OLS Results ===")
print(ols_model.summary())  # this prints a giant scary table of stats

# -----------------------------
# Step 4: IV regression (2SLS)
# -----------------------------
# use library_distance as instrument for study_hours
# formula looks weird but ok
iv_model = IV2SLS.from_formula(
    'confidence ~ 1 + [study_hours ~ library_distance]',
    data=df
).fit()

print("\n=== IV (2SLS) Results ===")
print(iv_model.summary)  # NOTE: no () here… weird but it works??

