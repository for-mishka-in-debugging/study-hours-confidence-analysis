# Import necessary libraries
import pandas as pd       # for creating and handling datasets
import numpy as np        # for numerical operations and random data generation
import statsmodels.api as sm   # for running OLS regression (ordinary least squares)
from linearmodels.iv import IV2SLS   # for running instrumental variables regression (2SLS)

# ---- Step 1: Simulate data ----
np.random.seed(42)  # ensures reproducibility (same random results every time)
n = 200             # number of observations (sample size)

# Step 2: Define instrument variable (distance to library)
# Instrument is something that influences study_hours but not directly confidence
library_distance = np.random.uniform(0.1, 5, n)  

# Step 3: Define endogenous regressor (study hours, affected by library distance)
# More distance → fewer study hours (negative relationship), plus some randomness
study_hours = 10 - 1.5 * library_distance + np.random.normal(0, 1, n)

# Step 4: Define the outcome variable (confidence level)
# Confidence depends positively on study_hours, with some noise
confidence = 2 + 0.4 * study_hours + np.random.normal(0, 1, n)

# Step 5: Put all variables into a DataFrame for easier handling
df = pd.DataFrame({
    'confidence': confidence,
    'study_hours': study_hours,
    'library_distance': library_distance
})

# ---- Step 6: Run OLS regression (basic regression without instrument) ----
# Add a constant term (intercept) to the model
X_ols = sm.add_constant(df['study_hours'])
# Fit the OLS model: confidence explained by study_hours
ols_model = sm.OLS(df['confidence'], X_ols).fit()

# Print OLS results
print("\n=== OLS Results ===")
print(ols_model.summary())

# ---- Step 7: Run IV regression (2SLS: two-stage least squares) ----
# Formula: confidence explained by study_hours, where study_hours is instrumented by library_distance
iv_model = IV2SLS.from_formula(
    'confidence ~ 1 + [study_hours ~ library_distance]',
    data=df
).fit()

# Print IV results
print("\n=== IV (2SLS) Results ===")
print(iv_model.summary)



#WRITTEN SUMMARY 

#This project simulates a dataset to explore the effect of study hours on confidence. However, study hours are potentially endogenous (i.e., correlated with unobserved factors). To address this, the code introduces an instrumental variable (library distance), which influences study hours but not confidence directly.
#First, an OLS regression is run, regressing confidence directly on study hours.
#Then, an IV regression (2SLS) is performed, using library distance as an instrument for study hours.
#This allows us to compare OLS vs IV estimates.
#The OLS model may suffer from bias if study hours are endogenous.
#The IV model attempts to correct this by isolating exogenous variation in study hours (coming from library distance).
#Conclusion: This code demonstrates how instrumental variables can be used in data analysis to handle endogeneity, showcasing Python skills in simulation, regression modeling, and drawing causal conclusions from data.
