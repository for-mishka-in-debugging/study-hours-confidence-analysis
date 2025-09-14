# Study Hours and Confidence Analysis  

Study Hours and Confidence: A Regression Analysis

Project Overview

I wanted to explore whether the number of hours students study has any relationship with how confident they feel academically. This connects to my broader interest in learning and teaching — since I already work with peers at the Academic Resource Center, I was curious to see if there’s some measurable link between preparation and self-confidence.

Methods

Created a small dataset with study hours and reported confidence levels.
Used pandas to structure the data and statsmodels for analysis.
Ran an Ordinary Least Squares (OLS) regression to see if study hours predict confidence.
For robustness, I also explored an instrumental variables (2SLS) regression, to mimic how more complex models deal with potential bias in real-world data.

Results

The OLS model suggested a positive relationship: more study hours were associated with higher confidence levels.
The regression coefficient was positive (≈ 0.55), meaning that for each additional study hour, confidence tended to increase by a modest amount.
The results weren’t from a large dataset, but they followed intuition: preparation helps students feel more in control.



