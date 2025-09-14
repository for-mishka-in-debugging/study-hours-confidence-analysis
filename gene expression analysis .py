# ===== Step 1: Load libraries =====
#pulling in the usual libraries…
# pandas/numpy for data handling, matplotlib/seaborn for plots
# sklearn for a simple classifier, and scipy for stats test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind


# ===== Step 2: Load the data =====
# I found this dataset on UCI (gene expression RNA-Seq). 
# should be a csv file with samples as rows and genes as columns.
df = pd.read_csv("gene_expression_cancer_rna_seq.csv")

# just checking what it looks like + the size so I know what I’m dealing with
print(df.head())
print(df.shape)  # want to confirm how many samples and how many genes


# ===== Step 3: Preprocess / simplify =====
# problem: there are ~20k genes, way too much to deal with as a beginner.
# idea: focus only on the ones that vary the most between samples.
variances = df.var(axis=0)

# grabbing the top 50 genes with biggest variance (hopefully the most “interesting”)
top_genes = variances.sort_values(ascending=False).head(50).index  

# now just keep those columns
df_small = df[top_genes]
print("Reduced to top 50 genes; new shape:", df_small.shape)

# need the labels (cancer type for each sample)
# not sure if they’re in a separate file or same df, so just assuming separate here
labels = pd.read_csv("gene_expression_targets.csv")  
y = labels['cancer_type']  # column name might differ depending on dataset

# split data into train/test so I can try classification later
X_train, X_test, y_train, y_test = train_test_split(
    df_small, y, test_size=0.3, random_state=42, stratify=y
)


# ===== Step 4: Exploratory Data Analysis (EDA) =====
# goal here = see if anything obvious jumps out visually/statistically

# a) pick the single most variable gene and check how its expression looks across cancers
gene = top_genes[0]
plt.figure(figsize=(8,6))
sns.boxplot(x=y, y=df_small[gene])  # boxplot to compare across groups
plt.xticks(rotation=45)
plt.title(f"Expression of {gene} across cancer types")
plt.show()

# b) heatmap — should help me see overall patterns 
# (but only taking a subset otherwise it’s unreadable)
mask = (y == 'BRCA') | (y == 'COAD')  # just two cancer types to simplify
subset_X = df_small[mask]
subset_y = y[mask]

plt.figure(figsize=(12,10))
sns.heatmap(subset_X.iloc[:30].T, cmap="viridis", cbar=True)
plt.title("Heatmap of first 30 samples (genes vs samples) for BRCA vs COAD")
plt.xlabel("Sample index")
plt.ylabel("Gene")
plt.show()

# c) basic stats test — wondering if this gene is significantly different between BRCA vs COAD
vals_brca = df_small.loc[mask & (y=='BRCA'), gene]
vals_coad = df_small.loc[mask & (y=='COAD'), gene]
t_stat, p_val = ttest_ind(vals_brca, vals_coad, equal_var=False)
print(f"T-test for {gene} between BRCA vs COAD: t={t_stat:.2f}, p={p_val:.3e}")


# ===== Step 5: Simple Classification Attempt =====
# let’s see if logistic regression can separate cancer types
# (probably won’t be amazing with only 50 genes, but worth trying)

clf = LogisticRegression(max_iter=1000)  # bumping max_iter just in case
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# ===== Step 6: Reflection / Conclusion =====
# okay so overall:
# - I reduced a giant dataset down to something manageable (top 50 genes)
# - did some quick plots to visualize expression differences
# - ran a t-test just to see if one gene differed between cancers
# - finally, tried a basic classifier
# clearly this is just scratching the surface but the aim was to at least 
# load biological data, ask a simple question, and try out basic analysis methods.
