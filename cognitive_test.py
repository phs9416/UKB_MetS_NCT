import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# --- Step 1: Load Data ---
cohort_df = pd.read_csv("cohort.csv") 
cog_df = pd.read_csv("cognitive_test.csv") 

# --- Step 2: Merge by eid ---
merged_df = pd.merge(cohort_df, cog_df, on="eid", how="inner")

# --- Step 3: Clean Cognitive Data ---
cognitive_columns = ['tmt_a', 'tmt_b', 'fi', 'bds', 'sds', 'pal', 'mpc']
cognitive_data = merged_df[cognitive_columns]

cognitive_data_clean = cognitive_data.replace(
    ["Trail not completed", "Abandoned"], np.nan
).astype(float)

# --- Step 4: MICE Imputation ---
imputer = IterativeImputer(random_state=0, max_iter=10)
imputed_data = imputer.fit_transform(cognitive_data_clean)

# --- Step 5: Replace with Imputed Data ---
merged_df_imputed = merged_df.copy()
merged_df_imputed[cognitive_columns] = imputed_data

# --- Optional: Save Output ---
merged_df_imputed.to_csv("cohort_with_imputed_cognitive_scores.csv", index=False)

###--------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Data ---
df = pd.read_csv("cohort_with_imputed_cognitive_scores.csv")

# --- Step 2: Define Variables ---
cognitive_columns = ['TMT_A', 'TMT_B', 'FI', 'BDS', 'SDS', 'PAL', 'MPC']
covariates = ['age', 'sex', 'education']

# --- Step 3: Standardize Cognitive Scores ---
df_std = df.copy()
scaler = StandardScaler()
df_std[cognitive_columns] = scaler.fit_transform(df_std[cognitive_columns])

# Reverse TMT scores 
df_std['TMT_A'] *= -1
df_std['TMT_B'] *= -1

# --- Step 4: Prepare Model DataFrame ---
df_std['sex'] = df_std['sex'].astype('category')
df_std['education'] = df_std['education'].astype('category')

df_model_std = pd.get_dummies(
    df_std[cognitive_columns + ['ms_diagnosis_x'] + covariates], 
    drop_first=True
)

# --- Step 5: Regression ---
adjusted_std_results = []
p_values_std = []

for col in cognitive_columns:
    X = df_model_std.drop(columns=cognitive_columns).copy()
    y = df_std[col].astype(float)
    X = sm.add_constant(X).astype(float)

    model = sm.OLS(y, X).fit()
    coef = model.params.get('ms_diagnosis_x', np.nan)
    pval = model.pvalues.get('ms_diagnosis_x', np.nan)

    adjusted_std_results.append({
        "Cognitive Test": col,
        "Standardized β (MetS)": coef,
        "Raw p-value": pval
    })
    p_values_std.append(pval)

# --- Step 6: FDR correction ---
fdr_corrected = multipletests(p_values_std, method='fdr_bh')[1]
for i, adj_p in enumerate(fdr_corrected):
    adjusted_std_results[i]["FDR-adjusted p-value"] = adj_p

adjusted_std_df = pd.DataFrame(adjusted_std_results)

# --- Step 7: visualize ---
plt.figure(figsize=(10, 6))
sns.barplot(data=adjusted_std_df, x="Cognitive Test", y="Standardized β (MetS)")
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel("Standardized β")
# plt.title("Effect of MetS on Standardized Cognitive Test Scores (with Covariates)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
