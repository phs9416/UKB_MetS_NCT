import pandas as pd
import numpy as np
import os
from patsy import dmatrix
from pcntoolkit.normative import estimate

cohort = pd.read_csv('cohort.csv')
activation = pd.read_csv('activation.csv') ## controllability.csv, metabolic_feature.csv
data = pd.merge(cohort, activation, how='inner', on='eid')
data.to_csv('whole_data.csv', index=False)

ability = [col for col in activation.columns if col != 'eid']
# ability = ['BMI', 'Weight', 'Waist circumference', 'Hip circumference', 'HbA1c', 'HDL', 'LDL', 'Cholesterol', 'Triglycerides', 'Systolic BP', 'Diastolic BP']

"""
==========================================================================
UKB Normative Modeling Pipeline
---------------------------------------------------------------------------
✓ preprocessing: sex, income, alcohol, education, ethnity(White vs Non-White),
          Townsend quintile(1–5), smoking/center one-hot
✓ spline(age)  : cubic B-spline, df=5, no intercept
✓ covariates   : covariates.txt
✓ site (dummy) : site.txt
✓ response     : <network>_response.txt
✓ BLR modeling : PCNtoolkit
==========================================================================
"""
# === [0] Basic setting & module =====================================================
import os
import numpy as np
import pandas as pd
from patsy import dmatrix
from pcntoolkit.normative import estimate

# -----------------------------------------------------------------
DATA_FILE   = "whole_data.csv" 
OUT_DIR     = "blr_spline_pcntoolkit_outputs"
NETWORKS    = ability

os.makedirs(OUT_DIR, exist_ok=True)

# === [1] Import data ======================================================
df = pd.read_csv(DATA_FILE)

# === [2] NaN processing ------------------------------------------------------
def _to_nan(x):
    if isinstance(x, str) and x.strip() in ["Do not know", "Prefer not to answer"]:
        return np.nan
    return x

cols_str = ['sex', 'household_income', 'alcohol',
            'education', 'smoking', 'center', 'ethnity']
df[cols_str] = df[cols_str].applymap(_to_nan)

# === [3] Individual covariate encoding ---------------------------------------------------
# 3-1) sex (0 = Male, 1 = Female) --------------------------------------------
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})

# 3-2) household_income (ordinal 0–4) ----------------------------------------
income_order = {
    'Less than 18,000': 0, '18,000 to 30,999': 1,
    '31,000 to 51,999': 2, '52,000 to 100,000': 3,
    'Greater than 100,000': 4
}
df['household_income'] = df['household_income'].map(income_order)
df['household_income'].fillna(int(df['household_income'].median()), inplace=True)

# 3-3) alcohol (ordinal 0–5) --------------------------------------------------
alcohol_order = {
    'Never': 0, 'Special occasions only': 1, 'One to three times a month': 2,
    'Once or twice a week': 3, 'Three or four times a week': 4,
    'Daily or almost daily': 5
}
df['alcohol'] = df['alcohol'].map(alcohol_order)
df['alcohol'].fillna(int(df['alcohol'].median()), inplace=True)

# 3-4) education (0–3) --------------------------------------------------------
def simplify_education(text):
    if pd.isna(text):                                 return np.nan
    if 'College or University degree' in text:        return 3
    if 'A levels/AS levels' in text:                  return 2
    if 'O levels/GCSEs' in text:                      return 1
    if 'None of the above' in text:                   return 0
    return 1
df['education'] = df['education'].apply(simplify_education)
df['education'].fillna(int(df['education'].median()), inplace=True)

# 3-5) ethnity → White vs Non-White (binary) ----------------------------------
white_cats = ['British', 'Irish', 'Any other white background', 'White']
df['is_nonwhite'] = df['ethnity'].apply(
    lambda x: 1.0 if pd.notna(x) and x not in white_cats
              else 0.0 if pd.notna(x) and x in white_cats
              else np.nan
)
df['is_nonwhite'].fillna(df['is_nonwhite'].median(), inplace=True)
df.drop(columns=['ethnity'], inplace=True)

# 3-6) Townsend index → 5-quantile (1–5) --------------------------------------
df['townsend_quintile'] = pd.qcut(
    df['townsend'], 5, labels=[1, 2, 3, 4, 5]
).astype(float)
df['townsend_quintile'].fillna(3, inplace=True)   

# 3-7) smoking & center one-hot  -----------
for col in ['smoking', 'center']:
    df[col].fillna('Unknown', inplace=True)

df = pd.get_dummies(df, columns=['smoking', 'center'], drop_first=False)

# ✅ one-hot bool type → float
df = df.astype(float)

# === [4] Type change ==================================
if len(df.select_dtypes('object').columns):
    raise ValueError(f"Still remain string: {df.select_dtypes('object').columns.tolist()}")

# === [5] Spline(age) generation =====================================================
spline_df = dmatrix(
    "bs(age, df=5, degree=3, include_intercept=False)",
    {"age": df["age"]}, return_type='dataframe'
)
spline_df.columns = [f"spline_{i}" for i in range(spline_df.shape[1])]

# === [6] covariate dataframe & save ======================================
# smoking_/center_ dummy column
smoking_dummies = [c for c in df.columns if c.startswith('smoking_')]
center_dummies  = [c for c in df.columns if c.startswith('center_')]

'''
non_spline_vars = [
    'sex', 'household_income', 'alcohol', 'education',
    'townsend_quintile', 'is_nonwhite'
] + smoking_dummies + center_dummies
'''

non_spline_vars = ['sex']

cov_df = pd.concat(
    [spline_df.reset_index(drop=True), df[non_spline_vars].reset_index(drop=True)],
    axis=1
)

cov_path  = os.path.join(OUT_DIR, "covariates.txt")
site_path = os.path.join(OUT_DIR, "site.txt")

cov_df.to_csv(cov_path,  index=False, header=False, sep=' ')
pd.DataFrame({'site': [0] * len(cov_df)}).to_csv(
    site_path, index=False, header=False, sep=' '
)

print(f"✅ covariates.txt saved → {cov_df.shape[0]} subjects × {cov_df.shape[1]} vars")

# === [7] BLR model training ========================================================
for net in NETWORKS:

    print(f"\n▶ BLR model training: {net}")

    resp_path = os.path.join(OUT_DIR, f"{net}_response.txt")
    df[[net]].to_csv(resp_path, index=False, header=False, sep=' ')

    # result directory
    net_outdir = os.path.join(OUT_DIR, net)
    os.makedirs(net_outdir, exist_ok=True)

    # PCNtoolkit BLR
    estimate(
        covfile=cov_path,
        respfile=resp_path,
        outdir=net_outdir,
        alg="blr",
        savemodel=True,
        standardize=True,
        variables=list(cov_df.columns),
        cvfolds=10,
        outputsuffix=net
    )

    print(f"✅ complete → result save: {net_outdir}")

print("\n🎉 All network BLR pipeline completed!")
