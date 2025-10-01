from patsy import dmatrix
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

def format_p(p):
    if pd.isna(p):
        return "P=NA"
    if p < 0.001:
        return "P<.001"
    if p > 0.99:
        return "P>.99"
    if p < 0.01:
        return "P=" + f"{p:.3f}".lstrip("0")
    return "P=" + f"{p:.2f}".lstrip("0")

# ------------------------
# Load data
# ------------------------
df_cohort = pd.read_csv('cohort_normative.csv')
df_activation = pd.read_csv('activation_normative.csv')
merged_df = pd.merge(df_cohort, df_activation, on='eid', how='inner')

network_names = ['CONT', 'DMN', 'DAN', 'LIM', 'VAN', 'SMN', 'SUB', 'VIS']
merged_df = merged_df.dropna(subset=['ms_score','age']).copy()
merged_df['ms_score'] = merged_df['ms_score'].astype(int)
ms_scores = sorted(merged_df['ms_score'].unique().tolist())

# ------------------------
# BAG calculation + age correction
# ------------------------
results_df = merged_df[['eid','age','ms_score']].copy()
for net in network_names:
    x = merged_df[net]
    spline_X = dmatrix("bs(x, df=5, degree=3, include_intercept=False)",
                       {"x": x}, return_type='dataframe')
    model = sm.OLS(merged_df['age'], sm.add_constant(spline_X), missing='drop').fit()
    predicted_age = model.predict(sm.add_constant(spline_X))
    bag = predicted_age - merged_df['age']
    gap_model = sm.OLS(bag, sm.add_constant(merged_df['age']), missing='drop').fit()
    results_df[f'{net}_gap'] = gap_model.resid

# ------------------------
# Tukey HSD 
# ------------------------
diff_mats, pval_mats, lower_mats, upper_mats, sig_mats = {},{},{},{},{}
for net in network_names:
    tmp = results_df[['ms_score',f'{net}_gap']].dropna()
    n_levels = len(ms_scores)
    diff_matrix  = np.zeros((n_levels,n_levels))
    pval_matrix  = np.full((n_levels,n_levels), np.nan)
    lower_matrix = np.zeros((n_levels,n_levels))
    upper_matrix = np.zeros((n_levels,n_levels))
    sig_matrix   = np.full((n_levels,n_levels), '', dtype=object)
    if tmp['ms_score'].nunique() >= 2:
        tukey = pairwise_tukeyhsd(tmp[f'{net}_gap'], tmp['ms_score'], alpha=0.05)
        tukey_df = pd.DataFrame(tukey._results_table.data[1:], 
                                columns=tukey._results_table.data[0])
        tukey_df[['group1','group2']] = tukey_df[['group1','group2']].astype(float).astype(int)
        for _,row in tukey_df.iterrows():
            g1,g2 = int(row['group1']), int(row['group2'])
            if g1 not in ms_scores or g2 not in ms_scores: continue
            i,j = ms_scores.index(g1), ms_scores.index(g2)
            d,p,lo,up = row['meandiff'], row['p-adj'], row['lower'], row['upper']
            diff_matrix[i,j] = d; diff_matrix[j,i] = -d
            pval_matrix[i,j] = p; pval_matrix[j,i] = p
            lower_matrix[i,j] = lo; upper_matrix[i,j] = up
            lower_matrix[j,i] = -up; upper_matrix[j,i] = -lo
            sig = '*' if p<0.05 else ''
            sig_matrix[i,j] = sig; sig_matrix[j,i] = sig
    np.fill_diagonal(diff_matrix,0.0)
    np.fill_diagonal(lower_matrix,0.0)
    np.fill_diagonal(upper_matrix,0.0)
    diff_mats[net]=diff_matrix; pval_mats[net]=pval_matrix
    lower_mats[net]=lower_matrix; upper_mats[net]=upper_matrix; sig_mats[net]=sig_matrix

max_abs = max(np.abs(m).max() for m in diff_mats.values())
max_abs = max(max_abs, 0.8)
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

# ------------------------
# 2×4 panel
# ------------------------
fig, axes = plt.subplots(2,4, figsize=(28,12))
axes = axes.flatten()

for k,net in enumerate(network_names):
    ax = axes[k]
    diff_mat  = diff_mats[net]
    pval_mat  = pval_mats[net]
    lower_mat = lower_mats[net]
    upper_mat = upper_mats[net]
    sig_mat   = sig_mats[net]

    annot = np.empty(diff_mat.shape, dtype=object)
    for r in range(len(ms_scores)):
        for c in range(len(ms_scores)):
            if r==c:
                annot[r,c] = "—"
            else:
                diff_txt = f"{diff_mat[r,c]:.2f}{sig_mat[r,c]}"
                p_txt = format_p(pval_mat[r,c])
                ci_txt = f"[{lower_mat[r,c]:.2f}, {upper_mat[r,c]:.2f}]"
                annot[r,c] = f"{diff_txt}\n{p_txt}\n{ci_txt}"

    sns.heatmap(diff_mat, ax=ax, annot=annot, fmt='',
                cmap='coolwarm', norm=norm, center=0,
                linewidths=0.5, linecolor='gray',
                cbar=True, annot_kws={"fontsize":7, "linespacing":0.9},
                xticklabels=ms_scores, yticklabels=ms_scores)

    ax.set_title(net, fontsize=14)
    ax.set_xlabel("MetS Score")
    ax.set_ylabel("MetS Score")

plt.tight_layout()
plt.show()
