import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def format_p(p):
    if pd.isna(p):
        return "P=NA"
    if p < 0.001:
        return "P<.001"
    elif p < 0.01:
        return f"P={p:.3f}"
    elif p < 0.99:
        return f"P={p:.2f}"
    else:
        return "P>.99"

# -----------------------------
# 1) load data
# -----------------------------
metabolic_path = "metabolic_normative.csv"
activation_path = "activation_normative.csv"

metabolic_data = pd.read_csv(metabolic_path)
activation_data = pd.read_csv(activation_path)

merged_data = pd.merge(metabolic_data, activation_data, on='eid', how='inner')

metabolic_cols = metabolic_data.columns.drop('eid')
activation_cols = activation_data.columns.drop('eid')

# -----------------------------
# 2) p_value calculation
# -----------------------------
corr = pd.DataFrame(index=metabolic_cols, columns=activation_cols, dtype=float)
pval = pd.DataFrame(index=metabolic_cols, columns=activation_cols, dtype=float)
ci_lo = pd.DataFrame(index=metabolic_cols, columns=activation_cols, dtype=float)
ci_hi = pd.DataFrame(index=metabolic_cols, columns=activation_cols, dtype=float)
n_mat = pd.DataFrame(index=metabolic_cols, columns=activation_cols, dtype=float)

for met in metabolic_cols:
    x_full = merged_data[met]
    for act in activation_cols:
        y_full = merged_data[act]

        # pairwise complete-case
        mask = x_full.notna() & y_full.notna()
        x = x_full[mask]
        y = y_full[mask]
        n = len(x)

        if n >= 4 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            r, p = pearsonr(x, y) 
            corr.loc[met, act] = r
            pval.loc[met, act] = p
            n_mat.loc[met, act] = n

            # Fisher z로 95% CI
            z = np.arctanh(r)
            se = 1.0 / np.sqrt(n - 3)
            z_ci = z + np.array([-1.96, 1.96]) * se
            r_ci = np.tanh(z_ci)
            ci_lo.loc[met, act] = r_ci[0]
            ci_hi.loc[met, act] = r_ci[1]
        else:
            corr.loc[met, act] = np.nan
            pval.loc[met, act] = np.nan
            n_mat.loc[met, act] = n
            ci_lo.loc[met, act] = np.nan
            ci_hi.loc[met, act] = np.nan

# -----------------------------
# 3) FDR correction
# -----------------------------
pvals_flat = pval.values.flatten()
valid_idx = ~np.isnan(pvals_flat)
pvals_valid = pvals_flat[valid_idx]

if pvals_valid.size > 0:
    _, pvals_corr_valid, _, _ = multipletests(pvals_valid, alpha=0.05, method='fdr_bh')
    pvals_corr_all = np.full_like(pvals_flat, np.nan, dtype=float)
    pvals_corr_all[valid_idx] = pvals_corr_valid
    fdr_pval = pd.DataFrame(
        pvals_corr_all.reshape(pval.shape),
        index=pval.index, columns=pval.columns
    )
else:
    fdr_pval = pval.copy()

# -----------------------------
# 4) (optional) metabolic feature subset
# -----------------------------
selected_metabolic = [
    "Waist circumference",
    "SBP",
    "DBP",
    "HbA1c",
    "Triglycerides",
    "HDL"
]

selected_metabolic = [m for m in selected_metabolic if m in corr.index]

corr_sub = corr.loc[selected_metabolic, activation_cols]
pval_sub = fdr_pval.loc[selected_metabolic, activation_cols]
ci_lo_sub = ci_lo.loc[selected_metabolic, activation_cols]
ci_hi_sub = ci_hi.loc[selected_metabolic, activation_cols]

# -----------------------------
# 5) heatmp annotation (r, 95% CI, FDR p)
# -----------------------------
annot = pd.DataFrame(index=corr_sub.index, columns=corr_sub.columns, dtype=str)

for met in corr_sub.index:
    for act in corr_sub.columns:
        r = corr_sub.loc[met, act]
        lo = ci_lo_sub.loc[met, act]
        hi = ci_hi_sub.loc[met, act]
        p = pval_sub.loc[met, act]

        p_str = format_p(p)
        star = "*" if (pd.notna(p) and p < 0.05) else ""
        r_str = "NA" if pd.isna(r) else f"{r:.2f}"
        ci_str = "[NA, NA]" if (pd.isna(lo) or pd.isna(hi)) else f"[{lo:.2f}, {hi:.2f}]"

        annot.loc[met, act] = f"{r_str}{star}\n{ci_str}\n{p_str}"

# -----------------------------
# 6) plot
# -----------------------------
plt.figure(figsize=(1.6*len(activation_cols)+4, 1.2*len(selected_metabolic)+3))

sns.heatmap(
    corr_sub.astype(float),
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    linecolor="white",
    annot=annot,
    fmt=""
    #cbar_kws={'label': 'Pearson r'}
)

#plt.title(
#    "Correlation between metabolic features and activation energy\n"
#    "(cells: r with 95% CI & FDR-adjusted P; * for FDR P<0.05)",
#    fontsize=12
#)
#plt.xlabel("Activation energy metrics")
#plt.ylabel("Metabolic features")

plt.tight_layout()
plt.show()
