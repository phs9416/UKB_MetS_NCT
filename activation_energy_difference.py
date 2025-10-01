import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -----------------------------
# 1) load data
# -----------------------------
activation_path = 'activation_normative.csv'
cohort_path = 'cohort.csv'

df = pd.read_csv(activation_path)
cohort_df = pd.read_csv(cohort_path)

merged_df = pd.merge(df, cohort_df[['eid', 'ms_diagnosis', 'ms_score']], 
                     on='eid', how='inner')

networks = ['CONT', 'DMN', 'DAN', 'LIM', 'VAN', 'SMN', 'SUB', 'VIS']
missing_nets = [c for c in networks if c not in merged_df.columns]

merged_df = merged_df.dropna(subset=['ms_score']).copy()
merged_df['ms_score'] = merged_df['ms_score'].astype(int)
merged_score_df = merged_df[['eid', 'ms_score'] + networks].copy()

# -----------------------------
# 2) ANOVA & Tukey HSD 
# -----------------------------
anova_results, tukey_results = {}, {}
for net in networks:
    model = ols(f'{net} ~ C(ms_score)', data=merged_score_df).fit()
    anova_results[net] = sm.stats.anova_lm(model, typ=2)
    tukey_results[net] = pairwise_tukeyhsd(
        endog=merged_score_df[net],
        groups=merged_score_df['ms_score'],
        alpha=0.05
    )

# -----------------------------
# 3) p-value
# -----------------------------
def format_p(p):
    if p < 0.001:
        return "P<.001"
    elif p < 0.01:
        return f"P={p:.3f}"
    elif p < 0.99:
        return f"P={p:.2f}"
    else:
        return "P>.99"

# -----------------------------
# 4) color scale
# -----------------------------
ms_levels = sorted(merged_score_df['ms_score'].unique().tolist())
vmin, vmax = float('inf'), float('-inf')
for net in networks:
    tukey_df = pd.DataFrame(tukey_results[net]._results_table.data[1:], 
                            columns=tukey_results[net]._results_table.data[0])
    diffs = tukey_df['meandiff'].astype(float).values
    if len(diffs) > 0:
        vmax = max(vmax, np.nanmax(np.abs(diffs)))
        vmin = min(vmin, -np.nanmax(np.abs(diffs)))
if not np.isfinite(vmin) or not np.isfinite(vmax):
    vmin, vmax = -1.0, 1.0

# -----------------------------
# 5) Heatmap (diff + * + CI + p)
# -----------------------------
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
axes = axes.flatten()

for i, net in enumerate(networks):
    tukey_df = pd.DataFrame(tukey_results[net]._results_table.data[1:], 
                            columns=tukey_results[net]._results_table.data[0])
    for col in ['group1','group2','meandiff','lower','upper','p-adj','reject']:
        if col in tukey_df.columns:
            if col in ['group1','group2']:
                tukey_df[col] = tukey_df[col].astype(float).astype(int)
            elif col == 'reject':
                tukey_df[col] = tukey_df[col].astype(bool)
            else:
                tukey_df[col] = tukey_df[col].astype(float)

    heatmap_matrix = pd.DataFrame(0.0, index=ms_levels, columns=ms_levels)
    annotation_matrix = pd.DataFrame('', index=ms_levels, columns=ms_levels)

    for g in ms_levels:
        annotation_matrix.loc[g, g] = "—"

    for _, row in tukey_df.iterrows():
        g1, g2 = int(row['group1']), int(row['group2'])
        diff, lower, upper, pval = row['meandiff'], row['lower'], row['upper'], row['p-adj']
        star = "*" if row['reject'] else ""
        p_text = format_p(pval)

        heatmap_matrix.loc[g1,g2] = diff
        heatmap_matrix.loc[g2,g1] = -diff

        annot_12 = f"{diff:.2f}{star}\n[{lower:.2f}, {upper:.2f}]\n{p_text}"
        annot_21 = f"{-diff:.2f}{star}\n[{(-upper):.2f}, {(-lower):.2f}]\n{p_text}"

        annotation_matrix.loc[g1,g2] = annot_12
        annotation_matrix.loc[g2,g1] = annot_21

    sns.heatmap(
        heatmap_matrix, ax=axes[i],
        annot=annotation_matrix, fmt='',
        cmap='coolwarm', center=0,
        vmin=vmin, vmax=vmax,
        linewidths=0.6, linecolor='gray',
        square=True, cbar=True,
        cbar_kws={"fraction":0.046, "pad":0.04, "aspect":25},
        annot_kws={"fontsize":8, "linespacing":0.9}
    )
    axes[i].set_title(net, fontsize=14, pad=10)
    axes[i].set_xticks(range(len(ms_levels)))
    axes[i].set_yticks(range(len(ms_levels)))
    axes[i].set_xticklabels(ms_levels, rotation=0)
    axes[i].set_yticklabels(ms_levels, rotation=0)
    axes[i].set_xlabel('MetS Score')
    axes[i].set_ylabel('MetS Score')

plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5)
plt.show()
