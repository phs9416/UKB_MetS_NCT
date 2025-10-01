import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pyls import behavioral_pls, meancentered_pls, pls_regression, save_results, load_results
from scipy.stats import spearmanr, linregress

df_activation = pd.read_csv('activation_normative.csv')
df_cohort = pd.read_csv('cohort.csv')
df = pd.merge(df_cohort, df_activation, on='eid', how='inner')
## df = df2[df2['age'].between(60, 70)] # (40, 50), (50, 60)

metabolic_column = ['Waist circumference', 'HDL', 'HbA1c', 'Triglycerides', 'Systolic BP', 'Diastolic BP']
network_column = ['CONT', 'DMN', 'DAN', 'LIM', 'VAN', 'SMN', 'SUB', 'VIS']

X = data[network_column]
Y = data[metabolic_column]

bpls = behavioral_pls(Y, X, n_boot=5000, n_perm=5000)

save_results("pls", bpls)
bpls = load_results("pls.hdf5")

sns.set_theme()

latent_variables_sig = pd.DataFrame({"Explained variance (%)":bpls["varexp"] * 100,"p-value":bpls["permres"]["pvals"]})

fig = plt.figure(figsize=(5,4))

ax = fig.add_subplot(111)
ax2 = ax.twinx()

# ax_color="#7693bc"
ax.plot(latent_variables_sig.index, latent_variables_sig["Explained variance (%)"], "o")
ax.set_xlabel("Latent variable")
ax.set_ylabel("Variance explained (%)")
ax.tick_params(axis="y")
ax.spines["top"].set_visible(False)
ax.set_ylim(-0.1,1)
ax.set_yticks(ax.get_yticks() * 100)
ax.locator_params(axis="y",nbins=4)

# ax2_color="darkgray"
ax2.plot(latent_variables_sig.index, latent_variables_sig["p-value"], "o", color='blue')
ax2.set_ylabel("p-value")
# ax2.spines["right"].set_color(ax2_color)
ax2.spines["right"].set_linewidth(3)
# ax2.spines["left"].set_color(ax_color)
ax2.spines["left"].set_linewidth(3)
ax2.tick_params(axis="y")
ax2.spines["top"].set_visible(False)
ax2.set_ylim(-0.1,1.1)
ax2.set_yticks([0.  , 0.4 , 0.8 , 0.05])
ax2.locator_params(axis="y",nbins=4)

plt.axhline(y=0.05, linestyle='dotted')

plt.tight_layout()

y_loadings = pd.DataFrame({"y_loadings":bpls["bootres"]["y_loadings"][:,0],
"y_loadings_ci_lower":bpls["bootres"]["y_loadings_ci"][:,0][:,0],
"y_loadings_ci_upper":bpls["bootres"]["y_loadings_ci"][:,0][:,1]})
y_loadings["y_loadings_ci_lower_offset"] = np.abs(y_loadings["y_loadings_ci_lower"] - y_loadings["y_loadings"])
y_loadings["y_loadings_ci_upper_offset"] = np.abs(y_loadings["y_loadings_ci_upper"] - y_loadings["y_loadings"])

# Plot vertical bar plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(
    x=network_column,
    height= y_loadings["y_loadings"],
    yerr=[y_loadings["y_loadings_ci_lower_offset"], y_loadings["y_loadings_ci_upper_offset"]],
    capsize=5,
    color='skyblue',
    edgecolor='black'
)

ax.bar_label(bars, fmt="%.2f", padding=3)
ax.axhline(0, color='gray', linewidth=1)
ax.set_ylabel("Covariance")
ax.set_xlabel("Brain Network")
ax.set_ylim(-0.20, 0.20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

scores_correlation = pd.DataFrame({"Imaging score":bpls["x_scores"][:,1],"Clinical score":bpls["y_scores"][:,1]}, index=X.index)

scores_correlation["Imaging score"] = scores_correlation["Imaging score"] + (np.nanmin(scores_correlation["Imaging score"]) * -1)
scores_correlation["Clinical score"] = scores_correlation["Clinical score"] + (np.nanmin(scores_correlation["Clinical score"]) * -1)

# Compute regression line
m, b = np.polyfit(scores_correlation["Imaging score"], scores_correlation["Clinical score"], 1)

# Compute Spearman correlation
corr, pval = spearmanr(scores_correlation["Imaging score"], scores_correlation["Clinical score"])
pval_text = "<0.001" if pval < 0.001 else f"{pval:.3f}"

# Plot
plt.figure(figsize=(6.5, 4))
sns.set(style="whitegrid")

ax = sns.regplot(
    data=scores_correlation,
    x="Imaging score",
    y="Clinical score",
    scatter_kws={'s': 35, 'alpha': 0.8, 'edgecolor': 'black'},
    line_kws={'color': 'red', 'lw': 2}
)

# Labels and annotation
plt.xlabel("Activation Energy PLS Score", fontsize=11)
plt.ylabel("Metabolic Feature PLS Score", fontsize=11)
plt.annotate(f"$r_{{sp}}$ = {corr:.3f}\n$p$ = {pval_text}",
             xy=(0.05, 0.90), xycoords="axes fraction",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

plt.tight_layout()
plt.show()
