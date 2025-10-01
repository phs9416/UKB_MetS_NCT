

'''
def get_metabolic_components(row):
    components = []

    if (row['sex'] == 'Male' and row['Waist circumference'] >= 102) or \
       (row['sex'] == 'Female' and row['Waist circumference'] >= 88):
        components.append('Waist circumference')
    
    if row['Triglycerides'] >= 1.7:
        components.append('Triglycerides')
    
    if (row['sex'] == 'Male' and row['HDL'] < 1.0) or \
       (row['sex'] == 'Female' and row['HDL'] < 1.3):
        components.append('HDL')
    
    if row['Systolic BP'] >= 130 or row['Diastolic BP'] >= 85:
        components.append('BP')
    
    if row['HbA1c'] >= 39:
        components.append('HbA1c')
'''

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# --- Load data
df_node = pd.read_csv("controllability_normative.csv")
df_names = pd.read_csv("node_names.csv", header=None, names=["region_name"])
df_cohort = pd.read_csv("cohort.csv")

# --- Merge
merged = pd.merge(
    df_node,
    df_cohort[['eid', 'sex', 'ms_diagnosis', 'Waist circumference', 'HbA1c', 'HDL', 'Triglycerides', 'Systolic BP', 'Diastolic BP']],
    on='eid',
    how='inner'
)

# --- Build sex-specific waist masks (IMPORTANT: use & and | with parentheses)
m0 = (merged['ms_diagnosis'] == 0)
m1 = (merged['ms_diagnosis'] == 1)

def compute_effects(prefix: str, n_nodes: int = 216):
    cols = [f'{prefix}{i}' for i in range(n_nodes)]
    t_list = []
    p_list = []
    for col in cols:
        # pull groups
        g0 = merged.loc[m0, col].astype(float).to_numpy()
        g1 = merged.loc[m1, col].astype(float).to_numpy()

        # Welch t-test with NaN handling
        t_stat, p_val = ttest_ind(g0, g1, equal_var=False, nan_policy='omit')
        p_list.append(p_val)
        t_list.append(t_stat)

    # FDR correction (ignore NaN p-values)
    p_arr = np.array(p_list, dtype=float)
    p_fdr = np.full_like(p_arr, np.nan, dtype=float)
    finite_mask = np.isfinite(p_arr)
    if finite_mask.any():
        _, p_fdr_sub, _, _ = multipletests(p_arr[finite_mask], method="fdr_bh")
        p_fdr[finite_mask] = p_fdr_sub

    df_out = pd.DataFrame({
        "region_name": df_names["region_name"].iloc[:len(cols)].reset_index(drop=True),
        "t_statistics": t_list,
        "p_value": p_arr,
        "p_value_fdr": p_fdr
    })
    return df_out

# --- Compute for ac and mc
df_ac = compute_effects('ac', n_nodes=216)
df_mc = compute_effects('mc', n_nodes=216)

# --- Save
df_ac.to_csv('tstats_ac_mets.csv', index=False)
df_mc.to_csv('tstats_mc_mets.csv', index=False)

print("Saved: tstats_ac_mets.csv, tstats_mc_mets.csv")

import numpy as np
import nibabel as nib
import pandas as pd

# file path
atlas_path = "Schaefer200_Tian_S1.nii.gz"        # atlas NIfTI
node_names_path = "node_names.csv"             
tstats_csv_path = "tstats_mc_mets.csv"     ## ac, mc  # region_name, tstats 
out_nii_path = "tstats_mc_mets.nii.gz"     ## ac, mc
idval_out_path = "label_values_mc_mets.csv"   ## ac, mc
unmatched_out_path = "unmatched_names_mc_mets.csv"   ## ac, mc

# ------------------------
# 1) load data
# ------------------------
# atlas
atlas_img = nib.load(atlas_path)
labels = atlas_img.get_fdata().astype(int)
print(f"Atlas shape={labels.shape}, label max={labels.max()}")

# node_names 
node_names = pd.read_csv(node_names_path, header=None)
node_names = node_names.rename(columns={0: "region_name"})
node_names["_name_std"] = node_names["region_name"].astype(str).str.strip().str.lower()
node_names["id"] = np.arange(1, len(node_names)+1, dtype=int)

# tstats csv
tstats = pd.read_csv(tstats_csv_path)
assert "region_name" in tstats.columns, "tstats CSV: need region_name "
assert "t_statistics" in tstats.columns, "tstats CSV: need t_statistics "
tstats["_name_std"] = tstats["region_name"].astype(str).str.strip().str.lower()

# ------------------------
# 2) matching
# ------------------------
merged = node_names.merge(tstats[["_name_std","t_statistics"]], on="_name_std", how="left")

unmatched = merged[merged["t_statistics"].isna()][["region_name"]]
if len(unmatched) > 0:
    unmatched.to_csv(unmatched_out_path, index=False)
    print(f"[Warn] Unmatched={len(unmatched)} → {unmatched_out_path}")
else:
    print("[OK] all nodes are matched")

# ------------------------
# 3) id-value table generation
# ------------------------
idval = merged[["id","t_statistics"]].rename(columns={"t_statistics":"value"})
idval["value"] = pd.to_numeric(idval["value"], errors="coerce")
idval.to_csv(idval_out_path, index=False)
print(f"[OK] id-value table saved: {idval_out_path}")

# ------------------------
# 4) LUT & NIfTI 
# ------------------------
lab_max = int(labels.max())
LUT = np.zeros(lab_max+1, dtype=np.float32)
LUT[0] = 0.0
valid = idval[idval["id"] <= lab_max]
LUT[valid["id"].to_numpy()] = valid["value"].to_numpy(dtype=np.float32)

out_data = LUT[labels]
out_img = nib.Nifti1Image(out_data.astype(np.float32), affine=atlas_img.affine, header=atlas_img.header)
nib.save(out_img, out_nii_path)
print(f"[OK] tstats NIfTI saved: {out_nii_path}")

