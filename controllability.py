import os
import numpy as np
import pandas as pd
from nctpy.utils import matrix_normalization
from nctpy.metrics import ave_control, modal_control
from nctpy.energies import get_control_inputs, integrate_u

base_dir = "unzip_file"
output_file = "controllability.csv"
results = []

for subject_dir in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_dir)
    if os.path.isdir(subject_path):
        subject_id = subject_dir.split("_")[0]
        csv_path = os.path.join(subject_path, "connectome_streamline_count_10M.csv")
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=",")
            B = np.array(data)
            b_norm = matrix_normalization(B, system='discrete')
            ac = ave_control(b_norm, system='discrete')
            mc = modal_control(b_norm)
            results.append([subject_id] + list(ac) + list(mc))

columns = ["subject_id"] + [f"ac_{i}" for i in range(216)] + [f"mc_{i}" for i in range(216)]
df = pd.DataFrame(results, columns=columns)
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
