import os
import numpy as np
import pandas as pd
from nctpy.utils import matrix_normalization, normalize_state
from nctpy.metrics import ave_control, modal_control
from nctpy.energies import get_control_inputs, integrate_u

base_dir = "unzip_file"
output_file = "transition_energy.csv"
state_csv = pd.read_csv('brain_state_network.csv')
state = state_csv.to_numpy()
results = []
n_nodes = 216

system = 'continuous'
T = 1  
rho = 1  
B = np.eye(n_nodes)  
S = np.eye(n_nodes)

for subject_dir in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_dir)
    if os.path.isdir(subject_path):
        subject_id = subject_dir.split("_")[0]
        csv_path = os.path.join(subject_path, "connectome_streamline_count_10M.csv")
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=",")
            A_norm = matrix_normalization(A=data, c=1, system=system)
            x0 = state[:, -1].reshape(-1, 1)
            energy_matrix = []
            for i in range(8):
                xf = state[:, i].reshape(-1, 1)
                xf_norm = normalize_state(xf)
                x, u, n_err = get_control_inputs(A_norm=A_norm, T=T, B=B, x0=x0, xf=xf_norm, system=system, rho=rho, S=S)
                node_energy = integrate_u(u)
                energy_matrix.append(np.sum(node_energy))
            results.append([subject_id] + list(energy_matrix))
            print(f"{subject_id} completed")

columns = ["subject_id"] + ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Subcortical']
df = pd.DataFrame(results, columns=columns)
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
