This repository provides scripts to study how metabolic syndrome (MetS) affects brain network control, aging, and cognition using UK Biobank connectomes.

# 1. Download connectomes
bash scripts/UKB_connectome_download.sh 

# 2. Compute activation energy and controllability
python scripts/activation_energy.py 
python scripts/controllability.py 

# 3. Normative modeling
python scripts/normative_modeling.py

# 4. Analyze effects of MetS
python scripts/activation_energy_difference.py 
python scripts/brain_age_gap.py 

# 5. Link with metabolic indicators
python scripts/multiple_regression.py 
python scripts/PLSC.py 

# 6. Cognitive vulnerability & validation
python scripts/cognitive_map_generation.py 
python scripts/cognitive_test.py 

# 7. Colocalization
python scripts/tmap_generation.py 
python scripts/colocalization.py 
