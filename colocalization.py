###---------------PET analysis------------------- 

# import nispace workflow function
from nispace.workflows import simple_colocalization

# import imaging functions
from nilearn.image import load_img
from nilearn.plotting import view_img

# the download function is a simple function to download a file from a url and save it to a temporary directory
file_path = "tstats_ac_mets.nii.gz"             ## ac, mc

# load the map into memory
effect_map = load_img(file_path)

colocalization, p_values, p_fdr_values, nispace_object = simple_colocalization(
    y=effect_map, # our effect map, with the dictionary we can pass a name, can also be just the effect map
    x="pet", # the PET data, here, a pre-parcellated set of PET maps is automatically downloaded
    permute_kwargs={"p_tails": "upper"}, # This tells the function to calculate right-tailed p values according to our hypothesis
    parcellation="Schaefer200TianS1", # the parcellation to use, here, Schaefer200
    seed=42, # the seed to use for the permutation analysis, for reproducibility
    n_proc=4 # number if parallel processes to use when possible
)

###---------------cell type-specific analysis------------------- 

# general imports
import matplotlib.pyplot as plt
import pandas as pd

# import imaging functions
from nilearn.image import load_img
from nilearn.plotting import view_img

# the download function is a simple function to download a file from a url and save it to a temporary directory
file_path = "tstats_mc_mets.nii.gz"

# load the map into memory
effect_map = load_img(file_path)

from nispace.datasets import fetch_collection

fetch_collection("CellTypesPsychEncodeTPM", dataset="mrna")

from nispace.datasets import fetch_reference

# parcellation to use throughout the analysis
parcellation = "Schaefer200TianS1"

# showcase data
gene_maps = fetch_reference("mrna", parcellation=parcellation, collection="CellTypesPsychEncodeTPM")


df = gene_maps.reset_index()
df["set"] = df["set"].str.split().str[0]
gene_maps = df.set_index(["set", "map"])

# set some parameters
# put into a dictionary to pass to the following functions
kwargs = {
    "parcellation": parcellation, # the parcellation to use, defined above
    "x_collection": "CellTypesPsychEncodeTPM", # the collection to use, here, CellTypesPsychEncodeTPM
    "seed": 42, # the seed to use for the permutation analysis, for reproducibility
    "n_proc": 4, # number of parallel processes to use when possible
    "plot": True, # dont plot, we will compare below
}

from nispace.workflows import simple_xsea

colocalization_randomgenes, p_values_randomgenes, p_fdr_values_randomgenes, nispace_object_randomgenes = simple_xsea(
    y={"MetS": effect_map}, # our effect map, with the dictionary we can pass a name, can also be just the effect map
    x=gene_maps, # Allen Human Brain Atlas mRNA data extracted with the abagen package
    permute_sets=True, # permuting sets is not the default as you will see below, so we set it to True here
    **kwargs
)

colocalization_randominput, p_values_randominput, p_fdr_values_randominput, nispace_object_randominput = simple_xsea(
    y={"MetS": effect_map}, # our effect map, with the dictionary we can pass a name, can also be just the effect map
    x=gene_maps, # Allen Human Brain Atlas mRNA data extracted with the abagen package
    **kwargs
)

# Concatenate p values from both methods
p_values = pd.DataFrame(
    {
        "randomgenes": p_values_randomgenes.loc["MetS"],
        "randominput": p_values_randominput.loc["MetS"],
    }
)
# Display, sorted by the input map permutation results
p_values.sort_values(by="randominput")

# create figure to host both plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

# plot both plots
nispace_object_randomgenes.plot(ax=axes[0], show=False, title="Random gene set sampling", fig=fig)
nispace_object_randominput.plot(ax=axes[1], show=False, title="Permutation of input map", fig=fig)

# hide legend of first plot
axes[0].legend_.set_visible(False)

# adjust layout
fig.tight_layout()
