import nimare
from pathlib import Path
from nilearn import plotting

data_dir = Path('/home/phs9416/UKB_NCT/neurosynth_data/data')
output_dir = Path('/home/phs9416/UKB_NCT/neurosynth_data/neurosynth_results')
output_dir.mkdir(parents=True, exist_ok=True)

# Neurosynth data loading
databases = nimare.extract.fetch_neurosynth(data_dir=str(data_dir))[0]
ds = nimare.io.convert_neurosynth_to_dataset(
    coordinates_file=databases['coordinates'],
    metadata_file=databases['metadata'],
    annotations_files=databases['features']
)
term_list = [
    'action', 'adaptation', 'addiction', 'anticipation', 'anxiety',
    'arousal', 'association', 'attention', 'autobiographical memory', 'balance',
    'belief', 'categorization', 'cognitive control', 'communication', 'competition',
    'concept', 'consciousness', 'consolidation', 'context', 'coordination',
    'decision', 'decision making', 'detection', 'discrimination', 'distraction',
    'eating', 'efficiency', 'effort', 'emotion', 'emotion regulation',
    'empathy', 'encoding', 'episodic memory', 'expectancy', 'expertise',
    'extinction', 'face recognition', 'facial expression', 'familiarity', 'fear',
    'fixation', 'focus', 'gaze', 'goal', 'hyperactivity',
    'imagery', 'impulsivity', 'induction', 'inference', 'inhibition',
    'insight', 'integration', 'intelligence', 'intention', 'interference',
    'judgment', 'knowledge', 'language', 'language comprehension', 'learning',
    'listening', 'localization', 'loss', 'maintenance', 'manipulation',
    'meaning', 'memory', 'memory retrieval', 'mental imagery', 'monitoring',
    'mood', 'morphology', 'motor control', 'movement', 'multisensory',
    'naming', 'navigation', 'object recognition', 'pain', 'perception',
    'planning', 'priming', 'psychosis', 'reading', 'reasoning',
    'recall', 'recognition', 'rehearsal', 'reinforcement learning', 'response inhibition',
    'response selection', 'retention', 'retrieval', 'reward anticipation', 'rhythm',
    'risk', 'rule', 'salience', 'search', 'selective attention',
    'semantic memory', 'sentence comprehension', 'skill', 'sleep', 'social cognition',
    'spatial attention', 'speech perception', 'speech production', 'strategy', 'strength',
    'stress', 'sustained attention', 'task difficulty', 'thought', 'uncertainty',
    'updating', 'utility', 'valence', 'verbal fluency', 'visual attention',
    'visual perception', 'word recognition', 'working memory'
]

prefixed_terms = [f"terms_abstract_tfidf__{term}" for term in term_list]

for term in prefixed_terms:
    try:
        term_name = term.split('__')[1]
        print(f"▶ Processing term: {term_name}")

        term_ids = ds.get_studies_by_label(labels=term, label_threshold=0.001)
        if len(term_ids) < 5:
            print(f"  ⤷ skip due to few study ({len(term_ids)})")
            continue

        notterm_ids = sorted(list(set(ds.ids) - set(term_ids)))
        term_dset = ds.slice(term_ids)
        notterm_dset = ds.slice(notterm_ids)

        meta = nimare.meta.cbma.mkda.MKDAChi2()
        results = meta.fit(term_dset, notterm_dset)
        results.save_maps(output_dir=output_dir, prefix=term_name)

        corrector = nimare.correct.FDRCorrector(alpha=0.01)
        results_corrected = corrector.transform(results)
        results_corrected.save_maps(output_dir=output_dir, prefix=term_name)

    except Exception as e:
        print(f"  ⤷ error: {e}")
        continue

print("✅ completed.")

labels = [
    "HIP-rh", "AMY-rh", "pTHA-rh", "aTHA-rh", "NAc-rh", "GP-rh", "PUT-rh", "CAU-rh",
    "HIP-lh", "AMY-lh", "pTHA-lh", "aTHA-lh", "NAc-lh", "GP-lh", "PUT-lh", "CAU-lh",
    "7Networks_LH_Vis_1", "7Networks_LH_Vis_2", "7Networks_LH_Vis_3", "7Networks_LH_Vis_4",
    "7Networks_LH_Vis_5", "7Networks_LH_Vis_6", "7Networks_LH_Vis_7", "7Networks_LH_Vis_8",
    "7Networks_LH_Vis_9", "7Networks_LH_Vis_10", "7Networks_LH_Vis_11", "7Networks_LH_Vis_12",
    "7Networks_LH_Vis_13", "7Networks_LH_Vis_14", "7Networks_LH_SomMot_1", "7Networks_LH_SomMot_2",
    "7Networks_LH_SomMot_3", "7Networks_LH_SomMot_4", "7Networks_LH_SomMot_5", "7Networks_LH_SomMot_6",
    "7Networks_LH_SomMot_7", "7Networks_LH_SomMot_8", "7Networks_LH_SomMot_9", "7Networks_LH_SomMot_10",
    "7Networks_LH_SomMot_11", "7Networks_LH_SomMot_12", "7Networks_LH_SomMot_13", "7Networks_LH_SomMot_14",
    "7Networks_LH_SomMot_15", "7Networks_LH_SomMot_16", "7Networks_LH_DorsAttn_Post_1", "7Networks_LH_DorsAttn_Post_2",
    "7Networks_LH_DorsAttn_Post_3", "7Networks_LH_DorsAttn_Post_4", "7Networks_LH_DorsAttn_Post_5",
    "7Networks_LH_DorsAttn_Post_6", "7Networks_LH_DorsAttn_Post_7", "7Networks_LH_DorsAttn_Post_8",
    "7Networks_LH_DorsAttn_Post_9", "7Networks_LH_DorsAttn_Post_10", "7Networks_LH_DorsAttn_FEF_1",
    "7Networks_LH_DorsAttn_FEF_2", "7Networks_LH_DorsAttn_PrCv_1", "7Networks_LH_SalVentAttn_ParOper_1",
    "7Networks_LH_SalVentAttn_ParOper_2", "7Networks_LH_SalVentAttn_ParOper_3", "7Networks_LH_SalVentAttn_FrOperIns_1",
    "7Networks_LH_SalVentAttn_FrOperIns_2", "7Networks_LH_SalVentAttn_FrOperIns_3", "7Networks_LH_SalVentAttn_FrOperIns_4",
    "7Networks_LH_SalVentAttn_PFCl_1", "7Networks_LH_SalVentAttn_Med_1", "7Networks_LH_SalVentAttn_Med_2",
    "7Networks_LH_SalVentAttn_Med_3", "7Networks_LH_Limbic_OFC_1", "7Networks_LH_Limbic_OFC_2",
    "7Networks_LH_Limbic_TempPole_1", "7Networks_LH_Limbic_TempPole_2", "7Networks_LH_Limbic_TempPole_3",
    "7Networks_LH_Limbic_TempPole_4", "7Networks_LH_Cont_Par_1", "7Networks_LH_Cont_Par_2",
    "7Networks_LH_Cont_Par_3", "7Networks_LH_Cont_Temp_1", "7Networks_LH_Cont_OFC_1", "7Networks_LH_Cont_PFCl_1",
    "7Networks_LH_Cont_PFCl_2", "7Networks_LH_Cont_PFCl_3", "7Networks_LH_Cont_PFCl_4", "7Networks_LH_Cont_PFCl_5",
    "7Networks_LH_Cont_pCun_1", "7Networks_LH_Cont_Cing_1", "7Networks_LH_Cont_Cing_2", "7Networks_LH_Default_Temp_1",
    "7Networks_LH_Default_Temp_2", "7Networks_LH_Default_Temp_3", "7Networks_LH_Default_Temp_4",
    "7Networks_LH_Default_Temp_5", "7Networks_LH_Default_Par_1", "7Networks_LH_Default_Par_2",
    "7Networks_LH_Default_Par_3", "7Networks_LH_Default_Par_4", "7Networks_LH_Default_PFC_1",
    "7Networks_LH_Default_PFC_2", "7Networks_LH_Default_PFC_3", "7Networks_LH_Default_PFC_4",
    "7Networks_LH_Default_PFC_5", "7Networks_LH_Default_PFC_6", "7Networks_LH_Default_PFC_7",
    "7Networks_LH_Default_PFC_8", "7Networks_LH_Default_PFC_9", "7Networks_LH_Default_PFC_10",
    "7Networks_LH_Default_PFC_11", "7Networks_LH_Default_PFC_12", "7Networks_LH_Default_PFC_13",
    "7Networks_LH_Default_pCunPCC_1", "7Networks_LH_Default_pCunPCC_2", "7Networks_LH_Default_pCunPCC_3",
    "7Networks_LH_Default_pCunPCC_4", "7Networks_LH_Default_PHC_1", "7Networks_RH_Vis_1", "7Networks_RH_Vis_2",
    "7Networks_RH_Vis_3", "7Networks_RH_Vis_4", "7Networks_RH_Vis_5", "7Networks_RH_Vis_6", "7Networks_RH_Vis_7",
    "7Networks_RH_Vis_8", "7Networks_RH_Vis_9", "7Networks_RH_Vis_10", "7Networks_RH_Vis_11", "7Networks_RH_Vis_12",
    "7Networks_RH_Vis_13", "7Networks_RH_Vis_14", "7Networks_RH_Vis_15", "7Networks_RH_SomMot_1", "7Networks_RH_SomMot_2",
    "7Networks_RH_SomMot_3", "7Networks_RH_SomMot_4", "7Networks_RH_SomMot_5", "7Networks_RH_SomMot_6",
    "7Networks_RH_SomMot_7", "7Networks_RH_SomMot_8", "7Networks_RH_SomMot_9", "7Networks_RH_SomMot_10",
    "7Networks_RH_SomMot_11", "7Networks_RH_SomMot_12", "7Networks_RH_SomMot_13", "7Networks_RH_SomMot_14",
    "7Networks_RH_SomMot_15", "7Networks_RH_SomMot_16", "7Networks_RH_SomMot_17", "7Networks_RH_SomMot_18",
    "7Networks_RH_SomMot_19", "7Networks_RH_DorsAttn_Post_1", "7Networks_RH_DorsAttn_Post_2",
    "7Networks_RH_DorsAttn_Post_3", "7Networks_RH_DorsAttn_Post_4", "7Networks_RH_DorsAttn_Post_5",
    "7Networks_RH_DorsAttn_Post_6", "7Networks_RH_DorsAttn_Post_7", "7Networks_RH_DorsAttn_Post_8",
    "7Networks_RH_DorsAttn_Post_9", "7Networks_RH_DorsAttn_Post_10", "7Networks_RH_DorsAttn_FEF_1",
    "7Networks_RH_DorsAttn_FEF_2", "7Networks_RH_DorsAttn_PrCv_1", "7Networks_RH_SalVentAttn_TempOccPar_1",
    "7Networks_RH_SalVentAttn_TempOccPar_2", "7Networks_RH_SalVentAttn_TempOccPar_3", "7Networks_RH_SalVentAttn_PrC_1",
    "7Networks_RH_SalVentAttn_FrOperIns_1", "7Networks_RH_SalVentAttn_FrOperIns_2", "7Networks_RH_SalVentAttn_FrOperIns_3",
    "7Networks_RH_SalVentAttn_FrOperIns_4", "7Networks_RH_SalVentAttn_Med_1", "7Networks_RH_SalVentAttn_Med_2",
    "7Networks_RH_SalVentAttn_Med_3", "7Networks_RH_Limbic_OFC_1", "7Networks_RH_Limbic_OFC_2",
    "7Networks_RH_Limbic_OFC_3", "7Networks_RH_Limbic_TempPole_1", "7Networks_RH_Limbic_TempPole_2",
    "7Networks_RH_Limbic_TempPole_3", "7Networks_RH_Cont_Par_1", "7Networks_RH_Cont_Par_2",
    "7Networks_RH_Cont_Par_3", "7Networks_RH_Cont_Temp_1", "7Networks_RH_Cont_PFCv_1", "7Networks_RH_Cont_PFCl_1",
    "7Networks_RH_Cont_PFCl_2", "7Networks_RH_Cont_PFCl_3", "7Networks_RH_Cont_PFCl_4", "7Networks_RH_Cont_PFCl_5",
    "7Networks_RH_Cont_PFCl_6", "7Networks_RH_Cont_PFCl_7", "7Networks_RH_Cont_pCun_1", "7Networks_RH_Cont_Cing_1",
    "7Networks_RH_Cont_Cing_2", "7Networks_RH_Cont_PFCmp_1", "7Networks_RH_Cont_PFCmp_2",
    "7Networks_RH_Default_Par_1", "7Networks_RH_Default_Par_2", "7Networks_RH_Default_Par_3",
    "7Networks_RH_Default_Temp_1", "7Networks_RH_Default_Temp_2", "7Networks_RH_Default_Temp_3",
    "7Networks_RH_Default_Temp_4", "7Networks_RH_Default_Temp_5", "7Networks_RH_Default_PFCv_1",
    "7Networks_RH_Default_PFCdPFCm_1", "7Networks_RH_Default_PFCdPFCm_2", "7Networks_RH_Default_PFCdPFCm_3",
    "7Networks_RH_Default_PFCdPFCm_4", "7Networks_RH_Default_PFCdPFCm_5", "7Networks_RH_Default_PFCdPFCm_6",
    "7Networks_RH_Default_PFCdPFCm_7", "7Networks_RH_Default_pCunPCC_1", "7Networks_RH_Default_pCunPCC_2",
    "7Networks_RH_Default_pCunPCC_3"
]

data = pd.DataFrame(index=labels)

atlas = '/home/phs9416/UKB_NCT/neurosynth_data/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_1mm.nii.gz'
mask = NiftiLabelsMasker(atlas, resampling_target='data')

map_dir = '/home/phs9416/UKB_NCT/neurosynth_data/neurosynth_results'

# parcellation
for term in term_list:
    nii_file = os.path.join(map_dir, f'{term}_z_desc-association_level-voxel_corr-FDR_method-indep.nii.gz')
    if os.path.exists(nii_file):
        print(f'Processing: {term}')
        img = check_niimg(nii_file, atleast_4d=True)
        vec = mask.fit_transform(img).squeeze()
        data[term] = vec
    else:
        print(f'File not found: {nii_file}')
        data[term] = np.nan  # 해당 term 결과 없을 시 NaN으로 채움

# 최종 CSV 저장
data.to_csv('cognitive_state_map_FDR.csv', sep=',')
