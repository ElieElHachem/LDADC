# Latent Dirichlet Allocation for Double Clustering (LDA-DC): Discovering patients phenotypes and cell populations within a single Bayesian framework

By Elie-Julien EL HACHEM<sup>1*</sup>, Nataliya SOKOLOVSKA<sup>1*</sup>, Hédi Soula<sup>1*</sup>

<sup>1.</sup> Sorbonne Université, INSERM, Nutrition and Obesities: systemic approaches, NutriOmique, 75013, Paris, France

<sup>*</sup> To whom correspondence should be addressed

Contact (elie-julien.el_hachem@sorbonne-universite.fr)

**Version 1.0.0**

## Abstract


Clinical routines rely more and more on ``omics'' data, from host and microbiota, as well as cytometry data. The variability of the measurements is very high, due to variability of cohorts, technical issues, and human subjects themselves are very strong source of heterogeneity. The underlying structure of these high-dimensional data is unknown. To make these complex data useful for real purposes, such as patients stratification and diagnostics, there is an acute need to develop novel statistical machine learning methods that are robust with respect to the data heterogeneity, efficient from the computational viewpoint, and can be understood by human experts.

## Results

 We propose a novel approach to stratify observations and huge-dimensional features within a single probabilistic framework, i.e., to identify patients phenotypes and cell types simultaneously. We define this problem as the double clustering problem, and we tackle it with the proposed approach. Our method is a practical extension of the Latent Dirichlet Allocation for the Double Clustering task (LDA-DC). First, we validate the method on artificial datasets, and second, we apply our method to two real problems of patients stratification from cytometry and microbiota data. We observe that the LDA-DC returns both identified cells populations and a very reasonable patients stratification. We also discuss graphical representation of the results which can be easily understood and is of a big help for human experts


## Real datasets

### **AML data** from  *(Aghaeepour et al., 2013)* 
Available at https://flowrepository.org under the accession FR-FCM-ZZYA. (Direct [link](https://flowrepository.org/id/FR-FCM-ZZYA))	

### **Microbiota**
- Cytometry data from *(Rubbens et al., 2021)*  are available at https://flowrepository.org under the accession FR-FCM-ZYVH. (Direct [link](https://flowrepository.org/id/FR-FCM-ZYVH))
- 16sRNA (genus data) from *(Vandeputte et al., 2017)* are available at [here](https://github.com/prubbens/PhenoGMM_CD/blob/master/Genus_tables/GenusAbundance_DiseaseCohort_nature24460.txt) (stored on github [PhenoGMM_CD](https://github.com/prubbens/PhenoGMM_CD) by [@prubbens](https://github.com/prubbens))

## Codes

### Simulated Data 
```
Simulated_data
├── Function_for_generate_data_and_LDADC.py
├── Simulation_for_2_phenotypes.py
└── Simulation_for_4_phenotypes.py
```
**Simulated_data** folder contain 2 scripts (Simulation_for_2_phenotypes.py and Simulation_for_4_phenotypes.py) to generate patients with different vectors. Function_for_generate_data_and_LDADC.py contain all the functions to generate patients for the analysis.
Note that you only need to put this three files to remake the synthethic simulation. 

### AML process
```
AML
├── AML_accuracy_tubes
│   ├── AML_file_for_analysis_discretized.py
│   └── function_to_use_for_AML_analysis.py 
└── AML_with_cluster_and_topics
    └── Analysis_file_AML.py
```
AML file contain two folder for analysis: 
1. **AML_accuracy_tubes** is for all tubes with two files : AML_file_for_analysis_discretized.py contain a script for the analysis and  function_to_use_for_AML_analysis.py contain all functions for the analysis
2. **AML_with_cluster_and_topics** folder contain a script and all the functions to perform analysis of words and project them on UMAP axis (Analysis_file_AML.py) 

All the results are stored in dataframe format in the Data_frame_LDA folder and figures are stored in Figures_LDA.
Please note that you need to update the path to aml_data for all the scripts

### Microbiota
```
Microbiota
├── Cytometry_data
│   ├── Multiple_run_with_network.py
│   ├── Notebook_adjusting_network_threshold.ipynb
│   └── Functions_for_analysis.py
├── GENUS_word_exploration
│   └── LDA_on_GENUS_with_normed_phi.py
└── Genus_LDA
    ├── LDA_on_GENUS.py
    └── Network_analysis.ipynb
```


Microbiota contain three folders:
1. **Cytometry_data** folder contain one script to make the analysis of the patients Multiple_run_with_network.py using cytometry data, and performing network stratification based on the number of topic. function_for_microbiota_analysis.py contain all the functions for the analysis and Notebook_adjusting_network_threshold.ipynb is a jupyter notebook to adjust the network.
2. **GENUS_word_exploration** is a folder with LDA_on_GENUS_with_normed_phi.py script and function allowing us to inspect words (i.e bacteria) involved in the prediction.
3. **Genus_LDA** contain LDA_on_GENUS.py allowing us to perform network stratification based on the number of topics using genus data. Its also contain Network_analysis.ipynb (a jupyter notebook) to adjust the network.

## Requirement

* Python 3
* Numpy
* fcsparser 0.2.4
* networkx 2.7.1
* Pandas
* Scikit-Learn
* Scipy
* Seaborn
* Tqdm
* Matplotlib
* Plotnine

## Computer characteristics

Simulation and data-analysis have been performed on:
``` Bash
CPU: Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz 
GPU: Nvidia QUADRO P5000 / CUDA version: 11.4
Ubuntu 20.04.3 LTS
```





