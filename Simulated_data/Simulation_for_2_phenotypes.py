import numpy as np
import os
import time
import random
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from Function_for_generate_data_and_LDADC import test_extreme_porba_v2,test_blob_left_right_proba,relabel_matrix,continuous_data_files_concat_kmeans_reattached_V2,storage_position,LDA_function_makeblop_form
from plotnine import *
import warnings
warnings.filterwarnings('ignore')
start_time = time.monotonic()
np.random.seed(5)
random.seed(5)

#Fix Directory
patient_frame_to_store = 'Patient_generated'

#Chose to work on continuous data
figures_storage ,frame_storage = 'Figures_LDA_on_data_generated' , 'Data_frame_LDA_on_data_generated'
path_to_store_figures,path_to_store_frame = storage_position(patient_frame_to_store,figures_storage ,frame_storage, abspath =  os.path.abspath(__file__) )

#Generate patients characteristics
number_of_patient = 100
number_of_cell = 10000
number_of_run = 10
test_std = [2,1,0.5,0.01]
dimension_to_test = [2,4] #number of dimension
proba_first_array = 0.8
pats = 0.05
n_pheno = 2
theta = 10


for dim in tqdm(dimension_to_test, leave = True ,desc='Dimension'):
    clusters_to_test = [2**dim]
    for cluters in tqdm(clusters_to_test, leave = True,desc='Cluster'):
        compiled_information = {'Dimension':[] ,'Number of Phenotypes':[],'Vectors of probability':[],'Std value': [],'Accuracy':[]}

        proba_to_test = test_extreme_porba_v2(first_array=proba_first_array,dimension=dim,c=pats)

        for value in tqdm(test_std,leave= True, desc = 'Std Value'): 
            for i in tqdm(range(number_of_run), desc='Run',leave = False):
                for proba in proba_to_test:
                    compiled_information['Std value'].append(value)
                    compiled_information['Dimension'].append(dim)
                    compiled_information['Number of Phenotypes'].append(n_pheno)

                    #Clustering possibility
                    number_of_cluster_k_means = cluters #Number_of_cluster_for_k_means

                    #LDA characteristics
                    diviser_of_matrix = 100
                    number_of_calculation = 50
                    cutting_tree = 1

                    #Lets Generate Patients and store them
                    phenotype_code , phenotype_combination = test_blob_left_right_proba(proba=proba,number_of_patient = number_of_patient,n_pheno = n_pheno,n_dim=dim,theta = theta,number_of_cell= number_of_cell, patient_frame_to_store=patient_frame_to_store, value=value, return_combination = True)
                    phenotype_code_frame = pd.DataFrame(phenotype_code)
                    compiled_information['Vectors of probability'].append(phenotype_combination)


                    #Now we cluster cells from different patients
                    full_dataframe , dataframe_for_LDA_to_use = continuous_data_files_concat_kmeans_reattached_V2(path_to_store_frame = patient_frame_to_store,number_of_cluster = number_of_cluster_k_means, number_of_cell = number_of_cell)

                    #Generate LDA
                    runrun = n_pheno #if you want to use len of vector = number of cluster
                    accuracy_score_store = []
                    norm_theta = LDA_function_makeblop_form(dataframe_for_LDA = dataframe_for_LDA_to_use,diviser_of_matrix = diviser_of_matrix,runrun = runrun,number_of_calculation = number_of_calculation, alpha= 1/number_of_calculation, beta=1/number_of_calculation)
                    pd.DataFrame(norm_theta).to_csv(path_to_store_frame + f'/Norm_Theta_frame_{runrun}_{n_pheno}_phenotypes.csv')
                    linked = linkage(norm_theta, method='complete', metric='euclidean', optimal_ordering = True)
                    labelList = dataframe_for_LDA_to_use.T.columns.to_list()
                    hierarchical_result = fcluster(linked, cutting_tree*norm_theta.max(),'distance')
                    frame_binary_with_threshold = pd.DataFrame(norm_theta,index=labelList,columns= [f'Topic {i}' for i in range(runrun)])
                    frame_binary_with_threshold['Patient_Number'] =  frame_binary_with_threshold.index
                    frame_binary_with_threshold.reset_index(drop=True, inplace=True)
                    frame_binary_with_threshold['Hierarchical_clustering'] = hierarchical_result
                    global_frame = pd.merge(frame_binary_with_threshold, phenotype_code_frame, on="Patient_Number")
                    global_frame.to_csv(path_to_store_frame + f'/Global_Dataframe_{runrun}_{n_pheno}_phenotypes_subset_selected.csv')
                    frame_for_accuracy = global_frame.groupby(['Phenotype_Number','Hierarchical_clustering']).size().unstack(fill_value=0)
                    frame_for_accuracy.to_csv(path_to_store_frame + f'/Dataframe_for_accuracy_{runrun}_{n_pheno}_phenotypes.csv')

                    #Relabeling
                    current_mat, current_cols, current_lines = relabel_matrix(frame_for_accuracy.to_numpy())

                    #Process for  accuracy
                    stacked_result_relabeled = pd.DataFrame(current_mat).stack()
                    accuracy_dataframed_stacked = stacked_result_relabeled.index.repeat(stacked_result_relabeled).to_frame().reset_index(drop=True)
                    calc_of_accuracy = accuracy_score(accuracy_dataframed_stacked.iloc[:,0], accuracy_dataframed_stacked.iloc[:,1])
                    compiled_information['Accuracy'].append(calc_of_accuracy)

                dataframe_compiled = pd.DataFrame(compiled_information)

            dataframe_compiled = pd.DataFrame(compiled_information)
            euc_dist_list = []
            for i in dataframe_compiled['Vectors of probability']:
                euc_dist = euclidean(*i)
                euc_dist_list.append(euc_dist)

            dataframe_compiled['Euclidean Distance'] = euc_dist_list
            dataframe_compiled['Euclidean Distance'] = dataframe_compiled['Euclidean Distance'].round(3)
            dataframe_compiled.to_csv(path_to_store_frame + f'/Compiled_information_for_{dim}_dim_tested_multiple_dim_{clusters_to_test}_clusters_with_eclidean_dist_subseted_{n_pheno}_phenotypes.csv')


        df_for_average = dataframe_compiled.groupby(['Euclidean Distance','Std value']).mean().reset_index()
        df_for_average['Accuracy_std'] = dataframe_compiled.groupby(['Euclidean Distance','Std value']).sem(ddof=1).reset_index()['Accuracy']
        df_for_average["Std value selected"] = 'std ='+ df_for_average["Std value"].astype(str)

        plot_average = ggplot(df_for_average, aes(x='Euclidean Distance', y='Accuracy')) +\
            geom_line(aes(color = 'Std value selected')) +\
            geom_point() +\
            scale_x_continuous(limits=(0, 1.2)) +\
            scale_y_continuous(limits=(0.4, 1)) +\
            geom_errorbar(aes(ymin = df_for_average['Accuracy'] - df_for_average['Accuracy_std'] , ymax =  df_for_average['Accuracy'] + df_for_average['Accuracy_std'], color = 'Std value selected'), position = position_dodge(0), width = 0.044) +\
            ggtitle("Accuracy plot for multiple Euclidean Distance and different std values")
        ggsave(plot_average,path_to_store_figures + f'/Mean_Accuracy_plot_for_multiple_different std {dim} dim {number_of_run} run and {number_of_cluster_k_means} clusters and {number_of_cell} cells wth zscore_subseted_{n_pheno}_phenotypes_scaled_v2.svg',format='svg' , dpi = 1200)

