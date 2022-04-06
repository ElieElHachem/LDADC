from pickle import NONE
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd 
from glob import glob 
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
np.random.seed(5)
from function_to_use_for_AML_analysis import binatodeci,storage_position,kmeans_reattached_patient,LDA_function_makeblop_form,relabel_matrix
start_time = time.monotonic()
import umap


if __name__ == '__main__':
    dirs = '~/aml_data'
    information_frame = pd.read_csv('~/aml_data/attachments/AML.csv')
    figures_storage ,frame_storage = 'Figures_LDA' , 'Data_frame_LDA'
    path_to_store_figures,path_to_store_frame = storage_position(figures_storage ,frame_storage, abspath =  os.path.abspath(__file__) )

    compiled_information = {'Tube':[] ,'Accuracy':[]}

    type_of_tube = [4]
    cell_to_sample = 10000
    sample_of_normal = 50
    div_mat = 100
    phenotypes_patients = 2
    number_of_calculation = 200
    time_of_calculation = 1
    LE = LabelEncoder()


    for tube in tqdm(type_of_tube, leave=True,desc='Tube'):
        list_of_phi_frame = []

        for calculation in tqdm(range(time_of_calculation),leave=True,desc='Iteration'):

            compiled_information['Tube'].append(tube)

            ref_val = information_frame.loc[information_frame['Tube number'] == tube]

            list_of_object = (ref_val.loc[ref_val['Condition'] == 'normal']).sample(n = sample_of_normal,random_state=42)['FCS file'].to_list() + ref_val.loc[ref_val['Condition'] == 'aml']['FCS file'].to_list()
            files = []
            for object in list_of_object:
                file_of_object = glob(f'{dirs}/{object}')
                files.extend(file_of_object) 

            dataframe_for_LDA ,cell_labs_with_kmeans= kmeans_reattached_patient(cell_to_sample=cell_to_sample,files=files)
            norm_theta, norm_phi = LDA_function_makeblop_form(dataframe_for_LDA=dataframe_for_LDA,diviser_of_matrix = div_mat,runrun = phenotypes_patients,number_of_calculation=number_of_calculation, alpha = 1/number_of_calculation, beta = 1/number_of_calculation)
            dataframe_norm_pheno = pd.DataFrame(norm_phi,columns= [f'Topic {i}' for i in range(phenotypes_patients)])
            list_of_phi_frame.append(dataframe_norm_pheno)
            labelList = dataframe_for_LDA.T.columns.to_list()
            frame_binary_with_threshold = pd.DataFrame(norm_theta,index=labelList,columns= [f'Topic {i}' for i in range(phenotypes_patients)])
            class_patient = [binatodeci(x) for x in norm_theta]
            frame_binary_with_threshold['Patient Statut'] =  LE.fit_transform(class_patient) +1
            frame_binary_with_threshold['FCS file'] =  frame_binary_with_threshold.index
            frame_binary_with_threshold.reset_index(drop=True, inplace=True)
            global_frame = pd.merge(frame_binary_with_threshold, ref_val, on="FCS file")
            global_frame['Condition code'] = LE.fit_transform(global_frame['Condition']) +1
            frame_for_accuracy = global_frame.groupby(['Condition code','Patient Statut']).size().unstack(fill_value=0)
            current_mat, current_cols, current_lines = relabel_matrix(frame_for_accuracy.to_numpy())
            stacked_result_relabeled = pd.DataFrame(current_mat).stack()
            accuracy_dataframed_stacked = stacked_result_relabeled.index.repeat(stacked_result_relabeled).to_frame().reset_index(drop=True)
            calc_of_accuracy = accuracy_score(accuracy_dataframed_stacked.iloc[:,0], accuracy_dataframed_stacked.iloc[:,1])
            compiled_information['Accuracy'].append(calc_of_accuracy)

            if (tube == 4) & (calculation==0) & (calc_of_accuracy>0.93):

                value_for_filtering = 0.9
                sampling = 1000
                fig, axs = plt.subplots(figsize=(7, 5))
                dataframe_norm_pheno.sort_values(by = 'Topic 0', ascending = False).plot.bar(ax=axs)
                axs.tick_params(axis='both', which='major', labelsize=11)
                axs.set_xlabel('Cell type',size=15)
                axs.set_ylabel('Probability', size=15)
                fig.savefig(path_to_store_figures + f'/Barplot_cells_run_{sampling}_cell_by_pheno.svg', dpi = 1200)
                plt.close()

                data_norm_pheno_sorted = dataframe_norm_pheno.sort_values(by = 'Topic 0', ascending = False)
                data_norm_pheno_sorted.reset_index(inplace = True)
                topic_0_best_cell_types = data_norm_pheno_sorted[data_norm_pheno_sorted['Topic 0'] > value_for_filtering]['index'].to_list()
                topic_1_best_cell_types = data_norm_pheno_sorted[data_norm_pheno_sorted['Topic 1'] > value_for_filtering]['index'].to_list()
                filtered_dataframe = pd.concat([data_norm_pheno_sorted[data_norm_pheno_sorted['Topic 0'] > value_for_filtering] ,data_norm_pheno_sorted[data_norm_pheno_sorted['Topic 1'] > value_for_filtering]])
                cell_types_topic_one = filtered_dataframe['index'].to_list()
                frame_to_plot_one = cell_labs_with_kmeans[cell_labs_with_kmeans['K-means labels'].isin(cell_types_topic_one)]

                cell_types_one = frame_to_plot_one['K-means labels'].to_list()
                cell_origin = frame_to_plot_one ['Cell origin'].to_list()
                data_to_reduce_one = frame_to_plot_one.drop(columns=['K-means labels','Cell origin'])

                UMAP_data = umap.UMAP().fit_transform(data_to_reduce_one)
                data_frame_test_one = pd.DataFrame(UMAP_data, index = cell_types_one)
                data_frame_test_one.reset_index(inplace=True)
                data_frame_test_one.rename(columns={'index': 'Cell Phenotype'},inplace=True)
                sampled = data_frame_test_one.copy()
                
                cell_topic = []
                for row in sampled['Cell Phenotype']:
                    if row in topic_0_best_cell_types:
                        cell_topic.append(0)
                    elif row in topic_1_best_cell_types:
                        cell_topic.append(1)

                sampled['Topic attribution'] = cell_topic
                sampled['FCS file'] = cell_origin
                sampled['Cell Phenotype'] = 'N°' + sampled["Cell Phenotype"].astype(str)
                merged_frame = pd.merge(sampled,global_frame[['FCS file','Condition']], on= "FCS file")

                merged_frame = merged_frame.groupby('Cell Phenotype').apply(lambda x: x.sample(sampling, replace=True,random_state=42))
                fig, axes = plt.subplots( figsize=(5,5))
                sns.scatterplot(
                    data=merged_frame,
                    x=merged_frame.iloc[:,1], y=merged_frame.iloc[:,2], hue=merged_frame["Cell Phenotype"],s=4)

                axes.set_xlabel('UMAP 1',size=15)
                axes.set_ylabel('UMAP 2', size=15)
                axes.tick_params(axis='both', which='major', labelsize=12)
                axes.legend(loc = 'lower right', title = 'Cell Phenotype', title_fontsize = 8, bbox_to_anchor=(1, 0), ncol = 4, prop = {'size' : 6.5})
                plt.tight_layout()
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_{sampling}_cell_by_pheno.svg', dpi = 300, transparent=True)
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_{sampling}_cell_by_pheno.png', dpi=300, transparent=True) 
                plt.close()

                fig, axes = plt.subplots( figsize=(5,5))
                sns.scatterplot(
                    data=merged_frame,
                    x=merged_frame.iloc[:,1], y=merged_frame.iloc[:,2], hue=merged_frame["Topic attribution"],s=4)

                axes.set_xlabel('UMAP 1',size=15)
                axes.set_ylabel('UMAP 2', size=15)
                axes.tick_params(axis='both', which='major', labelsize=12)
                axes.legend(loc = 'lower right', title = 'Topic attribution', title_fontsize = 12, bbox_to_anchor=(1, 0), ncol = 4, prop = {'size' : 12})
                plt.tight_layout()
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_Topic_{sampling}_cell_by_Topic_attribution.svg', dpi = 300,transparent=True)
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_Topic_{sampling}_cell_by_Topic_attribution.png', dpi=300, transparent=True) 
                plt.close()

                fig, axes = plt.subplots( figsize=(5,5))
                sns.scatterplot(
                    data=merged_frame,
                    x=merged_frame.iloc[:,1], y=merged_frame.iloc[:,2], palette = ['#FA462F','#89AD07'],hue=merged_frame["Condition"],s=4)
                axes.set_xlabel('UMAP 1',size=15)
                axes.set_ylabel('UMAP 2', size=15)
                axes.tick_params(axis='both', which='major', labelsize=12)
                axes.legend(loc = 'lower right', title = 'Condition', title_fontsize = 12, bbox_to_anchor=(1, 0), ncol = 4, prop = {'size' : 12})
                plt.tight_layout()
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_Topic_{sampling}_cell_by_condition.svg', dpi = 300,transparent=True)
                fig.savefig(path_to_store_figures + f'/UMAP_plot_{tube}_Topic_{sampling}_cell_by_condition.png', dpi=300, transparent=True) 
                plt.close()

                data_frame_test_one.to_csv(path_to_store_frame + f'/Global_frame_for_UMAP_reduced_data_all_cell.csv')
                merged_frame.to_csv((path_to_store_frame + f'/Global_frame_for_UMAP_reduced_data_{sampling}_cell_sampled.csv'))

    dataframe_compiled = pd.DataFrame(compiled_information)
    dataframe_compiled.to_csv(path_to_store_frame + f'/Compiled_information_rounded_{time_of_calculation}_runs_all_tubes_without_HC.csv')


    sns.set_context("paper")
    plt.figure(figsize=(7, 5))
    barfig = sns.catplot(x="Tube", y="Accuracy", data=dataframe_compiled, kind="bar", ci='sd', capsize=.2)
    barfig = sns.swarmplot(x="Tube", y="Accuracy", color='black', s = 3.5, dodge=True,data=dataframe_compiled)
    barfig.set_title("Accuracy based on different experiences", size = 14)
    barfig.set_xlabel('Experience (Tube)', size=14)
    barfig.set_ylabel('Accuracy', size=14)
    plt.gcf().set_size_inches(7, 5)
    barfig.get_figure().savefig(path_to_store_figures + f'/barplot_rounded_{time_of_calculation}_run_without_HC.pdf', dpi = 1000, format = 'pdf', bbox_inches='tight')
    plt.close()