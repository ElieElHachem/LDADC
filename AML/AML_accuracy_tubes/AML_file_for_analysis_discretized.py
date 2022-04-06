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

dirs = '~/aml_data' #Path for AML_data
information_frame = pd.read_csv('~/aml_data/attachments/AML.csv')

figures_storage ,frame_storage = 'Figures_LDA' , 'Data_frame_LDA'
path_to_store_figures,path_to_store_frame = storage_position(figures_storage ,frame_storage, abspath =  os.path.abspath(__file__) )

compiled_information = {'Tube':[] ,'Accuracy':[]}

type_of_tube = [1,2,3,4,5,6,7,8]
cell_to_sample = 10000
sample_of_normal = 50
div_mat = 100
phenotypes_patients = 2
number_of_calculation = 200
time_of_calculation = 20
LE = LabelEncoder()

for tube in tqdm(type_of_tube, leave=True,desc='Tube'):
    for calculation in tqdm(range(time_of_calculation),leave=False,desc='Iteration'):

        compiled_information['Tube'].append(tube)
        ref_val = information_frame.loc[information_frame['Tube number'] == tube]
        list_of_object = (ref_val.loc[ref_val['Condition'] == 'normal']).sample(n = sample_of_normal)['FCS file'].to_list() + ref_val.loc[ref_val['Condition'] == 'aml']['FCS file'].to_list()
        files = []
        for object in list_of_object:
            file_of_object = glob(f'{dirs}/{object}')
            files.extend(file_of_object) 

        dataframe_for_LDA = kmeans_reattached_patient(cell_to_sample=cell_to_sample,files=files)
        norm_theta = LDA_function_makeblop_form(dataframe_for_LDA=dataframe_for_LDA,diviser_of_matrix = div_mat,runrun = phenotypes_patients,number_of_calculation=number_of_calculation, alpha = 1/number_of_calculation, beta = 1/number_of_calculation)
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
