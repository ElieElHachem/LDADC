import time
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
from glob import glob 
from tqdm import tqdm
from itertools import combinations_with_replacement
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
np.random.seed(5)
from function_for_microbiota_analysis import binatodeci,storage_position,kmeans_reattached_patient_microbiota,LDA_function_makeblop_form,relabel_matrix
start_time = time.monotonic()
LE = LabelEncoder()



def symmetrize(a):
    return a +a .T-np.diag(a.diagonal())


def LDA_runs_multiple_times(information_frame,list_phenotypes_patients,time_of_calculation,markers,cell_for_sampling,path,list_of_object,components=None,markers_to_drop=None):
    list_of_dataframe = []
    compiled_information = {'Run':[] ,'Accuracy':[]}
    for phenotypes_patients in tqdm(list_phenotypes_patients,desc='Phenotypes'):

        for calculation in tqdm(range(time_of_calculation),leave=False,desc='Iteration'):
                
            compiled_information['Run'].append(calculation)
            dataframe_for_LDA = kmeans_reattached_patient_microbiota(cell_to_sample=cell_for_sampling,path=path,list_patient=list_of_object, components= None,markers_to_drop= None, selected_markers=markers)
            
            norm_theta = LDA_function_makeblop_form(dataframe_for_LDA=dataframe_for_LDA,diviser_of_matrix = div_mat,runrun = phenotypes_patients,number_of_calculation=number_of_calculation, alpha = 1/number_of_calculation, beta = 1/number_of_calculation)

            labelList = dataframe_for_LDA.T.columns.to_list()
            frame_binary_with_threshold = pd.DataFrame(norm_theta,index=labelList,columns= [f'Topic {i}' for i in range(phenotypes_patients)])

            class_patient = [binatodeci(x) for x in norm_theta]
            frame_binary_with_threshold['Patient Statut'] =  LE.fit_transform(class_patient) +1

            frame_binary_with_threshold['Individual'] =  frame_binary_with_threshold.index
            frame_binary_with_threshold.reset_index(drop=True, inplace=True)

            global_frame = pd.merge(frame_binary_with_threshold, information_frame, on="Individual")
            global_frame['Condition code'] = LE.fit_transform(global_frame['Health status']) +1
            global_frame['Enterotype code'] = LE.fit_transform(global_frame['Enterotype']) +1

            if microbiota_test == True:
                frame_for_accuracy = global_frame.groupby(['Enterotype code','Patient Statut']).size().unstack(fill_value=0)
                current_mat, current_cols, current_lines = relabel_matrix(frame_for_accuracy.to_numpy())

                stacked_result_relabeled = pd.DataFrame(current_mat).stack()
                accuracy_dataframed_stacked = stacked_result_relabeled.index.repeat(stacked_result_relabeled).to_frame().reset_index(drop=True)

                calc_of_accuracy = accuracy_score(accuracy_dataframed_stacked.iloc[:,0], accuracy_dataframed_stacked.iloc[:,1])
                compiled_information['Accuracy'].append(calc_of_accuracy)


            else:
                frame_for_accuracy = global_frame.groupby(['Condition code','Patient Statut']).size().unstack(fill_value=0)
                current_mat, current_cols, current_lines = relabel_matrix(frame_for_accuracy.to_numpy())

                stacked_result_relabeled = pd.DataFrame(current_mat).stack()
                accuracy_dataframed_stacked = stacked_result_relabeled.index.repeat(stacked_result_relabeled).to_frame().reset_index(drop=True)

                calc_of_accuracy = accuracy_score(accuracy_dataframed_stacked.iloc[:,0], accuracy_dataframed_stacked.iloc[:,1])
                compiled_information['Accuracy'].append(calc_of_accuracy)
            
            list_of_dataframe.append(global_frame)
            global_frame.to_csv(path_to_store_frame_multiple_run + f'/Unique_frame_LDA_rounded_microbiota_unreduced_markers_drop_{len(markers)}_microbiota_{microbiota_test}_run_{calculation}_on_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv')
    
    pd.concat(list_of_dataframe).to_csv(path_to_store_frame + f'/Global_frame_LDA_rounded_microbiota_genus_microbiota_{microbiota_test}_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv')

    dataframe_compiled = pd.DataFrame(compiled_information)
    dataframe_compiled.to_csv(path_to_store_frame + f'/Accuracy_Compiled_information_LDA_rounded_microbiota_unreduced_markers_selected_genus_microbiota_{microbiota_test}_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv') 



def dataframe_cymetrize(topic_to_test,runs,path_LDA_to_analyse,path_to_store_frame,r):
    os.chdir(path_LDA_to_analyse)
    dataframe_for_topic = []
    for topic in topic_to_test:
        array_to_add = np.zeros((len(information_frame['Individual'].to_list())+1,len(information_frame['Individual'].to_list())+1))
        list_of_dataframe_for_individual = []
        for file in glob(f"*{runs}_runs_{topic}_topics.csv"):
            dataframe_to_append = []
            dataframe_topic =  pd.read_csv(path_LDA_to_analyse + f'/{file}', index_col=0)
            for top in range(1,topic+1):
                dataframe_to_append.append(pd.DataFrame({f'{top}':dataframe_topic.loc[(dataframe_topic['Patient Statut'] == top)]['Individual'].to_list()}))
            list_of_dataframe_for_individual.append(pd.concat(dataframe_to_append, axis=1))
        
        for run_topic_frame in list_of_dataframe_for_individual:
            list_of_columns = run_topic_frame.columns.to_list()
            patients_selected = []
            for column in list_of_columns:
                list_of_patient = run_topic_frame[column].dropna().to_list()
                patients_selected.extend(list_of_patient)

            patients_selected_number = [int(s.replace("DC", "")) for s in patients_selected]
            for column in list_of_columns:
                list_of_patient = run_topic_frame[column].dropna().to_list()
                list_of_patient_number_only = [int(s.replace("DC", "")) for s in list_of_patient]
                patient_iterated_number_only = list(combinations_with_replacement(list_of_patient_number_only, r))
                for patient in patient_iterated_number_only:
                    array_to_add[patient] += 1
        array_to_add = symmetrize(array_to_add)
        array_to_add = np.delete(array_to_add, 0, axis=1)
        array_to_add = np.delete(array_to_add, 0, axis=0)

        df_for_network = pd.DataFrame(array_to_add,columns=information_frame['Individual'].to_list(), index=information_frame['Individual'].to_list())
        dataframe_to_use = df_for_network.loc[:, (df_for_network != 0).any(axis=0)]
        dataframe_to_use = dataframe_to_use.loc[~(dataframe_to_use==0).all(axis=1)]
        dataframe_to_use.to_csv(path_to_store_frame + f'/Dataframe_for_network_{topic}_topic_{runs}_run.csv')
        dataframe_for_topic.append(dataframe_to_use)
    return dataframe_for_topic


if __name__ == '__main__':
    figures_storage ,frame_storage,frame_multiple_run = 'Figures_LDA_microbiota' , 'Data_frame_LDA_on_microbiota','Data_frame_multiple_run_LDA'
    path_to_store_figures,path_to_store_frame,path_to_store_frame_multiple_run = storage_position(figures_storage ,frame_storage,frame_multiple_run, abspath =  os.path.abspath(__file__) )

    path = '~/FlowRepository_FR-FCM-ZYVH_files/'
    cell_for_sampling = 10000
    information_frame = pd.read_csv('~/FlowRepository_FR-FCM-ZYVH_files/attachments/Metadata_DC.csv')
    #sample_of_normal = 29
    #list_of_object = (information_frame.loc[information_frame['Health status'] == 'Healthy']).sample(n = sample_of_normal)['Individual'].to_list() + information_frame.loc[information_frame['Health status'] == 'CD']['Individual'].to_list()
    list_of_object = information_frame['Individual'].to_list()
    div_mat = 100
    number_of_calculation = 200
    time_of_calculation = 100
    markers = ['FL1-H','FL3-H','FSC-H','SSC-H'] 
    microbiota_test = True
    list_phenotypes_patients = [4,8]
    r=2
    threshold_links = 0.35

    LDA_runs_multiple_times(information_frame=information_frame,list_phenotypes_patients=list_phenotypes_patients,time_of_calculation=time_of_calculation,markers=markers,cell_for_sampling=cell_for_sampling,path=path,list_of_object=list_of_object,components=None,markers_to_drop=None)
    dataframe_globals=dataframe_cymetrize(topic_to_test=list_phenotypes_patients,runs=time_of_calculation,path_LDA_to_analyse=path_to_store_frame_multiple_run,path_to_store_frame=path_to_store_frame,r=r)
 
    for phenotypes,dataframe_for_tops in tqdm(zip(list_phenotypes_patients,dataframe_globals),desc='Phenotypes_drawn'):
        dataframe_for_tops_norm = dataframe_for_tops/time_of_calculation
        dataframe_for_tops_norm[dataframe_for_tops_norm<threshold_links]=0

        #Drawn network
        G = nx.from_numpy_matrix(dataframe_for_tops_norm.values)
        G = nx.relabel_nodes(G, dict(enumerate(dataframe_for_tops_norm.columns)))
        my_pos = nx.spring_layout(G, seed = 100)

        information_frame['Health status binary'] = LabelEncoder().fit_transform(information_frame['Health status'])
        N_colors=2
        cm_dis=np.linspace(0, 0.8 ,N_colors) 
        colors = [cm.RdBu(x) for x in cm_dis]
        color_edges=[]

        plt.figure(figsize=(30, 20))
        for node in G:
            temp=information_frame.loc[information_frame['Individual']==node] #Finding time of node 
            
            color=colors[int(temp['Health status binary'])]
            if color not in color_edges:
                plt.scatter([],[],color=color, label=temp['Health status'].values[0])
            color_edges.append(color)

        weights = [20*(G[u][v]['weight'])**4 for u,v in G.edges()]

        d = dict(G.degree)
        nx.draw(G,pos = my_pos,with_labels=True,node_color=color_edges,node_size=[v * 100 for v in d.values()],width=weights)
        plt.legend(loc="lower left",fontsize=24)
        plt.tight_layout()
        plt.savefig(path_to_store_figures + f'/Network_of_patient_healthy_stats_{phenotypes}_topics_{time_of_calculation}_runs.svg', format = 'svg', bbox_inches='tight')
        plt.close()

        #Color by bacterioides
        information_frame['Enterotype binary'] = LabelEncoder().fit_transform(information_frame['Enterotype'])
        N_colors=4
        cm_dis=np.linspace(0, 0.8 ,N_colors) 
        colors = [ cm.tab20c(x) for x in cm_dis]
        color_edges=[]
        plt.figure(figsize=(30, 20))

        for node in G:
            temp=information_frame.loc[information_frame['Individual']==node] #Finding time of node 
            
            color=colors[int(temp['Enterotype binary'])]
            if color not in color_edges:
                plt.scatter([],[],color=color, label=temp['Enterotype'].values[0])
            color_edges.append(color)

        weights = [20*(G[u][v]['weight'])**4 for u,v in G.edges()]
        d = dict(G.degree)
        nx.draw(G,pos= my_pos, with_labels=False,node_color=color_edges,width=weights, node_size=[v * 100 for v in d.values()])
        plt.legend(loc="lower left",fontsize=24)
        plt.tight_layout()
        plt.savefig(path_to_store_figures + f'/Network_of_patient_microbiota_stats_{phenotypes}_topics_{time_of_calculation}_runs.svg', format = 'svg', bbox_inches='tight')
        plt.close()







