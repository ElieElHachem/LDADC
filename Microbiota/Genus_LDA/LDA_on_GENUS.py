import seaborn as sns
import numpy as np
import pandas as pd
from random import choices
from collections import Counter
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
np.random.seed(5)




def storage_position(figures_storage,frame_storage,frame_multiple_run, abspath):
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path = os.getcwd()
    PATH_storage = path
    print(PATH_storage)

    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path = os.getcwd()
    PATH_storage = path
    print(PATH_storage)

    if not os.path.exists(figures_storage):
        os.makedirs(figures_storage)
    path_to_store_figures = os.path.join(PATH_storage, figures_storage)

    if not os.path.exists(frame_storage):
        os.makedirs(frame_storage)
    path_to_store_frame = os.path.join(PATH_storage, frame_storage)

    if not os.path.exists(frame_multiple_run):
        os.makedirs(frame_multiple_run)
    path_to_store_frame_multiple_run = os.path.join(PATH_storage, frame_multiple_run)

    return path_to_store_figures,path_to_store_frame,path_to_store_frame_multiple_run

def patient_genus_cell(dataframe,number_of_cell_to_sample):
    list_of_series = []
    patient = []
    for index, row in dataframe.iterrows():
        patient.append(index)
        million_samples = choices(dataframe.columns, row.values,k=number_of_cell_to_sample)
        from collections import Counter
        list_of_series.append(pd.Series(Counter(million_samples)))
    return list_of_series,patient



def LDA_function_makeblop_form(dataframe_for_LDA,diviser_of_matrix,runrun,number_of_calculation, alpha, beta):
    vocabulary  =  list(range(dataframe_for_LDA.shape[1]))
    raw_data_T4 = dataframe_for_LDA.T
    data_T4 = raw_data_T4.T
    t4_ = data_T4.to_numpy()/diviser_of_matrix

    #t4_ = np.around(t4_)
    docs = []
    npatients, nvocabulary = t4_.shape
    for n in range (npatients):
        current_doc = []
        doc = t4_[n,:]
        for i in range(nvocabulary):
            for _ in range(int(doc[i])):
                current_doc.append(i)
        docs.append(current_doc)
                
            

    D = len(docs)        # number of documents
    V = len(vocabulary)  # size of the vocabulary 
    T = runrun            # number of topics

    # the parameter of the Dirichlet prior on the per-document topic distributions  #Faire varier
    # the parameter of the Dirichlet prior on the per-topic word distribution


    z_d_n = [[0 for _ in range(len(d))] for d in docs]  # z_i_j
    theta_d_z = np.zeros((D, T))
    phi_z_w = np.zeros((T, V))
    n_d = np.zeros((D))
    n_z = np.zeros((T))

    ## Initialize the parameters
    # m: doc id
    for d, doc in enumerate(docs):  
        # n: id of word inside document, w: id of the word globally
        for n, w in enumerate(doc):
            # assign a topic randomly to words
            z_d_n[d][n] = int(np.random.randint(T))
            # get the topic for word n in document m
            z = z_d_n[d][n]
            # keep track of our counts
            theta_d_z[d][z] += 1
            phi_z_w[z, w] += 1
            n_z[z] += 1
            n_d[d] += 1

    #for iteration in range(number_of_calculation)):
    for iteration in range(number_of_calculation):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # get the topic for word n in document m
                z = z_d_n[d][n]

                # decrement counts for word w with associated topic z
                theta_d_z[d][z] -= 1
                phi_z_w[z, w] -= 1
                n_z[z] -= 1

                # sample new topic from a multinomial according to our formular
                p_d_t = (theta_d_z[d] + alpha) / (n_d[d] - 1 + T * alpha)
                p_t_w = (phi_z_w[:, w] + beta) / (n_z + V * beta)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                #new_z = np.random.multinomial(1, p_z).argmax()
                new_z = np.random.choice(len(p_z), 1, p=p_z)[0] 

                # set z as the new topic and increment counts
                z_d_n[d][n] = new_z
                theta_d_z[d][new_z] += 1 #prob / mot
                phi_z_w[new_z, w] += 1 #prob / patient
                n_z[new_z] += 1
 
    norm_theta = theta_d_z.copy()
    ns = np.sum(theta_d_z, axis=1)
    for i in range(ns.shape[0]):
        norm_theta[i, :] /= ns[i]
    #print(np.max(norm_theta))

    norm_theta_max = np.zeros_like(norm_theta)
    norm_theta_max[np.arange(len(norm_theta)), norm_theta.argmax(1)] = 1

    return np.round(norm_theta_max)

def relabel_matrix(matrix):
    ''' relabels columns and lines to max value in diagonal '''
    nx, ny = np.shape(matrix)
    current_mat = matrix.copy()
    current_lines = list(range(nx))
    current_cols = list(range(ny))
    rx = min(nx, ny)
    for i in range(rx):
        # find the max in mat[i:,i:]
        m = np.max(current_mat[i:, i:,])
        if m > 0:
            mx, my = np.unravel_index(current_mat[i:, i:].argmax(), current_mat[i:, i:].shape)
        
            mx += i
            my += i 
            # swap i line with mx
            cxi =  current_lines[i] 
            current_lines[i] = current_lines[mx] 
            current_lines[mx] = cxi 

            mxi =  current_mat[i, :].copy() 
            current_mat[i, :] = current_mat[mx, :] 
            current_mat[mx, :] = mxi

            # swap i col with my
            cyi =  current_cols[i] 
            current_cols[i] = current_cols[my] 
            current_cols[my] = cyi 

            myi =  current_mat[:, i].copy() 
            
            current_mat[:, i] = current_mat[:, my] 
            current_mat[:, my] = myi


    return current_mat , current_cols, current_lines

def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))    


if __name__ == '__main__':

    figures_storage ,frame_storage,frame_multiple_run = 'Figures_LDA_microbiota' , 'Data_frame_LDA_on_microbiota','Data_frame_multiple_run_LDA'
    path_to_store_figures,path_to_store_frame,path_to_store_frame_multiple_run = storage_position(figures_storage ,frame_storage,frame_multiple_run, abspath =  os.path.abspath(__file__) )

    compiled_information = {'Run':[] ,'Accuracy':[]}
    time_of_calculation = 20
    list_of_dataframe = []
    k = 10000
    div_mat = 100
    number_of_calculation = 200
    sample_of_normal = 29

    microbiota_test = False

    if microbiota_test == True:
        phenotypes_patients = 8
    else:
        phenotypes_patients = 2


    information_frame = pd.read_csv('~/FlowRepository_FR-FCM-ZYVH_files/attachments/Metadata_DC.csv')
    #list_of_object = (information_frame.loc[information_frame['Health status'] == 'Healthy']).sample(n = sample_of_normal)['Individual'].to_list() + information_frame.loc[information_frame['Health status'] == 'CD']['Individual'].to_list()
    genus_table = pd.read_table('~/FlowRepository_FR-FCM-ZYVH_files/attachments/GenusAbundance_DiseaseCohort_nature24460.txt', index_col=0, header=0)
    genus_comp = genus_table.div(genus_table.sum(axis=1), axis=0)
    #genus_com_selected = genus_table.loc[list_of_object]
    genus_com_selected = genus_comp.copy()
    series,patients = patient_genus_cell(dataframe=genus_com_selected,number_of_cell_to_sample=k)

    for calculation in tqdm(range(time_of_calculation),leave=False,desc='Iteration'):
        compiled_information['Run'].append(calculation)
        data_frame_for_LDA = pd.DataFrame(series,index=patients).fillna(0)
        norm_theta = LDA_function_makeblop_form(dataframe_for_LDA=data_frame_for_LDA,diviser_of_matrix = div_mat,runrun = phenotypes_patients,number_of_calculation=number_of_calculation, alpha = 1/number_of_calculation, beta = 1/number_of_calculation)
        labelList = data_frame_for_LDA.T.columns.to_list()
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
        global_frame.to_csv(path_to_store_frame_multiple_run + f'/Unique_frame_LDA_rounded_microbiota_unreduced_microbiota_{microbiota_test}_run_{calculation}_on_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv')


pd.concat(list_of_dataframe).to_csv(path_to_store_frame + f'/Global_frame_LDA_rounded_microbiota_genus_microbiota_{microbiota_test}_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv')

dataframe_compiled = pd.DataFrame(compiled_information)
dataframe_compiled.to_csv(path_to_store_frame + f'/Accuracy_Compiled_information_LDA_rounded_microbiota_unreduced_markers_selected_genus_microbiota_{microbiota_test}_{time_of_calculation}_runs_{phenotypes_patients}_topics.csv') 


sns.set_context("paper")
plt.figure(figsize=(7, 5))
barfig = sns.catplot( y="Accuracy", data=dataframe_compiled, kind="bar", ci='sd', capsize=.2)
barfig = sns.swarmplot( y="Accuracy", color='black', s = 3.5, dodge=True,data=dataframe_compiled)
if microbiota_test == True:
    barfig.set_title("Accuracy to separate microbiota community from crohn and healthy patient using Genus" , size = 14)
else:
    barfig.set_title("Accuracy to separate crohn and healthy patient based on microbiota using Genus" , size = 14)

barfig.set_xlabel('Experience', size=14)
barfig.set_ylabel('Accuracy', size=14)
plt.gcf().set_size_inches(7, 5)
barfig.get_figure().savefig(path_to_store_figures + f'/barplot_LDA_rounded_microbiota_{time_of_calculation}_run_genus_markers_microbiota_{microbiota_test}_{phenotypes_patients}_topics_all.svg',  format = 'svg', bbox_inches='tight')
plt.close()


