import numpy as np
import pandas as pd
from plotnine.labels import labs
import os
import random
from glob import glob 
from scipy import stats
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.datasets import make_blobs
 
def storage_position(patient_frame_to_store,figures_storage,frame_storage, abspath):
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path = os.getcwd()
    PATH_storage = path
    print(PATH_storage)

    patient_frame_to_store = 'Patient_generated_binary_only_mutliple_gaussian'
    if not os.path.exists(patient_frame_to_store):
        os.makedirs(patient_frame_to_store)
    path_to_store_patient_frame = os.path.join(PATH_storage, patient_frame_to_store)

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

    return path_to_store_figures,path_to_store_frame



def Generate_patient_with_make_blobs(number_of_patient,number_of_cell,vectors_of_probability, patient_frame_to_store, return_combination):
    for i,x in enumerate(vectors_of_probability):
        globals()[f"vector_of_probability_{i}"] = x

    splitted_vector = np.round(np.linspace(0, number_of_patient, len(vectors_of_probability)+1)[1:]).astype(int)
    list_of_array_patients = np.split(range(number_of_patient), splitted_vector)[:-1]


    patient_and_phenotype = {'Patient_Number':[] ,'Phenotype_Number':[], 'Patient_combination_phenotype':[]}
    first_step = 0


    for group_patient_phenotype , list_patient in enumerate(list_of_array_patients):
        vector_of_probability = globals()[f"vector_of_probability_{group_patient_phenotype}"]
        number_of_dimension = len(vector_of_probability)

        if first_step == 0:
            for number in range(number_of_dimension):
                value = 10
                #globals()[f"cluster_std_{number}"] = random.uniform(0, 1)
                globals()[f"cluster_std_{number}"] = value
                #value +=  0.1
            for number in range(number_of_dimension):
                advance = 42
                globals()[f"random_state{number}"] =  advance
                advance += random.uniform(1, 15)
            first_step += 1

        for patient in list_patient:
            phenotype_continuous_data = []
            phenotype_binary_data = []
            number_of_feature_to_generate = 1


            #Patient Phenotype
            patient_and_phenotype['Patient_Number'].append(f'patient_N°{str(patient).zfill(3)}')
            patient_and_phenotype['Phenotype_Number'].append(group_patient_phenotype+1)
            patient_and_phenotype['Patient_combination_phenotype'].append(vector_of_probability)


            for i in range(len(vector_of_probability)):
                globals()[f"cell_distribution_col_{i}"] = np.round([vector_of_probability[i], 1- vector_of_probability[i]],3)


            for i in range(number_of_dimension):
                #generate distribution
                distribution = 'cell_distribution_col_'+f'{i}' #integer for the distribution
                cells_distributed = (np.array(globals()[distribution]) * number_of_cell).astype(int) #distribution

                X, y = make_blobs(n_samples=cells_distributed, centers=None, n_features= number_of_feature_to_generate, cluster_std=globals()[f"cluster_std_{i}"], random_state= globals()[f"random_state{i}"])
                phenotype_continuous_data.append(X)
                phenotype_binary_data.append(y)

            phenotype_continuous_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_continuous_data))).T
            phenotype_binary_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_binary_data))).T

            phenotype_continuous_data_patient.to_csv(patient_frame_to_store +f'/Generated_file_for_patient_N°{str(patient).zfill(3)}.csv')
            phenotype_binary_data_patient.to_csv(patient_frame_to_store + f'/Generated_binary_file_for_patient_N°{str(patient).zfill(3)}.csv')
            

    pd.DataFrame(patient_and_phenotype).to_csv(patient_frame_to_store + f'/Patient_number_and_phenotype_code_{len(vectors_of_probability)}.csv')


    if return_combination == True:
        return patient_and_phenotype



def LDA_function_makeblop_form(dataframe_for_LDA,diviser_of_matrix,runrun,number_of_calculation, alpha, beta):
    excluded_patient = 0
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

    return np.round(norm_theta)

def continuous_data_files_concat_kmeans_reattached_V2(path_to_store_frame, number_of_cluster,number_of_cell):

    dirs = path_to_store_frame
    a = 'Generated_file'
    files = glob(f'{dirs}/{a}*.csv')
    list_name = []
    list_of_frame_to_append = []
    for file in files:
        name = '_'.join(file.split("_")[-2:]).replace('.csv','')
        list_name.append(name)
        df_to_generate = pd.read_csv(file, index_col= 0)
        list_of_frame_to_append.append(df_to_generate)
    overall_dataframe = pd.concat(list_of_frame_to_append, ignore_index=False)
    #overall_dataframe = pd.DataFrame(stats.zscore(overall_dataframe))

    kmeans = KMeans(n_clusters=number_of_cluster, random_state=41).fit(overall_dataframe)
    labels = kmeans.labels_
    overall_dataframe['K-means labels'] = labels 

    list_of_dataframe_sliced = []

    start_i = 0
    for i in range(number_of_cell,len(overall_dataframe)+number_of_cell,number_of_cell):
        sliced_frame = overall_dataframe.iloc[start_i:i,:]
        list_of_dataframe_sliced.append(sliced_frame)
        start_i =+ i

    list_of_series = []

    for i,frame in enumerate(list_of_dataframe_sliced):
        counter = Counter(frame['K-means labels'])
        list_of_series.append(pd.Series(counter))

    dataframe_for_LDA = pd.DataFrame(list_of_series, index = list_name )
    dataframe_for_LDA.fillna(0, inplace= True)

    return overall_dataframe, dataframe_for_LDA




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





#Code distributions

def make_exponential_distribution(n_dimension, theta):
    ''' create and shuffle an exponetial distribution 
    using this for testing purpose only '''
    vec = np.linspace(0,1, 2 ** n_dimension)
    exp_vec = np.exp(-theta * vec)
    norm_exp_vec = exp_vec / np.sum(exp_vec)
    #np.random.shuffle(norm_exp_vec)
    return norm_exp_vec

def test_distrib():
    n = 7 
    theta = 10
    exp_dis = make_exponential_distribution(n, theta)


def convert_to_binary(idx, n):
    assert idx < 2 ** n 
    rep = f'{idx:b}'
    ret = np.zeros(n)
    for ix, s in enumerate(rep[::-1]):
        ret[n - 1 - ix] = int(s)
    return ret 
    

def make_blob(n_dimension, distribution, n_cells, std=1., mean_high=1., mean_low=-1.):
    '''' function to create n_dimension * n_cells array of binary and float value whose distribrtuion follows distribution '''
    assert 2 ** n_dimension == len(distribution) 

    idxs = list(range(2 ** n_dimension))
    cells_idx = np.random.choice(idxs, p=distribution, size=n_cells)
    cells_bins = np.array([convert_to_binary(idx, n_dimension) for idx in cells_idx])
    cells_float = mean_low + std * np.random.randn(*cells_bins.shape)
    cells_float[cells_bins==1] += mean_high - mean_low
    return cells_float, cells_bins



def test_blob_left_right_proba(proba,number_of_patient,n_pheno,n_dim,theta,number_of_cell,patient_frame_to_store,value, return_combination):
    splitted_vector = np.round(np.linspace(0, number_of_patient, n_pheno+1)[1:]).astype(int)
    list_of_array_patients = np.split(range(number_of_patient), splitted_vector)[:-1]
    phenotype_comb = []

    patient_and_phenotype = {'Patient_Number':[] ,'Phenotype_Number':[],'Patient_combination_phenotype':[] }

    for group_patient_phenotype , list_patient in enumerate(list_of_array_patients):
        exp_dis = proba[group_patient_phenotype]
        phenotype_comb.append(exp_dis)

        for patient in list_patient:

            #Patient Phenotype
            patient_and_phenotype['Patient_Number'].append(f'patient_N°{str(patient).zfill(3)}')
            patient_and_phenotype['Phenotype_Number'].append(group_patient_phenotype+1)
            patient_and_phenotype['Patient_combination_phenotype'].append(exp_dis)

            ci, cb = make_blob(n_dim, exp_dis, number_of_cell, std = value)
            pd.DataFrame(ci).to_csv(patient_frame_to_store +f'/Generated_file_for_patient_N°{str(patient).zfill(3)}.csv')
            pd.DataFrame(cb).to_csv(patient_frame_to_store + f'/Generated_binary_file_for_patient_N°{str(patient).zfill(3)}.csv')

        
    pd.DataFrame(patient_and_phenotype).to_csv(patient_frame_to_store + f'/Patient_number_and_phenotype_code_{n_pheno}.csv')

    if return_combination == True:
        return patient_and_phenotype , phenotype_comb


def test_extreme_porba_v2(first_array,dimension,c):
    array = np.full(shape=2**dimension -2 , fill_value=(1-first_array)/(2**dimension -2))

    firstphenotypes = np.concatenate(([first_array], array, [0]))
    sndphenotypes = firstphenotypes[::-1]

    proba_to_test = []#, 'euc_dist' : [] }


    for configuration in np.arange(0,first_array,c):
        new_arr = firstphenotypes.copy()
        new_arr_2 = sndphenotypes.copy()

        new_arr[0] = new_arr[0] - configuration
        new_arr[-1] = new_arr[-1] + configuration

        new_arr_2[0] = new_arr_2[0] + configuration
        new_arr_2[-1] = new_arr_2[-1] - configuration

        proba_to_test.append([new_arr,new_arr_2])


    return proba_to_test



#To generate multiple vectors of probability 4phenotypes

def swap(seq,n_pheno):
    pheno = []
    pheno.append(seq)
    for i in range(1,len(seq)):
        seq_2 = seq.copy()
        i1 = 0
        i2 = i
        seq_2[i1], seq_2[i2] = seq_2[i2], seq_2[i1]
        pheno.append(seq_2)
    return pheno[:n_pheno]

def generate_multiple_probability(dimension,first_array,c,phenotypes):
    array_of_phenotypes = []
    for value in np.arange(0,first_array,c):
        array_to_use = first_array - value 
        array = np.full(shape=2**dimension -1 , fill_value=(1-array_to_use)/(2**dimension -1))
        firstphenotypes = np.concatenate(([array_to_use], array))
        probability_to_append = swap(firstphenotypes,phenotypes)
        array_of_phenotypes.append(probability_to_append)
    
    return array_of_phenotypes

