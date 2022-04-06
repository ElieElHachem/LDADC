import numpy as np
import pandas as pd
import os
from glob import glob 
from scipy import stats
from sklearn.cluster import KMeans
from collections import Counter
import fcsparser

 
def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))    


def storage_position(figures_storage,frame_storage, abspath):
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

    return path_to_store_figures,path_to_store_frame

def common_type(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False

def kmeans_reattached_patient(cell_to_sample,files):
    list_name = []
    list_of_frame_to_append = []
    columns = []
    for file in files:
        name = file.split("/")[-1]
        meta, len_to_see = fcsparser.parse(file, reformat_meta=True)
        if len_to_see.shape[0] > cell_to_sample:
            list_name.append(name)
            sampling_cell = (len_to_see.sample(n = cell_to_sample, axis=0)).drop(columns =['FS Lin', 'SS Log'])
            columns.append(sampling_cell.columns)
            if common_type(['sKappa-FITC', 'sLambda-PE'],sampling_cell.columns):
                sampling_cell.rename(index=str, columns={'sKappa-FITC': 'Kappa-FITC', 'sLambda-PE': 'Lambda-PE'}, inplace=True)
            if common_type(['HLA DR-FITC'],sampling_cell.columns):
                sampling_cell.rename(index=str,columns={"HLA DR-FITC": "HLA-DR-FITC"}, inplace=True)
             
            list_of_frame_to_append.append(sampling_cell)
    

    overall_dataframe = pd.concat(list_of_frame_to_append, ignore_index=False)
    overall_dataframe = pd.DataFrame(stats.zscore(overall_dataframe))
    number_of_cluster = 2**overall_dataframe.shape[1]

    kmeans = KMeans(n_clusters=number_of_cluster).fit(overall_dataframe)
    labels = kmeans.labels_
    overall_dataframe['K-means labels'] = labels
    list_of_dataframe_sliced = []
    number_of_cell = cell_to_sample
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

    return dataframe_for_LDA




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
                theta_d_z[d][new_z] += 1 
                phi_z_w[new_z, w] += 1 
                n_z[new_z] += 1
 
    norm_theta = theta_d_z.copy()
    ns = np.sum(theta_d_z, axis=1)
    for i in range(ns.shape[0]):
        norm_theta[i, :] /= ns[i]

    norm_theta_max = np.zeros_like(norm_theta)
    norm_theta_max[np.arange(len(norm_theta)), norm_theta.argmax(1)] = 1

    return np.around(norm_theta_max), np.around(phi_z_w)


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













