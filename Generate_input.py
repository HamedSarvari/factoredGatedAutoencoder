import pandas as pd
import numpy as np
from scipy.io import arff
from Autoencoder_utils_torch import *
import itertools
import random
from gatedAutoencoder import FactoredGatedAutoencoder
import sys

###########################################################################################################################

def Read_data(dataset, type= 'arff'):
    if type == 'arff':
        address = './DataSets/' + dataset + '.arff'
        #address='/home/hamed/PycharmProjects/DeepEnsembles/DataSets/' + dataset + '.arff'

        data, meta = arff.loadarff(address)
        data = pd.DataFrame(data)
        if 'id' in data.columns:
            data = data.drop(labels='id', axis=1)
        labels = list(data['outlier'])
        data = data.drop(labels='outlier', axis=1)
        data=check_data_modify(data)

    return data,labels
###########################################################################################################################
def data_init(dataset):

    data, label = Read_data(dataset)
    print('data shape', data.shape)

    data_dic = data_to_dic(data)
    labels_dic = data_to_dic(label)

    return  data_dic,labels_dic

def generate_pairs(data_dic, labels_dic, GT_prcnt):


    outlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "yes"]
    inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]

    random_outlier_inds = random.sample(outlier_inds, int(GT_prcnt* len(outlier_inds)))
    unlabeled_outlier_inds = [i for i in outlier_inds if i not in random_outlier_inds]

    print('lableed', random_outlier_inds, len(random_outlier_inds),len(outlier_inds))
    print('unlableed',unlabeled_outlier_inds,len(unlabeled_outlier_inds))

    inlier_pairs = list(itertools.combinations(inlier_inds, 2))
    outlier_inlier_pairs = [(a, b) for a in random_outlier_inds for b in inlier_inds if a != b]


    List_X = []
    List_Y = []
    List_labels = []

    for element in inlier_pairs:
        List_X.append(data_dic[element[0]])
        List_Y.append(data_dic[element[1]])
        List_labels.append(1)

    for element in outlier_inlier_pairs:
        List_X.append(data_dic[element[0]])
        List_Y.append(data_dic[element[1]])
        List_labels.append(0)

    X = np.array(List_X)
    Y = np.array(List_Y)

    L = np.matrix(List_labels)
    L = L.transpose()

    return X,Y,L, unlabeled_outlier_inds
###########################################################################################################################

# def pair_with_inliers(data_dic, labels_dic, unlabeled_ind):
#
#     inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]
#     print('inlier',inlier_inds)
#     unlabeled_inlier_pairs = [(a, b) for a in [unlabeled_ind] for b in inlier_inds]
#     List_X = []
#     List_Y = []
#
#     for element in unlabeled_inlier_pairs:
#
#         unlabeled_inlier_pairs = [(a, b) for a in [unlabeled_ind] for b in inlier_inds]
#         List_X = []
#         List_Y = []
#
#         for element in unlabeled_inlier_pairs:
#             List_X.append(data_dic[element[0]])
#             List_Y.append(data_dic[element[1]])
#         X = np.array(List_X)
#         Y = np.array(List_Y)
#
#         List_X.append(data_dic[element[0]])
#         List_Y.append(data_dic[element[1]])
#     X = np.array(List_X)
#     Y = np.array(List_Y)
#
#     return X,Y

