#import pandas as pd
#import numpy as np
#from scipy.io import arff
from Autoencoder_utils_torch import *
#import itertools
import random
#from gatedAutoencoder import FactoredGatedAutoencoder
#import sys

########################################################################################################################
def data_init(dataset):

    data, label = Read_data(dataset)
    print('data shape', data.shape)

    data_dic = data_to_dic(data)
    labels_dic = data_to_dic(label)

    data_np = data
    return data_dic,labels_dic, data_np
#######################################################################################################################

def generate_pairs(data_dic, labels_dic, GT_prcnt):

    outlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "yes"]
    inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]

    print('number of outliers=', len(outlier_inds))
    random_outlier_inds = random.sample(outlier_inds, int(np.ceil(GT_prcnt*len(outlier_inds))))
    print('number of selected=', len(random_outlier_inds))
    print('selected outliers', random_outlier_inds, len(random_outlier_inds))
    random_inlier_inds = random.sample(inlier_inds, int(GT_prcnt * len(inlier_inds)))
    print('selected inliers', len(random_inlier_inds))

    unlabeled_outlier_inds = [i for i in outlier_inds if i not in random_outlier_inds]
    unlabeled_inlier_inds = [i for i in inlier_inds if i not in random_inlier_inds]

    outlier_inlier_pairs = [(a, b) for a in random_outlier_inds for b in random_inlier_inds if a != b]
    inlier_inlier_pairs = [(a, b) for a in random_inlier_inds for b in random_inlier_inds if a != b]

    List_X = []
    List_Y = []
    List_labels = []

    for element in inlier_inlier_pairs:
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

    return X,Y,L, unlabeled_inlier_inds, unlabeled_outlier_inds
###########################################################################################################################
