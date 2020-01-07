import numpy as np 
import matplotlib.pyplot as plt
import pylab
import sys
import pandas as pd
import tensorflow as tf
from gatedAutoencoder import FactoredGatedAutoencoder
from Generate_input import *
from utils import plot_mats
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, './')
########################################################################################################################
# Import random dots dataset
# X = np.load('./shiftsuniform_x.npy').astype('float32')
# Y = np.load('./shiftsuniform_y.npy').astype('float32')
#L=np.ones(X.shape[0])
# L=np.matrix(np.array([1]*X.shape[0]))
# L=L.transpose()
#print(X.shape,Y.shape, L.shape)
#print(type(L))

#######
# Import related and unrelated pairs
data_dic, labels_dic = data_init('WPBC')
X,Y,L,unlabeled_inds= generate_pairs(data_dic,labels_dic,0.2)


print(X.shape, Y.shape, L.shape)


model = FactoredGatedAutoencoder(
    numFactors=15,
    numHidden=5,
    corrutionLevel=0.0)

#model.train(X, Y, L, epochs=5, batch_size=1, print_debug=True)
#model.save('WPBC_20_')


##########################
# Inference
##########################

def Inference_with_inliers(dataset, output_file):

    data_dic, labels_dic = data_init(dataset)
    #inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]
    inlier_inds= data_dic.keys()
    print('number of inliers', len(inlier_inds))
    print('number of data points', len(data_dic.keys()))
    Final_scores =[]

    for ind in data_dic.keys():
        print('data inlierrrrrrrrrrrrrrrrr ',ind)
        test_inlier_pairs = [(a, b) for a in [ind] for b in inlier_inds]
        print('XXXXXXXXXXXXXXXXXXXX',len(inlier_inds))

        List_X = []
        List_Y = []

        for element in test_inlier_pairs:
            List_X.append(data_dic[element[0]])
            List_Y.append(data_dic[element[1]])

        X = np.array(List_X)
        Y = np.array(List_Y)
        Final_scores.append(model.inference(X, Y).flatten())

    pd.DataFrame(Final_scores).to_csv(output_file,header=False, index= False)



model.load_from_weights('WPBC_20_')
Inference_with_inliers('WPBC',"WPBC20_Inference_scores")

#datasets=['WPBC','Glass','Lympho','SatImage','PageBlocks','WDBC','Yeast05679v4','Wilt','Stamps','Pima','Ecoli4','SpamBase']