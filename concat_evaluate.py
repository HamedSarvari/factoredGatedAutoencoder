import pandas as pd
import numpy as np
from Generate_input import *

######################################################################################################################
def eval_model(scores, labels_dic):
    y_true = np.array(list(labels_dic.values()))
    y_true = (y_true == 'yes') + 0
    return average_precision_score(y_true,scores)
######################################################################################################################

# Generating the 2-d matrix of inference could be done in parallel
# This method takes a dataset name and indices of the different chops and build a square matrix
def concat_files(dataset, start_inds):

    final_mat = None
    for i in range(len(start_inds)):

        add = str('./results/' + dataset + '_Inference_scores_starting' + str(start_inds[i]) + '.csv')
        if final_mat is None:
            final_mat = np.array(pd.read_csv(add, header=None))
            # drop any addit
            final_mat = final_mat[:start_inds[i+1], :]
        else:
            partial_mat = np.array(pd.read_csv(add, header=None))
            final_mat = np.concatenate((final_mat, partial_mat))
            if i < len(start_inds) - 1:
                final_mat = final_mat[:start_inds[i+1], :]

        #print('iter', i, final_mat.shape)
    return final_mat

######################################################################################################################

scores = concat_files('Wilt', start_inds= [0,3700,4000,4500])
print('scores shape', scores.shape)
scores_mean= np.mean(scores,axis=1)
scores_updated = 1-scores_mean


data_dic, labels_dic, data_np = data_init('Wilt')
print(len(labels_dic.values()))
print(eval_model(scores_updated,labels_dic))

