import pandas as pd
import numpy as np
from Generate_input import *

def eval_model(scores, labels_dic):
    y_true = np.array(list(labels_dic.values()))
    y_true = (y_true == 'yes') + 0
    return average_precision_score(y_true,scores)

def concat_files(dataset, start_inds):

    final_mat = None
    for i in range(len(start_inds)-1):

        add = './results/' + dataset + '_Inference_scores_' + str(start_inds[i]) + '_to_' + str(start_inds[i+1]) + '.csv'

        if final_mat is None:
            final_mat = np.array(pd.read_csv(add, header= None))
            # drop any addit
            #final_mat = final_mat[:start_inds[i+1], :]
        else:
            partial_mat = np.array(pd.read_csv(add, header= None))
            print('Partial  Matrix ', partial_mat.shape)
            final_mat = np.concatenate((final_mat, partial_mat))
            #if i < len(start_inds) - 1:
            #    final_mat = final_mat[:start_inds[i+1], :]

        print('iter', i, add)
        print('iter', i, final_mat.shape)

    return final_mat


scores = concat_files('Glass', start_inds= [0,100,150,200,214])

print('scores shape', scores.shape)
scores_mean= np.max(scores,axis=1)
scores_updated= 1-scores_mean

#
data_dic, labels_dic, data_np = data_init('Glass')
print(len(labels_dic.values()))
print(eval_model(scores_updated,labels_dic))


