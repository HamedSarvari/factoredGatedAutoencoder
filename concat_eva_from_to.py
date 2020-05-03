# This scripts takes a dataset name and concatenates the processed data
# The dataset has been split into chunks for parallel computation
# Starting indices of chunks are passed as the second argument of the concat function
###########################################################################################################################
import pandas as pd
import numpy as np
from tensorflow.python.ops.metrics_impl import average_precision_at_k

from Generate_input import *
from universal_utils import *
from sklearn.metrics import average_precision_score
###########################################################################################################################
data_name = 'TwoGauss_data_7dim'
loaded_data = load_obj(data_name)
data = loaded_data['data']
labels = loaded_data['labels']
class_labels = loaded_data['class_labels']
###########################################################################################################################

def eval_model(scores, labels_dic):
    y_true = np.array(list(labels_dic.values()))
    y_true = (y_true == 'yes') + 0
    return average_precision_score(y_true,scores)
###########################################################################################################################

def concat_files(dataset, exp_code, start_inds):

    final_mat = None
    for i in range(len(start_inds)-1):

        add = './results/' + dataset + '_code' + str(exp_code) + '_Inference_scores_' + str(start_inds[i]) + '_to_' + str(start_inds[i+1]) + '.csv'

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

###########################################################################################################################


# scores = concat_files('TwoGauss_data_7dim', start_inds= [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
scores = concat_files('TwoGauss_data_7dim',1, start_inds= [0,500,1000,2000,3000,4000,5000])
# print('scores shape', scores.shape)
# save_obj(scores,'All_scores_twoGauss_data_7dim')
###########################################################################################################################
#scores = load_obj('All_scores_twoGauss_data_7dim')
scores_mean = np.mean(scores,axis=1)
scores_updated = 1- scores_mean
print(len(scores_updated))
print(average_precision_score(labels, scores_updated))

#
# saved_labels = load_obj('TwoGauss_data_7dim_labels')
# outlier_inds = saved_labels['random_outlier_inds']
# inlier_inds = saved_labels['random_inlier_inds']
# selected_inlier_inds = random.sample(inlier_inds, len(outlier_inds))
#
# all_selected_inds = outlier_inds+selected_inlier_inds
# selected_mean =  1 - np.mean(scores[:,all_selected_inds], axis=1)
# print(average_precision_score(labels,selected_mean))
#


#
# data_dic, labels_dic, data_np = data_init('Glass')
# print(len(labels_dic.values()))
# print(eval_model(scores_updated,labels_dic))

###########################################################################################################################

