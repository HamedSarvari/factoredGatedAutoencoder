import pandas as pd
import numpy as np
from Generate_input import *

def eval_model(scores, labels_dic):
    y_true = np.array(labels_dic.values())
    y_true = (y_true == 'yes') + 0

    return average_precision_score(y_true,scores)

scores= pd.DataFrame(pd.read_csv('WPBC10_Inference_scores',header=None))
print(scores.shape)
scores_mean= np.max(scores,axis=1)
scores_updated= 1-scores_mean


data_dic, labels_dic = data_init('WPBC')
print(eval_model(scores_updated,labels_dic))

