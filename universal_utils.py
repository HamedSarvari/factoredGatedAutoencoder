import os
import pickle
import numpy as np
########################################################################################################################


def save_obj(obj, name ):
    if os.path.isfile('obj/'+ name + '.pkl'):
        print("File already exists")
    with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def combine_scores(scores, agg_func='mean', reversed_scores='true'):
    if agg_func == 'mean':
        scores_mean = np.mean(scores, axis=1)
    elif agg_func == 'max':
        scores_mean = np.max(scores, axis=1)

    if reversed_scores:
        scores_updated = 1 - scores_mean
    else:
        scores_updated = scores_mean

    return scores_updated
