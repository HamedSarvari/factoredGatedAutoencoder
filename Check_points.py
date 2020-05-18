from universal_utils import *
import numpy as np
########################################################################################################################
exp_name = 'TwoGauss_data_7dim'
exp_code = 2

def print_info(data,outlier_inds, inlier_inds, class_labels,final_scores, ind, type):
    if type == 'outlier':
        print('outlier', data[outlier_inds[ind]], class_labels[outlier_inds[ind]], final_scores[outlier_inds[ind]])
    elif type == 'inlier':
        print('inlier', data[inlier_inds[ind]], class_labels[inlier_inds[ind]], final_scores[inlier_inds[ind]])

def Load_labels(exp_name, exp_code):
    exp_code = str(exp_code)
    selected_indexes = load_obj(exp_name + '_' + 'code' + exp_code + '_selected_labels')
    original_labels = load_obj(exp_name)
    print(selected_indexes.keys())
    print(original_labels.keys())

    data = original_labels['data']
    all_labels = original_labels['labels']
    class_labels = original_labels['class_labels']
    IDs = np.array(range(len(all_labels)))
    outlier_inds = IDs[all_labels == 1]
    inlier_inds = IDs[all_labels == 0]
    random_outlier_inds = selected_indexes['random_outlier_inds']
    random_inlier_inds = selected_indexes['random_inlier_inds']

    outlier_scores = load_obj('outlier_scores_' + exp_name + '_exp' + exp_code)
    final_scores = np.array(combine_scores(outlier_scores))

    for i in range(5):
        print_info(data, outlier_inds, inlier_inds, class_labels,final_scores,i,'outlier')
        print_info(data, outlier_inds, inlier_inds, class_labels, final_scores, i, 'inlier')
        print('-----------------------------------------------------------------------------------')

    #print('dataaaa', data[2], all_labels[1])
    #print('inlier', data[inlier_inds[0]], class_labels[inlier_inds[0]], final_scores[inlier_inds[0]])

    #print(final_scores[2707])
    #print(np.argsort(-final_scores))
    #print(random_outlier_inds)


    #print(final_scores[random_outlier_inds])
    #print(final_scores[random_inlier_inds])
Load_labels(exp_name,5)


