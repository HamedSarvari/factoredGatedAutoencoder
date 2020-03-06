from gatedAutoencoder import FactoredGatedAutoencoder
from Generate_input import *
from utils import plot_mats
import tensorflow as tf
import os
import csv
from itertools import cycle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.path.insert(0, './')
CUDA_VISIBLE_DEVICES=1

########################################################################################################################

def Inference_with_inliers(dataset, model, start_ind, end_ind):

    #with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0 '):
    #with tf.device('/gpu:0'):
        data_dic, labels_dic, data_np = data_init(dataset)
        # inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]
        iter_inds = data_dic.keys()
        # print('number of inliers', len(inlier_inds))
        # print('number of data points', len(data_dic.keys()))


        output_file = './results/' + dataset + "_Inference_scores_" + str(start_ind) + '_to_' + str(end_ind) + '.csv'

        with open(output_file, mode='a') as csvfile:

            file_writer = csv.writer(csvfile, delimiter=',')
            for ind in range(start_ind, end_ind):
                print(ind)
                size= len(iter_inds)
                X = np.array([list(data_dic[ind])]*size)
                Y = data_np
                file_writer.writerow(model.inference(X, Y).flatten())
                del X
                del Y

########################################################################################################################

def train_infer(dataset, fac_num, hid_num, start_ind , end_ind ,GT_prcnt=0.2, train=True, infer=True):
    data_dic, labels_dic, data_np = data_init(dataset)
    X, Y, L, unlabeled_inlier_inds, unlabeled_outlier_inds = generate_pairs(data_dic, labels_dic, GT_prcnt)
    print(X.shape, Y.shape, L.shape)
    # if end ind is not specified iterate to the very last index
    if end_ind is None:
        end_ind = len(data_dic.keys())
    print('end ind', end_ind)

    model = FactoredGatedAutoencoder(
        numFactors=fac_num,
        numHidden=hid_num,
        corrutionLevel=0.0)
    if train:
        model.train(X, Y, L, epochs=3, batch_size=1, print_debug=True)
        model.save('./Weights/' + dataset + '_')

    if infer:
        model.load_from_weights('./Weights/' + dataset + '_')
        Inference_with_inliers(dataset, model, start_ind, end_ind)

########################################################################################################################
# Galss, shuttle, wilt
dataset = 'Wilt'
start_index = 200
end_index = None


train_infer(dataset,fac_num=3,hid_num=3, start_ind = start_index, end_ind= end_index,
            GT_prcnt=0.1, train=False, infer=True)

# datasets=['WPBC','Glass','Lympho','SatImage','PageBlocks','WDBC','Yeast05679v4','Wilt','Stamps','Pima','Ecoli4','SpamBase']
