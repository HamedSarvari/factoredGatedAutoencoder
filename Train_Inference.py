from gatedAutoencoder import FactoredGatedAutoencoder
from create_toy_dataset import *
from utils import plot_mats
import tensorflow as tf
import os
import csv
from itertools import cycle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#sys.path.insert(0, './')
CUDA_VISIBLE_DEVICES=1

########################################################################################################################

def Inference_with_inliers(dataset, model, start_ind, end_ind, mu1, mu2, sigma, dim, size):

    #with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0 '):
    #with tf.device('/gpu:0'):

        dataset='TwoGauss'
        data, labels, class_labels = Gen_2_gaussians(mu1, mu2, sigma, dim, size)


        output_file = './results/' + dataset + "_Inference_scores_" + str(start_ind) + '_to_' + str(end_ind) + '.csv'

        with open(output_file, mode='a') as csvfile:

            file_writer = csv.writer(csvfile, delimiter=',')
            for ind in range(start_ind, end_ind):
                print(ind)
                size = data.shape[0]
                X = np.array([list(data[ind,:])]*size)
                Y = data
                file_writer.writerow(model.inference(X, Y).flatten())
                del X
                del Y

########################################################################################################################

def train_infer_two_gauss(dataset,fac_num, hid_num, start_ind , end_ind ,mu1=0, mu2=5, sigma=1, dim=7, size=2500, GT_prcnt=0.1,
                          ep_num=1, train=True, infer=True):


    X, Y, L, unlabeled_inlier_inds, unlabeled_outlier_inds = generate_pairs_two_gauss(mu1, mu2, sigma, dim, size, GT_prcnt)


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
        model.train(X, Y, L, epochs= ep_num, batch_size=1, print_debug=True)
        model.save('./Weights/' + dataset + '_')

    if infer:
        model.load_from_weights('./Weights/' + dataset + '_')
        Inference_with_inliers(dataset, model, start_ind, end_ind, mu1, mu2, sigma, dim, size)

########################################################################################################################
mu1 = 0
mu2 = 5
sigma = 1
dim = 7
size = 2500
GT_prcnt = 0.1
########################################################################################################################

start_index = 0
end_index = 100

dataset = 'TwoGauss'

train_infer_two_gauss(dataset, fac_num=3, hid_num=3, start_ind = start_index, end_ind= end_index,
            GT_prcnt=0.1, train=True, infer=True)

# datasets=['WPBC','Glass','Lympho','SatImage','PageBlocks','WDBC','Yeast05679v4','Wilt','Stamps','Pima','Ecoli4','SpamBase']
