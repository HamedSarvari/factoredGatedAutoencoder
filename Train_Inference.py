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

def Inference_with_inliers(data_name, exp_code, model, start_ind, end_ind):

    #with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0 '):
    #with tf.device('/gpu:0'):

        loaded_data = load_obj(data_name)
        data = loaded_data['data']


        labels = loaded_data['labels']
        class_labels = loaded_data['class_labels']

        # retrieve labels of the data points labeled and used for training
        selected_labels = load_obj(data_name +'_code' + str(exp_code) + '_selected_labels')
        random_outlier_inds = selected_labels['random_outlier_inds']
        random_inlier_inds = selected_labels['random_inlier_inds']

        output_file = './results/' + data_name +'_code' + str(exp_code) + "_Inference_scores_" + str(start_ind) + '_to_' + str(end_ind) + '.csv'

        with open(output_file, mode='a') as csvfile:

            file_writer = csv.writer(csvfile, delimiter=',')
            for ind in range(start_ind, end_ind):
                print(ind)
                size = data.shape[0]
                # total
                #X = np.array([list(data[ind,:])]*size)
                # partial
                X = np.array([list(data[ind, :])] * len(random_inlier_inds))

                # only use the indices of inliers selected for training in inference
                # (X,Y) --> (o,i) and (i,i) pairs have been used to train
                # (unknown,i) is now used for inference --> if it's inlier it should output closer to 1, if outlier it should output closer to 0

                print('hereeeeeeee', len(random_inlier_inds))
                Y = data[random_inlier_inds,:]
                file_writer.writerow(model.inference(X, Y).flatten())
                print('random inlier inds', len(random_inlier_inds))
                del X
                del Y

########################################################################################################################

def train_infer_two_gauss(data_name, exp_code, fac_num, hid_num, start_ind , end_ind , GT_prcnt=0.1,
                          ep_num = 10, train=True, infer=True):
    if end_ind is None:
        # Two gaussians so number of data points is size * 2
        end_ind = size * 2
    print('end ind', end_ind)

    model = FactoredGatedAutoencoder(
        numFactors=fac_num,
        numHidden=hid_num,
        corrutionLevel=0.0)
    if train:
        #X, Y, L, unlabeled_inlier_inds, unlabeled_outlier_inds = generate_pairs_two_gauss(data_name, GT_prcnt)
        X_pos, Y_pos, L_pos, X_neg, Y_neg, L_neg = generate_pairs_two_gauss(data_name, GT_prcnt, exp_code)


        #model.train_gen(X, Y, L, epochs= ep_num, batch_size=1, print_debug=True)
        # Train the genrative weights only with positive pairs
        model.train_gen(X_pos, Y_pos, L_pos, epochs= ep_num, batch_size= 1, print_debug= True)
        # Train the discriminative weights with both positive and negative weights

        X_all = np.concatenate((X_pos, X_neg))
        Y_all = np.concatenate((Y_pos, Y_neg))
        L_all = np.concatenate((L_pos, L_neg))

        model.train_disc(X_all, Y_all, L_all, epochs= ep_num, batch_size= 1, print_debug= True)
        model.save('./Weights/' + dataset + '_code' + str(exp_code) + '_')

    if infer:
        model.load_from_weights('./Weights/' + dataset + '_code' + str(exp_code) + '_')
        Inference_with_inliers(data_name, exp_code, model, start_ind, end_ind)

########################################################################################################################
mu1 = 0
mu2 = 5
sigma = 1
dim = 7
size = 2500
GT_prcnt = 0.1
num_epochs = 1
########################################################################################################################
# For Argo
current_ind = int(os.environ["SLURM_ARRAY_TASK_ID"])
index_list = list(range(0,5001,500))
start_index = index_list[current_ind]
end_index = index_list[current_ind + 1]
dataset = 'TwoGauss'

#train_infer_two_gauss('TwoGauss_data_7dim', exp_code = 1, fac_num=3, hid_num=3, start_ind = start_index, end_ind= end_index,
#            GT_prcnt= 0.1, ep_num= num_epochs, train= True, infer= False)
# datasets=['WPBC','Glass','Lympho','SatImage','PageBlocks','WDBC','Yeast05679v4','Wilt','Stamps','Pima','Ecoli4','SpamBase']



