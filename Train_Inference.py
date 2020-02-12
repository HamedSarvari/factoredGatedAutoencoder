from gatedAutoencoder import FactoredGatedAutoencoder
from Generate_input import *
from utils import plot_mats
import tensorflow as tf
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, './')
CUDA_VISIBLE_DEVICES=1
########################################################################################################################

def Inference_with_inliers(dataset, output_file, model):

    #with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0 '):
    with tf.device('/gpu:0'):
        data_dic, labels_dic = data_init(dataset)
        # inlier_inds = [i for i, x in enumerate(labels_dic.values()) if x == "no"]
        iter_inds = data_dic.keys()
        # print('number of inliers', len(inlier_inds))
        # print('number of data points', len(data_dic.keys()))

        with open(output_file, mode='a') as csvfile:
            for ind in data_dic.keys():
                print(ind)
                test_inlier_pairs = [(a, b) for a in [ind] for b in iter_inds]

                List_X = []
                List_Y = []

                for element in test_inlier_pairs:
                    List_X.append(data_dic[element[0]])
                    List_Y.append(data_dic[element[1]])

                X = np.array(List_X)
                Y = np.array(List_Y)

                file_writer = csv.writer(csvfile, delimiter=',')
                file_writer.writerow(model.inference(X, Y).flatten())



########################################################################################################################

def train_infer(dataset, fac_num, hid_num, out_file,GT_prcnt=0.2, train=True, infer=True):
    data_dic, labels_dic = data_init(dataset)
    X, Y, L, unlabeled_inlier_inds, unlabeled_outlier_inds = generate_pairs(data_dic, labels_dic, GT_prcnt)
    print(X.shape, Y.shape, L.shape)

    model = FactoredGatedAutoencoder(
        numFactors=fac_num,
        numHidden=hid_num,
        corrutionLevel=0.0)
    if train:
        model.train(X, Y, L, epochs=30, batch_size=1, print_debug=True)
        model.save(dataset + '_')

    if infer:
        model.load_from_weights(dataset + '_')
        Inference_with_inliers(dataset, out_file, model)

########################################################################################################################
# Galss, shuttle, wilt
dataset = 'Glass'

out_file=  dataset + "_Inference_scores"
train_infer(dataset,3,3,out_file,GT_prcnt=0.1,train=False, infer=True)

    # datasets=['WPBC','Glass','Lympho','SatImage','PageBlocks','WDBC','Yeast05679v4','Wilt','Stamps','Pima','Ecoli4','SpamBase']