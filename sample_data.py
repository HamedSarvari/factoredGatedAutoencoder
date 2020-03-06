import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from utils_gmm.data_gmm import GMM_distribution, sample_GMM, plot_GMM
from utils_gmm.data_utils import shuffle, iter_data, ToyDataset
from tqdm import tqdm
import sklearn.datasets

slim = tf.contrib.slim
ds = tf.contrib.distributions
graph_replace = tf.contrib.graph_editor.graph_replace

# %% md
### Parameters

# %%

DATASET = '4gaussians'  # 5gaussians, swiss_roll, s_curve
N_NOISY = 100

# %%

""" parameters """
n_epoch = 300
batch_size = 1024
dataset_size_x = 512 * 4
dataset_size_z = 512 * 4

# dataset_size_x_test = 512 * 2
# dataset_size_z_test = 512 * 2

input_dim = 2
latent_dim = 2
eps_dim = 2

n_layer_disc = 2
n_hidden_disc = 256
n_layer_gen = 3
n_hidden_gen = 256
n_layer_inf = 2
n_hidden_inf = 256

# %%

""" Create directory for results """
result_dir = 'results/alad_toy/'
directory = result_dir
if not os.path.exists(directory):
    os.makedirs(directory)

# %% md

## Training dataset

# %% md

#### 4 or 5 GMM

# %%

""" Create dataset """


def four_five_gaussians(p1=0):
    # create X dataset
    global dataset_size_x
    means_x = map(lambda x: np.array(x), [[2, 2],
                                          [-2, -2],
                                          [2, -2],
                                          [-2, 2],
                                          [0, 0]])

    means_x = list(means_x)
    std_x = 0.02
    variances_x = [np.eye(2) * std_x for _ in means_x]

    # contamination = 4.0*p/(1-p)
    priors_x = np.array([1.0, 1.0, 1.0, 1.0, p1])
    priors_x /= sum(priors_x)
    # print(priors_x)
    gaussian_mixture = GMM_distribution(means=means_x,
                                        variances=variances_x,
                                        priors=priors_x, rng=33)
    dataset_x = sample_GMM(dataset_size_x, means_x, variances_x, priors_x, sources=('features',))
    return dataset_x


dataset_x = four_five_gaussians(p1=0)
save_path_x = result_dir + 'X_4gmm_data_train.png'


# %%

## input x
X_dataset = dataset_x.data['samples']
X_targets = dataset_x.data['label']

# i=1
# print('DS', X_dataset[i])
# print('T', X_targets[i])
# print(dataset_x.means)
# print(X_dataset[i]-dataset_x.means[X_targets[i]])
# print(np.linalg.norm(X_dataset[i]-dataset_x.means[X_targets[i]]))
# print(dataset_x.variances[0][0][0]*2)

threshold = np.sqrt(dataset_x.variances[0][0][0])*2
outlier_labels =[]
for i in range(len(X_dataset)):
    dist_to_mean = np.linalg.norm(X_dataset[i]-dataset_x.means[X_targets[i]])
    if dist_to_mean > threshold:
        outlier_labels.append('yes')
    else:
        outlier_labels.append('no')

out_labels= np.array(outlier_labels)
print(sum(out_labels == 'no')/len(X_dataset))

# fig_mx, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
# ax.scatter(X_dataset[:, 0], X_dataset[:, 1], c=cm.tab20(X_targets.astype(float) / input_dim / 3.0),
#            edgecolor='none', alpha=0.5)
# # ax.set_xlim(-3, 3); ax.set_ylim(-3.5, 3.5)
# ax.set_xlabel('$x_1$');
# ax.set_ylabel('$x_2$')
# ax.set_title("X distribution in the training set")
# ax.axis('on')
# plt.savefig(save_path_x, transparent=True, bbox_inches='tight')
# plt.show()

