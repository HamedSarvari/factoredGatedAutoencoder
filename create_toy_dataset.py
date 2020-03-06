import numpy as np
import random
########################################################################################################################
# generates as many as size, 'dim' dimensional guassian points with mean mu and sd sigma.
# Returns a matrix with generated points
def gauss_nd(mu, sigma, dim, size):
    all_points=[]
    for i in range(size):
        point=[]
        for j in range(dim):
            point.append(list(np.random.normal(mu, sigma, 1))[0])
        all_points.append(point)
    return np.array(all_points)

########################################################################################################################

# Takes a dataset genrated by gauss_nd and a threshold and returns labels of inliers/outliers
def check_outlierness(dataset, mu, threshold):

    dim = dataset.shape[1]

    # Find points that are more than threshold away from the mean in 70 percent of the features
    outliers = np.array([sum(np.array(np.abs(dataset[ind, :] - mu)) > threshold) > int(0.7*dim) for ind in range(dataset.shape[0])])
    labels = outliers+0

    return labels
########################################################################################################################

# Generates two gaussian with means mu1 and mu2 and returns data/ labels/ class labels
def Gen_2_gaussians(mu1, mu2, sigma, dim, size):

    dataset1 = gauss_nd(mu1, sigma, dim, size)
    dataset2 = gauss_nd(mu2, sigma, dim, size)

    ds1_labels = check_outlierness(dataset1, mu1, 1 * sigma)
    ds2_labels = check_outlierness(dataset2, mu2, 1 * sigma)

    dataset = np.concatenate((dataset1, dataset2))
    labels = np.concatenate((ds1_labels, ds2_labels))
    class_labels = [1] * len(ds1_labels) + [2] * len(ds2_labels)

    return dataset, labels, class_labels

########################################################################################################################


def generate_pairs_two_gauss(mu1, mu2, sigma, dim, size, GT_prcnt):

    data, labels, class_labels = Gen_2_gaussians(mu1, mu2, sigma, dim, size)
    IDs = np.array(range(data.shape[0]))
    outlier_inds = IDs[labels == 1]
    inlier_inds = IDs[labels == 0]

    # outlier_inds = labelslabels == 1
    print('number of outliers=', len(outlier_inds))

    random_outlier_inds = random.sample(list(outlier_inds), int(np.ceil(GT_prcnt * len(outlier_inds))))

    print('number of selected=', len(random_outlier_inds))
    print('selected outliers', random_outlier_inds, len(random_outlier_inds))
    random_inlier_inds = random.sample(list(inlier_inds), int(GT_prcnt * len(inlier_inds)))
    print('selected inliers', len(random_inlier_inds))

    unlabeled_outlier_inds = [i for i in outlier_inds if i not in random_outlier_inds]
    unlabeled_inlier_inds = [i for i in inlier_inds if i not in random_inlier_inds]


    outlier_inlier_pairs = [(a, b) for a in random_outlier_inds for b in random_inlier_inds if a != b] + \
                           [(a, b) for b in random_outlier_inds for a in random_inlier_inds if a != b]

    inlier_inlier_pairs = [(a, b) for a in random_inlier_inds for b in random_inlier_inds if a != b and
                           class_labels[a] == class_labels[b]]


    print(len(inlier_inlier_pairs),'in-in')
    print(len(outlier_inlier_pairs), 'out-in')

    List_X = []
    List_Y = []
    List_labels = []

    for element in inlier_inlier_pairs:
        List_X.append(data[element[0],:])
        List_Y.append(data[element[1],:])
        List_labels.append(1)

    for element in outlier_inlier_pairs:
        List_X.append(data[element[0], :])
        List_Y.append(data[element[1], :])
        List_labels.append(0)


    X = np.array(List_X)
    Y = np.array(List_Y)

    L = np.matrix(List_labels)
    L = L.transpose()

    return X,Y,L, unlabeled_inlier_inds, unlabeled_outlier_inds

########################################################################################################################

