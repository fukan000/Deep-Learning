import numpy as np
from numpy import unique
from numpy import where
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans, \
    MeanShift, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture

##########################################################################
####################  K- means clustering data points  ###################
##########################################################################

# load data
##########################################################################

try:
    mat = loadmat("ex7data2.mat")
except:
    raise FileExistsError
else:
    print("Load data successfully!\n")

X = mat["X"]

# Gimps the data
########################################################################

print(X.shape)
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Raw Data for clustering")
plt.text(6, 5, '%d data points' % X.shape[0])
plt.show()


# two main functions for k-means clustering
##########################################################################

# Assign data points to centroids
def assign_data(X, centroids):
    """
    Returns the closest centroids in idx for a dataset X where each row is a single example.
    """
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    temp = np.zeros((centroids.shape[0], 1))

    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i, :] - centroids[j, :]
            length = np.sum(dist ** 2)
            temp[j] = length
        idx[i] = np.argmin(temp) + 1
    return idx


# test code
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = assign_data(X, initial_centroids)
print("Closest centroids for the first 3 examples:\n", idx[0:3])


# use data points to update centroids
def update_centroids(X, idx, K):
    """
    returns the new centroids by computing the means of the data points assigned to each centroid.
    """
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))
    count = np.zeros((K, 1))

    for i in range(m):
        index = int((idx[i] - 1)[0])
        centroids[index, :] += X[i, :]
        count[index] += 1

    return centroids / count


# test code
centroids = update_centroids(X, idx, K)
print("Centroids computed after initial finding of closest centroids:\n", centroids)


# Visualizing K-means Clustering
##########################################################################

def plot_k_means(X, centroids, idx, K, num_iters):
    """
    plots the data points with colors assigned to each centroid
    """
    m, n = X.shape[0], X.shape[1]

    fig, ax = plt.subplots(nrows=1, ncols=num_iters, figsize=(30, 3))

    for i in range(num_iters):
        # Visualisation of data
        color = "rgb"
        for k in range(1, K + 1):
            grp = (idx == k).reshape(m, 1)
            ax[i].scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)

        # visualize the new centroids
        ax[i].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        ax[i].set_title(title)

        # Compute the centroids mean
        centroids = update_centroids(X, idx, K)

        # assign each training example to the nearest centroid
        idx = assign_data(X, centroids)

    plt.tight_layout()


# test code
plot_k_means(X, initial_centroids, idx, K, 7)
plt.show()


# random initialization
##########################################################################
def random_init_k_centroids(X, K):
    """
    This function randomly pick K data points as centroids for K-Means clustering
    """
    m, n = X.shape[0], X.shape[1]
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i] = X[np.random.randint(0, m), :]

    return centroids


# test code
centroids = random_init_k_centroids(X, K)
idx = assign_data(X, centroids)
plot_k_means(X, centroids, idx, K, 7)
plt.show()


# view diff random initial outcomes
##########################################################################
def random_results(X, K, num_iters, num_trails):
    m, n = X.shape[0], X.shape[1]
    num_rows = 3
    num_cols = num_trails // 3 + (0 if num_trails % 3 == 0 else 1)

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3, num_rows * 3))

    for i in range(num_trails):
        centroids = random_init_k_centroids(X, K)
        idx = assign_data(X, centroids)

        # run num_iters k-means
        for _ in range(num_iters):
            # update centroids
            centroids = update_centroids(X, idx, K)
            # assign data points
            idx = assign_data(X, centroids)

        # Visualisation of data
        color = "rgb"
        for k in range(1, K + 1):
            grp = (idx == k).reshape(m, 1)
            ax[i // num_cols, i % num_cols].scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)

        # visualize the new centroids
        ax[i // num_cols, i % num_cols].scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black",
                                                linewidth=3)
        title = str(i + 1) + "Trail"
        ax[i // num_cols, i % num_cols].set_title(title)

        plt.tight_layout()


# test code for random starts
random_results(X, K, 2, 12)
plt.show()
random_results(X, K, 10, 9)
plt.show()

def clustering(X, model, i, title):
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        ax[i // num_cols, i % num_cols].scatter(X[row_ix, 0], X[row_ix, 1])
        ax[i // num_cols, i % num_cols].set_title(title)
    plt.tight_layout()




num_rows, num_cols = 2, 5
fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 3, num_rows * 3))

model0 = AffinityPropagation(damping=0.7)
model1 = AgglomerativeClustering(n_clusters=3)
model2 = KMeans(n_clusters=3)
model3 = Birch(threshold=0.01, n_clusters=3)
model4 = DBSCAN(eps=0.30, min_samples=9)
model5 = MiniBatchKMeans(n_clusters=3)
model6 = MeanShift()
model7 = OPTICS(eps=0.8, min_samples=10)
model8 = SpectralClustering(n_clusters=3)
model9 = GaussianMixture(n_components=3)
title = ['AffinityPropagation', 'AgglomerativeClustering', 'KMeans', 'Birch', 'DBSCAN', \
         'MiniBatchKMeans', 'MeanShift', 'OPTICS', 'SpectralClustering', 'GaussianMixture']

clustering(X, model0, 0, title[0])
clustering(X, model1, 1, title[1])
clustering(X, model2, 2, title[2])
clustering(X, model3, 3, title[3])
clustering(X, model4, 4, title[4])
clustering(X, model5, 5, title[5])
clustering(X, model6, 6, title[6])
clustering(X, model7, 7, title[7])
clustering(X, model8, 8, title[8])
clustering(X, model9, 9, title[9])

plt.show()

##########################################################################
####################  image Compression with K-means  ####################
##########################################################################

# # load data
# mat2 = loadmat("bird_small.mat")
# A = mat2["A"]
#
# # preprocess and reshape the image
# X2 = (A / 255).reshape(128 * 128, 3)
#
#
# def runKmeans(X, initial_centroids, num_iters, K):
#     idx = assign_data(X, initial_centroids)
#
#     for i in range(num_iters):
#         # Compute the centroids mean
#         centroids = update_centroids(X, idx, K)
#
#         # assign each training example to the nearest centroid
#         idx = assign_data(X, initial_centroids)
#
#     return centroids, idx
#
#
# # Running K-means algorithm on the data
# K2 = 16
# num_iters = 10
# initial_centroids2 = random_init_k_centroids(X2, K2)
# centroids2, idx2 = runKmeans(X2, initial_centroids2, num_iters, K2)
#
# m2, n2 = X.shape[0], X.shape[1]
# X2_recovered = X2.copy()
# for i in range(1, K2 + 1):
#     X2_recovered[(idx2 == i).ravel(), :] = centroids2[i - 1]
#
# # Reshape the recovered image into proper dimensions
# X2_recovered = X2_recovered.reshape(128, 128, 3)
#
# # Display the image
# import matplotlib.image as mpimg
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(X2.reshape(128, 128, 3))
# ax[1].imshow(X2_recovered)
