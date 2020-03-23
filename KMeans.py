import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.cluster import KMeans

DATA = genfromtxt('data/d_reg_val.csv', delimiter=',')
TRAIN_DATA = genfromtxt('data/d_reg_tra.csv', delimiter=',')


def k_means_(n_cluster, data):
    k_means = KMeans(n_clusters=n_cluster)
    k_means.fit(data)
    print(k_means.cluster_centers_)
    print(k_means.labels_)
    plt.scatter(data[:, 0], data[:, 1], label='True Position')
    plt.scatter(data[:, 0], data[:, 1], c=k_means.labels_, cmap='rainbow')
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], color='black')
    plt.show()


k_means_(3, DATA)
k_means_(3, TRAIN_DATA)