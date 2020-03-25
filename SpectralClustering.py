import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

from numpy import genfromtxt
from scipy import linalg as LA

dataTesting1 = genfromtxt('data/d_reg_tra.csv', delimiter=',')
dataTesting2 = genfromtxt('data/d_reg_val.csv', delimiter=',')

# define params
k = 3  # numb of clusters
iteration_counter = 0  # clustering iteration counter
input = dataTesting2
var = 1.5  # var in RFB kernel
initCentroidMethod = "kmeans++"  # options: random, kmeans++, badInit, zeroInit

print("starting...")
oldTime = np.around(time.time(), decimals=0)


def init_centroid(data_in, method, k):
    global result
    if method == "random":
        result = random_init_centroid(data_in, k)
    if method == "kmeans++":
        result = kmeans_init_centroid(data_in, k)
    if method == "badInit":
        result, data_in = bad_init_centroid(data_in, k)
    if method == "zeroInit":
        result = np.asmatrix(np.full((k, data_in.shape[1]), 0))
    return result


def bad_init_centroid(data_in, k):
    all_centroid = np.ndarray(shape=(0, data_in.shape[1]))
    first_index = np.random.randint(0, data_in.shape[0])
    first = np.asmatrix(data_in[first_index, :])
    data_in = np.delete(data_in, first_index, 0)
    all_centroid = np.concatenate((all_centroid, first), axis=0)
    repeated_cent = np.repeat(first, data_in.shape[0], axis=0)
    delta_matrix = abs(np.subtract(data_in, repeated_cent))
    euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
    index_next_cent = (np.argmin(np.asmatrix(euclidean_matrix)))
    if k > 1:
        for a in range(1, k):
            next_cent = np.asmatrix(data_in[np.asscalar(index_next_cent), :])
            data_in = np.delete(data_in, np.asscalar(index_next_cent), 0)
            euclidean_matrix_all_centroid = np.ndarray(shape=(data_in.shape[0], 0))
            all_centroid = np.concatenate((all_centroid, next_cent), axis=0)
            for i in range(0, all_centroid.shape[0]):
                repeated_cent = np.repeat(all_centroid[i, :], data_in.shape[0], axis=0)
                delta_matrix = abs(np.subtract(data_in, repeated_cent))
                euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
                euclidean_matrix_all_centroid = \
                    np.concatenate((euclidean_matrix_all_centroid, euclidean_matrix), axis=1)
            euclidean_final = np.min(np.asmatrix(euclidean_matrix_all_centroid), axis=1)
            index_next_cent = np.argmin(np.asmatrix(euclidean_final))
    return all_centroid, data_in


def kmeans_init_centroid(data_in, k):
    euclidean_matrix_all_centroid = np.ndarray(shape=(data_in.shape[0], 0))
    all_centroid = np.ndarray(shape=(0, data_in.shape[1]))
    first = random_init_centroid(data_in, 1)
    all_centroid = np.concatenate((all_centroid, first), axis=0)
    repeated_cent = np.repeat(first, data_in.shape[0], axis=0)
    delta_matrix = abs(np.subtract(data_in, repeated_cent))
    euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
    index_next_cent = (np.argmax(np.asmatrix(euclidean_matrix)))
    if k > 1:
        for a in range(1, k):
            next_cent = np.asmatrix(data_in[np.asscalar(index_next_cent), :])
            all_centroid = np.concatenate((all_centroid, next_cent), axis=0)
            for i in range(0, all_centroid.shape[0]):
                repeated_cent = np.repeat(all_centroid[i, :], data_in.shape[0], axis=0)
                delta_matrix = abs(np.subtract(data_in, repeated_cent))
                euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
                euclidean_matrix_all_centroid = \
                    np.concatenate((euclidean_matrix_all_centroid, euclidean_matrix), axis=1)
            euclidean_final = np.min(np.asmatrix(euclidean_matrix_all_centroid), axis=1)
            index_next_cent = np.argmax(np.asmatrix(euclidean_final))
    return all_centroid


def random_init_centroid(data_in, k):
    return data_in[np.random.choice(data_in.shape[0], k, replace=False)]


def rbf_kernel(data1, data2, sigma):
    delta = np.asmatrix(abs(np.subtract(data1, data2)))
    squared_euclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-squared_euclidean / (2 * sigma ** 2))
    return result


def plot_cluster_result(list_cluster_members, centroid, iteration, converged):
    n = list_cluster_members.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("result")
    plt.clf()
    plt.title("iteration-" + iteration)
    for i in range(n):
        member_cluster = np.asmatrix(list_cluster_members[i])
        plt.scatter(np.ravel(member_cluster[:, 0]), np.ravel(member_cluster[:, 1]), marker=".", s=100, c=next(color))
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(centroid[i, 0], centroid[i, 1], marker="*", s=400, c=col, edgecolors="black")
    if converged == 0:
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if converged == 1:
        plt.show(block=True)


def build_simmilarity_matrix(data_in):
    n_data = data_in.shape[0]
    result = np.asmatrix(np.full((n_data, n_data), 0, dtype=np.float))
    for i in range(0, n_data):
        for j in range(0, n_data):
            weight = rbf_kernel(data_in[i, :], data_in[j, :], var)
            result[i, j] = weight
    return result


def build_degree_matrix(similarity_matrix):
    diag = np.array(similarity_matrix.sum(axis=1)).ravel()
    return np.diag(diag)


def unnormalized_laplacian(sim_matrix, deg_matrix):
    return deg_matrix - sim_matrix


def transform_to_spectral(laplacian):
    global k
    e_vals, e_vecs = LA.eig(np.asmatrix(laplacian))
    ind = e_vals.real.argsort()[:k]
    result = np.ndarray(shape=(laplacian.shape[0], 0))
    for i in range(1, ind.shape[0]):
        cor_e_vec = np.transpose(np.asmatrix(e_vecs[:, np.asscalar(ind[i])]))
        result = np.concatenate((result, cor_e_vec), axis=1)
    return result


sim_mat = build_simmilarity_matrix(input)
deg_mat = build_degree_matrix(sim_mat)
lap_mat = unnormalized_laplacian(sim_mat, deg_mat)
transformed_data = transform_to_spectral(lap_mat)
new_time = np.around(time.time(), decimals=0)
print("time needed until eigen decomposition: ", new_time - oldTime, " s")


def k_means(data_ori, transformed_data, centroid_init_transf):
    n_cluster = centroid_init_transf.shape[0]
    # looping until converged
    global iteration_counter, k
    while True:
        iteration_counter += 1
        euclidean_matrix_all_cluster = np.ndarray(shape=(transformed_data.shape[0], 0))
        # assign data to cluster whose centroid is the closest one
        for i in range(0, n_cluster):
            centroid_repeated = np.repeat(centroid_init_transf[i, :], transformed_data.shape[0], axis=0)
            delta_matrix = abs(np.subtract(transformed_data, centroid_repeated))
            euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
            euclidean_matrix_all_cluster = \
                np.concatenate((euclidean_matrix_all_cluster, euclidean_matrix), axis=1)
        cluster_matrix = np.ravel(np.argmin(np.asmatrix(euclidean_matrix_all_cluster), axis=1))
        list_cluster_member_transf = [[] for i in range(k)]
        list_cluster_member_ori = [[] for i in range(k)]
        for i in range(0, transformed_data.shape[0]):  # assign data to cluster regarding cluster matrix
            list_cluster_member_transf[np.asscalar(cluster_matrix[i])].append(np.array(transformed_data[i, :]).ravel())
            list_cluster_member_ori[np.asscalar(cluster_matrix[i])].append(np.array(data_ori[i, :]).ravel())
        # calculate new centroid
        new_centroid_transf = np.ndarray(shape=(0, centroid_init_transf.shape[1]))
        new_centroid_ori = np.ndarray(shape=(0, data_ori.shape[1]))
        print("iteration: ", iteration_counter)
        for i in range(0, n_cluster):
            member_cluster_transf = np.asmatrix(list_cluster_member_transf[i])
            member_cluster_ori = np.asmatrix(list_cluster_member_ori[i])
            print("cluster members number-", i + 1, ": ", member_cluster_transf.shape)
            centroid_cluster_transf = member_cluster_transf.mean(axis=0)
            centroid_cluster_ori = member_cluster_ori.mean(axis=0)
            new_centroid_transf = np.concatenate((new_centroid_transf, centroid_cluster_transf), axis=0)
            new_centroid_ori = np.concatenate((new_centroid_ori, centroid_cluster_ori), axis=0)
        # break when converged
        if (centroid_init_transf == new_centroid_transf).all():
            break
        # update new centroid
        centroid_init_transf = new_centroid_transf
        plot_cluster_result(list_cluster_member_ori, new_centroid_ori, str(iteration_counter), 0)
        time.sleep(2)
    return list_cluster_member_ori, new_centroid_ori


centroid_init = init_centroid(transformed_data, initCentroidMethod, k)
cluster_member_result, centroid = k_means(input, transformed_data, centroid_init)
plot_cluster_result(cluster_member_result, centroid, str(iteration_counter) + " (converged)", 1)
print("converged!")
