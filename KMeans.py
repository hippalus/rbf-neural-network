import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as  np
from sklearn.__check_build import raise_build_error
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


def calculate(dlt):
    result = 0;
    temp = 1;

    for i in range(1, dlt):
        temp = temp * i
        for k in range(i):
            result += temp
    return result


print("Result is: ", calculate(4))

a = [[1, 2], [2, 1]]
b = [[4, 1], [2, 2]]
print(np.cross(a,b))

def y(x):
    global a
    a=4
    return 0
def f(a):
    a=3
    print(a)
    return a
a=5
f(a)
print(a)
y(a)
print(a)

arr = [1.4,3.7,4.8,6.3,99.9]

x = arr.pop(2)
print(x)
x=np.array([[1, 2, 3], [4, 5, 6]])
print(np.transpose(x))


