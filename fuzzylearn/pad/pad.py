from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import math
A = np.array(
    [
    [1,2,3],
    [1,2,4],
    [1,5,5],
    [1,2,3],
    ])
B = np.array(
    [
    [1,2,3],
    [1,2,4],
    [1,5,5],
    [1,2,3],
    ])



check = pairwise_distances(A, B,metric="cosine")

print(A)
print(B)
print(np.round(check,decimals = 3))



A = np.array(
    [
    [1,2],
    [1,5],
    ])
 
B = np.array(
    [
    [1],
    [2],
    ])

#print(A)
#print(B)
print(np.dot(A,B))


l = [np.array([1,2]),np.array([1,3])]

print([np.mean(l, 0).tolist()])


A = np.array(
    [
    [1,20,30],
    [1,2,40],
    [1,5,5],
    ])
A *= np.tri(*A.shape)
print(A)