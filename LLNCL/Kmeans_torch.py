import torch
import numpy as np


def train(data,num_clustres,max_iterations):

    centroids = centroids_init(data,num_clustres)
    centroids = centroids.cuda()
    num_example = data.shape[0]
    closest_centroids_ids = torch.empty((num_example,1))
    for _ in range(max_iterations):
 
        closest_centroids_ids = centroids_find_closest(data,centroids)
        centroids = centroids_compute(data,closest_centroids_ids,num_clustres)
        centroids = centroids.cuda()
    return centroids


def centroids_init(data,num_clustres):
    num_example = data.shape[0]
    random_ids = np.random.permutation(num_example)
    centroids = data[random_ids[:num_clustres]]
    return centroids

def centroids_find_closest(data,centroids):
    num_example = data.shape[0]
    num_centroids = centroids.shape[0]
    closest_centroids_ids = torch.empty((num_example, 1))
    for example_index in range(num_example):
        distance = torch.empty((num_centroids,1))
        for centroid_index in range(num_centroids):
            distance_diff = data[example_index] - centroids[centroid_index]
            distance[centroid_index] = torch.sum(distance_diff**2)
        closest_centroids_ids[example_index] = torch.argmin(distance,dim=0)
    return closest_centroids_ids

def centroids_compute(data,closest_centroids_ids,num_clustres):
    num_feature = data.shape[1]
    centroids = torch.empty((num_clustres,num_feature))
    closest_centroids_ids_np = closest_centroids_ids.numpy()
    for centroid_id in range(num_clustres):
        centroids_ids = np.where(closest_centroids_ids_np.flatten() == centroid_id)
        centroids[centroid_id] = torch.mean(data[centroids_ids],dim=0)

    return centroids


