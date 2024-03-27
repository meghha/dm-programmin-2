import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import *

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(data,linkage_type,n_clusters):
    Z = AgglomerativeClustering(linkage=linkage_type,n_clusters=n_clusters).fit(data)
    return Z.labels_


def fit_modified(data,linkage_type,n_clusters):
    Z = linkage(data, method=linkage_type)

    distances = np.diff(Z[:, 2], 2)
    max_rate_idx = np.argmax(distances)
    cutoff_distance = Z[max_rate_idx, 2]

    Z = AgglomerativeClustering(linkage=linkage_type,n_clusters=n_clusters,distance_threshold=cutoff_distance).fit(data)
    return Z.labels_

def plots(datasets, dataset_names, linkage_list,filenm,ques='B'):
    # Create a big figure
    plt.figure(figsize=(20,16))

    # Loop over each dataset
    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names), start=1):

        for j, linkage_ in enumerate(linkage_list, start=1):
            
            if ques=="B":
                predicted_labels = fit_hierarchical_cluster(dataset, linkage_type=linkage_,n_clusters=2)
            else:
                predicted_labels = fit_modified(dataset, linkage_type=linkage_,n_clusters=None)
            # Plot the scatter plot
            plt.subplot(4, 5, i + (j - 1) * 5)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=predicted_labels, cmap='viridis')
            plt.title(f"{dataset_name}\nK={linkage_}")
            plt.xlabel("Dataset")

    plt.tight_layout()
    plt.savefig(filenm)

def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    random_state = 42
    nc, nc_labels = make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    nm, nm_labels = make_moons(n_samples=100, noise=.05, random_state=random_state)
    b, b_labels = make_blobs(n_samples=100, random_state=random_state)
    bvv, bvv_labels = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    add, add_labels = make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = np.dot(add, transformation)

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {}

    dct["nc"] = [nc, nc_labels]
    dct["nm"] = [nm, nm_labels]
    dct["bvv"] = [bvv, bvv_labels]
    dct["add"] = [add, add_labels]
    dct["b"] = [b, b_labels]


    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    datasets = [nc, nm, bvv, add, b]
    dataset_names = ["Noisy Circles", "Noisy Moons", "Blobs with Varied Variances", 
                    "Anisotropicly Distributed Data", "Blobs"]

    # Define the number of clusters for each row
    linkage_list = ["single","complete","ward","average"]

    plots(datasets, dataset_names, linkage_list,"4B: Hierarchical Clusters")

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    plots(datasets, dataset_names, linkage_list,"4C: Hierarchical Clusters with distance cutoff",'C')
    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
