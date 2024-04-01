from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
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
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data,n_clusters,eval):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42, n_init='auto')

    # Fit KMeans to the standardized data
    kmeans.fit(standardized_data)

    # Get predicted labels
    predicted_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sse = 0
    
    if eval=="sse":
        for i in range(len(data)):
            cluster_idx = predicted_labels[i]
            centroid = centroids[cluster_idx]
            sse += np.linalg.norm(data[i] - centroid) ** 2
        return predicted_labels, sse
    else:
        inertia = kmeans.inertia_
        return predicted_labels, inertia



def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    b, b_labels = make_blobs(center_box=(-20,20), n_samples=20, centers=5, random_state=12)

    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = [b[0],b[1], b_labels]

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    sse_values = []
    k_sse = []
    # Define range of k values
    k_values = range(1, 9)

    for k in k_values:
        predicted_labels, sse = fit_kmeans(b,n_clusters=k,eval="sse")
        sse_values.append(sse)
        k_sse.append([k,sse])
    
    plt.plot(k_values, sse_values, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig("2C: SSE plot")

    # Find the optimal k using the elbow method
    # Optimal k is often the point where the rate of decrease of SSE slows down (elbow point)
    deltas = np.diff(sse_values, 2)
    optimal_k = k_values[np.argmax(deltas) + 1]

    print("Optimal number of clusters (k) based on the elbow method:", optimal_k)
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = k_sse

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    inertia_values = []
    k_inertia = []
    # Define range of k values
    k_values = range(1, 9)

    for k in k_values:
        predicted_labels, inertia = fit_kmeans(b,n_clusters=k,eval="sse")
        inertia_values.append(inertia)
        k_inertia.append([k,inertia])
    
    plt.plot(k_values, inertia_values, marker='o', linestyle='-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia from K Means function')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig("2D: Inertia plot")

    # Find the optimal k using the elbow method
    # Optimal k is often the point where the rate of decrease of SSE slows down (elbow point)
    deltas = np.diff(inertia_values, 2)
    optimal_k = k_values[np.argmax(deltas) + 1]

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = k_inertia

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
