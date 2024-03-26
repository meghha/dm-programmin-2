import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #
from sklearn.datasets import *
# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data,n_clusters):

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="random")

    # Fit KMeans to the standardized data
    kmeans.fit(standardized_data)

    # Get predicted labels
    predicted_labels = kmeans.labels_
    return predicted_labels

def plots(datasets, dataset_names, n_clusters_list, filenm):
    # Create a big figure
    plt.figure(figsize=(20, 16))

    # Loop over each dataset
    for i, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names), start=1):
        # Loop over each number of clusters for each dataset
        for j, n_clusters in enumerate(n_clusters_list, start=1):
            # Fit KMeans and get predicted labels
            predicted_labels = fit_kmeans(dataset, n_clusters)

            # Plot the scatter plot
            plt.subplot(4, 5, i + (j - 1) * 5)
            plt.scatter(dataset[:, 0], dataset[:, 1], c=predicted_labels, cmap='viridis')
            plt.title(f"{dataset_name}\nK={n_clusters}")
            plt.xlabel("Dataset")
            plt.ylabel("Number of clusters")

    plt.tight_layout()
    plt.savefig(filenm)

def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Set random state for reproducibility
    random_state = 42
    nc, nc_labels = make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    nm, nm_labels = make_moons(n_samples=100, noise=.05, random_state=random_state)
    b, b_labels = make_blobs(n_samples=100, random_state=random_state)
    bvv, bvv_labels = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    add, add_labels = make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    add = np.dot(add, transformation)
    

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)

    
    dct = answers["1A: datasets"] = {}

    dct["nc"] = [nc, nc_labels]
    dct["nm"] = [nm, nm_labels]
    dct["bvv"] = [bvv, bvv_labels]
    dct["add"] = [add, add_labels]
    dct["b"] = [b, b_labels]

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    # Define the datasets and their names

    datasets = [nc, nm, bvv, add, b]
    dataset_names = ["Noisy Circles", "Noisy Moons", "Blobs with Varied Variances", 
                    "Anisotropicly Distributed Data", "Blobs"]

    # Define the number of clusters for each row
    n_clusters_list = [2, 3, 5, 10]

    plots(datasets, dataset_names, n_clusters_list,"1C: KMeans Clusters")

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {"bvv": [2,3], "add": [3],"b":[3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc", "nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """
    n_clusters_list = [2, 3]

    plots(datasets,dataset_names,n_clusters_list,"Testing 1")
    plots(datasets,dataset_names,n_clusters_list,"Testing 2")
    plots(datasets,dataset_names,n_clusters_list,"Testing 3")
    plots(datasets,dataset_names,n_clusters_list,"Testing 4")
    plots(datasets,dataset_names,n_clusters_list,"Testing 5")

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm","add"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)

"Why do you say all the datasets are sensitive to choice of initialization?"