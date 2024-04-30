"""
Work with Spectral clustering.
Do not use global variables!
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
import pickle

######################################################################
#####     CHECK THE PARAMETERS     ########
######################################################################
def simple_kmeans(data, k, iterations):
    """A simple k-means implementation."""
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(iterations):
        # Assign clusters
        distances = np.array([np.sum((data - centroid) ** 2, axis=1) for centroid in centroids])
        clusters = np.argmin(distances, axis=0)
        
        # Update centroids
        for i in range(k):
            if np.any(clusters == i):
                centroids[i] = np.mean(data[clusters == i], axis=0)
    return clusters

def compute_sse(data, labels, centroids):
    sse = 0.0
    for i in range(5):
        cluster_data = data[labels == i]
        centroid = centroids[i]
        sse += np.sum((cluster_data - centroid)**2)
    return sse

def adjusted_rand_score(labels_true, labels_pred):
    # Find the contingency table
    contingency_matrix = np.histogram2d(labels_true, labels_pred, bins=(np.unique(labels_true).size, np.unique(labels_pred).size))[0]

    # Sum the combinatorics for each row and column
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency_matrix, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency_matrix, axis=0))

    # Sum the combinatorics for the whole matrix
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency_matrix.flatten())

    # Calculate the expected index (as if the agreement is purely random)
    expected_index = sum_comb_c * sum_comb_k / comb(contingency_matrix.sum(), 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari


params_dict = {
    'sigma': 0.4,  # Creates 100 values from 0.1 to 10 evenly spaced
    'k': 5  # Number of clusters
}
def spectral(
    data: NDArray[np.floating], labels: NDArray[np.int32], params_dict: dict
) -> tuple[
    NDArray[np.int32] | None, float | None, float | None, NDArray[np.floating] | None
]:
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            distance_matrix[i, j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))
    W =np.exp(- (distance_matrix ** 2) / (2 * (params_dict['sigma']) ** 2))
    D = np.diag(W.sum(axis=1))
    L = D-W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = eigenvalues[0:params_dict['k']]
    # Sort eigenvectors by eigenvalues
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    computed_labels = simple_kmeans(eigenvectors[:, :params_dict['k']], params_dict['k'], iterations=300)
    ARI = adjusted_rand_score(labels, computed_labels)
    centroids = np.array([data[computed_labels == i].mean(axis=0) for i in range(params_dict['k'])])
    SSE = compute_sse(data, computed_labels, centroids)
    """
    Implementation of the Spectral clustering  algorithm only using the `numpy` module.

    Arguments:
    - data: a set of points of shape 50,000 x 2.
    - dict: dictionary of parameters. The following two parameters must
       be present: 'sigma', and 'k'. There could be others.
       params_dict['sigma']:  in the range [.1, 10]
       params_dict['k']: the number of clusters, set to five.

    Return values:
    - computed_labels: computed cluster labels
    - SSE: float, sum of squared errors
    - ARI: float, adjusted Rand index
    - eigenvalues: eigenvalues of the Laplacian matrix
    """
    return computed_labels, SSE, ARI, eigenvalues


def spectral_clustering():
    """
    Performs DENCLUE clustering on a dataset.

    Returns:
        answers (dict): A dictionary containing the clustering results.
    """

    answers = {}

    # Return your `spectral` function
    answers["spectral_function"] = spectral
    cluster_data = np.load("question1_cluster_data.npy")
    cluster_labels = np.load("question1_cluster_labels.npy")
    # Generate random indices from the total number of rows in cluster_data
    selected_indices = np.random.choice(cluster_data.shape[0], 5000, replace=False)

    # Use the selected indices to retrieve the corresponding points and labels
    selected_points = cluster_data[selected_indices]
    selected_labels = cluster_labels[selected_indices]

    indices = np.arange(selected_points.shape[0])
    np.random.shuffle(indices)

    # Split the shuffled indices into 5 groups of 1000 each
    data = [selected_points[indices[i:i + 1000]] for i in range(0, 5000, 1000)]
    labels = [selected_labels[indices[i:i + 1000]] for i in range(0, 5000, 1000)]

    # Work with the first 10,000 data points: data[0:10000]
    # Do a parameter study of this data using Spectral clustering.
    # Minimmum of 10 pairs of parameters ('sigma' and 'xi').

    # Create a dictionary for each parameter pair ('sigma' and 'xi').
    groups = {}
    sigma_values = np.linspace(0.1, 10, 5)
    eigenvalues_group = []
    for i, sigma in enumerate(sigma_values):
        params_dict = {'sigma': sigma, 'k': 5}
        computed_labels, SSE, ARI, eigenvalues = spectral(data[i], labels[i],params_dict)
        groups[i] = {"sigma": sigma, "ARI": ARI, "SSE": SSE}
        eigenvalues_group.append(eigenvalues)

        if i == 0:
            answers["1st group, SSE"] = SSE

    # For the spectral method, perform your calculations with 5 clusters.
    # In this cas,e there is only a single parameter, σ.

    # data for data group 0: data[0:10000]. For example,
    # groups[0] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # data for data group i: data[10000*i: 10000*(i+1)], i=1, 2, 3, 4.
    # For example,
    # groups[i] = {"sigma": 0.1, "ARI": 0.1, "SSE": 0.1}

    # groups is the dictionary above
    answers["cluster parameters"] = groups

    # Identify the cluster with the lowest value of ARI. This implies
    # that you set the cluster number to 5 when applying the spectral
    # algorithm.

    # Create two scatter plots using `matplotlib.pyplot`` where the two
    # axes are the parameters used, with \sigma on the horizontal axis
    # and \xi and the vertical axis. Color the points according to the SSE value
    # for the 1st plot and according to ARI in the second plot.

    # Choose the cluster with the largest value for ARI and plot it as a 2D scatter plot.
    # Do the same for the cluster with the smallest value of SSE.
    # All plots must have x and y labels, a title, and the grid overlay.

    # Plot is the return value of a call to plt.scatter()
    sigmas = [group["sigma"] for group in groups.values()]
    ARI_values = [group["ARI"] for group in groups.values()]
    SSE_values = [group["SSE"] for group in groups.values()]

    #Choose the cluster with the largest value
    eigenvalues = list(eigenvalues_group)
    max_key = max(groups, key=lambda k: groups[k]['ARI'])
    data_largest_ARI = data[max_key]
    labels_largest_ARI = labels[max_key]

    plt.figure()
    plot_ARI = plt.scatter(data_largest_ARI[:, 0], data_largest_ARI[:, 1], c=labels_largest_ARI, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatter Plot with Largest ARI (σ = {groups[max_key]["sigma"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with Largest SSE"] = plot_ARI

    #Choose the cluster with the smallest value
    min_key  = min(groups, key=lambda k: groups[k]['ARI'])
    data_smallest_SSE = data[min_key]
    labels_smallest_SSE = labels[min_key]
    plt.figure()
    plot_SSE = plt.scatter(data_smallest_SSE[:, 0], data_smallest_SSE[:, 1], c=labels_smallest_SSE, cmap='viridis', label='Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Scatter Plot with Largest ARI (σ = {groups[min_key]["sigma"]})')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    answers["cluster scatterplot with smallest SSE"] = plot_SSE


    # Plot of the eigenvalues (smallest to largest) as a line plot.
    # Use the plt.plot() function. Make sure to include a title, axis labels, and a grid.
    sorted_eigenvalues = np.sort(eigenvalues)
    plt.figure()
    plot_eig = plt.plot(sorted_eigenvalues, marker='o')  # marker='o' makes the individual eigenvalues more visible
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues from Smallest to Largest')
    plt.grid(True)
    answers["eigenvalue plot"] = plot_eig

    # Pick the parameters that give the largest value of ARI, and apply these
    # parameters to datasets 1, 2, 3, and 4. Compute the ARI for each dataset.
    # Calculate mean and standard deviation of ARI for all five datasets.
    data_set_0 = [data[0], labels[0]]
    data_set_1 = [data[1], labels[1]]
    data_set_2 = [data[2], labels[2]]
    data_set_3 = [data[3], labels[3]]
    data_set_4 = [data[4], labels[4]]
    data_sets = [data_set_0, data_set_1, data_set_2, data_set_3, data_set_4]
    largest_ARI_parameters = {'sigma': 0.1, 'k': 5}
    ARIs = []
    SSEs = []
    for data, labels in data_sets:
        computed_labels, SSE, ARI, eigenvalues = spectral(data, labels, largest_ARI_parameters)
        ARIs.append(ARI)
        SSEs.append(SSE)

    answers["mean_ARIs"] = np.mean(ARIs)
    answers["std_ARIs"] = np.std(ARIs)
    answers["mean_SSEs"] = np.mean(SSEs)
    answers["std_SSEs"] = np.std(SSEs)

    return answers
# ----------------------------------------------------------------------
if __name__ == "__main__":
    all_answers = spectral_clustering()
    with open("spectral_clustering.pkl", "wb") as fd:
        pickle.dump(all_answers, fd, protocol=pickle.HIGHEST_PROTOCOL)
