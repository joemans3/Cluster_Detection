import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN
from cluster_detection.utils import reshape_col2d
from cluster_detection.blob_detection import blob_detection
import matplotlib.pyplot as plt


def scale_space_plus_blob_detection(
    img, blob_parameters, fitting_parameters, show=False
):
    """
    Wrapper for the blob_detection function in the blob_detection.py file
    See the blob_detection.py file for more details on the parameters
    """

    blob_class = blob_detection(
        img,
        threshold=blob_parameters.get("threshold", 1e-4),
        overlap=blob_parameters.get("overlap", 0.5),
        median=blob_parameters.get("median", False),
        min_sigma=blob_parameters.get("min_sigma", 1),
        max_sigma=blob_parameters.get("max_sigma", 2),
        num_sigma=blob_parameters.get("num_sigma", 500),
        logscale=blob_parameters.get("log_scale", False),
        verbose=True,
    )
    blob_class._update_fitting_parameters(kwargs=fitting_parameters)
    blob = blob_class.detection(type=blob_parameters.get("detection", "bp"))
    fitted = blob["Fitted"]
    scale = blob["Scale"]
    blob["Fitted"] = reshape_col2d(fitted, [1, 0, 2, 3])
    blob["Scale"] = reshape_col2d(scale, [1, 0, 2])
    blobs = blob
    if show:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        # make a circle with the radius of the blob
        for i in range(len(blobs["Fitted"])):
            # get the radius
            radius = fitting_parameters["radius_func"](blobs["Fitted"][i][2:4])
            # get the center
            center = blobs["Fitted"][i][0:2]
            # get the circle
            circle = plt.Circle(center, radius, color="r", fill=False)
            # add the circle to the axis
            ax.add_patch(circle)
        # aspect ratio
        ax.set_aspect("equal")
        plt.show()
    print(
        "Scale-space plus blob detection found {0} blobs".format(len(blobs["Fitted"]))
    )
    print("Fitted blobs (x,y,r): \n", blobs["Fitted"])
    print("Scale-space plus blobs (x,y,r): \n", blobs["Scale"])

    return blobs


def perfrom_DBSCAN_Cluster(localizations, D, minP, show=False):
    """
    Parameters:
    -----------
    localizations: np.ndarray
        Numpy array of the localizations in the form [[x,y],...]
    D: float, in the units of the localizations
        The maximum distance between two samples for one to be considered as in the neighborhood of the other
    minP: int
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    --------
    cluster_labels: np.ndarray
        Numpy array of the cluster labels in the form [0,0,1,1,2,2,...]
    cluster_centers: np.ndarray
        Numpy array of the cluster centers in the form [[x,y],...]
    cluster_radii: np.ndarray
        Numpy array of the cluster radii in the form [r1,r2,...]
    loc_per_cluster: np.ndarray
        Numpy array of the number of localizations per cluster in the form [n1,n2,...]
    """
    # get the DBSCAN object
    db = DBSCAN(eps=D, min_samples=minP)
    # fit the data
    db.fit(localizations)
    # get the labels
    cluster_labels = db.labels_
    # get the unique labels without -1
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    # get the cluster centers
    cluster_centers = np.zeros((len(unique_labels), 2))
    # get the cluster radii
    cluster_radii = np.zeros(len(unique_labels))
    # get the number of localizations per cluster
    loc_per_cluster = np.zeros(len(unique_labels))
    # loop over the unique labels
    for i in range(len(unique_labels)):
        # get the cluster label
        cluster_label = unique_labels[i]
        # get the cluster
        cluster = localizations[cluster_labels == cluster_label]
        # get the convex hull
        hull = ConvexHull(cluster)
        # get the cluster center
        cluster_centers[i] = np.mean(cluster[hull.vertices], axis=0)
        # get the cluster radius
        cluster_radii[i] = np.mean(
            np.linalg.norm(cluster[hull.vertices] - cluster_centers[i], axis=1)
        )
        # get the number of localizations per cluster
        loc_per_cluster[i] = len(cluster)

    if show:
        fig, ax = plt.subplots()
        ax.scatter(
            localizations[:, 0], localizations[:, 1], c=cluster_labels, marker="o", s=10
        )
        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            c=cluster_labels[unique_labels],
            marker="x",
            s=50,
        )
        for i in range(len(cluster_radii)):
            circle = plt.Circle(
                cluster_centers[i], cluster_radii[i], color="r", fill=False
            )
            ax.add_patch(circle)
        # aspect ratio
        ax.set_aspect("equal")
        plt.show()
    # print the number of clusters
    print("DBSCAN found {0} clusters".format(len(unique_labels)))
    print("Cluster centers (x,y): \n", cluster_centers)
    print("Cluster radii: \n", cluster_radii)
    print("Number of localizations per cluster: \n", loc_per_cluster)
    return cluster_labels, cluster_centers, cluster_radii, loc_per_cluster


# function for performing hdbscan clustering in a similar way to the DBSCAN clustering
def perform_HDBSCAN_Cluster(localizations, min_cluster_size, min_samples, show=False):
    """
    Parameters:
    -----------
    localizations: np.ndarray
        Numpy array of the localizations in the form [[x,y],...]
    min_cluster_size: int
        The minimum size of clusters
    min_samples: int
        The number of samples in a neighborhood for a point to be considered as a core point.
    show: bool
        Whether or not to display a plot of the clusters

    Returns:
    --------
    cluster_labels: np.ndarray
        Numpy array of the cluster labels in the form [0,0,1,1,2,2,...]
    cluster_centers: np.ndarray
        Numpy array of the cluster centers in the form [[x,y],...]
    cluster_radii: np.ndarray
        Numpy array of the cluster radii in the form [r1,r2,...]
    loc_per_cluster: np.ndarray
        Numpy array of the number of localizations per cluster in the form [n1,n2,...]
    """
    # get the HDBSCAN object
    hdbscan_clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    # fit the data
    hdbscan_clusterer.fit(localizations)
    # get the labels
    cluster_labels = hdbscan_clusterer.labels_
    # get the unique labels without -1
    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    # get the cluster centers
    cluster_centers = []
    cluster_radii = []
    loc_per_cluster = []
    for label in unique_labels:
        # get the indices of the localizations in the cluster
        indices = np.where(cluster_labels == label)[0]
        # get the localizations in the cluster
        cluster_localizations = localizations[indices]
        # get the center of the cluster
        center = np.mean(cluster_localizations, axis=0)
        cluster_centers.append(center)
        # get the radius of the cluster
        radius = np.max(np.linalg.norm(cluster_localizations - center, axis=1))
        cluster_radii.append(radius)
        # get the number of localizations in the cluster
        loc_per_cluster.append(len(indices))
    cluster_centers = np.array(cluster_centers)
    cluster_radii = np.array(cluster_radii)
    loc_per_cluster = np.array(loc_per_cluster)
    # plot the clusters if show is True
    if show:
        plt.scatter(
            localizations[:, 0], localizations[:, 1], c=cluster_labels, cmap="viridis"
        )
        plt.axis("equal")
        plt.show()
    print("HDBSCAN found {0} clusters".format(len(unique_labels)))
    print("Cluster centers (x,y): \n", cluster_centers)
    print("Cluster radii: \n", cluster_radii)
    print("Number of localizations per cluster: \n", loc_per_cluster)

    return cluster_labels, cluster_centers, cluster_radii, loc_per_cluster
