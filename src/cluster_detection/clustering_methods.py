from dataclasses import asdict, dataclass
from typing import Dict, List, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN

from .blob_detection import blob_detection
from .utils import create_all_points, identity, make_circle, reshape_col2d


@dataclass
class ClusterResult:
    """Data class for clustering results."""

    cluster_labels: NDArray
    cluster_centers: NDArray
    cluster_radii: NDArray
    loc_per_cluster: NDArray

    def __getitem__(self, key: str) -> NDArray:
        """
        Make the class subscriptable using dictionary-style access.

        Args:
            key: String key to access the attribute

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' is not a valid attribute")


@dataclass
class AnalysisResult:
    """Data class for true/false positive analysis results."""

    true_positive_num: int
    false_positives: int
    true_center_TP_center_error: NDArray
    true_center_TP_scale_error: NDArray

    def __getitem__(self, key: str) -> Union[int, NDArray]:
        """
        Make the class subscriptable using dictionary-style access.

        Args:
            key: String key to access the attribute

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' is not a valid attribute")

    def to_dict(self) -> dict[str, Union[int, NDArray]]:
        """Convert the analysis result to a dictionary."""
        return asdict(self)

    def keys(self) -> list[str]:
        """Return available keys, similar to dictionary keys."""
        return list(self.__annotations__.keys())


def DBSCAN_TP_FP_center_scale_error(
    initial_dict: Dict[str, Union[List, NDArray]],
    localization_form: Union[Dict, NDArray],
    D: Union[float, int],
    minPts: int,
    convert: bool = False,
    threshold: float = 1.0,
) -> AnalysisResult:
    """
    Analyze DBSCAN clustering results to identify true/false positives and calculate errors.

    Args:
        initial_dict: Dictionary with true centers and scales
            Format: {"initial_centers": [[x1,y1],...], "initial_scale": [r1,r2,...]}
        localization_form: Points as dict or array
        D: Maximum distance between points
        minPts: Minimum points for core point
        convert: Whether to convert dict to array
        threshold: Maximum distance for true positive classification

    Returns:
        AnalysisResult containing true/false positive counts and errors
    """
    points = create_all_points(localization_form) if convert else localization_form
    labels = perform_DBSCAN(points, D, minPts, convert=False)

    unique_labels = np.unique(labels[labels != -1])
    if len(unique_labels) == 0:
        return AnalysisResult(0, 0, np.array([]), np.array([]))

    # Process clusters
    cluster_data = _process_cluster_metrics(points, labels, unique_labels)
    cluster_scales = np.array(cluster_data["scales"])
    cluster_centers = np.array(cluster_data["centers"])

    # Analyze results
    true_centers = np.array(initial_dict["initial_centers"])[:, 0:2]
    true_scale = np.array(initial_dict["initial_scale"])

    analysis_results = _calculate_tp_fp_errors(
        true_centers, true_scale, cluster_centers, cluster_scales, threshold
    )

    return AnalysisResult(**analysis_results)


def _process_cluster_metrics(
    points: NDArray, labels: NDArray, unique_labels: NDArray
) -> Dict[str, List]:
    """Helper function to process cluster metrics."""
    results = {"centers": [], "scales": []}

    for label in unique_labels:
        if label != -1:
            cluster_points = points[labels == label]
            circle_data = make_circle(cluster_points)
            results["scales"].append(circle_data[2])
            results["centers"].append(circle_data[0:2])

    return results


def _calculate_tp_fp_errors(
    true_centers: NDArray,
    true_scale: NDArray,
    cluster_centers: NDArray,
    cluster_scales: NDArray,
    threshold: float,
) -> Dict:
    """Helper function to calculate true/false positives and errors."""
    true_center_TP_center_error = np.full(len(true_centers), np.nan)
    true_center_TP_scale_error = np.full(len(true_centers), np.nan)

    for i, true_center in enumerate(true_centers):
        min_error = np.inf
        best_match_idx = -1

        for j, found_center in enumerate(cluster_centers):
            error = np.linalg.norm(true_center - found_center)
            if error < threshold and error < min_error:
                min_error = error
                best_match_idx = j

        if best_match_idx != -1:
            true_center_TP_center_error[i] = min_error
            true_center_TP_scale_error[i] = (
                np.abs(true_scale[i] - cluster_scales[best_match_idx]) / true_scale[i]
            )

    true_positives = np.sum(~np.isnan(true_center_TP_center_error))
    false_positives = len(cluster_centers) - true_positives

    return {
        "true_positive_num": true_positives,
        "false_positives": false_positives,
        "true_center_TP_center_error": true_center_TP_center_error,
        "true_center_TP_scale_error": true_center_TP_scale_error,
    }


def true_positive_and_error(
    initial_dict: Dict[str, Union[List, NDArray]],
    found_dict: Dict[str, List],
    center_threshold: float = 0.5,
) -> AnalysisResult:
    """
    Calculate true/false positives and errors between initial and found blobs.

    Args:
        initial_dict: Dictionary with true centers and scales
            Format: {"initial_centers": [[x1,y1],...], "initial_scale": [r1,r2,...]}
        found_dict: Dictionary with found centers and scales
            Format: {"Fitted": [[x1,y1,r1,r2],...], "Scale": [r1,r2,...]}
        center_threshold: Maximum distance for true positive classification

    Returns:
        AnalysisResult containing true/false positive counts and errors
    """
    # Extract centers and scales
    true_centers = np.array(initial_dict["initial_centers"])
    true_scale = np.array(initial_dict["initial_scale"])

    found_centers = np.array([blob[:2] for blob in found_dict["Fitted"]])
    found_scale = np.array(
        [np.mean([blob[2], blob[3]]) for blob in found_dict["Fitted"]]
    )

    # Initialize error arrays
    true_center_TP_center_error = np.full(len(true_centers), np.nan)
    true_center_TP_scale_error = np.full(len(true_centers), np.nan)

    # Calculate errors for each true center
    for i, true_center in enumerate(true_centers):
        min_error = np.inf
        best_match_idx = -1

        for j, found_center in enumerate(found_centers):
            error = np.linalg.norm(true_center - found_center)
            if error < center_threshold and error < min_error:
                min_error = error
                best_match_idx = j

        if best_match_idx != -1:
            true_center_TP_center_error[i] = min_error
            true_center_TP_scale_error[i] = (
                np.abs(true_scale[i] - found_scale[best_match_idx]) / true_scale[i]
            )

    # Calculate final metrics
    true_positives = np.sum(~np.isnan(true_center_TP_center_error))
    false_positives = len(found_centers) - true_positives

    return AnalysisResult(
        true_positive_num=true_positives,
        false_positives=false_positives,
        true_center_TP_center_error=true_center_TP_center_error,
        true_center_TP_scale_error=true_center_TP_scale_error,
    )


def scale_space_plus_blob_detection(
    img: NDArray, blob_parameters: Dict, fitting_parameters: Dict, show: bool = False
) -> Dict[str, NDArray]:
    """
    Perform scale-space blob detection with optional visualization.

    Args:
        img: Input image array
        blob_parameters: Parameters for blob detection
            {
                'threshold': float = 1e-4,
                'overlap': float = 0.5,
                'median': bool = False,
                'min_sigma': float = 1,
                'max_sigma': float = 2,
                'num_sigma': int = 500,
                'logscale': bool = False,
                'detection': str = 'bp'
            }
        fitting_parameters: Parameters for blob fitting
        show: Whether to display visualization

    Returns:
        Dict containing fitted and scale-space blobs
    """
    detector = blob_detection(
        img,
        threshold=blob_parameters.get("threshold", 1e-4),
        overlap=blob_parameters.get("overlap", 0.5),
        median=blob_parameters.get("median", False),
        min_sigma=blob_parameters.get("min_sigma", 1),
        max_sigma=blob_parameters.get("max_sigma", 2),
        num_sigma=blob_parameters.get("num_sigma", 500),
        logscale=blob_parameters.get("logscale", False),
        verbose=True,
    )

    detector._update_fitting_parameters(kwargs=fitting_parameters)
    blobs = detector.detection(type=blob_parameters.get("detection", "bp"))

    # Reshape the blob coordinates
    blobs["Fitted"] = reshape_col2d(blobs["Fitted"], [1, 0, 2, 3])
    blobs["Scale"] = reshape_col2d(blobs["Scale"], [1, 0, 2])

    if show:
        if not fitting_parameters["radius_func"]:
            fitting_parameters["radius_func"] = identity
        _visualize_blobs(img, blobs, fitting_parameters)

    return blobs


def _visualize_blobs(
    img: NDArray, blobs: Dict[str, NDArray], fitting_parameters: Dict
) -> None:
    """Helper function to visualize detected blobs."""
    _, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    for blob in blobs["Fitted"]:
        radius = np.max(fitting_parameters["radius_func"](blob[2:4]))
        circle = plt.Circle(blob[0:2], radius, color="r", fill=False)
        ax.add_patch(circle)

    ax.set_aspect("equal")
    plt.show()

    print(f"Scale-space plus blob detection found {len(blobs['Fitted'])} blobs")
    print("Fitted blobs (x,y,r):\n", blobs["Fitted"])
    print("Scale-space plus blobs (x,y,r):\n", blobs["Scale"])


def tp_fp_scale_error_combined(
    img: NDArray,
    blob_parameters: Dict,
    fitting_parameters: Dict,
    initial_dict: Dict[str, Union[List, NDArray]],
    center_threshold: float = 0.5,
    show: bool = False,
) -> AnalysisResult:
    """combined function for true positive and false positive analysis  and scale_space_plus_blob_detection"""

    blobs = scale_space_plus_blob_detection(
        img=img,
        blob_parameters=blob_parameters,
        fitting_parameters=fitting_parameters,
        show=show,
    )
    return true_positive_and_error(initial_dict, blobs, center_threshold)


def perfrom_DBSCAN_Cluster(
    localizations: NDArray, D: Union[float, int], minP: int, show: bool = False
) -> ClusterResult:
    """
    Perform DBSCAN clustering on localizations.

    Args:
        localizations: Array of points [[x,y],...]
        D: Maximum distance between points for neighborhood consideration
        minP: Minimum points for core point
        show: Whether to display visualization

    Returns:
        ClusterResult object containing cluster information
    """
    points = localizations[:, 0:2]
    cluster_labels = perform_DBSCAN(points, D, minP, convert=False)

    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    n_clusters = len(unique_labels)

    cluster_centers = np.zeros((n_clusters, 2))
    cluster_radii = np.zeros(n_clusters)
    loc_per_cluster = np.zeros(n_clusters)

    for i, label in enumerate(unique_labels):
        cluster = points[cluster_labels == label]
        if len(cluster) < 3:
            continue
        hull = ConvexHull(cluster)
        cluster_centers[i] = np.mean(cluster[hull.vertices], axis=0)
        cluster_radii[i] = np.mean(
            np.linalg.norm(cluster[hull.vertices] - cluster_centers[i], axis=1)
        )
        loc_per_cluster[i] = len(cluster)

    if show:
        _visualize_clusters(
            points, cluster_labels, cluster_centers, cluster_radii, unique_labels
        )

    return ClusterResult(
        cluster_labels, cluster_centers, cluster_radii, loc_per_cluster
    )


def perform_HDBSCAN_Cluster(
    localizations: NDArray,
    min_cluster_size: int,
    min_samples: int,
    show: bool = False,
) -> ClusterResult:
    """
    Perform HDBSCAN clustering on localizations.

    Args:
        localizations: Array of points [[x,y],...]
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples for core point
        show: Whether to display visualization

    Returns:
        ClusterResult object containing cluster information
    """
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(localizations)

    unique_labels = np.unique(cluster_labels[cluster_labels != -1])
    cluster_data = _process_clusters(localizations, cluster_labels, unique_labels)

    if show:
        _visualize_clusters(localizations, cluster_labels, cluster_data)

    return ClusterResult(
        cluster_labels,
        np.array(cluster_data["centers"]),
        np.array(cluster_data["radii"]),
        np.array(cluster_data["sizes"]),
    )


def _process_clusters(
    points: NDArray, labels: NDArray, unique_labels: NDArray
) -> Dict[str, List]:
    """Helper function to process cluster data."""
    results = {"centers": [], "radii": [], "sizes": []}

    for label in unique_labels:
        cluster_points = points[labels == label]
        center = np.mean(cluster_points, axis=0)
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1))

        results["centers"].append(center)
        results["radii"].append(radius)
        results["sizes"].append(len(cluster_points))

    return results


def _visualize_clusters(
    points: NDArray,
    labels: NDArray,
    centers: NDArray,
    radii: NDArray,
    unique_labels: NDArray,
) -> None:
    """Helper function to visualize clusters."""
    _, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=labels, marker="o", s=10)
    ax.scatter(centers[:, 0], centers[:, 1], c=labels[unique_labels], marker="x", s=50)

    for center, radius in zip(centers, radii):
        circle = plt.Circle(center, radius, color="r", fill=False)
        ax.add_patch(circle)

    ax.set_aspect("equal")
    plt.show()
    print(f"Found {len(unique_labels)} clusters")
    print("Cluster centers (x,y):\n", centers)
    print("Cluster radii:\n", radii)


def perform_DBSCAN(
    localization_form: Union[Dict, NDArray],
    D: Union[float, int],
    minPts: int,
    convert: bool = False,
) -> NDArray:
    """
    Perform DBSCAN clustering.

    Args:
        localization_form: Points as dict or array
        D: Maximum distance between points
        minPts: Minimum points for core point
        convert: Whether to convert dict to array

    Returns:
        Array of cluster labels
    """
    points = create_all_points(localization_form) if convert else localization_form
    return DBSCAN(eps=D, min_samples=minPts).fit(points).labels_


def scale_utility(img_map: NDArray, threshold: float) -> Dict:
    """Deprecated: Use blob_detection instead."""
    warn(
        "This function is deprecated. Use blob_detection instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    detector = blob_detection(
        path=img_map,
        median=False,
        threshold=threshold,
        min_sigma=1 / np.sqrt(2),
        max_sigma=10 / np.sqrt(2),
        num_sigma=30,
        overlap=0.5,
        logscale=False,
        verbose=True,
    )
    detector._update_fitting_parameters(
        kwargs={
            "mask_size": 5,
            "plot_fit": False,
            "fitting_image": "Original",
            "radius_func": None,
            "sigma_range": 2,
            "centroid_range": 2,
        }
    )
    return detector.detection(type="bp")
