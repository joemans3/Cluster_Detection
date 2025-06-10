# Detecting Clusters in Microscopy Images

A Python library providing advanced clustering and blob detection capabilities for scientific image analysis and point cloud data processing, with automated parameter optimization.

## Key Features

- Scale-space blob detection with customizable parameters
- Automated parameter optimization for both DBSCAN and Scale-space detection
- Ground truth comparison and metric evaluation


## Installation

```bash
pip install cluster_detection
```


## Function Documentation

### DBSCAN Clustering

```python
from cluster_detection import perform_DBSCAN_Cluster
results = perform_DBSCAN_Cluster(localizations, D, minP, show=False)
```

Performs DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering on point cloud data.

**Parameters:**
- `localizations`: numpy.ndarray - Array of points in format [[x,y],...]
- `D`: float or int - Maximum distance between points for neighborhood consideration
- `minP`: int - Minimum points required to form a core point
- `show`: bool - Whether to display visualization of clusters (default: False)

**Returns:**
- `ClusterResult` object containing:
  - cluster_labels: Labels for each point
  - cluster_centers: Centers of detected clusters
  - cluster_radii: Radii of detected clusters
  - loc_per_cluster: Number of points per cluster

### HDBSCAN Clustering

```python
from cluster_detection import perform_HDBSCAN_Cluster
results = perform_HDBSCAN_Cluster(localizations, min_cluster_size, min_samples, show=False)
```

Performs HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering for more robust density-based clustering.

**Parameters:**
- `localizations`: numpy.ndarray - Array of points in format [[x,y],...]
- `min_cluster_size`: int - Minimum size of clusters
- `min_samples`: int - Minimum samples required for core point
- `show`: bool - Whether to display visualization of clusters (default: False)

**Returns:**
- `ClusterResult` object containing:
  - cluster_labels: Labels for each point
  - cluster_centers: Centers of detected clusters
  - cluster_radii: Radii of detected clusters
  - loc_per_cluster: Number of points per cluster

### Scale-Space Blob Detection

```python
from cluster_detection import scale_space_plus_blob_detection
results = scale_space_plus_blob_detection(img, blob_parameters, fitting_parameters, show=False)
```

Performs scale-space blob detection with customizable parameters and optional visualization.

**Parameters:**
- `img`: numpy.ndarray - Input image array
- `blob_parameters`: dict - Parameters for blob detection:
  ```python
  {
      'path': np.ndarray,             # the image (repeat of img)
      'threshold': float = 1e-4,      # Detection threshold
      'overlap': float = 0.5,         # Allowed overlap between blobs
      'median': bool = False,         # Whether to apply median filter
      'min_sigma': float = 1,         # Minimum scale for detection
      'max_sigma': float = 2,         # Maximum scale for detection
      'num_sigma': int = 500,         # Number of scales to consider
      'logscale': bool = False,       # Whether to use logarithmic scale
      'detection': str = 'bp'         # Detection method
  }
  ```
- `fitting_parameters`: dict - Parameters for blob subpixel localization fitting:
  ```python
  {
        "mask_size": 5,               # mask area around localization used for fitting
        "plot_fit": False,            # show fit
        "fitting_image": "Original",  # Fit to the original image or to the Laplacian of the Gaussian image
        "radius_func": None,          # How to handle anisotropic cluster radii
        "sigma_range": 2,             # Radius in pixel units around which to constrain the fitting. Relative to the Scale estimate
        "centroid_range": 2,          # Same as above, but now for the centeroid fit. (constrained to 2 pixel units around the original Scale estimate) 
  }
  ```
- `show`: bool - Whether to display visualization (default: False)

**Returns:**
- Dictionary containing:
  - 'Fitted': Array of fitted (subpixel localization) blob parameters [x, y, r1, r2] - ([x,y] -> center pixel; [r1,r2] -> radii in x and y)
  - 'Scale': Array of scale-space blob parameters (same as above but not subpixel)
  - Placeholder for verbose fitting results for 'Fitted'

## Parameter Optimization

The library includes optimization capabilities for both DBSCAN and Scale-space blob detection parameters. Optimization of parameters based on known ground truth data uses at most 4 metrics:

### Available Metrics

```python
from cluster_detection import MetricType
class MetricType(Enum):
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    CENTER_ERROR = "center_error"
    SCALE_ERROR = "scale_error"
```

### Structure of Ground Truth Values

```python
ground_truth = {
    "initial_centers": [[x1,y1], [x2,y2], ... , [xn,yn]], # center location of the true cluster centers 
    "initial_scale": [s1, s2, ... , sn], # radius of the clusters defined before
}

```

### DBSCAN Optimization

```python
from cluster_detection import DBSCANOptimizer
optimizer = DBSCANOptimizer(metrics=[MetricType.TRUE_POSITIVES, MetricType.FALSE_POSITIVES]) # only consider these two metrics when optimizing for DBSCAN parameters.
result = optimizer.optimize(
    data=points,
    ground_truth=ground_truth, # ground_truth structure as defined above
    distance_range=np.linspace(0.1, 1.0, 10), # range of D (distance) values to consider
    minpoints_range=range(3, 10) # range of minP (minimum points) values to consider
)
```

### Scale-Space Optimization

#### Optimizing with a given ground truth value

```python
from cluster_detection import ScaleSpaceOptimizerTruth
optimizer = ScaleSpaceOptimizerTruth(metrics=[MetricType.TRUE_POSITIVES, MetricType.FALSE_POSITIVES])
result = optimizer.optimize(
    blob_params=blob_params,
    fit_params=fit_params,
    ground_truth=ground_truth,
    threshold_range=np.logspace(-6, -2, 20) # range of scale-space threshold values to scan across
)
```

#### Optimization Without Ground Truth
This will find the first instance (at the threshold value) at which the first cluster is detected.
If the input image a uniform localization control, this threshold value represents the smallest value to consider a cluster a true cluster. Only applies if the total intensity (# localizations) in the control image and applied image is conserved.

```python
from cluster_detection import ScaleSpaceOptimizerExplore
optimizer = ScaleSpaceOptimizerExplore(metrics=[MetricType.TRUE_POSITIVES, MetricType.FALSE_POSITIVES])
result = optimizer.optimize(
    blob_params=blob_params,
    fit_params=fit_params,
    threshold_range=np.logspace(-6, -2, 20) # range of scale-space threshold values to scan across
)
```

The optimization results include:
- Optimal parameters
- Metric values at optimal parameters
- Success status
- Status message

```python
@dataclass
class OptimizationResult:
    """Contains the results of parameter optimization."""

    optimal_parameters: np.ndarray
    metric_values: Dict[MetricType, float]
    success: bool
    message: str
```

for the ScaleSpaceOptimizerExplore version:
```python
OptimizationResult(
        optimal_parameters=np.array([optimal_threshold]),
        metric_values={MetricType.FALSE_POSITIVES: false_positives[optimal_idx]},
        success=True,
        message="Optimization successful",
    )
```


## Usage Example

```python
import numpy as np
from cluster_detection import (
    perform_DBSCAN_Cluster,
    perform_HDBSCAN_Cluster,
    scale_space_plus_blob_detection,
    DBSCANOptimizer,
    ScaleSpaceOptimizerTruth
)

# DBSCAN clustering with optimization
points = np.random.rand(100, 2)
ground_truth = {
    "initial_centers": np.array([[0.2, 0.3], [0.7, 0.8]]),
    "initial_scale": np.array([0.1, 0.1])
}

optimizer = DBSCANOptimizer()
opt_result = optimizer.optimize(
    points,
    ground_truth,
    distance_range=np.linspace(0.1, 0.5, 5),
    minpoints_range=range(3, 8)
)

if opt_result.success:
    optimal_distance, optimal_minpoints = opt_result.optimal_parameters[0]
    clusters = perform_DBSCAN_Cluster(points, optimal_distance, optimal_minpoints)

# Scale-space blob detection with optimization
image = np.random.rand(100, 100)
blob_params = {
    'threshold': 1e-4,
    'overlap': 0.5,
    'min_sigma': 1,
    'max_sigma': 2
}
fitting_params = {'mask_size': 5}

optimizer = ScaleSpaceOptimizerTruth()
opt_result = optimizer.optimize(
    blob_params,
    fitting_params,
    ground_truth,
    threshold_range=np.logspace(-6, -2, 20)
)

if opt_result.success:
    blob_params['threshold'] = opt_result.optimal_parameters[0]
    blobs = scale_space_plus_blob_detection(image, blob_params, fitting_params)
```
