from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cluster_detection.clustering_methods import (
    DBSCAN_TP_FP_center_scale_error,
    scale_space_plus_blob_detection,
    true_positive_and_error,
)
from cluster_detection.parameter_template import BlobParams, FitParams


class MetricType(Enum):
    """Enumeration of available optimization metrics."""

    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    CENTER_ERROR = "center_error"
    SCALE_ERROR = "scale_error"


@dataclass
class OptimizationResult:
    """Contains the results of parameter optimization."""

    optimal_parameters: np.ndarray
    metric_values: Dict[MetricType, float]
    success: bool
    message: str


@dataclass
class MetricGrid:
    """Stores and manages metric calculations across parameter combinations."""

    parameter_ranges: Dict[str, Sequence]
    metrics: List[MetricType]
    values: Dict[MetricType, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        grid_shape = tuple(len(range_) for range_ in self.parameter_ranges.values())
        self.values = {metric: np.full(grid_shape, np.nan) for metric in self.metrics}

    def update(self, metric: MetricType, indices: Tuple[int, ...], value: float):
        """Update a specific metric value in the grid."""
        self.values[metric][indices] = value

    def get_common_minima(self) -> List[Tuple[int, ...]]:
        """Find indices where all metrics are minimized."""
        return list(common_min_indices(*[self.values[m] for m in self.metrics]))


class ClusterOptimizer:
    """Base class for cluster detection parameter optimization."""

    def __init__(self, metrics: Optional[List[MetricType]] = None):
        # check if input strings match the MetricType enum values
        if metrics:
            for metric in metrics:
                try:
                    MetricType(metric)
                except ValueError:
                    raise ValueError(
                        "Metrics supported are:{0}".format(
                            [m.value for m in MetricType]
                        )
                    )
            # convert the input strings to MetricType enum values
            self.metrics = [MetricType(metric) for metric in metrics]
        else:
            self.metrics = [
                MetricType.TRUE_POSITIVES,
                MetricType.FALSE_POSITIVES,
            ]

    @staticmethod
    def _validate_input(data: np.ndarray, ground_truth: Dict) -> None:
        """Validate input data and ground truth."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array")
        if not isinstance(ground_truth, dict) or "initial_centers" not in ground_truth:
            raise ValueError("Invalid ground truth format")


class DBSCANOptimizer(ClusterOptimizer):
    """Optimizer for DBSCAN clustering parameters."""

    def optimize(
        self,
        data: np.ndarray,
        ground_truth: Dict,
        distance_range: Sequence[float],
        minpoints_range: Sequence[int],
        threshold: float = 1.0,
    ) -> OptimizationResult:
        """
        Find optimal DBSCAN parameters by minimizing error metrics.

        Args:
            data: Input localization data (N x 2 array)
            ground_truth: Dictionary containing ground truth centers and scales
            distance_range: Sequence of distance values to evaluate
            minpoints_range: Sequence of minimum points values
            threshold: Comparison threshold

        Returns:
            OptimizationResult containing optimal parameters and metrics
        """
        self._validate_input(data, ground_truth)

        metric_grid = MetricGrid(
            parameter_ranges={"distance": distance_range, "minpoints": minpoints_range},
            metrics=self.metrics,
        )

        for i, dist in enumerate(distance_range):
            for j, minp in enumerate(minpoints_range):
                result = self._evaluate_parameters(
                    data, ground_truth, dist, minp, threshold
                )

                for metric, value in result.items():
                    metric_grid.update(metric, (i, j), value)

        optimal_indices = metric_grid.get_common_minima()

        if not optimal_indices:
            return OptimizationResult(
                optimal_parameters=np.array([]),
                metric_values={},
                success=False,
                message="No common optimal parameters found",
            )

        optimal_params = np.array(
            [(distance_range[i], minpoints_range[j]) for i, j in optimal_indices]
        )

        return OptimizationResult(
            optimal_parameters=optimal_params,
            metric_values={
                metric: grid[optimal_indices[0]]
                for metric, grid in metric_grid.values.items()
            },
            success=True,
            message="Optimization successful",
        )

    def _evaluate_parameters(
        self,
        data: np.ndarray,
        ground_truth: Dict,
        distance: float,
        min_points: int,
        threshold: float,
    ) -> Dict[MetricType, float]:
        """Evaluate metrics for a specific parameter combination."""
        result = DBSCAN_TP_FP_center_scale_error(
            ground_truth, data, distance, min_points, convert=False, threshold=threshold
        )

        metrics = {}
        if MetricType.TRUE_POSITIVES in self.metrics:
            metrics[MetricType.TRUE_POSITIVES] = abs(
                len(ground_truth["initial_centers"]) - result["true_positive_num"]
            )
        if MetricType.FALSE_POSITIVES in self.metrics:
            metrics[MetricType.FALSE_POSITIVES] = result["false_positives"]
        if MetricType.CENTER_ERROR in self.metrics:
            metrics[MetricType.CENTER_ERROR] = np.nanmean(
                result["true_center_TP_center_error"]
            )
        if MetricType.SCALE_ERROR in self.metrics:
            metrics[MetricType.SCALE_ERROR] = np.nanmean(
                result["true_center_TP_scale_error"]
            )

        return metrics


class ScaleSpaceOptimizerTruth(ClusterOptimizer):
    """Optimizer for Scale Space blob detection parameters."""

    def optimize(
        self,
        blob_params: BlobParams,
        fit_params: FitParams,
        ground_truth: Dict,
        threshold_range: Sequence[float],
        threshold: float = 1.0,
    ) -> OptimizationResult:
        """
        Find optimal Scale Space threshold by minimizing the specified metrics.

        Args:
            blob_params: BlobParams instance containing blob detection parameters
            fit_params: FitParams instance containing fitting parameters
            ground_truth: Ground truth data
                {"inital_centers": np.array([[x1,y1,...], [x2,y2,...], ...]),
                    "initial_scales": np.array([s1, s2, ...])}
                Note: initial_scales and initial_centers have the same length
            threshold_range: Range of threshold values to evaluate
            threshold: Comparison threshold

        Returns:
            OptimizationResult containing optimal threshold
        """
        self._validate_input(ground_truth["initial_centers"], ground_truth)
        metric_grid = MetricGrid(
            parameter_ranges={"threshold": threshold_range},
            metrics=self.metrics,
        )
        # Convert to sorted numpy array for consistent handling
        sorted_thresholds = np.sort(np.array(threshold_range))[::-1]

        # Create working copies of the parameters
        blob_params_dict = blob_params.to_dict()
        fit_params_dict = fit_params.to_dict()

        for i, thresh in enumerate(sorted_thresholds):
            # Update threshold in the params copy
            curr_blob_params = blob_params_dict.copy()
            curr_blob_params["threshold"] = thresh

            # Perform blob detection
            blobs = scale_space_plus_blob_detection(
                curr_blob_params["path"], curr_blob_params, fit_params_dict, show=False
            )
            result = self._evaluate_parameters(ground_truth, blobs, threshold)
            for metric, value in result.items():
                metric_grid.update(metric, (i), value)

        optimal_indices = metric_grid.get_common_minima()
        optimal_params = np.array([sorted_thresholds[i] for i in optimal_indices])
        if not optimal_indices:
            return OptimizationResult(
                optimal_parameters=np.array([]),
                metric_values={},
                success=False,
                message="No common optimal parameters found",
            )
        return OptimizationResult(
            optimal_parameters=optimal_params,
            metric_values={
                metric: grid[optimal_indices]
                for metric, grid in metric_grid.values.items()
            },
            success=True,
            message="Optimization successful",
        )

    def _evaluate_parameters(
        self,
        ground_truth: Dict,
        found_blobs: Dict,
        threshold: float,
    ) -> Dict[MetricType, float]:
        """Evaluate metrics for a specific parameter combination."""
        result = true_positive_and_error(
            ground_truth, found_blobs, center_threshold=threshold
        )
        metrics = {}
        if MetricType.TRUE_POSITIVES in self.metrics:
            metrics[MetricType.TRUE_POSITIVES] = abs(
                len(ground_truth["initial_centers"]) - result["true_positive_num"]
            )
        if MetricType.FALSE_POSITIVES in self.metrics:
            metrics[MetricType.FALSE_POSITIVES] = result["false_positives"]
        if MetricType.CENTER_ERROR in self.metrics:
            metrics[MetricType.CENTER_ERROR] = np.nanmean(
                result["true_center_TP_center_error"]
            )
        if MetricType.SCALE_ERROR in self.metrics:
            metrics[MetricType.SCALE_ERROR] = np.nanmean(
                result["true_center_TP_scale_error"]
            )

        return metrics


class ScaleSpaceOptimizerExplore:
    """Optimizer for Scale Space blob detection parameters. This optimizer finds the first threshold value at which any cluster is identified. As such it should only be applied to a randomized control in which no true clusters exist."""

    def optimize(
        self,
        blob_params: BlobParams,
        fit_params: FitParams,
        threshold_range: Sequence[float],
    ) -> OptimizationResult:
        """
        Find optimal Scale Space threshold by minimizing false positives.

        Args:
            blob_params: BlobParams instance containing blob detection parameters
            fit_params: FitParams instance containing fitting parameters
            threshold_range: Range of threshold values to evaluate

        Returns:
            OptimizationResult containing optimal threshold
        """
        # Convert to sorted numpy array for consistent handling
        sorted_thresholds = np.sort(np.array(threshold_range))[::-1]
        false_positives = []

        # Create working copies of the parameters
        blob_params_dict = blob_params.to_dict()
        fit_params_dict = fit_params.to_dict()

        for thresh in sorted_thresholds:
            # Update threshold in the params copy
            curr_blob_params = blob_params_dict.copy()
            curr_blob_params["threshold"] = thresh

            # Perform blob detection
            blobs = scale_space_plus_blob_detection(
                curr_blob_params["path"], curr_blob_params, fit_params_dict, show=False
            )
            false_positives_num = len(blobs["Scale"])

            false_positives.append(false_positives_num)

        # Find first threshold with non-zero false positives
        optimal_idx = next((i for i, fp in enumerate(false_positives) if fp > 0), None)

        if optimal_idx is None:
            return OptimizationResult(
                optimal_parameters=np.array([]),
                metric_values={},
                success=False,
                message="No suitable threshold found. Consider using a larger range.",
            )

        optimal_threshold = sorted_thresholds[optimal_idx]
        return OptimizationResult(
            optimal_parameters=np.array([optimal_threshold]),
            metric_values={MetricType.FALSE_POSITIVES: false_positives[optimal_idx]},
            success=True,
            message="Optimization successful",
        )


def common_min_indices(*arrays: np.ndarray) -> List[Tuple[int, ...]]:
    """Find common indices where minimum values occur in multiple arrays."""
    if not arrays:
        return []

    min_indices_sets = [
        set(map(tuple, np.argwhere(arr == np.min(arr)))) for arr in arrays
    ]

    return sorted(set.intersection(*min_indices_sets))
