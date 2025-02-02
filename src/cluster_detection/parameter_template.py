# template for the parameter dictionary for blob_detection and dbscan
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

import numpy as np


class HasDict(Protocol):
    __dict__: dict[str, Any]


@dataclass
class BlobParams:
    path: str | np.ndarray
    median: bool
    threshold: float
    min_sigma: float
    max_sigma: float
    num_sigma: int
    overlap: float
    logscale: bool
    verbose: bool
    exclude_border: bool

    def __post_init__(self):
        errors = []

        if not isinstance(self.path, (str, np.ndarray)):
            errors.append("path must be a string or numpy.ndarray")
        if isinstance(self.path, str) and not os.path.isfile(self.path):
            errors.append(f"path must be a valid path to a file: {self.path}")

        if not isinstance(self.median, bool):
            errors.append("median must be a bool")
        if not isinstance(self.threshold, float):
            errors.append("threshold must be a float")
        if not isinstance(self.min_sigma, float):
            errors.append("min_sigma must be a float")
        if not isinstance(self.max_sigma, float):
            errors.append("max_sigma must be a float")
        if not isinstance(self.num_sigma, int):
            errors.append("num_sigma must be an int")
        if self.num_sigma <= 0:
            errors.append("num_sigma must be greater than 0")
        if not isinstance(self.overlap, float):
            errors.append("overlap must be a float")
        if not 0 <= self.overlap <= 1:
            errors.append("overlap must be between 0.0 and 1.0")
        if not isinstance(self.logscale, bool):
            errors.append("logscale must be a bool")
        if not isinstance(self.verbose, bool):
            errors.append("verbose must be a bool")
        if not isinstance(self.exclude_border, bool):
            errors.append("exclude_border must be a bool")

        if errors:
            raise ValueError("\n".join(errors))


@dataclass
class FitParams:
    mask_size: int
    centroid_range: int | float
    sigma_range: int | float
    height_range: int | float
    fitting_image: Literal["Original", "Laplacian"]
    radius_func: callable

    def __post_init__(self):
        errors = []

        if not isinstance(self.mask_size, int):
            errors.append("mask_size must be an int")
        if self.mask_size <= 2:
            errors.append("mask_size must be greater than 2")

        if not isinstance(self.centroid_range, (int, float)):
            errors.append("centroid_range must be an int or float")
        if self.centroid_range > self.mask_size:
            errors.append("centroid_range must be less than or equal to mask_size")

        if not isinstance(self.sigma_range, (int, float)):
            errors.append("sigma_range must be an int or float")
        if self.sigma_range > self.mask_size:
            errors.append("sigma_range must be less than or equal to mask_size")

        if not isinstance(self.height_range, (int, float)):
            errors.append("height_range must be an int or float")

        if self.fitting_image not in ["Original", "Laplacian"]:
            errors.append("fitting_image must be one of ['Original', 'Laplacian']")

        if not callable(self.radius_func):
            errors.append("radius_func must be callable")

        if errors:
            raise ValueError("\n".join(errors))


def to_dict(instance: HasDict) -> dict:
    """Convert the dataclass instance to a dictionary, preserving callable fields."""

    dict_repr = asdict(instance)
    dict_added = {}
    for key, value in dict_repr.items():
        if callable(value):  # Check if any field is callable
            dict_repr[key] = value  # Keep the function pointer in the dict
            # add string representation of the callable
            dict_added[key + "_str"] = (
                value.__name__ if hasattr(value, "__name__") else str(value)
            )
    dict_repr = {**dict_repr, **dict_added}
    return dict_repr


# Assign the generic to_dict function to the dataclasses
BlobParams.to_dict = to_dict
FitParams.to_dict = to_dict
