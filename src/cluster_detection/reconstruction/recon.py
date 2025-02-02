from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import skimage.io
from numpy._typing import NDArray
from scipy.stats import multivariate_normal

# Type aliases for clarity
Coordinates = Union[Tuple[float, float], List[float], np.ndarray]
ImageDimensions = Union[Tuple[int, int], List[int], np.ndarray]


class ImageFormat(str, Enum):
    """Supported image formats for saving."""

    PNG = "png"
    JPG = "jpg"
    TIF = "tif"
    SVG = "svg"

    def __getitem__(self, key: str) -> str:
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


class ConversionType(str, Enum):
    """Types of coordinate conversion."""

    RC_TO_ORIGINAL = "RC_to_Original"
    ORIGINAL_TO_RC = "Original_to_RC"


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction parameters."""

    bounding_box_padding: int = 5
    random_seed: int = 666
    gaussian_scale_factor: float = 5.0


def create_gaussian(
    mu: Coordinates, sigma: Union[float, np.ndarray], domain: List[np.ndarray]
) -> np.ndarray:
    """
    Create a 2D Gaussian distribution.

    Args:
        mu: Center position (x,y)
        sigma: Standard deviation(s)
        domain: x,y domain arrays

    Returns:
        2D array of Gaussian values
    """
    x = domain[0]
    y = domain[1]
    xx, yy = np.meshgrid(x, y)
    # generate the multivariate normal distribution
    rv = multivariate_normal(mu, sigma)
    # generate the probability distribution
    gauss = rv.pdf(np.dstack((xx, yy)))
    # reshape the distribution on the grid
    return gauss


class ScaleSpacePlus(ABC):
    """Abstract base class for scale space reconstruction."""

    @abstractmethod
    def make_reconstruction(self, *args, **kwargs) -> np.ndarray:
        """Create a reconstruction from localizations."""
        pass

    @abstractmethod
    def save_image(self, path: Union[str, Path], name: str, fmt: ImageFormat) -> None:
        """Save the reconstruction image."""
        pass


class SMReconstruction(ScaleSpacePlus):
    """Single Molecule reconstruction implementation."""

    def __init__(
        self,
        img_dims: ImageDimensions,
        pixel_size: float = 130,
        rescale_pixel_size: float = 10,
        config: Optional[ReconstructionConfig] = None,
    ):
        self.config = config or ReconstructionConfig()
        self.img_dims_normal = np.array(img_dims)
        self.pixel_size_normal = pixel_size
        self.rescale_pixel_size = rescale_pixel_size

        # Calculate scaled dimensions
        self.scale_factor = self.pixel_size_normal / self.rescale_pixel_size
        self._img_dims = (self.img_dims_normal * self.scale_factor).astype(int)

        # Initialize image space
        self.img_space = np.zeros(self._img_dims)
        self._setup_domain()

    def _setup_img_space(self) -> None:
        self.img_space = np.zeros(self._img_dims)

    def _setup_domain(self) -> None:
        """Setup the reconstruction domain."""
        self.domain = [np.arange(self._img_dims[0]), np.arange(self._img_dims[1])]

    def make_reconstruction(
        self, localizations: np.ndarray, localization_error: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Create a reconstruction from localizations.

        Args:
            localizations: Array of (x,y) coordinates
            localization_error: Error values for each localization

        Returns:
            Reconstructed image array
        """
        self._setup_img_space()
        # Validate and prepare localization data
        self.df_localizations = self._prepare_localizations(
            localizations, localization_error
        )

        # Process each localization
        for _, loc in self.df_localizations.iterrows():
            self._add_localization_gaussian(loc)

        return self.img_space.T

    def _prepare_localizations(
        self, localizations: np.ndarray, localization_error: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        """Prepare localization data for reconstruction."""
        if np.isscalar(localization_error):
            localization_error = np.full(len(localizations), localization_error)

        df = pd.DataFrame(
            {
                "x": localizations[:, 0],
                "y": localizations[:, 1],
                "error": localization_error,
            }
        )

        # Scale coordinates
        df[["x", "y"]] *= self.scale_factor
        return df

    def _add_localization_gaussian(self, loc: pd.Series) -> None:
        """Add a Gaussian contribution from a single localization."""
        error_pixels = loc["error"] / self.rescale_pixel_size
        domain_size = int(self.config.gaussian_scale_factor * error_pixels)

        # Create local domain for Gaussian
        local_domain = [np.arange(domain_size)] * 2

        # Generate Gaussian
        gaussian = create_gaussian(
            mu=[domain_size / 2.0, domain_size / 2.0],
            sigma=error_pixels,
            domain=local_domain,
        )
        # Add to image space
        self._embed_gaussian(gaussian, loc["x"], loc["y"], domain_size)

    def _embed_gaussian(
        self,
        gaussian: np.ndarray,
        x: float,
        y: float,
        size: int,
        img_space: Optional[np.ndarray] = None,
    ) -> None:
        """Embed Gaussian into main image space."""
        x_start = int(x - size / 2)
        y_start = int(y - size / 2)

        if img_space is None:
            img_space = self.img_space

        try:
            img_space[x_start : x_start + size, y_start : y_start + size] += gaussian
        except Exception as e:
            print(f"Warning: Could not embed Gaussian at ({x}, {y}): {e}")

    def save_image(self, path: Union[str, Path], name: str, fmt: ImageFormat) -> None:
        """Save reconstruction image."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        full_path = path / f"{name}.{fmt}"
        skimage.io.imsave(str(full_path), self.img_space.T)


@dataclass
class MaskedReconstructionConfig(ReconstructionConfig):
    """Additional configuration for masked reconstruction."""

    mask_threshold: int = 255
    uniform_distribution: bool = False


class SMReconstructionMasked(SMReconstruction):
    """Single Molecule reconstruction with masking support."""

    def __init__(
        self,
        img_dims: ImageDimensions,
        pixel_size: float = 130,
        rescale_pixel_size: float = 10,
        config: Optional[MaskedReconstructionConfig] = None,
    ):
        super().__init__(
            img_dims,
            pixel_size,
            rescale_pixel_size,
            config or MaskedReconstructionConfig(),
        )
        self._masked_img_space: Optional[np.ndarray] = None
        self._masked_domain: Optional[np.ndarray] = None
        self._bounding_box: Optional[np.ndarray] = None

    def make_reconstruction(
        self,
        localizations: np.ndarray,
        localization_error: Union[float, np.ndarray],
        masked_img: np.ndarray,
        uniform: bool = False,
    ) -> np.ndarray:
        """
        Create a masked reconstruction from localizations.

        Args:
            localizations: Array of (x,y) coordinates
            localization_error: Error values for each localization
            masked_img: Binary mask defining the region of interest
            uniform: If True, uses uniform distribution within mask

        Returns:
            Reconstructed image array
        """
        self._setup_masked_space(masked_img)
        self._setup_bounding_box()

        if uniform:
            return self._make_uniform_reconstruction(
                len(localizations), localization_error
            )

        return self._make_masked_reconstruction(localizations, localization_error)

    def _setup_masked_space(self, masked_img: np.ndarray) -> None:
        """Setup the masked image space and domain."""
        self._masked_img_space = masked_img

        # Find coordinates within mask
        mask_coords = np.where(masked_img == self.config.mask_threshold)
        self._masked_domain = np.column_stack(
            (mask_coords[1], mask_coords[0])  # x,y order
        )

    def _setup_bounding_box(self) -> None:
        """Calculate the bounding box of the masked region."""
        if self._masked_domain is None:
            raise ValueError(
                "Masked domain not initialized. Call _setup_masked_space(...)"
            )

        padding = self.config.bounding_box_padding
        mins = np.min(self._masked_domain, axis=0) - padding
        maxs = np.max(self._masked_domain, axis=0) + padding
        self._bounding_box = np.array([mins, maxs])

    def _make_masked_reconstruction(
        self, localizations: np.ndarray, localization_error: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Create reconstruction using actual localizations."""
        df = self._prepare_masked_localizations(localizations, localization_error)

        img_space = self._create_bounded_image_space()

        for _, loc in df.iterrows():
            self._add_masked_gaussian(loc, img_space)

        return img_space.T

    def _make_uniform_reconstruction(
        self, num_points: int, localization_error: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Create reconstruction using uniform random points within mask."""
        if num_points == 0:
            return self._create_bounded_image_space()

        localizations = self._generate_uniform_points(num_points)
        df = self._prepare_masked_localizations(localizations, localization_error)
        img_space = self._create_bounded_image_space()

        for _, loc in df.iterrows():
            self._add_masked_gaussian(loc, img_space)

        return img_space.T

    def _prepare_masked_localizations(
        self, localizations: np.ndarray, localization_error: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        """Prepare localizations for masked reconstruction."""
        df = pd.DataFrame(
            {
                "x": localizations[:, 0],
                "y": localizations[:, 1],
                "error": localization_error
                if np.isscalar(localization_error)
                else np.array(localization_error),
            }
        )

        # Adjust coordinates relative to mask
        df[["x", "y"]] = (
            df[["x", "y"]]
            - np.min(self._masked_domain, axis=0)
            + self.config.bounding_box_padding
        )

        # Scale to reconstruction space
        df[["x", "y"]] *= self.scale_factor

        return df

    def _generate_uniform_points(self, num_points: int) -> NDArray:
        """Generate uniform random points within the mask."""
        np.random.seed(self.config.random_seed)

        # Random selection from masked domain points
        indices = np.random.choice(
            len(self._masked_domain), size=num_points, replace=True
        )
        points = self._masked_domain[indices]

        # Add random subpixel offsets
        points = points + np.random.rand(*points.shape)

        return points

    def _create_bounded_image_space(self) -> np.ndarray:
        """Create image space sized to the bounding box."""
        bbox_size = (self._bounding_box[1] - self._bounding_box[0]) * self.scale_factor
        return np.zeros(bbox_size.astype(int))

    def _add_masked_gaussian(self, loc: pd.Series, img_space: np.ndarray) -> None:
        """Add a Gaussian contribution from a single localization in masked space."""
        error_pixels = loc["error"] / self.rescale_pixel_size
        domain_size = int(self.config.gaussian_scale_factor * error_pixels)

        # Create local domain for Gaussian
        local_domain = [np.arange(domain_size)] * 2
        # Generate Gaussian
        gaussian = create_gaussian(
            mu=[domain_size / 2.0, domain_size / 2.0],
            sigma=error_pixels,
            domain=local_domain,
        )
        # Add to image space
        self._embed_gaussian(gaussian, loc["x"], loc["y"], domain_size, img_space)

    def coordinate_conversion(
        self, coords: np.ndarray, radius: np.ndarray, conv_type: ConversionType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert coordinates between reconstruction and original space.

        Args:
            coords: Coordinates to convert
            radius: Associated radii to convert
            conv_type: Direction of conversion

        Returns:
            Converted coordinates and radii
        """
        if conv_type == ConversionType.RC_TO_ORIGINAL:
            # Reconstruction to original space
            conv_coords = (
                coords * self.rescale_pixel_size / self.pixel_size_normal
                - self.config.bounding_box_padding
                + np.min(self._masked_domain, axis=0)
            )
            conv_radius = radius * self.rescale_pixel_size / self.pixel_size_normal

        elif conv_type == ConversionType.ORIGINAL_TO_RC:
            # Original to reconstruction space
            conv_coords = (
                (
                    coords
                    - np.min(self._masked_domain, axis=0)
                    + self.config.bounding_box_padding
                )
                * self.pixel_size_normal
                / self.rescale_pixel_size
                # - self.config.bounding_box_padding
            )
            conv_radius = radius * self.pixel_size_normal / self.rescale_pixel_size

        else:
            raise ValueError(f"Unknown conversion type: {conv_type}")

        return conv_coords, conv_radius

    @property
    def masked_domain(self) -> np.ndarray:
        """Get the masked domain coordinates."""
        if self._masked_domain is None:
            raise ValueError("Masked domain not initialized")
        return self._masked_domain

    @property
    def bounding_box(self) -> np.ndarray:
        """Get the bounding box coordinates."""
        if self._bounding_box is None:
            raise ValueError("Bounding box not initialized")
        return self._bounding_box
