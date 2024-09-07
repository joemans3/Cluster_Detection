"""
Documentation for blob_detection.py

This file contains the class blob_detection which is used to detect blobs in an image. It uses the skimage.blob_log() function to detect blobs in an image.
It also has the option to fit the blobs with a 2D gaussian function.
The class also has the option to fit the blobs with a 2D gaussian function. This is done by using the lmfit package to fit the blobs with a 2D gaussian function.
Detection can be done using the skimage.blob_log() function or a custom function. The custom function is a modified version of the skimage.blob_log() function. Named blob_logv2()

Classes:
--------
blob_detection: see class docstring for more info, this is the main class that is used to detect blobs in an image
"""

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize, report_fit
from scipy import spatial
from scipy.ndimage import filters
from skimage.feature import blob
from skimage.util import dtype

from utils import rescale_range, read_file


# global vars for fitting functions
FWHM_FACTOR = 2.0 * (np.log(2.0 + np.sqrt(3)))


class blob_detection:
    """
    Parameters TODO
    ----------
    Path : string
        Full path of the image to be read
    median : bool
        if true apply a median filter to the image before blog detection
    threshold : float
        threshold for the blob detection
    min_sigma : float
        Minimum value of the gaussian sigma for the blobs
    max_sigma : float
        Maximum value of the gaussian sigma for the blobs
    num_sigma : int
        Eqidistant values between min_sigma and max_sigma to consider
    overlap : float
        Allowed overlap of identified blobs. If 1, full overlap is allowed

    Methods TODO
    -------
    open_file()
        opens the file and applied media filter if true
        retuns an array
    detection()
        applies blob detection using np.blob_log
        returns array of blob attributes or dictionary of blob attributes
    Notes
    -----
    theory: https://www.cse.psu.edu/~rtc12/CSE586/lectures/featureExtractionPart2_6pp.pdf
    https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/lecture10_detector_descriptors_2015_notes.pdf
    """

    def __init__(
        self,
        path,
        median=False,
        threshold=0.0005,
        min_sigma=1.0,
        max_sigma=1.5,
        num_sigma=500,
        overlap=1.0,
        logscale=False,
        verbose=False,
        exclude_border=False,
    ):
        """
        Initilizes the class object with the parameters for the blob detection

        Parameters:
        -----------
        Path : string or 2d array
            Full path of the image to be read or the 2d array of the image
        median : bool
            if true apply a median filter to the image before blog detection
        threshold : float
            threshold for the blob detection
        min_sigma : float
            Minimum value of the gaussian sigma for the blobs
        max_sigma : float
            Maximum value of the gaussian sigma for the blobs
        num_sigma : int
            Eqidistant values between min_sigma and max_sigma to consider
        overlap : float
            Allowed overlap of identified blobs. If 1, full overlap is allowed
        logscale : bool
            if True, use a log scale for the sigma values
        verbose : bool
            if True, return out the parameters used for the blob detection and fitting
        exclude_border : tuple of ints, int, or False, optional, Default is False.
            If tuple of ints, the length of the tuple must match the input array's
            dimensionality.  Each element of the tuple will exclude peaks from
            within `exclude_border`-pixels of the border of the image along that
            dimension.
            If nonzero int, `exclude_border` excludes peaks from within
            `exclude_border`-pixels of the border of the image.
            If zero or False, peaks are identified regardless of their
            distance from the border. See method "blob_logv2" for more info
        Notes:
        ------
        1. The blob detection is done using the skimage.blob_log() function or a custom function. The custom function is a modified version of the skimage.blob_log() function. Named blob_logv2()
        2. To use the custom function, call the method detection() with the argument type='bp' else the default is 'skimage'
        """
        self.img = path
        self.median = median
        self.threshold = threshold
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.overlap = overlap
        self.log_scale = logscale

        self.median_filter_size = 1
        self.fitting_parameters = {}
        self.verbose = verbose

    def _update_fitting_parameters(self, kwargs={}):
        """
        Updates the fitting_parameters to be used in each iteration of this class object

        Kwargs
        ------
        mask_size: int
            when fitting the image with a function this is size of square round a reference point to use for fit
        residual_func: functional
            function to use when defining the residuals for the fitting
        fit_method: string, default 'least squares'
            method of the fitting to use
        radius_func: functional, default numpy.mean
            function to use as a method to take two sigams and convert to one radius
        plot_fit: bool
            if True, plots each fit with the fit statistics
        centroid_range: int or float-like
            controls the bounds on the fit for the centroid (x,y). Ie: the min fit is x-centroid_range, and max is x+centroid_range
            same for y.
        sigma_range: int or float-like
            controls the bounds on the fit for the sigmas (s_x,s_y). Ie: the min fit is s_x-sigma_range, and max is s_x+sigma_range
            same for y.
        fitting_image: string
            if "Original" use the original image to fit function
            else use the Laplacian image created with the sigma that maximized the laplacian

        Notes
        -----
        Some of these expect a certain type to work. This is not fully coded yet and might break if you give inputs which dont make sense
        to it.
        """

        for i, j in kwargs.items():
            self.fitting_parameters[i] = j

    def open_file(self):
        """
        Opens and retuns array of the image data

        Parameters
        ----------
        self.img: string of a path or path object, or 2D image
            if path provided opens the file and reads in image data to apply filter or applies filter is self.img is 2D array
        self.median: bool
            if True applies a median filter of size self.median_filter_size before returning 2D array


        Returns
        -------
        array-like
            2D array of the image data
        """
        file_gray = read_file(self.img)
        if file_gray.ndim == 3:
            file_gray = file_gray[:, :, 0]
        if self.median:
            file_gray = filters.median_filter(file_gray, size=self.median_filter_size)
        return file_gray

    def detection(self, type="skimage", **kwargs):
        """
        Applies the blob_log scheme to detect blobs in an image using the parameters of this class object
        Full list see __init__().__doc__

        Parameters
        ----------
        type: string
            if "bp" use the custom blob_logv2() elif "skimage" use the skimage implimentation of blob_log

        Returns
        -------
        if verbose is True: return type is a dictionary
            returns a dictionary of the parameters used for the blob detection and fitting and the fitted objects
        else: return type is a numpy array of size 3 tuples
            returns the scale space blobs found in the image

        Notes:
        ------
        1. For 2D images the blob radius estimate is the standard deviation of the gaussian fit to the image times sqrt(2) (see theory)
        2. Scale fits are isotropic
        3. Fitted fits are anisotropic and are size 4 tuples with simga_x and sigma_y
        """
        if isinstance(self.img, str):
            file = self.open_file()
        else:
            file = self.img
        if type == "skimage":  # default method based on blob_log from skimage
            blobs = blob.blob_log(
                file,
                threshold=self.threshold,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                overlap=self.overlap,
                log_scale=self.log_scale,
            )
            blobs[:, 2] *= np.sqrt(
                2
            )  # converting the standard deviation of the gaussian fit to radius of the circle
            return np.array(
                blobs
            )  # blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot
        elif type == "bp":
            blobs = self.blob_logv2(
                file,
                threshold=self.threshold,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                overlap=self.overlap,
                log_scale=self.log_scale,
            )
            if kwargs.get("testing", False):
                blobs["Fitted"][:, 2:] *= np.sqrt(
                    2
                )  # converting the standard deviation of the gaussian fit to radius of the circle
                blobs["Scale"][:, 2] *= np.sqrt(2)
                return blobs  # blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot
            if self.verbose:
                blobs["Fitted"][:, 2:] *= np.sqrt(
                    2
                )  # converting the standard deviation of the gaussian fit to radius of the circle
                blobs["Scale"][:, 2] *= np.sqrt(2)
                return blobs  # blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot
            else:
                blobs["Scale"][:, 2] *= np.sqrt(2)
                return blobs[
                    "Scale"
                ]  # blobs returns array of size 3 tuples (x,y,radius) defining the circle defining the spot

    def _prune_blobs(self, blobs_array, overlap, *, sigma_dim=1, **kwargs):
        """Eliminated blobs with area overlap. UPDATED: compared to the skimage implimentation this prunes based on the
        maximum value of the laplacian of the blobs rather than if the blob is bigger or not.

        Parameters
        ----------
        blobs_array : ndarray
            A 2d array with each row representing 3 (or 4) values,
            ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
            where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
            and ``sigma`` is the standard deviation of the Gaussian kernel which
            detected the blob.
            This array must not have a dimension of size 0.
        overlap : float
            A value between 0 and 1. If the fraction of area overlapping for 2
            blobs is greater than `overlap` the smaller blob is eliminated.
        sigma_dim : int, optional
            The number of columns in ``blobs_array`` corresponding to sigmas rather
            than positions.

        KWARGS
        ------
        max_lap: N-array
            array of values for each blob indicating the max value of the laplacian that created it
        sigma_indx: n-array
            array of indexes of the sigmas for the blob, this is just inputted for conveniecne in later calulations. Not used here

        Returns
        -------
        A : ndarray
            `array` with overlapping blobs removed.

        Notes
        -----
        Example: blob1 = [100,100,3], blob2 = [101,101,3]
        max_lap = [1,2]. If the overlap is larger than the threshold used then blob2 is choosen because it produces a larger max_lap
        """

        max_lap = kwargs.get("max_lap", None)
        sigma_indx = kwargs.get("sigma_indx", None)

        if max_lap is None:
            raise TypeError(
                "max_lap cannot be None, if intended use skimage.blob_log implimentation"
            )
        if sigma_indx is None:
            raise TypeError(
                "sigma_indx cannot be None, if intended use skimage.blob_log implimentation"
            )

        sigma = blobs_array[:, -sigma_dim:].max()
        distance = 2 * sigma * np.sqrt(blobs_array.shape[1] - sigma_dim)
        tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
        pairs = np.array(list(tree.query_pairs(distance)))
        if len(pairs) == 0:
            return blobs_array, sigma_indx
        else:
            for (
                i,
                j,
            ) in pairs:  ####turns out that for each pair it assigns -1 to the sigma if it fails the call,
                # but depending on the pairs which are choosen first it assigns a -1 to a blob that is likely larger than others
                # find a way to do a ranked list of sorts for this.
                blob1, blob2 = blobs_array[i], blobs_array[j]

                overlap_blob = blob._blob_overlap(blob1, blob2, sigma_dim=sigma_dim)
                if overlap_blob > overlap:
                    # note: this test works even in the anisotropic case because
                    # all sigmas increase together.
                    if max_lap[i] > max_lap[j]:
                        blob2[-1] = -1
                    else:
                        blob1[-1] = -1
        blobs_pruned = []
        sigma_indx_pruned = []
        for inx, val in enumerate(blobs_array):
            if val[-1] > -1:
                blobs_pruned.append(val)
                sigma_indx_pruned.append(sigma_indx[inx])
        # return np.stack([b for b in blobs_array if b[-1] > -1]) #save for testing
        return np.stack(blobs_pruned), np.stack(sigma_indx_pruned)

    def blob_logv2(
        self,
        image,
        min_sigma=1,
        max_sigma=50,
        num_sigma=10,
        threshold=0.2,
        overlap=0.5,
        log_scale=False,
        *,
        exclude_border=False,
        **kwargs,
    ):
        r"""Finds blobs in the given grayscale image. Adapted from the implimentation of skimage blob-log:
        https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

        Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
        For each blob found, the method returns its coordinates and the standard
        deviation of the Gaussian kernel that detected the blob.

        Parameters
        ----------
        image : 2D or 3D ndarray
            Input grayscale image, blobs are assumed to be light on dark
            background (white on black).
        min_sigma : scalar or sequence of scalars, optional
            the minimum standard deviation for Gaussian kernel. Keep this low to
            detect smaller blobs. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number, in
            which case it is equal for all axes.
        max_sigma : scalar or sequence of scalars, optional
            The maximum standard deviation for Gaussian kernel. Keep this high to
            detect larger blobs. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number, in
            which case it is equal for all axes.
        num_sigma : int, optional
            The number of intermediate values of standard deviations to consider
            between `min_sigma` and `max_sigma`.
        threshold : float, optional.
            The absolute lower bound for scale space maxima. Local maxima smaller
            than thresh are ignored. Reduce this to detect blobs with less
            intensities.
        overlap : float, optional
            A value between 0 and 1. If the area of two blobs overlaps by a
            fraction greater than `threshold`, the blob with the smaller maximum
            laplacian value is eliminated. If set to 1, then all overlapping blobs
            are kept.
        log_scale : bool, optional
            If set intermediate values of standard deviations are interpolated
            using a logarithmic scale to the base `10`. If not, linear
            interpolation is used.
        exclude_border : tuple of ints, int, or False, optional
            If tuple of ints, the length of the tuple must match the input array's
            dimensionality.  Each element of the tuple will exclude peaks from
            within `exclude_border`-pixels of the border of the image along that
            dimension.
            If nonzero int, `exclude_border` excludes peaks from within
            `exclude_border`-pixels of the border of the image.
            If zero or False, peaks are identified regardless of their
            distance from the border.

        Returns
        -------
        A : (n, image.ndim + sigma) ndarray
            A 2d array with each row representing 2 coordinate values for a 2D
            image, and 3 coordinate values for a 3D image, plus the sigma(s) used.
            When a single sigma is passed, outputs are:
            ``(r, c, sigma)`` or ``(p, r, c, sigma)`` where ``(r, c)`` or
            ``(p, r, c)`` are coordinates of the blob and ``sigma`` is the standard
            deviation of the Gaussian kernel which detected the blob. When an
            anisotropic gaussian is used (sigmas per dimension), the detected sigma
            is returned for each dimension.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

        Examples
        --------
        >>> from skimage import data, feature, exposure
        >>> img = data.coins()
        >>> img = exposure.equalize_hist(img)  # improves detection
        >>> feature.blob_log(img, threshold = .3)
        array([[124.        , 336.        ,  11.88888889],
            [198.        , 155.        ,  11.88888889],
            [194.        , 213.        ,  17.33333333],
            [121.        , 272.        ,  17.33333333],
            [263.        , 244.        ,  17.33333333],
            [194.        , 276.        ,  17.33333333],
            [266.        , 115.        ,  11.88888889],
            [128.        , 154.        ,  11.88888889],
            [260.        , 174.        ,  17.33333333],
            [198.        , 103.        ,  11.88888889],
            [126.        , 208.        ,  11.88888889],
            [127.        , 102.        ,  11.88888889],
            [263.        , 302.        ,  17.33333333],
            [197.        ,  44.        ,  11.88888889],
            [185.        , 344.        ,  17.33333333],
            [126.        ,  46.        ,  11.88888889],
            [113.        , 323.        ,   1.        ]])

        Notes
        -----
        The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
        a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
        """
        image2 = image
        image = dtype.img_as_float(image)

        # if both min and max sigma are scalar, function returns only one sigma
        scalar_sigma = (
            True if np.isscalar(max_sigma) and np.isscalar(min_sigma) else False
        )

        # Gaussian filter requires that sequence-type sigmas have same
        # dimensionality as image. This broadcasts scalar kernels
        if np.isscalar(max_sigma):
            max_sigma = np.full(image.ndim, max_sigma, dtype=float)
        if np.isscalar(min_sigma):
            min_sigma = np.full(image.ndim, min_sigma, dtype=float)

        # Convert sequence types to array
        min_sigma = np.asarray(min_sigma, dtype=float)
        max_sigma = np.asarray(max_sigma, dtype=float)

        if log_scale:
            # for anisotropic data, we use the "highest resolution/variance" axis
            standard_axis = np.argmax(min_sigma)
            start = np.log10(min_sigma[standard_axis])
            stop = np.log10(max_sigma[standard_axis])
            scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
            sigma_list = scale * min_sigma / np.max(min_sigma)
        else:
            scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
            sigma_list = scale * (max_sigma - min_sigma) + min_sigma

        # computing gaussian laplace
        # average s**2 provides scale invariance
        gl_images = [
            -filters.gaussian_laplace(image, s) * np.mean(s) ** 2 for s in sigma_list
        ]

        image_cube = np.stack(gl_images, axis=-1)

        exclude_border = blob._format_exclude_border(image.ndim, exclude_border)
        local_maxima = blob.peak_local_max(
            image_cube,
            threshold_abs=threshold,
            footprint=np.ones((3,) * (image.ndim + 1)),
            threshold_rel=0.0,
            exclude_border=exclude_border,
        )

        # # view laplacian slices for all local maxima sigmas
        # for i in local_maxima:
        # 	x,y,s_indx = i
        # 	plt.imshow(image_cube[:,:,s_indx])
        # 	plt.show()

        # Catch no peaks
        if local_maxima.size == 0:
            return {
                "Fitted": np.empty((0, 4)),
                "Scale": np.empty((0, 3)),
                "Fit": np.empty((0, 3)),
            }
        # find the max of the laplacian for each peak found
        # figure out a way to vectorize it using slicing: https://numpy.org/doc/stable/user/basics.indexing.html
        max_lap = image_cube[local_maxima[:, 0], local_maxima[:, 1], local_maxima[:, 2]]

        # Convert local_maxima to float64
        lm = local_maxima.astype(np.float64)
        local_max_sigma_indx = local_maxima[:, -1]
        # translate final column of lm, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = sigma_list[local_max_sigma_indx]
        if scalar_sigma:
            # select one sigma column, keeping dimension
            sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

        # Remove sigma index and replace with sigmas
        lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

        sigma_dim = sigmas_of_peaks.shape[1]

        # return blob_detection._prune_blobs(lm, overlap, sigma_dim=sigma_dim,max_lap = max_lap) #save for testing
        blobs_pruned, sigma_indx_pruned = self._prune_blobs(
            lm,
            overlap,
            sigma_dim=sigma_dim,
            max_lap=max_lap,
            sigma_indx=local_max_sigma_indx,
        )

        if self.fitting_parameters.get("fitting_image", "Original") == "Original":
            fit_objects = self._create_mask(
                image2,
                blobs_pruned,
                size=self.fitting_parameters.get("mask_size", 5),
                sigma_indx=sigma_indx_pruned,
            )
        else:
            fit_objects = self._create_mask(
                image_cube,
                blobs_pruned,
                size=self.fitting_parameters.get("mask_size", 5),
                sigma_indx=sigma_indx_pruned,
            )

        return {
            "Fitted": self._update_blob_estimate(
                blobs_pruned=blobs_pruned,
                fit_object=fit_objects,
                radius_func=self.fitting_parameters.get("radius_func", identity),
            ),
            "Scale": blobs_pruned,
            "Fit": fit_objects,
            "lap_cube": gl_images,
            "local_maxima": local_maxima,
        }

    def _update_blob_estimate(self, blobs_pruned, fit_object, radius_func=None):
        """Using fitted parameters update the esimates of the centroid and sigmas for blob fit

        Parameters
        ----------
        blobs_pruned: (ndarry)
            output from self._prune_blobs()
        fit_object: (lmfit.minimize.fit object)
            instances of the .fit object from lmfit's minimize
        radius_func: (functional, optional)
            Which function to use to remap anisotropic sigmas into isotropic sigma. Defaults to None.


        Returns
        -------
        (N,3) array
            first output is [(x,y,s),...] of the blobs updates
        (N,3) array
            second is the original blobs inputed for comparison
        """
        blobs = []
        for i, obj in enumerate(blobs_pruned):
            x = fit_object[i].params["centroid_x"].value
            y = fit_object[i].params["centroid_y"].value
            if radius_func is not None:
                radius = radius_func(
                    [
                        fit_object[i].params["sigma_x"].value,
                        fit_object[i].params["sigma_y"].value,
                    ]
                )
            else:
                radius = [
                    fit_object[i].params["sigma_x"].value,
                    fit_object[i].params["sigma_y"].value,
                ]
            if np.isscalar(radius):
                radius = [radius, radius]
            blobs.append([x, y, *radius])

        return np.stack(blobs)

    def _create_mask(self, img, coords, size, sigma_indx):
        """mask of the image at the center point of the pixel coordinate. Also applies the fits
        TODO make this modular and seperate the mask creation and the fitting to allow better user control.

        Parameters
        ----------
        img : 2D array
            array defining the image
        coords : (N,3) array
            array defining the [x,y,radius] of the blobs
        size : int
            size of the square box around coord[:2] in which to fit
        sigma_indx : N array
            array defining the index of the sigmas used for each blob in coords

        Returns
        -------
        lmfit.minimize.fit objects in a list: [object, object, ... ]
            the fit objects created by lmfit.minizie for each coord.

        Raises
        ------
        TypeError
            "size needs to an integer value"
        Exception
            "simga_indx needs to be same shape as coords"
        """

        if not isinstance(size, int):
            raise TypeError("size needs to an integer value")
        if len(sigma_indx) != len(coords):
            raise Exception("simga_indx needs to be same shape as coords")

        fit_objects = []

        for inx, val in enumerate(coords):
            if img.ndim == 3:
                # find the lap image that created this blob and get a mask
                lap_img = img[:, :, sigma_indx[inx]]
            else:
                lap_img = img
            if (
                val[-1] >= 30 * size
            ):  # fix this condition, right now defalts to using defined size
                x, y, view, _ = self._gaussian_mesh_helper(
                    lap_img,
                    val[:2],
                    sub_arr=[int(val[-1] * FWHM_FACTOR), int(val[-1] * FWHM_FACTOR)],
                )

            else:
                x, y, view, _ = self._gaussian_mesh_helper(
                    lap_img, val[:2], sub_arr=[size, size]
                )

            # initialize the fitter
            initials = self.initalize_2dgaus(
                height=np.max(view) - np.min(view),
                centroid_x=val[0],
                centroid_y=val[1],
                sigma_x=val[-1],
                sigma_y=val[-1],
                background=np.min(view),
            )
            fit = minimize(
                self.fitting_parameters.get("residual_func", residuals_gaus2d),
                initials,
                args=(x, y, view),
                method=self.fitting_parameters.get("fit_method", "least_squares"),
            )
            fit_objects.append(fit)

            # check fit
            if self.fitting_parameters.get("plot_fit", False):
                # if this condition is called more than 3 time it will overflow memory so be careful.
                # a condition is made to check if this is called more than 5 times and if so it will not plot

                z1 = gaussian2D(
                    x,
                    y,
                    height=fit.params["height"],
                    sig_x=fit.params["sigma_x"],
                    sig_y=fit.params["sigma_y"],
                    cen_x=fit.params["centroid_x"],
                    cen_y=fit.params["centroid_y"],
                    offset=fit.params["background"],
                )
                print(report_fit(fit))
                fig = plt.figure()
                ax = plt.axes(projection="3d")
                ax.plot_wireframe(x, y, view)
                ax.plot_wireframe(x, y, z1, color="green")
                plt.show()
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.imshow(lap_img, cmap="gray")
                plt.show()
        return fit_objects

    def _gaussian_mesh_helper(self, mesh_2d, initial_xy, sub_arr=[3, 3]):
        """takes a 2d_mesh (image data) and a bounding box to return a list of (x,y,z) in that bounding box
        box is implimented from the center point of the pixel. this function works similar to Analysis_functions.subarray2D()

        Parameters
        ----------
        mesh_2d : 2D array
            image data in 2D array format
        initial_xy : tuple or array-like of len 2
            [x,y] coordinate of the center point around which to create the subarray
        sub_arr : tuple or 1D array, optional
            the grid size from the center to create the mesh, by default [3,3]

        Returns
        -------
        1D list
            array containing:
                x,y: the x,y coordinates of the meshgrid
                mesh_view: subarray defined from the original image
                centers: centers of the mesh
        """
        # make x,y,z list from mesh data
        # find dims
        sub_arr = np.array(sub_arr)
        initial_xy = np.array(initial_xy)
        minx, miny = initial_xy - sub_arr
        maxx, maxy = initial_xy + sub_arr
        # make sure the bounds are within the size of mesh_2d
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx > mesh_2d.shape[0]:
            maxx = mesh_2d.shape[0] - 1
        if maxy > mesh_2d.shape[1]:
            maxy = mesh_2d.shape[1] - 1
        minx, miny = int(minx), int(miny)
        maxx, maxy = int(maxx), int(maxy)
        centers = [
            rescale_range(initial_xy[0], minx, maxx, 0, 2 * sub_arr[1] + 1),
            rescale_range(initial_xy[1], miny, maxy, 0, 2 * sub_arr[0] + 1),
        ]
        x, y = np.meshgrid(
            np.arange(minx, maxx, 1), np.arange(miny, maxy, 1), indexing="ij"
        )
        mesh_view = mesh_2d[minx:maxx, miny:maxy]

        return [x, y, mesh_view, centers]

    def initalize_2dgaus(self, **kwargs):
        """initalizes lmfit parameters

        Returns
        -------
        Parameter() class object
            class object defining the parameters with bounds for the fit later on
        """
        initial = Parameters()
        for i, j in kwargs.items():
            if (i == "centroid_x") or (i == "centroid_y"):
                disp = self.fitting_parameters.get("centroid_range", 1)
                initial.add(i, value=j, min=j - disp, max=j + disp)
            elif (i == "sigma_x") or (i == "sigma_y"):
                disp = self.fitting_parameters.get("sigma_range", 1)
                initial.add(i, value=j, min=j - disp, max=j + disp)
            elif i == "height":
                disp = self.fitting_parameters.get("height_range", 1)
                initial.add(i, value=j, min=j - disp, max=j + disp)
            else:
                initial.add(i, value=j)
        # initial.add("height",value=.3)
        # #initial.add("centroid_x",value=100.)
        # #initial.add("centroid_y",value=100.)
        # initial.add("sigma_x",value=20.)
        # #initial.add("sigma_y",value=20.)
        # initial.add("background",value=0.015)
        return initial


# take a list and return it as is
def identity(x):
    """identity function, returns the input

    Parameters:
    -----------
    x : any
        input to be returned

    Returns:
    --------
    x : any
        input

    Uses:
    -----
    identity(x) = x
    """
    return x


# isotropic residual gaussain
def iso_gaus(p, x, y, z):
    p["sigma_x"] = p["sigma_y"]
    return residuals_gaus2d(p, x, y, z)


def residuals_gaus2d(p, x, y, z, **kwargs):
    """Residual calculator for a 2D gaussian for lmfit.minimize

    Parameters
    ----------
    p : Parameter() object
        Parameters
    x : independent variable
        x values
    y : independent variable
        y values
    z : z = f(x,y)
        function values at x,y. Where function is the one we are trying to fit

    Returns
    -------
    array-like
        residuals of the function z-f(x,y) (2D gaussian)
    """
    height = p["height"].value
    cen_x = p["centroid_x"].value
    cen_y = p["centroid_y"].value
    sigma_x = p["sigma_x"].value
    sigma_y = p["sigma_y"].value
    offset = p["background"].value
    return z - gaussian2D(
        x=x,
        y=y,
        cen_x=cen_x,
        cen_y=cen_y,
        sig_x=sigma_x,
        sig_y=sigma_y,
        offset=offset,
        height=height,
        kwargs=kwargs,
    )


def gaussian2D(x, y, cen_x, cen_y, sig_x, sig_y, offset, height, kwargs={}):
    """2d gaussian anistropic"""
    return (height) * np.exp(
        -(((cen_x - x) / sig_x) ** 2 + ((cen_y - y) / sig_y) ** 2) / 2.0
    ) + offset
