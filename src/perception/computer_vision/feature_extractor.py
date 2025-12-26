"""
Feature extraction utilities for computer vision.

Provides methods for extracting visual features from images including edges,
corners, keypoints, and descriptors.
"""
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import cv2


class FeatureExtractor:
    """
    Utility class for extracting visual features from images.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        # Initialize feature detectors
        self.sift = None
        self.orb = None
        self.fast = None

        # Try to initialize SIFT (may not be available in all OpenCV builds)
        try:
            self.sift = cv2.SIFT_create()
        except AttributeError:
            print("Warning: SIFT not available in this OpenCV build")

        # Initialize ORB (always available)
        self.orb = cv2.ORB_create()

        # Initialize FAST corner detector
        self.fast = cv2.FastFeatureDetector_create()

    def extract_edges(self, image: np.ndarray, low_threshold: int = 50,
                      high_threshold: int = 150) -> np.ndarray:
        """
        Extract edges using Canny edge detection.

        Args:
            image: Input image (grayscale or BGR)
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny

        Returns:
            Edge map as binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Detect edges
        edges = cv2.Canny(blurred, low_threshold, high_threshold)

        return edges

    def extract_corners_harris(self, image: np.ndarray,
                               block_size: int = 2,
                               ksize: int = 3,
                               k: float = 0.04) -> np.ndarray:
        """
        Extract corners using Harris corner detection.

        Args:
            image: Input image (grayscale or BGR)
            block_size: Size of neighborhood for corner detection
            ksize: Aperture parameter for Sobel operator
            k: Harris detector free parameter

        Returns:
            Corner response map
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Ensure float32 type
        gray = np.float32(gray)

        # Detect corners
        corners = cv2.cornerHarris(gray, block_size, ksize, k)

        return corners

    def extract_keypoints_sift(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        if self.sift is None:
            raise RuntimeError("SIFT not available")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        return keypoints, descriptors

    def extract_keypoints_orb(self, image: np.ndarray,
                              max_features: int = 500) -> Tuple[List, np.ndarray]:
        """
        Extract ORB keypoints and descriptors.

        Args:
            image: Input image (grayscale or BGR)
            max_features: Maximum number of features to extract

        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Set maximum features
        self.orb.setMaxFeatures(max_features)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        return keypoints, descriptors

    def extract_fast_keypoints(self, image: np.ndarray,
                               threshold: int = 10) -> List:
        """
        Extract FAST corner keypoints.

        Args:
            image: Input image (grayscale or BGR)
            threshold: Threshold for FAST detector

        Returns:
            List of keypoints
        """
        # Set threshold
        self.fast.setThreshold(threshold)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect keypoints
        keypoints = self.fast.detect(gray, None)

        return keypoints

    def extract_color_histogram(self, image: np.ndarray,
                                bins: int = 256,
                                normalized: bool = True) -> np.ndarray:
        """
        Extract color histogram features.

        Args:
            image: Input image (BGR)
            bins: Number of bins for histogram
            normalized: Whether to normalize the histogram

        Returns:
            Color histogram (concatenated for all channels)
        """
        if len(image.shape) != 3:
            raise ValueError("Color histogram requires BGR image")

        # Calculate histogram for each channel
        histograms = []
        for i in range(3):  # B, G, R channels
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            if normalized:
                hist = hist / hist.sum()
            histograms.append(hist.flatten())

        # Concatenate histograms
        color_hist = np.concatenate(histograms)

        return color_hist

    def extract_hog_features(self, image: np.ndarray,
                            win_size: Tuple[int, int] = (64, 128),
                            block_size: Tuple[int, int] = (16, 16),
                            block_stride: Tuple[int, int] = (8, 8),
                            cell_size: Tuple[int, int] = (8, 8),
                            nbins: int = 9) -> np.ndarray:
        """
        Extract Histogram of Oriented Gradients (HOG) features.

        Args:
            image: Input image (grayscale or BGR)
            win_size: Window size for HOG
            block_size: Block size for normalization
            block_stride: Block stride
            cell_size: Cell size
            nbins: Number of orientation bins

        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to win_size
        resized = cv2.resize(gray, win_size)

        # Create HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Compute features
        features = hog.compute(resized)

        return features.flatten()

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using GLCM (Gray Level Co-occurrence Matrix) approach.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Dictionary of texture features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate simple texture metrics
        features = {}

        # Variance (measure of texture roughness)
        features['variance'] = float(np.var(gray))

        # Standard deviation
        features['std_dev'] = float(np.std(gray))

        # Entropy (measure of randomness)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero entries
        features['entropy'] = float(-np.sum(hist * np.log2(hist)))

        # Edge density
        edges = self.extract_edges(gray)
        features['edge_density'] = float(np.sum(edges > 0) / edges.size)

        return features

    def match_features(self, descriptors1: np.ndarray,
                      descriptors2: np.ndarray,
                      method: str = 'bf',
                      k: int = 2) -> List:
        """
        Match features between two sets of descriptors.

        Args:
            descriptors1: First set of descriptors
            descriptors2: Second set of descriptors
            method: Matching method ('bf' for brute force, 'flann' for FLANN)
            k: Number of nearest neighbors

        Returns:
            List of matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []

        if method == 'bf':
            # Brute force matcher
            if descriptors1.dtype == np.uint8:
                # For binary descriptors (ORB, etc.)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                # For float descriptors (SIFT, etc.)
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            matches = matcher.knnMatch(descriptors1, descriptors2, k=k)

        elif method == 'flann':
            # FLANN matcher
            if descriptors1.dtype == np.uint8:
                # For binary descriptors
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
            else:
                # For float descriptors
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

            matches = matcher.knnMatch(descriptors1, descriptors2, k=k)
        else:
            raise ValueError(f"Unknown matching method: {method}")

        return matches

    def filter_matches_ratio_test(self, matches: List,
                                  ratio: float = 0.7) -> List:
        """
        Filter matches using Lowe's ratio test.

        Args:
            matches: List of matches from knnMatch
            ratio: Ratio threshold

        Returns:
            Filtered list of good matches
        """
        good_matches = []

        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                # Ratio test
                if m.distance < ratio * n.distance:
                    good_matches.append(m)

        return good_matches

    def visualize_keypoints(self, image: np.ndarray,
                           keypoints: List,
                           color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Visualize keypoints on an image.

        Args:
            image: Input image
            keypoints: List of keypoints
            color: Color for keypoint visualization

        Returns:
            Image with visualized keypoints
        """
        vis_image = cv2.drawKeypoints(image, keypoints, None, color,
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return vis_image

    def get_feature_statistics(self, keypoints: List,
                               descriptors: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get statistics about extracted features.

        Args:
            keypoints: List of keypoints
            descriptors: Feature descriptors (optional)

        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'num_keypoints': len(keypoints),
            'descriptor_shape': None,
            'descriptor_type': None
        }

        if descriptors is not None:
            stats['descriptor_shape'] = descriptors.shape
            stats['descriptor_type'] = str(descriptors.dtype)

        # Calculate keypoint statistics if available
        if keypoints:
            sizes = [kp.size for kp in keypoints]
            responses = [kp.response for kp in keypoints]

            stats['avg_size'] = float(np.mean(sizes))
            stats['avg_response'] = float(np.mean(responses))
            stats['max_response'] = float(np.max(responses))

        return stats
