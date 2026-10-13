"""Conservative offline similarity alignment for visual-wandering crop pairs.

extract_features(path) lazily computes CPU SIFT features on a <=512px image and
caches at most 512 images, invalidating entries when file size or mtime changes.
Unreadable or featureless images return empty features, never a forced transform.

fit_alignment(source, destination) maps destination canonical coordinates into
source canonical coordinates:

    q_source = scale * R(angle) @ q_destination + [x, y]
    q = (pixel - [image_width/2, image_height/2]) / max(width, height)

Angles are degrees; x points right and y down, so a positive rotation appears
clockwise on screen. Translation and median fit error use source canonical units.
The returned coverage is the smaller of the two inlier convex-hull area fractions
relative to its own crop's area. None means insufficient evidence for alignment.
This is a presentation transform, not proof that two crops show the same object.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 512
MAX_FEATURES = 500
RATIO = .75


@dataclass(frozen=True)
class FeatureData:
    points: np.ndarray
    descriptors: np.ndarray
    width: float
    height: float


def _empty():
    return FeatureData(np.empty((0, 2), np.float32), np.empty((0, 128), np.float32), 0, 0)


@lru_cache(maxsize=512)
def _extract(path, size, modified):
    del size, modified  # Part of the cache identity, not image processing inputs.
    try:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('L')
            width, height = image.size
            image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            gray = np.asarray(image)
        resized_height, resized_width = gray.shape
        if min(resized_width, resized_height) < 16:
            return _empty()
        keypoints, descriptors = cv2.SIFT_create(nfeatures=MAX_FEATURES).detectAndCompute(gray, None)
    except (OSError, ValueError, cv2.error):
        return _empty()
    if descriptors is None or len(keypoints) < 8:
        return _empty()
    # SIFT can return slightly more than nfeatures when responses tie.
    order = sorted(range(len(keypoints)), key=lambda i: (-keypoints[i].response, i))[:MAX_FEATURES]
    points = np.asarray([keypoints[i].pt for i in order], dtype=np.float32)
    points *= np.asarray([width/resized_width, height/resized_height], np.float32)
    points = (points - np.asarray([width/2, height/2], np.float32)) / max(width, height)
    descriptors = np.ascontiguousarray(descriptors[order], dtype=np.float32)
    points.flags.writeable = False
    descriptors.flags.writeable = False
    return FeatureData(points, descriptors, width, height)


def extract_features(path):
    """Read crop features, with a bounded stat-keyed cache; missing files are empty."""
    try:
        path = Path(path).absolute()
        stat = path.stat()
    except (OSError, ValueError):
        return _empty()
    return _extract(str(path), stat.st_size, stat.st_mtime_ns)



def detection_features(features, box, crop_box):
    """Convert padded-crop features into detection-box canonical coordinates.

    box and crop_box are [left, top, right, bottom] in the SAME original image
    pixel coordinate system. Pass the image-clamped detection box used by the
    published context overlay. Crop dimensions must match the feature image;
    a one-pixel tolerance accounts for historical integer rounding. Features
    outside the detection are discarded so background cannot drive the fit.
    The returned width/height describe the detection, not the padded JPEG.
    """
    if not _valid(features):
        return _empty()
    try:
        box = np.asarray(box, dtype=np.float64)
        crop_box = np.asarray(crop_box, dtype=np.float64)
        if box.shape != (4,) or crop_box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(crop_box).all():
            return _empty()
        dimensions = box[2:] - box[:2]
        crop_dimensions = crop_box[2:] - crop_box[:2]
        if (dimensions <= 0).any() or (crop_dimensions <= 0).any():
            return _empty()
        if (np.abs(crop_dimensions - [features.width, features.height]) > 1).any():
            return _empty()
        if (box[:2] < crop_box[:2] - 1).any() or (box[2:] > crop_box[2:] + 1).any():
            return _empty()
        pixels = features.points.astype(np.float64) * max(features.width, features.height)
        pixels += [features.width/2, features.height/2]
        # Map a possible rounding difference back to the exact crop rectangle.
        pixels *= crop_dimensions / [features.width, features.height]
        pixels += crop_box[:2]
        inside = ((pixels >= box[:2]) & (pixels <= box[2:])).all(axis=1)
        if inside.sum() < 8:
            return _empty()
        points = np.ascontiguousarray((pixels[inside] - (box[:2]+box[2:])/2) / dimensions.max(), dtype=np.float32)
        descriptors = np.ascontiguousarray(features.descriptors[inside], dtype=np.float32)
        points.flags.writeable = False
        descriptors.flags.writeable = False
        return FeatureData(points, descriptors, float(dimensions[0]), float(dimensions[1]))
    except (TypeError, ValueError, FloatingPointError):
        return _empty()


def _valid(features):
    return (features.width > 0 and features.height > 0 and
            features.points.ndim == 2 and features.points.shape[1] == 2 and
            features.descriptors.ndim == 2 and features.descriptors.shape[1] == 128 and
            len(features.points) == len(features.descriptors) and len(features.points) >= 8 and
            np.isfinite(features.points).all() and np.isfinite(features.descriptors).all())


def _ratio_matches(first, second):
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(first, second, k=2)
    return {pair[0].queryIdx: pair[0].trainIdx for pair in pairs
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance}


def _coverage(points, features):
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    image_area = features.width * features.height / max(features.width, features.height)**2
    return float(cv2.contourArea(hull)) / image_area


def fit_alignment(source_features, dest_features):
    """Return accepted destination-to-source similarity parameters, or None.

    Acceptance requires >=8 mutual ratio-test inliers, >=55% consensus,
    >=15% inlier hull coverage of each crop, <=0.025 median residual, scale in
    [0.8, 1.25], rotation <=20 degrees and translation norm <=0.15.
    """
    if not _valid(source_features) or not _valid(dest_features):
        return None
    try:
        forward = _ratio_matches(dest_features.descriptors, source_features.descriptors)
        reverse = _ratio_matches(source_features.descriptors, dest_features.descriptors)
        matches = [(d, s) for d, s in forward.items() if reverse.get(s) == d]
        if len(matches) < 8:
            return None
        dest = np.asarray([dest_features.points[d] for d, _ in matches], dtype=np.float32)
        source = np.asarray([source_features.points[s] for _, s in matches], dtype=np.float32)
        matrix, mask = cv2.estimateAffinePartial2D(dest, source, method=cv2.RANSAC,
            ransacReprojThreshold=.015, maxIters=1500, confidence=.995, refineIters=10)
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None
        inliers = mask.ravel().astype(bool)
        count = int(inliers.sum())
        if count < 8 or count / len(matches) < .55:
            return None
        linear, translation = matrix[:, :2], matrix[:, 2]
        if np.linalg.det(linear) <= 0:
            return None
        scale = float(np.linalg.norm(linear[:, 0]))
        angle = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
        if not .8 <= scale <= 1.25 or abs(angle) > 20 or np.linalg.norm(translation) > .15:
            return None
        coverage = min(_coverage(dest[inliers], dest_features), _coverage(source[inliers], source_features))
        if coverage < .15:
            return None
        prediction = dest @ linear.T + translation
        error = float(np.median(np.linalg.norm(prediction[inliers] - source[inliers], axis=1)))
        if error > .025:
            return None
        return {'angle': round(angle, 5), 'scale': round(scale, 6),
                'x': round(float(translation[0]), 6), 'y': round(float(translation[1]), 6),
                'inliers': count, 'coverage': round(coverage, 5), 'error': round(error, 6)}
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return None
