"""Vessel centerline, thin-branch, bifurcation and boundary measurements."""

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


def cldice(prediction, truth):
    pred_skeleton, true_skeleton = skeletonize(prediction), skeletonize(truth)
    if not prediction.any() and not truth.any():
        return 1.0
    precision = (pred_skeleton & truth).sum() / max(int(pred_skeleton.sum()), 1)
    recall = (true_skeleton & prediction).sum() / max(int(true_skeleton.sum()), 1)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def thin_vessel_recall(prediction, truth, max_diameter=3.0):
    """Recall on GT centerline pixels with local EDT diameter <= max_diameter."""
    if max_diameter <= 0:
        raise ValueError("Thin-vessel diameter must be positive")
    reference = skeletonize(truth) & (2 * ndimage.distance_transform_edt(truth) <= max_diameter)
    return float(prediction[reference].mean()) if reference.any() else float("nan")


def bifurcation_points(mask):
    """Cluster skeleton pixels having >=3 eight-connected neighbors."""
    skeleton = skeletonize(mask)
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                                 mode="constant") - skeleton
    candidates = skeleton & (neighbors >= 3)
    components, count = ndimage.label(candidates, structure=np.ones((3, 3)))
    if not count:
        return np.empty((0, 2), dtype=float)
    return np.asarray(ndimage.center_of_mass(candidates, components, range(1, count + 1)))


def bifurcation_f1(prediction, truth, tolerance=2.0):
    """Maximum-cardinality one-to-one matching within a pixel tolerance."""
    if tolerance < 0:
        raise ValueError("Bifurcation tolerance cannot be negative")
    predicted, reference = bifurcation_points(prediction), bifurcation_points(truth)
    if not len(predicted) and not len(reference):
        return 1.0
    if not len(predicted) or not len(reference):
        return 0.0
    distances = cdist(predicted, reference)
    penalty = (min(distances.shape) + 1) * (tolerance + 1)
    rows, columns = linear_sum_assignment(np.where(distances <= tolerance, distances, penalty))
    matches = int((distances[rows, columns] <= tolerance).sum())
    return float(2 * matches / (len(predicted) + len(reference)))


def hd95(prediction, truth):
    """Maximum of directional 95th-percentile surface distances in pixels."""
    if not prediction.any() and not truth.any():
        return 0.0
    if not prediction.any() or not truth.any():
        return float(np.hypot(prediction.shape[0] - 1, prediction.shape[1] - 1))
    footprint = ndimage.generate_binary_structure(2, 2)
    pred_edge = prediction & ~ndimage.binary_erosion(prediction, structure=footprint, border_value=0)
    true_edge = truth & ~ndimage.binary_erosion(truth, structure=footprint, border_value=0)
    pred_to_true = ndimage.distance_transform_edt(~true_edge)[pred_edge]
    true_to_pred = ndimage.distance_transform_edt(~pred_edge)[true_edge]
    return float(max(np.percentile(pred_to_true, 95), np.percentile(true_to_pred, 95)))


def vessel_structure_metrics(probabilities, truth, threshold=0.5, thin_diameter=3.0,
                              bifurcation_tolerance=2.0, roi=None):
    probabilities, truth = np.asarray(probabilities), np.asarray(truth, dtype=bool)
    if probabilities.ndim != 2 or probabilities.shape != truth.shape:
        raise ValueError("Structure metrics expect matching 2D masks")
    if not np.isfinite(probabilities).all() or not 0 <= threshold <= 1:
        raise ValueError("Predictions must be finite and threshold must be in [0,1]")
    prediction = probabilities >= threshold
    if roi is not None:
        prediction, truth = prediction & roi, truth & roi
    return {"cldice": cldice(prediction, truth),
            "thin_recall": thin_vessel_recall(prediction, truth, thin_diameter),
            "bifurcation_f1": bifurcation_f1(prediction, truth, bifurcation_tolerance),
            "hd95": hd95(prediction, truth)}
