"""
Non-Maximum Suppression (NMS)
Section 2.2: "we apply a greedy non-maximum suppression (for each class independently)"
"""

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Non-maximum suppression.

    Paper: "we apply a greedy non-maximum suppression (for each class independently)
    that rejects a region if it has an intersection-over-union (IoU) overlap with a
    higher scoring selected region larger than a learned threshold."

    Args:
        boxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        scores: (N,) confidence scores
        threshold: IoU threshold (paper doesn't specify exact value)

    Returns:
        keep: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by score (descending)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU below threshold
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def multi_class_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    num_classes: int,
    threshold: float = 0.3
) -> list:
    """
    Apply NMS independently per class.

    Args:
        boxes: (N, 4) bounding boxes
        scores: (N, num_classes) class scores
        num_classes: Number of classes
        threshold: NMS IoU threshold

    Returns:
        detections: List of (boxes, scores, class_ids) tuples per class
    """
    detections = []

    for class_id in range(num_classes):
        class_scores = scores[:, class_id]

        # Apply NMS for this class
        keep = nms(boxes, class_scores, threshold)

        if len(keep) > 0:
            detections.append({
                'boxes': boxes[keep],
                'scores': class_scores[keep],
                'class_id': class_id
            })

    return detections
