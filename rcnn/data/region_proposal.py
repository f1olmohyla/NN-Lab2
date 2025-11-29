"""
Region proposal generation using Selective Search
Section 2.1: "We use selective search to enable a controlled comparison"
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import cv2


class SelectiveSearchProposer:
    
    def __init__(self, mode: str = "fast", num_proposals: int = 2000):
        self.mode = mode
        self.num_proposals = num_proposals

    def generate_proposals(self, image: Image.Image) -> np.ndarray:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Initialize Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_cv)

        if self.mode == "fast":
            ss.switchToSelectiveSearchFast()
        else:
            ss.switchToSelectiveSearchQuality()

        # Generate proposals
        rects = ss.process()

        # Convert to (x1, y1, x2, y2) format
        proposals = []
        for (x, y, w, h) in rects[:self.num_proposals]:
            proposals.append([x, y, x + w, y + h])

        return np.array(proposals, dtype=np.float32)

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.generate_proposals(image)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    # Intersection area
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Union area
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def assign_proposals_to_ground_truth(
    proposals: np.ndarray,
    ground_truth: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    num_proposals = len(proposals)
    labels = np.zeros(num_proposals, dtype=np.int32)  # 0 = background
    max_ious = np.zeros(num_proposals, dtype=np.float32)

    for i, proposal in enumerate(proposals):
        if len(ground_truth) == 0:
            continue

        # Compute IoU with all ground truth boxes
        ious = np.array([compute_iou(proposal, gt) for gt in ground_truth])
        max_iou = ious.max()
        max_ious[i] = max_iou

        if max_iou >= iou_threshold:
            # Assign to class of best matching ground truth
            labels[i] = 1  # For binary classification (aircraft)

    return labels, max_ious
