"""
R-CNN Inference Pipeline
Complete pipeline: Selective Search → CNN → SVM → NMS → Bbox Regression
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import pickle

from rcnn.config.config import RCNNConfig
from rcnn.models.cnn import AlexNetRCNN
from rcnn.models.bbox_regressor import BoundingBoxRegressor
from rcnn.data.region_proposal import SelectiveSearchProposer, compute_iou
from rcnn.data.transforms import RegionWarper


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


class RCNNDetector:

    def __init__(self, config: RCNNConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        self.cnn = AlexNetRCNN(num_classes=config.num_classes, pretrained=False)
        self.cnn.to(device)
        self.cnn.eval()

        self.svms = {}
        self.bbox_regressors = {}

        self.proposer = SelectiveSearchProposer(mode="fast", num_proposals=2000)
        self.warper = RegionWarper(output_size=(227, 227), context_padding=16)

    def load_models(self, artifacts_dir: Path):
        artifacts_dir = Path(artifacts_dir)

        cnn_path = artifacts_dir / "finetuned_cnn_best.pth"
        if not cnn_path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {cnn_path}")

        checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=False)
        self.cnn.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CNN from {cnn_path}")

        for svm_path in artifacts_dir.glob("svm_class_*.pkl"):
            class_id = int(svm_path.stem.split("_")[-1])
            with open(svm_path, 'rb') as f:
                svm = pickle.load(f)
            self.svms[class_id] = svm
            print(f"Loaded SVM for class {class_id}")

        for bbox_path in artifacts_dir.glob("bbox_regressor_class_*.npz"):
            class_id = int(bbox_path.stem.split("_")[-1])
            regressor = BoundingBoxRegressor(feature_dim=4096, lambda_reg=self.config.bbox_reg_lambda)
            regressor.load(bbox_path)
            self.bbox_regressors[class_id] = regressor

        print(f"\nLoaded {len(self.svms)} SVMs and {len(self.bbox_regressors)} bbox regressors")

    @torch.no_grad()
    def extract_features(self, images_tensor: torch.Tensor) -> np.ndarray:
        images_tensor = images_tensor.to(self.device)
        features = self.cnn.extract_features(images_tensor)
        return features.cpu().numpy()

    def detect(
        self,
        image: Image.Image,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        use_bbox_regression: bool = True
    ) -> List[Dict]:
        proposals = self.proposer.generate_proposals(image)
        print(f"Generated {len(proposals)} proposals")

        warped_regions = [self.warper.warp_region(image, prop) for prop in proposals]
        warped_batch = torch.stack(warped_regions)

        features = self.extract_features(warped_batch)

        detections = []

        for class_id, svm in self.svms.items():
            scores = svm.decision_function(features)

            positive_idx = np.where(scores > score_threshold)[0]

            if len(positive_idx) == 0:
                continue

            class_scores = scores[positive_idx]
            class_boxes = proposals[positive_idx]

            keep_idx = nms(class_boxes, class_scores, nms_threshold)

            final_boxes = class_boxes[keep_idx]
            final_scores = class_scores[keep_idx]
            final_features = features[positive_idx][keep_idx]

            if use_bbox_regression and class_id in self.bbox_regressors:
                regressor = self.bbox_regressors[class_id]
                final_boxes = regressor.predict(final_features, final_boxes)

            for box, score in zip(final_boxes, final_scores):
                detections.append({
                    'class_id': class_id,
                    'score': float(score),
                    'bbox': box.tolist()
                })

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        return detections

    def detect_batch(
        self,
        images: List[Image.Image],
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        use_bbox_regression: bool = True
    ) -> List[List[Dict]]:
        return [
            self.detect(img, score_threshold, nms_threshold, use_bbox_regression)
            for img in images
        ]
