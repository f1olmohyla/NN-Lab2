"""
Patch-based detector that slides an AlexNet classifier over large images.

The detector loads weights from the standalone classifier and evaluates
all overlapping patches across a grid, returning patch-level hits and
merged regions that approximate aircraft locations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from alexnet_classifier import IMAGENET_MEAN, IMAGENET_STD, build_alexnet_classifier


Box = Tuple[int, int, int, int]


class PatchGridDetector:
    """Slide a classifier over an image grid to localize aircraft."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        patch_size: int = 256,
        stride: int = 128,
        score_threshold: float = 0.6,
        merge_iou: float = 0.3,
        batch_size: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {self.weights_path}")

        self.patch_size = patch_size
        self.stride = stride
        self.score_threshold = score_threshold
        self.merge_iou = merge_iou
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_alexnet_classifier(num_classes=2)
        state_dict = torch.load(self.weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _compute_positions(length: int, patch_size: int, stride: int) -> List[int]:
        if length <= patch_size:
            return [0]
        positions = list(range(0, length - patch_size + 1, stride))
        last_start = length - patch_size
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    @staticmethod
    def _preprocess_patch(crop: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(crop)
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor

    def _prepare_patches(self, image: Image.Image) -> List[Dict[str, object]]:
        width, height = image.size
        xs = self._compute_positions(width, self.patch_size, self.stride)
        ys = self._compute_positions(height, self.patch_size, self.stride)

        patches: List[Dict[str, object]] = []
        for top in ys:
            for left in xs:
                bbox = (
                    int(left),
                    int(top),
                    int(min(left + self.patch_size, width)),
                    int(min(top + self.patch_size, height)),
                )
                crop = image.crop(bbox).resize((224, 224), Image.BILINEAR)
                tensor = self._preprocess_patch(crop)
                patches.append({"bbox": bbox, "tensor": tensor})
        return patches

    def detect_image(self, image: Image.Image, *, aggregate: bool = True) -> Dict[str, object]:
        """Run detection on a PIL image."""

        image = image.convert("RGB")
        patch_entries = self._prepare_patches(image)
        if not patch_entries:
            return {"patch_detections": [], "detections": [], "patch_total": 0}

        detections: List[Dict[str, object]] = []
        tensors: Sequence[torch.Tensor] = [entry["tensor"] for entry in patch_entries]

        with torch.no_grad():
            for start in range(0, len(tensors), self.batch_size):
                end = min(start + self.batch_size, len(tensors))
                batch = torch.stack(list(tensors[start:end])).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                class_scores = probs[:, 1]
                pred_labels = torch.argmax(probs, dim=1)
                for offset, (score_tensor, label_tensor) in enumerate(zip(class_scores, pred_labels)):
                    score = float(score_tensor.item())
                    is_aircraft = int(label_tensor.item()) == 1
                    entry = patch_entries[start + offset]
                    if is_aircraft and score >= self.score_threshold:
                        detections.append({"bbox": entry["bbox"], "score": score, "label": 1})

        merged = self.merge_detections(detections) if aggregate else detections
        return {
            "patch_detections": detections,
            "detections": merged,
            "patch_total": len(patch_entries),
        }

    def detect_path(self, image_path: Path | str, *, aggregate: bool = True) -> Dict[str, object]:
        with Image.open(image_path) as img:
            return self.detect_image(img, aggregate=aggregate)

    def merge_detections(self, detections: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        if not detections:
            return []

        remaining = sorted(detections, key=lambda item: item["score"], reverse=True)
        merged: List[Dict[str, object]] = []
        while remaining:
            base = remaining.pop(0)
            cluster = [base]
            keep: List[Dict[str, object]] = []
            for det in remaining:
                iou = self._bbox_iou(base["bbox"], det["bbox"])
                if iou >= self.merge_iou:
                    cluster.append(det)
                else:
                    keep.append(det)
            merged_box = (
                int(min(det["bbox"][0] for det in cluster)),
                int(min(det["bbox"][1] for det in cluster)),
                int(max(det["bbox"][2] for det in cluster)),
                int(max(det["bbox"][3] for det in cluster)),
            )
            merged_score = float(np.mean([det["score"] for det in cluster]))
            merged_label = cluster[0].get("label", 1)
            merged.append(
                {
                    "bbox": merged_box,
                    "score": merged_score,
                    "label": merged_label,
                    "members": len(cluster),
                    "max_score": float(max(det["score"] for det in cluster)),
                }
            )
            remaining = keep
        return merged

    @staticmethod
    def _bbox_iou(box_a: Box, box_b: Box) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(inter_x2 - inter_x1, 0)
        inter_h = max(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
        union = area_a + area_b - inter_area
        return float(inter_area / union) if union > 0 else 0.0


__all__ = ["PatchGridDetector"]

