"""
Stage 4: Bounding Box Regression Training
Paper Appendix C: Train ridge regression on pool5 features with IoU ≥ 0.6 pairs
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from rcnn.config.config import RCNNConfig
from rcnn.models.cnn import AlexNetRCNN
from rcnn.models.bbox_regressor import BoundingBoxRegressor
from rcnn.data.region_proposal import SelectiveSearchProposer, compute_iou
from rcnn.data.transforms import RegionWarper


class BBoxRegressionTrainer:

    def __init__(self, config: RCNNConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        self.cnn = AlexNetRCNN(num_classes=config.num_classes, pretrained=False)
        self.cnn.to(device)
        self.cnn.eval()

        checkpoint_path = config.output_dir / "finetuned_cnn_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.cnn.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded fine-tuned CNN from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Fine-tuned CNN not found at {checkpoint_path}")

        self.proposer = SelectiveSearchProposer(mode="fast", num_proposals=2000)
        self.warper = RegionWarper(output_size=(227, 227), context_padding=16)

        self.bbox_regressors = {}

    @torch.no_grad()
    def extract_features(self, images_tensor: torch.Tensor) -> np.ndarray:
        images_tensor = images_tensor.to(self.device)
        features = self.cnn.extract_features(images_tensor)
        return features.cpu().numpy()

    def train(
        self,
        annotations_csv: Path,
        images_dir: Path,
        iou_threshold: float = 0.6,
        batch_size: int = 64
    ):
        print(f"\nStage 4: Bounding Box Regression Training")
        print(f"IoU threshold for positive pairs: {iou_threshold}")
        print(f"Regularization λ: {self.config.bbox_reg_lambda}\n")

        annotations = pd.read_csv(annotations_csv)
        image_ids = annotations['image_id'].unique()

        all_features = []
        all_proposals = []
        all_gt_boxes = []
        all_class_ids = []

        print(f"Collecting training pairs from {len(image_ids)} images...")

        for img_idx, image_id in enumerate(tqdm(image_ids)):
            image_path = images_dir / image_id

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                proposals = self.proposer.generate_proposals(img)

                img_annotations = annotations[annotations['image_id'] == image_id]

                for _, row in img_annotations.iterrows():
                    coords = eval(row['geometry'])
                    xs = [pt[0] for pt in coords]
                    ys = [pt[1] for pt in coords]
                    gt_box = np.array([min(xs), min(ys), max(xs), max(ys)])
                    class_id = row['category_id']

                    ious = np.array([compute_iou(prop, gt_box) for prop in proposals])
                    valid_idx = np.where(ious >= iou_threshold)[0]

                    if len(valid_idx) == 0:
                        continue

                    valid_proposals = proposals[valid_idx]

                    warped_regions = []
                    for prop in valid_proposals:
                        warped = self.warper.warp_region(img, prop)
                        warped_regions.append(warped)

                    if len(warped_regions) == 0:
                        continue

                    warped_batch = torch.stack(warped_regions)
                    features = self.extract_features(warped_batch)

                    all_features.append(features)
                    all_proposals.append(valid_proposals)
                    all_gt_boxes.append(np.tile(gt_box, (len(valid_proposals), 1)))
                    all_class_ids.append(np.full(len(valid_proposals), class_id))

        if len(all_features) == 0:
            raise RuntimeError(f"No training pairs found with IoU ≥ {iou_threshold}")

        all_features = np.vstack(all_features)
        all_proposals = np.vstack(all_proposals)
        all_gt_boxes = np.vstack(all_gt_boxes)
        all_class_ids = np.concatenate(all_class_ids)

        print(f"\nCollected {len(all_features)} training pairs")

        unique_classes = np.unique(all_class_ids)
        print(f"Training regressors for {len(unique_classes)} classes: {unique_classes}")

        for class_id in unique_classes:
            class_mask = all_class_ids == class_id
            class_features = all_features[class_mask]
            class_proposals = all_proposals[class_mask]
            class_gt_boxes = all_gt_boxes[class_mask]

            print(f"\nClass {class_id}: {len(class_features)} training pairs")

            regressor = BoundingBoxRegressor(
                feature_dim=4096,
                lambda_reg=self.config.bbox_reg_lambda
            )

            regressor.train(class_features, class_proposals, class_gt_boxes)

            self.bbox_regressors[int(class_id)] = regressor

        print("\nBounding box regression training complete!")

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for class_id, regressor in self.bbox_regressors.items():
            save_path = output_dir / f"bbox_regressor_class_{class_id}.npz"
            regressor.save(save_path)

        print(f"\nSaved {len(self.bbox_regressors)} bbox regressors to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/rcnn")
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = RCNNConfig()
    config.output_dir = Path(args.output_dir)

    trainer = BBoxRegressionTrainer(config, device=args.device)

    trainer.train(
        annotations_csv=Path(args.annotations),
        images_dir=Path(args.images),
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size
    )

    trainer.save(config.output_dir)


if __name__ == "__main__":
    main()
