"""
Stage 3: SVM Training with Hard Negative Mining
Section 2.3, Appendix B: "Detection SVMs"

Paper: "Once features are extracted and training labels are applied, we optimize
one linear SVM per class. Since the training data is too large to fit in memory,
we adopt the standard hard negative mining method."
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from rcnn.config.config import RCNNConfig
from rcnn.models.cnn import AlexNetFeatureExtractor
from rcnn.data import AirbusRCNNDataset


class SVMTrainer:

    def __init__(self, config: RCNNConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        print("Loading fine-tuned CNN...")
        self.cnn = AlexNetFeatureExtractor(
            pretrained=True,
            feature_layer=config.feature_layer,
            num_classes=config.num_classes
        )
        
        checkpoint_path = config.output_dir / "finetuned_cnn_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.cnn.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded fine-tuned model from {checkpoint_path}")
        else:
            print(f"⚠️  Fine-tuned model not found at {checkpoint_path}")
            print("Using ImageNet pre-trained weights instead")

        self.cnn = self.cnn.to(device)
        self.cnn.eval()

        self.svms = {}

    def extract_features(
        self,
        dataset: AirbusRCNNDataset,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"\nExtracting features from {len(dataset)} samples...")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        all_features = []
        all_labels = []
        all_ious = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].cpu().numpy()
                ious = batch['iou'].cpu().numpy()

                # Extract features from fc7 (4096-dim)
                features = self.cnn.extract_features(images)
                features = features.cpu().numpy()

                all_features.append(features)
                all_labels.append(labels)
                all_ious.append(ious)

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        ious = np.concatenate(all_ious)

        print(f"✓ Extracted features: {features.shape}")
        return features, labels, ious

    def prepare_svm_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        ious: np.ndarray,
        class_id: int = 1  # 1 = aircraft
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos_mask = (labels == class_id) & (ious >= 0.99)
        pos_features = features[pos_mask]
        pos_labels = np.ones(len(pos_features), dtype=np.int32)

        # Negative samples: IoU < 0.3
        neg_mask = ious < self.config.svm_neg_iou_threshold
        neg_features = features[neg_mask]
        neg_labels = np.zeros(len(neg_features), dtype=np.int32)

        print(f"\nSVM training data for class {class_id}:")
        print(f"  Positives (IoU ≈ 1.0): {len(pos_features)}")
        print(f"  Negatives (IoU < 0.3): {len(neg_features)}")
        print(f"  Ignored (0.3 ≤ IoU < 1.0): {np.sum((ious >= 0.3) & (ious < 0.99))}")

        return pos_features, pos_labels, neg_features, neg_labels

    def train_svm_with_hard_negatives(
        self,
        pos_features: np.ndarray,
        pos_labels: np.ndarray,
        neg_features: np.ndarray,
        neg_labels: np.ndarray,
        class_id: int = 1
    ) -> LinearSVC:
        print(f"\nTraining SVM for class {class_id}...")

        max_initial_neg = min(len(neg_features), len(pos_features) * 10)

        if max_initial_neg < len(neg_features):
            print(f"  Using subset of negatives: {max_initial_neg}/{len(neg_features)}")
            neg_indices = np.random.choice(len(neg_features), max_initial_neg, replace=False)
            initial_neg_features = neg_features[neg_indices]
            initial_neg_labels = neg_labels[neg_indices]
        else:
            initial_neg_features = neg_features
            initial_neg_labels = neg_labels

        X_train = np.vstack([pos_features, initial_neg_features])
        y_train = np.concatenate([pos_labels, initial_neg_labels])

        print(f"  Initial training: {len(pos_labels)} pos + {len(initial_neg_labels)} neg")

        # Train initial SVM
        svm = LinearSVC(
            C=0.001,  # Paper uses C=0.001 for PASCAL VOC
            class_weight='balanced',
            max_iter=10000,
            random_state=1337,
            verbose=0
        )

        svm.fit(X_train, y_train)

        if self.config.svm_hard_neg_mining and len(neg_features) > max_initial_neg:
            print("  Hard negative mining...")

            remaining_neg_features = neg_features[max_initial_neg:]
            scores = svm.decision_function(remaining_neg_features)

            hard_neg_mask = scores > 0
            hard_neg_features = remaining_neg_features[hard_neg_mask]

            if len(hard_neg_features) > 0:
                print(f"  Found {len(hard_neg_features)} hard negatives")

                X_train_hard = np.vstack([X_train, hard_neg_features])
                y_train_hard = np.concatenate([
                    y_train,
                    np.zeros(len(hard_neg_features), dtype=np.int32)
                ])

                print(f"  Retraining with {len(y_train_hard)} samples...")
                svm.fit(X_train_hard, y_train_hard)
            else:
                print("  No hard negatives found")
        
        train_predictions = svm.predict(X_train)
        train_acc = (train_predictions == y_train).mean()
        print(f"  ✓ Training accuracy: {train_acc:.4f}")

        return svm

    def train(
        self,
        annotations_csv: Path,
        images_dir: Path,
        train_val_split: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 4
    ):

        print("="*60)
        print("R-CNN Stage 3: SVM Training")
        print("="*60)

        print("\nCreating SVM training dataset...")
        print("Note: Using stricter IoU thresholds than fine-tuning")
        print(f"  Positives: IoU ≈ 1.0 (ground-truth boxes only)")
        print(f"  Negatives: IoU < {self.config.svm_neg_iou_threshold}")

        dataset = AirbusRCNNDataset(
            annotations_csv=annotations_csv,
            images_dir=images_dir,
            stage="svm",
            num_proposals=1000,
            cache_proposals=True,
            iou_positive_threshold=self.config.svm_pos_iou_threshold,
            iou_negative_threshold=self.config.svm_neg_iou_threshold,
        )

        features, labels, ious = self.extract_features(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        class_id = 1  # Aircraft

        pos_features, pos_labels, neg_features, neg_labels = \
            self.prepare_svm_training_data(features, labels, ious, class_id)

        if len(pos_features) == 0:
            raise ValueError("No positive samples found! Check dataset annotations.")

        svm = self.train_svm_with_hard_negatives(
            pos_features, pos_labels,
            neg_features, neg_labels,
            class_id
        )

        self.svms[class_id] = svm

        # Save SVM
        svm_path = self.config.output_dir / f"svm_class_{class_id}.pkl"
        with open(svm_path, 'wb') as f:
            pickle.dump(svm, f)
        print(f"\n✓ Saved SVM to {svm_path}")

        # Save all SVMs together
        svms_path = self.config.output_dir / "svms_all.pkl"
        with open(svms_path, 'wb') as f:
            pickle.dump(self.svms, f)
        print(f"✓ Saved all SVMs to {svms_path}")

        print("\n" + "="*60)
        print("SVM Training Complete!")
        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='R-CNN SVM Training')
    parser.add_argument('--dataset', type=str, default='airbus',
                       help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    args = parser.parse_args()

    # Configuration
    config = RCNNConfig()
    device = torch.device(config.device if torch.cuda.is_available()
                         else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Dataset paths
    BASE_DIR = Path("dataset/airbus-aircrafts-sample-dataset")
    ANNOTATIONS_CSV = BASE_DIR / "annotations.csv"
    IMAGES_DIR = BASE_DIR / "images"

    if not ANNOTATIONS_CSV.exists():
        print(f"Error: Annotations not found at {ANNOTATIONS_CSV}")
        exit(1)

    # Train SVM
    trainer = SVMTrainer(config, device=str(device))
    trainer.train(
        annotations_csv=ANNOTATIONS_CSV,
        images_dir=IMAGES_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
