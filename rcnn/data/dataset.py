"""
R-CNN Dataset for fine-tuning and SVM training
Adapted from existing AlexNet classification pipeline but with:
- Dynamic proposal generation (not pre-generated crops)
- IoU tracking for all proposals
- Support for different training stages (fine-tuning vs SVM)
- Stratified batch sampling (32 positive + 96 negative)

Paper reference: Section 2.3, Appendix B
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from ast import literal_eval
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image

from .region_proposal import SelectiveSearchProposer, compute_iou
from .transforms import RegionWarper


def polygon_to_bbox(geometry: str) -> np.ndarray:
    coords = literal_eval(geometry)
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


class AirbusRCNNDataset(Dataset):

    def __init__(
        self,
        annotations_csv: Path,
        images_dir: Path,
        stage: str = "finetune",  # "finetune" or "svm"
        num_proposals: int = 2000,
        cache_proposals: bool = True,
        iou_positive_threshold: float = 0.5,
        iou_negative_threshold: float = 0.3,
        context_padding: int = 16,
    ):
        self.annotations_csv = Path(annotations_csv)
        self.images_dir = Path(images_dir)
        self.stage = stage
        self.num_proposals = num_proposals
        self.cache_proposals = cache_proposals
        self.iou_positive_threshold = iou_positive_threshold
        self.iou_negative_threshold = iou_negative_threshold

        # Load annotations
        self.annotations = pd.read_csv(annotations_csv)
        self.annotations['bbox'] = self.annotations['geometry'].apply(polygon_to_bbox)

        # Group by image
        self.image_ids = self.annotations['image_id'].unique().tolist()
        self.image_to_boxes = {}
        for image_id in self.image_ids:
            boxes = self.annotations[self.annotations['image_id'] == image_id]['bbox'].tolist()
            self.image_to_boxes[image_id] = np.stack(boxes) if boxes else np.array([])

        # Proposal generator
        self.proposer = SelectiveSearchProposer(mode="fast", num_proposals=num_proposals)

        # Region warper
        self.warper = RegionWarper(
            output_size=(227, 227),
            context_padding=context_padding,
        )

        # Proposal cache
        self.proposal_cache: Dict[str, Dict] = {}

        # Build proposal index (image_idx, proposal_idx) for each sample
        self._build_proposal_index()

    def _build_proposal_index(self):
        print(f"Building proposal index for stage: {self.stage}")
        print(f"Total images to process: {len(self.image_ids)}\n")

        self.positive_samples = []  # (image_idx, proposal_idx)
        self.negative_samples = []  # (image_idx, proposal_idx)

        for img_idx, image_id in enumerate(self.image_ids):
            # Log which image is being processed
            print(f"[{img_idx + 1}/{len(self.image_ids)}] Processing: {image_id}", end="")

            # Generate or load cached proposals
            if self.cache_proposals and image_id in self.proposal_cache:
                proposals = self.proposal_cache[image_id]['proposals']
                ious = self.proposal_cache[image_id]['ious']
                print(" (cached)")
            else:
                print(" (generating proposals...)", end="", flush=True)
                image_path = self.images_dir / image_id
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    proposals = self.proposer.generate_proposals(img)

                # Compute IoU with GT boxes
                gt_boxes = self.image_to_boxes[image_id]
                ious = np.zeros(len(proposals), dtype=np.float32)

                if len(gt_boxes) > 0:
                    for i, proposal in enumerate(proposals):
                        # Max IoU with any GT box
                        proposal_ious = [compute_iou(proposal, gt) for gt in gt_boxes]
                        ious[i] = max(proposal_ious) if proposal_ious else 0.0

                if self.cache_proposals:
                    self.proposal_cache[image_id] = {
                        'proposals': proposals,
                        'ious': ious,
                    }
                print(" done")

            # Assign proposals based on IoU and training stage
            if self.stage == "finetune":
                # Fine-tuning: IoU ≥ 0.5 = positive, IoU < 0.5 = background
                for prop_idx, iou in enumerate(ious):
                    if iou >= self.iou_positive_threshold:
                        self.positive_samples.append((img_idx, prop_idx))
                    else:
                        self.negative_samples.append((img_idx, prop_idx))

            elif self.stage == "svm":
                # SVM: Only GT boxes = positive (IoU ≈ 1.0), IoU < 0.3 = negative
                for prop_idx, iou in enumerate(ious):
                    if iou >= 0.99:  # Only GT boxes (effectively IoU = 1.0)
                        self.positive_samples.append((img_idx, prop_idx))
                    elif iou < self.iou_negative_threshold:
                        self.negative_samples.append((img_idx, prop_idx))
                    # Ignore 0.3 ≤ IoU < 1.0 (ambiguous regions)
                    
            if (img_idx + 1) % 10 == 0:
                print(f"  → Progress: {img_idx + 1}/{len(self.image_ids)} images | "
                      f"Positives: {len(self.positive_samples)} | "
                      f"Negatives: {len(self.negative_samples)}\n")

        print(f"Total samples - positives: {len(self.positive_samples)}, "
              f"negatives: {len(self.negative_samples)}")

        # Combined index
        self.sample_index = (
            [(idx, 1) for idx in self.positive_samples] +  # (sample_idx, label=1)
            [(idx, 0) for idx in self.negative_samples]    # (sample_idx, label=0)
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        (img_idx, prop_idx), label = self.sample_index[idx]
        image_id = self.image_ids[img_idx]

        # Load image
        image_path = self.images_dir / image_id
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Get cached proposal
            if image_id not in self.proposal_cache:
                raise RuntimeError(f"Proposal not cached for {image_id}")

            proposals = self.proposal_cache[image_id]['proposals']
            ious = self.proposal_cache[image_id]['ious']

            proposal = proposals[prop_idx]
            iou = ious[prop_idx]

            # Warp region to 227x227 with context padding
            warped_region = self.warper.warp_region(img, proposal)

        return {
            'image': warped_region,
            'label': torch.tensor(label, dtype=torch.long),
            'iou': torch.tensor(iou, dtype=torch.float32),
        }


class StratifiedBatchSampler(Sampler):

    def __init__(
        self,
        dataset: AirbusRCNNDataset,
        batch_size: int = 128,
        num_positive: int = 32,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_positive = num_positive
        self.num_negative = batch_size - num_positive  # 96 in paper
        self.drop_last = drop_last

        # Separate positive and negative indices
        self.positive_indices = [i for i, ((_, _), label) in enumerate(dataset.sample_index) if label == 1]
        self.negative_indices = [i for i, ((_, _), label) in enumerate(dataset.sample_index) if label == 0]

        print(f"Stratified sampler: {len(self.positive_indices)} positives, "
              f"{len(self.negative_indices)} negatives")
        print(f"Batch composition: {num_positive} pos + {self.num_negative} neg = {batch_size}")

    def __iter__(self):
        # Shuffle indices
        pos_perm = torch.randperm(len(self.positive_indices)).tolist()
        neg_perm = torch.randperm(len(self.negative_indices)).tolist()

        pos_indices = [self.positive_indices[i] for i in pos_perm]
        neg_indices = [self.negative_indices[i] for i in neg_perm]

        # Determine number of batches
        num_batches = min(
            len(pos_indices) // self.num_positive,
            len(neg_indices) // self.num_negative,
        )

        for batch_idx in range(num_batches):
            batch = []

            # Sample positives
            pos_start = batch_idx * self.num_positive
            batch.extend(pos_indices[pos_start:pos_start + self.num_positive])

            # Sample negatives
            neg_start = batch_idx * self.num_negative
            batch.extend(neg_indices[neg_start:neg_start + self.num_negative])

            # Shuffle batch
            perm = torch.randperm(len(batch)).tolist()
            batch = [batch[i] for i in perm]

            yield batch

    def __len__(self) -> int:
        return min(
            len(self.positive_indices) // self.num_positive,
            len(self.negative_indices) // self.num_negative,
        )


def create_rcnn_dataloaders(
    annotations_csv: Path,
    images_dir: Path,
    stage: str = "finetune",
    batch_size: int = 128,
    num_positive_per_batch: int = 32,
    num_workers: int = 4,
    train_val_split: float = 0.8,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    # Load all images and split
    annotations = pd.read_csv(annotations_csv)
    image_ids = annotations['image_id'].unique()

    # Split images (not annotations) to avoid data leakage
    num_train = int(len(image_ids) * train_val_split)
    np.random.seed(1337)
    np.random.shuffle(image_ids)

    train_images = image_ids[:num_train]
    val_images = image_ids[num_train:]

    # Create train/val annotation CSVs (temporary)
    train_annotations = annotations[annotations['image_id'].isin(train_images)]
    val_annotations = annotations[annotations['image_id'].isin(val_images)]

    # Save temporary CSVs
    temp_dir = Path(annotations_csv).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / f"train_{stage}.csv"
    val_csv = temp_dir / f"val_{stage}.csv"

    train_annotations.to_csv(train_csv, index=False)
    val_annotations.to_csv(val_csv, index=False)

    # Create datasets
    train_dataset = AirbusRCNNDataset(
        annotations_csv=train_csv,
        images_dir=images_dir,
        stage=stage,
    )

    val_dataset = AirbusRCNNDataset(
        annotations_csv=val_csv,
        images_dir=images_dir,
        stage=stage,
    )

    # Create samplers for fine-tuning stage
    if stage == "finetune":
        train_sampler = StratifiedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            num_positive=num_positive_per_batch,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        # SVM stage can use standard sampling
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
