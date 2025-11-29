"""
Configuration for R-CNN training and inference
Based on Girshick et al., 2014
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RCNNConfig:
    """R-CNN hyperparameters from the paper"""

    # Dataset paths
    dataset_dir: Path = Path("dataset/airbus-aircrafts-sample-dataset")
    output_dir: Path = Path("artifacts/rcnn")

    # Region proposals
    num_proposals: int = 2000
    selective_search_mode: str = "fast"  # Selective search mode

    # CNN architecture
    input_size: Tuple[int, int] = (227, 227)
    context_padding: int = 16
    feature_layer: str = "fc7"
    feature_dim: int = 4096

    # Fine-tuning (Section 2.3)
    finetune_learning_rate: float = 0.001
    finetune_batch_size: int = 128
    finetune_pos_per_batch: int = 32
    finetune_iou_threshold: float = 0.5
    finetune_epochs: int = 50

    # SVM training (Section 2.3, Appendix B)
    svm_pos_iou_threshold: float = 1.0
    svm_neg_iou_threshold: float = 0.3
    svm_hard_neg_mining: bool = True

    # Bounding box regression (Appendix C)
    bbox_reg_lambda: float = 1000.0 
    bbox_reg_iou_threshold: float = 0.6

    # Test-time detection (Section 2.2)
    nms_threshold: float = 0.3 
    score_threshold: float = 0.5

    # Hardware
    device: str = "cuda"  # "cpu", "mps"
    num_workers: int = 4

    # Classes (Airbus aircraft detection)
    num_classes: int = 1
    class_names: list = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ["aircraft"]
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Default configuration
config = RCNNConfig()
