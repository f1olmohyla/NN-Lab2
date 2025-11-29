"""
Linear SVM Classifier
Section 2.3: "Once features are extracted and training labels are applied,
we optimize one linear SVM per class"
"""

import numpy as np
from sklearn.svm import LinearSVC
from typing import Dict, List
import pickle
from pathlib import Path


class LinearSVMClassifier:

    def __init__(self, num_classes: int = 1, C: float = 0.001):
        self.num_classes = num_classes
        self.C = C
        self.svms: Dict[int, LinearSVC] = {}

        for class_id in range(num_classes):
            self.svms[class_id] = LinearSVC(
                C=C,
                max_iter=10000,
                dual=True,
                random_state=42
            )

    def train_single_class(
        self,
        class_id: int,
        positive_features: np.ndarray,
        negative_features: np.ndarray,
        hard_negative_mining: bool = True,
        max_iterations: int = 1
    ):
        print(f"Training SVM for class {class_id}...")
        print(f"  Positives: {len(positive_features)}")
        print(f"  Negatives: {len(negative_features)}")

        # Prepare initial training data
        X = np.vstack([positive_features, negative_features])
        y = np.hstack([
            np.ones(len(positive_features)),
            np.zeros(len(negative_features))
        ])

        # Train initial SVM
        self.svms[class_id].fit(X, y)

        # Hard negative mining (Section 2.3)
        if hard_negative_mining:
            for iteration in range(max_iterations):
                print(f"  Hard negative mining iteration {iteration + 1}/{max_iterations}")

                # Find hard negatives (false positives)
                neg_scores = self.svms[class_id].decision_function(negative_features)
                hard_neg_mask = neg_scores > 0  # Misclassified negatives

                if not hard_neg_mask.any():
                    print("  No hard negatives found, stopping.")
                    break

                # Retrain with hard negatives
                hard_negatives = negative_features[hard_neg_mask]
                X = np.vstack([positive_features, hard_negatives])
                y = np.hstack([
                    np.ones(len(positive_features)),
                    np.zeros(len(hard_negatives))
                ])

                self.svms[class_id].fit(X, y)
                print(f"    Added {len(hard_negatives)} hard negatives")

        print(f"  SVM training complete for class {class_id}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(features), self.num_classes))

        for class_id in range(self.num_classes):
            scores[:, class_id] = self.svms[class_id].decision_function(features)

        return scores

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.svms, f)
        print(f"Saved SVMs to {path}")

    def load(self, path: Path):
        with open(path, 'rb') as f:
            self.svms = pickle.load(f)
        print(f"Loaded SVMs from {path}")


def prepare_svm_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    ious: np.ndarray,
    pos_threshold: float = 1.0,
    neg_threshold: float = 0.3
) -> tuple:
    pos_mask = ious >= pos_threshold
    positive_features = features[pos_mask]
    
    neg_mask = ious < neg_threshold
    negative_features = features[neg_mask]

    return positive_features, negative_features
