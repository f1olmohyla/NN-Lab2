"""
Bounding Box Regression
Appendix C: "We use a simple bounding-box regression stage to improve localization"
"""

import torch
import numpy as np
from typing import Tuple


class BoundingBoxRegressor:

    def __init__(self, feature_dim: int = 4096, lambda_reg: float = 1000.0):

        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg

        # Learned weights for each transformation
        self.w_x: np.ndarray = None
        self.w_y: np.ndarray = None
        self.w_w: np.ndarray = None
        self.w_h: np.ndarray = None

    def compute_targets(
        self,
        proposals: np.ndarray,
        ground_truth: np.ndarray
    ) -> np.ndarray:

        # Convert to center coordinates and width/height
        P_w = proposals[:, 2] - proposals[:, 0]
        P_h = proposals[:, 3] - proposals[:, 1]
        P_x = proposals[:, 0] + 0.5 * P_w
        P_y = proposals[:, 1] + 0.5 * P_h

        G_w = ground_truth[:, 2] - ground_truth[:, 0]
        G_h = ground_truth[:, 3] - ground_truth[:, 1]
        G_x = ground_truth[:, 0] + 0.5 * G_w
        G_y = ground_truth[:, 1] + 0.5 * G_h

        # Compute targets
        t_x = (G_x - P_x) / P_w
        t_y = (G_y - P_y) / P_h
        t_w = np.log(G_w / P_w)
        t_h = np.log(G_h / P_h)

        targets = np.stack([t_x, t_y, t_w, t_h], axis=1)
        return targets

    def train(
        self,
        features: np.ndarray,
        proposals: np.ndarray,
        ground_truth: np.ndarray
    ):

        targets = self.compute_targets(proposals, ground_truth)

        X = features
        lambda_I = self.lambda_reg * np.eye(self.feature_dim)

        XtX = X.T @ X + lambda_I

        self.w_x = np.linalg.solve(XtX, X.T @ targets[:, 0])
        self.w_y = np.linalg.solve(XtX, X.T @ targets[:, 1])
        self.w_w = np.linalg.solve(XtX, X.T @ targets[:, 2])
        self.w_h = np.linalg.solve(XtX, X.T @ targets[:, 3])

        print("Bounding box regressor trained")

    def predict(
        self,
        features: np.ndarray,
        proposals: np.ndarray
    ) -> np.ndarray:

        # Compute transformations
        d_x = features @ self.w_x
        d_y = features @ self.w_y
        d_w = features @ self.w_w
        d_h = features @ self.w_h

        # Convert proposals to center coordinates
        P_w = proposals[:, 2] - proposals[:, 0]
        P_h = proposals[:, 3] - proposals[:, 1]
        P_x = proposals[:, 0] + 0.5 * P_w
        P_y = proposals[:, 1] + 0.5 * P_h

        # Apply transformations
        G_x = P_w * d_x + P_x
        G_y = P_h * d_y + P_y
        G_w = P_w * np.exp(d_w)
        G_h = P_h * np.exp(d_h)

        # Convert back to corner coordinates
        x1 = G_x - 0.5 * G_w
        y1 = G_y - 0.5 * G_h
        x2 = G_x + 0.5 * G_w
        y2 = G_y + 0.5 * G_h

        refined_boxes = np.stack([x1, y1, x2, y2], axis=1)
        return refined_boxes

    def save(self, path):
        np.savez(
            path,
            w_x=self.w_x,
            w_y=self.w_y,
            w_w=self.w_w,
            w_h=self.w_h
        )
        print(f"Saved bbox regressor to {path}")

    def load(self, path):
        data = np.load(path)
        self.w_x = data['w_x']
        self.w_y = data['w_y']
        self.w_w = data['w_w']
        self.w_h = data['w_h']
        print(f"Loaded bbox regressor from {path}")
