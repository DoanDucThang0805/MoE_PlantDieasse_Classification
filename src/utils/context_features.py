import cv2
import numpy as np


def extract_context_features(image_np: np.ndarray) -> np.ndarray:
    """
    Extract simple context features from an RGB image.

    Features (6 dims):
        1. Mean brightness
        2. Brightness std
        3. Blur score (Laplacian variance proxy)
        4. Edge density
        5. Saturation mean
        6. Green ratio
    """

    img = image_np.astype(np.uint8)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # brightness statistics
    brightness_mean = gray.mean() / 255.0
    brightness_std = gray.std() / 255.0

    # blur proxy (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    blur_score = np.tanh(lap_var / 1000.0)

    # edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0

    # saturation proxy
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat_mean = hsv[..., 1].mean() / 255.0

    # green ratio
    green_ratio = (img[..., 1].astype(np.float32) + 1.0) / (
        img[..., 0].astype(np.float32) + img[..., 2].astype(np.float32) + 2.0
    )
    green_ratio = np.clip(green_ratio.mean() / 2.0, 0.0, 1.0)

    features = np.array(
        [
            brightness_mean,
            brightness_std,
            blur_score,
            edge_density,
            sat_mean,
            green_ratio,
        ],
        dtype=np.float32,
    )

    return features
