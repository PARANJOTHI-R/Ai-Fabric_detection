"""
Image quality assessment and frame validation for fabric inspection.
Detects dark/standby frames, extreme lighting, and motion blur.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .config import InspectionConfig


class ImageQualityResult:
    def __init__(
        self,
        is_standby: bool,
        mean_brightness: float,
        variance: float,
        sharpness: float,
        warning: Optional[str] = None
    ):
        self.is_standby = is_standby
        self.mean_brightness = mean_brightness
        self.variance = variance
        self.sharpness = sharpness
        self.warning = warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_standby": self.is_standby,
            "mean_brightness": round(self.mean_brightness, 1),
            "variance": round(self.variance, 1),
            "sharpness": round(self.sharpness, 1),
            "warning": self.warning
        }


def assess_image_quality(
    image: np.ndarray,
    config: InspectionConfig = InspectionConfig
) -> ImageQualityResult:
    """
    Evaluates frame brightness, variance, and focus sharpness.

    Parameters
    ----------
    image : np.ndarray
        BGR image array.
    config : InspectionConfig
        Configured thresholds.

    Returns
    -------
    ImageQualityResult
    """
    if image is None or image.size == 0:
        return ImageQualityResult(
            is_standby=True,
            mean_brightness=0.0,
            variance=0.0,
            sharpness=0.0,
            warning="Corrupted or empty image buffer"
        )

    # Convert to grayscale for illumination & texture assessment
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    var_val = float(np.var(gray))

    # Fast Laplacian variance for sharpness / motion blur
    # Downscale large frames to max 640px for sub-millisecond Laplacian computation
    h, w = gray.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray

    sharpness = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())

    # Check for dark / standby feed (e.g. covered camera lens or offline frame)
    # A frame is standby if it is truly dark (mean < 15) or extremely dim with no texture (mean < 30 and var < 5) or all-zero
    is_standby = (mean_val < config.DARK_MEAN_THRESHOLD) or (mean_val < 30.0 and var_val < 5.0) or (var_val == 0.0)

    warning = None
    if is_standby:
        warning = "Feed dark / Camera standby (No fabric visible)"
    elif mean_val < config.DIM_MEAN_THRESHOLD:
        warning = "Low illumination: Fabric lighting is dim, detection accuracy may be degraded"
    elif mean_val > config.OVEREXPOSED_MEAN_THRESHOLD:
        warning = "Overexposed: Intense glare or washed out lighting detected"
    elif sharpness < config.BLUR_LAPLACIAN_THRESHOLD:
        warning = "Motion blur or low focus detected: Fabric texture may be indistinct"

    return ImageQualityResult(
        is_standby=is_standby,
        mean_brightness=mean_val,
        variance=var_val,
        sharpness=sharpness,
        warning=warning
    )
