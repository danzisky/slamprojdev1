"""Chair detection and observation extraction utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import ChairObservation

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from rfdetr import RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES
except ImportError:  # pragma: no cover - optional dependency
    RFDETRBase = None
    COCO_CLASSES = None


class ChairObservationDetector:
    """
    Convert a camera frame into chair range-bearing observations.

    The detector backend is optional. If RF-DETR is unavailable, callers can still
    provide their own detection dictionaries and use observations_from_detections().
    """

    CHAIR_CLASS_IDS = (56, 62, 63)

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_size: Optional[Tuple[int, int]] = None,
        depth_model=None,
        depth_scale: float = 1.0,
        detection_threshold: float = 0.35,
        chair_height_m: float = 0.9,
        default_range_std_m: float = 0.35,
        default_bearing_std_rad: float = math.radians(4.0),
        load_default_detector: bool = True,
        device: Optional[str] = None,
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.image_size = image_size
        self.depth_model = depth_model
        self.depth_scale = float(depth_scale)
        self.detection_threshold = float(detection_threshold)
        self.chair_height_m = float(chair_height_m)
        self.default_range_std_m = float(default_range_std_m)
        self.default_bearing_std_rad = float(default_bearing_std_rad)
        self.device = device or self._choose_device()

        self.detector = None
        if load_default_detector and RFDETRBase is not None:
            try:
                self.detector = RFDETRBase(device=self.device)
                # self.detector.optimize_for_inference()
            except Exception as exc:  # pragma: no cover - external dependency behavior
                print(f"Warning: failed to load RF-DETR detector: {exc}")
                self.detector = None

    @staticmethod
    def _choose_device() -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def set_camera_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the intrinsics used for bearing estimation."""
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        if image_size is not None:
            self.image_size = image_size

    def horizontal_fov_rad(self, image_width: Optional[int] = None) -> float:
        """Return the horizontal field of view implied by the intrinsics."""
        width = image_width or (self.image_size[0] if self.image_size is not None else None)
        if width is None:
            raise ValueError("Image width is required to compute the camera FOV.")
        return float(2.0 * math.atan(width / (2.0 * self.fx)))

    def detect(self, image: np.ndarray) -> List[Dict[str, float]]:
        """Detect chair bounding boxes in a BGR image."""
        if self.detector is None:
            return []

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.predict(image_rgb, threshold=self.detection_threshold)
        if results is None or len(results.class_id) == 0:
            return []

        detections: List[Dict[str, float]] = []
        for index in range(len(results.class_id)):
            class_id = int(results.class_id[index])
            if class_id not in self.CHAIR_CLASS_IDS:
                continue

            x1, y1, x2, y2 = results.xyxy[index]
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(results.confidence[index]),
                    "class_name": COCO_CLASSES[class_id] if COCO_CLASSES is not None else "chair",
                }
            )

        return detections

    def infer_depth_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Run the configured depth model if one is available."""
        if self.depth_model is None:
            return None

        if torch is None:
            depth_map = self.depth_model.infer_image(image)
        else:
            with torch.no_grad():
                use_autocast = self.device == "cuda"
                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=torch.float16):
                    depth_map = self.depth_model.infer_image(image)

        return np.asarray(depth_map, dtype=np.float32) * self.depth_scale

    def observations_from_image(self, image: np.ndarray) -> Tuple[ChairObservation, ...]:
        """Run the detector backend and convert detections into observations."""
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        detections = self.detect(image)
        depth_map = self.infer_depth_map(image)
        return self.observations_from_detections(detections, image=image, depth_map=depth_map)

    def observations_from_detections(
        self,
        detections: Sequence[Dict[str, float]],
        image: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[ChairObservation, ...]:
        """
        Convert bounding boxes to range-bearing observations.

        Each detection must expose a bbox in (x1, y1, x2, y2) format and can
        optionally include a confidence value.
        """
        if depth_map is None and image is not None and self.depth_model is not None:
            depth_map = self.infer_depth_map(image)

        observations: List[ChairObservation] = []
        for detection in detections:
            bbox = tuple(float(value) for value in detection["bbox"])
            confidence = float(detection.get("confidence", 1.0))

            range_m = None
            source = "geometry"
            if depth_map is not None:
                range_m = self._estimate_depth_from_bbox(depth_map, bbox)
                source = "depth"

            if range_m is None:
                range_m = self._estimate_range_from_bbox_geometry(bbox)

            if range_m is None or not np.isfinite(range_m) or range_m <= 0.0:
                continue

            x1, _, x2, _ = bbox
            center_x = 0.5 * (x1 + x2)
            bearing_rad = math.atan((center_x - self.cx) / self.fx)

            range_std_m = self.default_range_std_m if source == "depth" else max(self.default_range_std_m * 1.6, 0.6)
            range_std_m *= 1.0 + max(0.0, 1.0 - confidence)
            bearing_std_rad = self.default_bearing_std_rad * (1.0 + 0.5 * max(0.0, 1.0 - confidence))

            observations.append(
                ChairObservation(
                    range_m=float(range_m),
                    bearing_rad=float(bearing_rad),
                    confidence=confidence,
                    bbox=bbox,
                    source=source,
                    range_std_m=float(range_std_m),
                    bearing_std_rad=float(bearing_std_rad),
                )
            )

        observations.sort(key=lambda item: item.bearing_rad)
        return tuple(observations)

    def _estimate_depth_from_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[float]:
        """
        Sample the lower-central part of the box.

        That region tends to be more stable than the geometric center when the chair
        backrest spans a wide depth range.
        """
        x1, y1, x2, y2 = bbox
        height, width = depth_map.shape[:2]

        left = max(0, int(round(x1 + 0.30 * (x2 - x1))))
        right = min(width, int(round(x1 + 0.70 * (x2 - x1))))
        top = max(0, int(round(y1 + 0.45 * (y2 - y1))))
        bottom = min(height, int(round(y1 + 0.95 * (y2 - y1))))
        if right <= left or bottom <= top:
            return None

        region = depth_map[top:bottom, left:right]
        valid_values = region[np.isfinite(region) & (region > 0.0)]
        if valid_values.size == 0:
            return None

        # Bias toward the front-most quartile to avoid averaging in distant background.
        return float(np.percentile(valid_values, 30.0))

    def _estimate_range_from_bbox_geometry(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[float]:
        """Fallback depth estimate using a nominal chair height."""
        _, y1, _, y2 = bbox
        pixel_height = max(0.0, y2 - y1)
        if pixel_height < 8.0:
            return None
        return float((self.chair_height_m * self.fy) / pixel_height)