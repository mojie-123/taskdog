"""YOLOE text-prompt detection followed by SAM2 mask refinement."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
from typing import Iterable

import cv2
import numpy as np

from .config import PerceptionConfig
from .types import Detection2D


_TEXT_ENCODER_CWD_LOCK = threading.Lock()


@contextmanager
def _working_directory(path: Path):
    """Temporarily expose MobileCLIP under the filename Ultralytics expects."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class YOLOESAM2Perceptor:
    """Lazy GPU perception backend with replaceable task prompts.

    YOLOE supplies class labels and boxes.  SAM2 receives each box as a prompt
    and returns the object mask; no MuJoCo body/geom segmentation is consulted.
    """

    def __init__(self, config: PerceptionConfig, *, refine_with_sam2: bool = True):
        self.config = config
        self.refine_with_sam2 = bool(refine_with_sam2)
        self._yolo = None
        self._sam_predictor = None
        self._prompts: tuple[str, ...] = ()

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLOE

            if not self.config.yoloe_weights.is_file():
                raise FileNotFoundError(self.config.yoloe_weights)
            self._yolo = YOLOE(str(self.config.yoloe_weights))
        return self._yolo

    def _load_sam(self):
        if not self.refine_with_sam2:
            return None
        if self._sam_predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if not self.config.sam2_checkpoint.is_file():
                raise FileNotFoundError(self.config.sam2_checkpoint)
            model = build_sam2(
                self.config.sam2_model_config,
                str(self.config.sam2_checkpoint),
                device=self.config.device,
            )
            self._sam_predictor = SAM2ImagePredictor(model)
        return self._sam_predictor

    def set_prompts(self, prompts: Iterable[str]) -> tuple[str, ...]:
        """Encode a deduplicated task vocabulary once, then reuse it per frame."""

        normalised: list[str] = []
        for value in prompts:
            label = " ".join(str(value).strip().lower().split())
            if label and label not in normalised:
                normalised.append(label)
        if not normalised:
            raise ValueError("at least one non-empty perception prompt is required")
        requested = tuple(normalised)
        if requested == self._prompts:
            return requested
        if not self.config.mobileclip_weights.is_file():
            raise FileNotFoundError(self.config.mobileclip_weights)
        if self.config.mobileclip_weights.name != "mobileclip2_b.ts":
            raise ValueError("Ultralytics YOLOE expects MobileCLIP file mobileclip2_b.ts")
        model = self._load_yolo()
        # Ultralytics resolves the fixed text-encoder filename from cwd.  Keep
        # this tiny global section locked; ordinary frame inference is outside.
        with _TEXT_ENCODER_CWD_LOCK:
            with _working_directory(self.config.mobileclip_weights.parent):
                model.set_classes(list(requested))
        self._prompts = requested
        return requested

    def detect(self, image_rgb: np.ndarray) -> list[Detection2D]:
        """Return prompt-labelled masks sorted by detector confidence."""

        rgb = np.asarray(image_rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be uint8 with shape (H, W, 3)")
        if not rgb.flags.writeable or not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb).copy()
        if not self._prompts:
            raise RuntimeError("call set_prompts() before detect()")
        model = self._load_yolo()
        # Ultralytics treats ndarray sources as OpenCV BGR.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = model.predict(
            source=bgr,
            device=self.config.device,
            conf=self.config.confidence,
            iou=self.config.iou,
            max_det=self.config.max_detections,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        masks: list[np.ndarray]
        predictor = self._load_sam()
        if predictor is not None:
            predictor.set_image(rgb)
            masks = []
            for box in boxes:
                candidates, scores, _ = predictor.predict(
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                )
                masks.append(np.asarray(candidates[int(np.argmax(scores))], dtype=bool))
        elif result.masks is not None:
            masks = [
                np.asarray(mask.detach().cpu().numpy() > 0.5, dtype=bool)
                for mask in result.masks.data
            ]
        else:
            masks = []
            height, width = rgb.shape[:2]
            for x1, y1, x2, y2 in boxes:
                mask = np.zeros((height, width), dtype=bool)
                mask[
                    max(0, int(np.floor(y1))) : min(height, int(np.ceil(y2))),
                    max(0, int(np.floor(x1))) : min(width, int(np.ceil(x2))),
                ] = True
                masks.append(mask)

        detections: list[Detection2D] = []
        for box, confidence, class_id, mask in zip(
            boxes, confidences, class_ids, masks, strict=True
        ):
            if int(mask.sum()) < self.config.min_mask_pixels:
                continue
            label = str(result.names[int(class_id)])
            detections.append(
                Detection2D(
                    label=label,
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in box),
                    mask=mask,
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def close(self) -> None:
        self._sam_predictor = None
        self._yolo = None
        self._prompts = ()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def annotate_detection(
    image_rgb: np.ndarray,
    detection: Detection2D,
    *,
    show_label: bool = True,
) -> np.ndarray:
    """Draw the candidate contour and label used for VLM verification/debugging."""

    output = np.asarray(image_rgb).copy()
    contours, _ = cv2.findContours(
        detection.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    if show_label:
        x1, y1, _, _ = detection.bbox_xyxy
        cv2.putText(
            output,
            f"{detection.label} {detection.confidence:.2f}",
            (max(0, int(x1)), max(18, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output
