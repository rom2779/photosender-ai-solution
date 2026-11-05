import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any
import logging
import config

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self):
        self._yolo_model = None
        self._ocr_reader = None

    def _get_yolo(self):
        if self._yolo_model is None:
            from ultralytics import YOLO
            logger.info("Initializing YOLOv8-nano model for person/jersey detection...")
            self._yolo_model = YOLO("yolov8n.pt")
        return self._yolo_model

    def _get_ocr(self):
        if self._ocr_reader is None:
            import easyocr
            logger.info("Initializing EasyOCR reader (Apple Silicon / MPS)...")
            use_gpu = config.DEVICE.type in ("cuda", "mps")
            self._ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        return self._ocr_reader

    def calculate_sharpness(self, image_np: np.ndarray) -> float:
        """Measures focus sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def detect_faces(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces using face_recognition (dlib) on a scaled preview for high speed.
        """
        import face_recognition

        h, w = image_np.shape[:2]
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Scale down for fast processing on high-res photos
        max_dim = 1200
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            small_rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        else:
            small_rgb = rgb

        # Detect face locations [(top, right, bottom, left)]
        raw_faces = face_recognition.face_locations(small_rgb, model="hog")

        results = []
        for (top, right, bottom, left) in raw_faces:
            # Rescale back to original coords
            oy = int(top / scale)
            ox = int(left / scale)
            ow = int((right - left) / scale)
            oh = int((bottom - top) / scale)

            face_crop = rgb[oy:oy+oh, ox:ox+ow]
            sharpness = self.calculate_sharpness(face_crop) if face_crop.size > 0 else 0.0

            relative_height = oh / float(h)
            framing = "CLOSE_UP" if relative_height > 0.22 else ("MEDIUM_SHOT" if relative_height > 0.07 else "WIDE_SHOT")

            results.append({
                "box": [ox, oy, ow, oh],
                "sharpness": round(sharpness, 1),
                "framing": framing
            })

        # Sort largest face first
        results.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        return results

    def detect_persons_and_jerseys(self, image_path: str) -> Dict[str, Any]:
        """
        Runs person detection and extracts potential jersey numbers.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image at {image_path}")

        img_h, img_w = img.shape[:2]
        faces = self.detect_faces(img)

        yolo = self._get_yolo()
        # Class 0 in COCO is person
        pred = yolo.predict(source=img, classes=[0], conf=0.35, verbose=False, device=config.DEVICE)

        detected_persons = []
        candidate_numbers = []

        if pred and len(pred) > 0 and pred[0].boxes is not None:
            boxes = pred[0].boxes.xyxy.cpu().numpy()
            confs = pred[0].boxes.conf.cpu().numpy()

            person_indices = sorted(range(len(boxes)), key=lambda i: (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1]), reverse=True)

            for idx in person_indices[:3]:
                x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                conf = float(confs[idx])
                pw, ph = x2 - x1, y2 - y1

                # Torso crop
                torso_y1 = max(0, y1 + int(ph * 0.15))
                torso_y2 = min(img_h, y1 + int(ph * 0.65))
                torso_x1 = max(0, x1)
                torso_x2 = min(img_w, x2)

                detected_persons.append({
                    "box": [x1, y1, pw, ph],
                    "conf": round(conf, 2)
                })

                if torso_y2 > torso_y1 and torso_x2 > torso_x1:
                    torso_crop = img[torso_y1:torso_y2, torso_x1:torso_x2]
                    numbers = self._ocr_jersey_numbers(torso_crop)
                    for num, ocr_conf in numbers:
                        candidate_numbers.append({
                            "number": num,
                            "confidence": round(ocr_conf, 2),
                            "person_index": len(detected_persons) - 1
                        })

        return {
            "image_size": [img_w, img_h],
            "overall_sharpness": round(self.calculate_sharpness(img), 1),
            "face_count": len(faces),
            "primary_face": faces[0] if faces else None,
            "detected_persons_count": len(detected_persons),
            "candidate_jersey_numbers": candidate_numbers
        }

    def _ocr_jersey_numbers(self, crop_bgr: np.ndarray) -> List[Tuple[str, float]]:
        """Extracts 1 or 2 digit numbers from shirt crop using EasyOCR."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
            return []

        ocr = self._get_ocr()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        results = ocr.readtext(rgb, allowlist="0123456789", paragraph=False)
        valid_numbers = []
        for bbox, text, conf in results:
            clean = text.strip()
            if clean.isdigit() and 1 <= int(clean) <= 99:
                valid_numbers.append((clean, float(conf)))

        return valid_numbers
