import os
import cv2
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import face_recognition
from sqlalchemy import text
from database.connection import engine
import config

logger = logging.getLogger(__name__)

FACES_DIR = Path(config.BASE_DIR) / "data" / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)

class FaceEngine:
    """
    Football Player Face Recognition Engine.
    Uses deep neural face detection (RetinaFace) with high sensitivity for action,
    angled, profile and running shots.
    Generates 128-d face embeddings and matches against active match rosters.
    """

    @classmethod
    def get_embedding_path(cls, player_id: int) -> Path:
        return FACES_DIR / f"{player_id}.npy"

    @classmethod
    def save_player_face_embedding(cls, player_id: int, embedding: np.ndarray, update_db_flag: bool = True):
        """Saves or averages a player face embedding and marks has_face=1 in DB."""
        path = cls.get_embedding_path(player_id)
        if path.exists():
            try:
                existing = np.load(path)
                embedding = (existing * 0.6) + (embedding * 0.4)
                embedding = embedding / np.linalg.norm(embedding)
            except Exception as e:
                logger.warning(f"Error loading existing embedding for player {player_id}: {e}")

        np.save(path, embedding)

        if update_db_flag:
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("UPDATE roster_player SET has_face = 1 WHERE id = :id"),
                        {"id": player_id}
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Error updating has_face flag in DB for player {player_id}: {e}")

    @classmethod
    def load_player_face_embedding(cls, player_id: int) -> Optional[np.ndarray]:
        path = cls.get_embedding_path(player_id)
        if path.exists():
            try:
                return np.load(path)
            except Exception as e:
                logger.error(f"Failed to read face embedding for player {player_id}: {e}")
        return None

    @classmethod
    def detect_and_encode_faces(cls, image_path: str) -> List[Dict[str, Any]]:
        """
        Detects faces using RetinaFace (deep neural net) with fallback to dlib HOG.
        Extracts 128-d embeddings and creates a base64 thumbnail of the detected face.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Scale down for fast RetinaFace inference on high-res photos
        max_dim = 1400
        scale = 1.0
        if max(img_h, img_w) > max_dim:
            scale = max_dim / float(max(img_h, img_w))
            small_rgb = cv2.resize(img_rgb, (int(img_w * scale), int(img_h * scale)))
        else:
            small_rgb = img_rgb

        detected_boxes = []

        # 1. Primary Deep Detector: RetinaFace
        try:
            from retinaface import RetinaFace
            raw_faces = RetinaFace.detect_faces(small_rgb)
            if isinstance(raw_faces, dict):
                for k, f in raw_faces.items():
                    score = float(f.get("score", 0.0))
                    if score >= 0.65:
                        sx1, sy1, sx2, sy2 = f["facial_area"]
                        ox1 = max(0, int(sx1 / scale))
                        oy1 = max(0, int(sy1 / scale))
                        ox2 = min(img_w, int(sx2 / scale))
                        oy2 = min(img_h, int(sy2 / scale))
                        if ox2 > ox1 and oy2 > oy1:
                            detected_boxes.append((ox1, oy1, ox2 - ox1, oy2 - oy1, score))
        except Exception as e:
            logger.warning(f"RetinaFace detection error: {e}")

        # 2. Fallback: dlib HOG with upsample if RetinaFace found nothing
        if not detected_boxes:
            try:
                hog_faces = face_recognition.face_locations(small_rgb, number_of_times_to_upsample=1, model="hog")
                for (top, right, bottom, left) in hog_faces:
                    ox1 = max(0, int(left / scale))
                    oy1 = max(0, int(top / scale))
                    ox2 = min(img_w, int(right / scale))
                    oy2 = min(img_h, int(bottom / scale))
                    detected_boxes.append((ox1, oy1, ox2 - ox1, oy2 - oy1, 0.85))
            except Exception as e:
                logger.warning(f"HOG fallback error: {e}")

        if not detected_boxes:
            return []

        # Sort largest face first
        detected_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        results = []
        for (ox, oy, ow, oh, score) in detected_boxes:
            # Expand slightly around face for crop thumbnail
            pad_x = int(ow * 0.15)
            pad_y = int(oh * 0.15)
            cx1 = max(0, ox - pad_x)
            cy1 = max(0, oy - pad_y)
            cx2 = min(img_w, ox + ow + pad_x)
            cy2 = min(img_h, oy + oh + pad_y)

            face_crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
            thumb_b64 = None
            if face_crop_bgr.size > 0:
                # Resize thumbnail to 120x120
                thumb_resized = cv2.resize(face_crop_bgr, (120, 120))
                ret, buf = cv2.imencode(".jpg", thumb_resized)
                if ret:
                    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            # Extract 128-d face embedding
            loc = [(oy, ox + ow, oy + oh, ox)]
            encs = face_recognition.face_encodings(img_rgb, known_face_locations=loc)
            enc = encs[0] if encs else None

            # Calculate sharpness
            gray_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY) if face_crop_bgr.size > 0 else None
            sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()) if gray_crop is not None else 0.0

            rel_h = oh / float(img_h)
            framing = "CLOSE_UP" if rel_h > 0.18 else ("MEDIUM_SHOT" if rel_h > 0.05 else "WIDE_SHOT")

            results.append({
                "box": [ox, oy, ow, oh],
                "framing": framing,
                "sharpness": round(sharpness, 1),
                "embedding": enc,
                "score": round(score, 2),
                "area": ow * oh,
                "thumbnail_b64": thumb_b64
            })

        return results

    @classmethod
    def match_face_against_roster(
        cls,
        face_embedding: np.ndarray,
        team_players: List[Dict[str, Any]],
        distance_threshold: float = 0.58
    ) -> List[Dict[str, Any]]:
        """
        Compares face_embedding with saved embeddings of players in the match roster.
        Returns sorted list of matches with confidence score.
        """
        if face_embedding is None:
            return []

        matches = []
        for p in team_players:
            p_id = p["id"]
            known_enc = cls.load_player_face_embedding(p_id)
            if known_enc is not None:
                dist = float(np.linalg.norm(face_embedding - known_enc))
                confidence = max(0.0, min(100.0, (1.0 - (dist / 0.8)) * 100.0))
                matches.append({
                    "player": p,
                    "distance": round(dist, 3),
                    "confidence": round(confidence, 1),
                    "is_match": dist <= distance_threshold
                })

        matches.sort(key=lambda m: m["distance"])
        return matches
