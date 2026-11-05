import os
import cv2
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
    Detects faces, generates 128-dimensional biometric embeddings,
    and matches against the players of the active match roster.
    Also allows one-click auto-learning from approved match photos.
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
                # Weighted average: refine the embedding with new photo
                embedding = (existing * 0.7) + (embedding * 0.3)
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
        Detects faces in image and extracts 128-d embeddings for each detected face.
        Returns list with box, embedding, sharpness and crop preview.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}")

        img_h, img_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Scale down for ultra-fast face search on 24MP match photos
        max_dim = 1400
        scale = 1.0
        if max(img_h, img_w) > max_dim:
            scale = max_dim / float(max(img_h, img_w))
            small_rgb = cv2.resize(img_rgb, (int(img_w * scale), int(img_h * scale)))
        else:
            small_rgb = img_rgb

        # Detect face locations [(top, right, bottom, left)]
        raw_boxes = face_recognition.face_locations(small_rgb, model="hog")
        if not raw_boxes:
            return []

        # Extract embeddings
        embeddings = face_recognition.face_encodings(small_rgb, known_face_locations=raw_boxes)

        detected = []
        for box, enc in zip(raw_boxes, embeddings):
            top, right, bottom, left = box
            # Rescale back to original coords
            oy = int(top / scale)
            ox = int(left / scale)
            ow = int((right - left) / scale)
            oh = int((bottom - top) / scale)

            # Crop face for sharpness score
            crop = img_rgb[oy:oy+oh, ox:ox+ow]
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()) if crop.size > 0 else 0.0

            # Framing estimation
            rel_h = oh / float(img_h)
            framing = "CLOSE_UP" if rel_h > 0.20 else ("MEDIUM_SHOT" if rel_h > 0.06 else "WIDE_SHOT")

            detected.append({
                "box": [ox, oy, ow, oh],
                "framing": framing,
                "sharpness": round(sharpness, 1),
                "embedding": enc,
                "area": ow * oh
            })

        # Sort largest face first (primary subject)
        detected.sort(key=lambda f: f["area"], reverse=True)
        return detected

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
        matches = []
        for p in team_players:
            p_id = p["id"]
            known_enc = cls.load_player_face_embedding(p_id)
            if known_enc is not None:
                # Euclidean distance: 0 = identical, 0.6 = threshold for dlib
                dist = float(np.linalg.norm(face_embedding - known_enc))
                # Convert distance to confidence %
                confidence = max(0.0, min(100.0, (1.0 - (dist / 0.8)) * 100.0))
                matches.append({
                    "player": p,
                    "distance": round(dist, 3),
                    "confidence": round(confidence, 1),
                    "is_match": dist <= distance_threshold
                })

        # Sort by best distance (closest match first)
        matches.sort(key=lambda m: m["distance"])
        return matches
