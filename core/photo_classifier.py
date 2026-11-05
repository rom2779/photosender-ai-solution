from typing import Dict, Any, List

class PhotoClassifier:
    """
    Classifies the football situation based on composition, detected elements,
    and editorial metadata.
    """
    PHOTO_TYPES = {
        "GOAL_CELEBRATION": {
            "label": "Esultanza Gol",
            "action_verb": "celebrates after scoring",
            "keyword_sets": ["GOAL_CELEBRATION", "PLAYER_CELEBRATING"]
        },
        "BALL_CHALLENGE": {
            "label": "Contrasto / Azione",
            "action_verb": "in action",
            "keyword_sets": ["BALL_CHALLENGE", "PLAYER_RUNNING", "PLAYER_ACTIONS"]
        },
        "PLAYER_SHOOTING": {
            "label": "Tiro in Porta",
            "action_verb": "shoots on goal",
            "keyword_sets": ["PLAYER_SHOOTING", "SCORING_OPPORTUNITY"]
        },
        "GOALKEEPER_ACTION": {
            "label": "Azione Portiere / Parata",
            "action_verb": "makes a save",
            "keyword_sets": ["GOALKEEPER", "GOALKEEPER_SAVES"]
        },
        "PLAYER_PORTRAIT": {
            "label": "Primo Piano / Reazione",
            "action_verb": "looks on",
            "keyword_sets": ["PLAYER_EMOTIONS", "PLAYER_SITUATIONS"]
        },
        "BENCH_COACH": {
            "label": "Allenatore / Panchina",
            "action_verb": "gestures from the technical area",
            "keyword_sets": ["TEAM_AND_STAFF"]
        },
        "SUPPORTERS": {
            "label": "Tifosi / Curva",
            "action_verb": "cheer in the stands",
            "keyword_sets": ["SUPPORTERS", "STADIUM_AND_ENVIRONMENT"]
        },
        "WARM_UP": {
            "label": "Riscaldamento Pre-Gara",
            "action_verb": "during warm up before the match",
            "keyword_sets": ["PLAYER_ACTIONS", "STADIUM_AND_ENVIRONMENT"]
        }
    }

    @classmethod
    def classify(cls, vision_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Determines the most probable photo type based on visual features and context.
        """
        face_count = vision_data.get("face_count", 0)
        primary_face = vision_data.get("primary_face")
        
        # 1. Warm up check from context/time
        if context.get("is_warmup"):
            return "WARM_UP"

        # 2. Composition heuristics:
        # If single large face occupying > 25% of height -> Portrait / Reaction
        if primary_face and primary_face.get("framing") == "CLOSE_UP" and face_count <= 2:
            return "PLAYER_PORTRAIT"

        # If many faces without strong single jersey number -> Supporters or Team celebration
        if face_count >= 5:
            return "GOAL_CELEBRATION"

        # Default standard match action
        return "BALL_CHALLENGE"

    @classmethod
    def get_details(cls, photo_type_code: str) -> Dict[str, Any]:
        return cls.PHOTO_TYPES.get(photo_type_code, cls.PHOTO_TYPES["BALL_CHALLENGE"])

