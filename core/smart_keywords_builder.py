from typing import List, Dict, Any, Optional
from database.repositories import SmartKeywordRepository

class SmartKeywordsBuilder:
    @staticmethod
    def generate_keywords(
        player_name: Optional[str],
        team_name: Optional[str],
        opponent_team: Optional[str],
        competition: Optional[str],
        venue: Optional[str],
        city: Optional[str],
        country: Optional[str],
        photo_type_code: str,
        orientation: str = "Horizontal",
        extra_terms: Optional[List[str]] = None
    ) -> List[str]:
        """
        Builds IMAGO and agency compliant keywords list directly from database sets.
        """
        keywords: List[str] = []

        def add_kw(val: Optional[str]):
            if val:
                clean = val.strip()
                if clean and clean.lower() not in [k.lower() for k in keywords]:
                    keywords.append(clean)

        # 1. Base sports & editorial mandatory terms
        base_editorial = [
            "Football", "Soccer", "Sport", "Match", "Action",
            orientation
        ]
        for b in base_editorial:
            add_kw(b)

        # 2. Main entities
        add_kw(player_name)
        add_kw(team_name)
        add_kw(opponent_team)
        add_kw(competition)
        add_kw(venue)
        add_kw(city)
        add_kw(country)

        # 3. Dynamic terms from PhotoSender DB Keyword Sets
        # Map photo_type to set codes
        type_mapping = {
            "GOAL_CELEBRATION": ["GOAL_CELEBRATION", "PLAYER_CELEBRATING", "TEAM_CELEBRATION"],
            "BALL_CHALLENGE": ["BALL_CHALLENGE", "PLAYER_RUNNING", "PLAYER_ACTIONS"],
            "PLAYER_SHOOTING": ["PLAYER_SHOOTING", "SCORING_OPPORTUNITY"],
            "GOALKEEPER_ACTION": ["GOALKEEPER", "GOALKEEPER_SAVES"],
            "PLAYER_PORTRAIT": ["PLAYER_EMOTIONS", "PLAYER_SITUATIONS"],
            "BENCH_COACH": ["TEAM_AND_STAFF"],
            "SUPPORTERS": ["SUPPORTERS", "STADIUM_AND_ENVIRONMENT"],
            "WARM_UP": ["PLAYER_ACTIONS", "STADIUM_AND_ENVIRONMENT"]
        }

        set_codes = type_mapping.get(photo_type_code, ["PLAYER_ACTIONS"])
        try:
            db_terms = SmartKeywordRepository.get_terms_for_sets_by_codes(set_codes)
            for t in db_terms:
                add_kw(t)
        except Exception as e:
            # Fallback if DB disconnected
            pass

        # 4. Extra custom terms if passed
        if extra_terms:
            for et in extra_terms:
                add_kw(et)

        return keywords

