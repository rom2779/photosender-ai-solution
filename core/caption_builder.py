from datetime import datetime
from typing import Dict, Any, Optional

class CaptionBuilder:
    @staticmethod
    def build_caption(
        player_name: str,
        team_name: str,
        action_verb: str,
        competition_name: str,
        home_team: str,
        away_team: str,
        venue: str,
        city: str,
        country: str,
        event_date: Optional[datetime],
        photographer_credit: str = "Marco Canoniero",
        agency: str = "Sportphoto24"
    ) -> str:
        """
        Builds international agency editorial standard caption.
        Example:
        'Rafael Leao of AC Milan celebrates after scoring during the Serie A match between AC Milan and FC Internazionale at San Siro on October 31, 2026 in Milan, Italy. (Photo Marco Canoniero/Sportphoto24)'
        """
        if event_date is None:
            event_date = datetime.now()

        date_str = event_date.strftime("%B %d, %Y")

        subject_part = f"{player_name} of {team_name}" if (player_name and team_name) else (player_name or team_name or "Players")
        action_part = action_verb if action_verb else "in action"
        
        match_part = f"the {competition_name} match between {home_team} and {away_team}" if (home_team and away_team) else f"the {competition_name} event"
        location_part = f"at {venue}" if venue else ""
        if city and country:
            geo_part = f"in {city}, {country}"
        elif city or country:
            geo_part = f"in {city or country}"
        else:
            geo_part = ""

        # Assemble main sentence
        elements = [subject_part, action_part, "during", match_part]
        if location_part:
            elements.append(location_part)
        elements.append(f"on {date_str}")
        if geo_part:
            elements.append(geo_part)

        main_text = " ".join(elements) + "."

        # Photographer credit
        if agency and agency not in photographer_credit:
            credit_sig = f"(Photo {photographer_credit}/{agency})"
        else:
            credit_sig = f"(Photo {photographer_credit})"

        return f"{main_text} {credit_sig}"

