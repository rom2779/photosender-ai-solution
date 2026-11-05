from typing import List, Dict, Optional, Any
from sqlalchemy import text
from database.connection import engine

class RosterRepository:
    @staticmethod
    def get_matchups(limit: int = 20) -> List[Dict[str, Any]]:
        query = text("""
            SELECT m.id, m.match_date, m.status,
                   ht.id as home_team_id, ht.name as home_team_name, ht.logo_url as home_logo,
                   at.id as away_team_id, at.name as away_team_name, at.logo_url as away_logo
            FROM roster_matchup m
            LEFT JOIN roster_team ht ON m.home_team_id = ht.id
            LEFT JOIN roster_team at ON m.away_team_id = at.id
            ORDER BY m.match_date DESC
            LIMIT :limit
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {"limit": limit}).fetchall()
            return [dict(r._mapping) for r in rows]

    @staticmethod
    def get_players_by_team(team_id: int) -> List[Dict[str, Any]]:
        query = text("""
            SELECT id, team_id, number, first_name, last_name, position, has_face
            FROM roster_player
            WHERE team_id = :team_id
            ORDER BY CAST(number AS UNSIGNED) ASC, last_name ASC
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {"team_id": team_id}).fetchall()
            return [dict(r._mapping) for r in rows]

    @staticmethod
    def find_player_by_number_and_teams(number: str, team_ids: List[int]) -> List[Dict[str, Any]]:
        valid_ids = [int(t) for t in team_ids if t is not None]
        if not valid_ids:
            return []
        ids_str = ",".join(str(i) for i in valid_ids)
        sql = f"""
            SELECT p.id, p.team_id, p.number, p.first_name, p.last_name, p.position,
                   t.name as team_name
            FROM roster_player p
            LEFT JOIN roster_team t ON p.team_id = t.id
            WHERE p.number = :number AND p.team_id IN ({ids_str})
        """
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"number": str(number).strip()}).fetchall()
            return [dict(r._mapping) for r in rows]

    @staticmethod
    def search_players_by_name(query_str: str, team_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        valid_ids = [int(t) for t in team_ids if t is not None] if team_ids else []
        sql = """
            SELECT p.id, p.team_id, p.number, p.first_name, p.last_name, p.position,
                   t.name as team_name
            FROM roster_player p
            LEFT JOIN roster_team t ON p.team_id = t.id
            WHERE (LOWER(p.last_name) LIKE LOWER(:q) OR LOWER(p.first_name) LIKE LOWER(:q))
        """
        params: Dict[str, Any] = {"q": f"%{query_str.strip()}%"}
        if valid_ids:
            ids_str = ",".join(str(i) for i in valid_ids)
            sql += f" AND p.team_id IN ({ids_str})"
        sql += " LIMIT 15"
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            return [dict(r._mapping) for r in rows]


class SmartKeywordRepository:
    @staticmethod
    def get_imago_profile() -> Optional[Dict[str, Any]]:
        query = text("""
            SELECT id, agency_code, agency_name, keyword_limit, active
            FROM agency_keyword_profiles
            WHERE agency_code = 'IMAGO' AND active = 1
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = conn.execute(query).fetchone()
            return dict(row._mapping) if row else None

    @staticmethod
    def get_keyword_sets_by_category(category: str, sport: str = "FOOTBALL") -> List[Dict[str, Any]]:
        query = text("""
            SELECT id, code, sport, category, name, description
            FROM keyword_sets
            WHERE category = :cat AND sport = :sport AND active = 1
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {"cat": category, "sport": sport}).fetchall()
            return [dict(r._mapping) for r in rows]

    @staticmethod
    def get_terms_for_set(set_id: int) -> List[str]:
        query = text("""
            SELECT keyword
            FROM keyword_set_terms
            WHERE keyword_set_id = :set_id AND active = 1
            ORDER BY priority DESC, keyword ASC
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, {"set_id": set_id}).fetchall()
            return [r[0] for r in rows]

    @staticmethod
    def get_terms_for_sets_by_codes(set_codes: List[str]) -> List[str]:
        if not set_codes:
            return []
        quoted = ",".join(f"'{c.strip()}'" for c in set_codes if c.strip())
        if not quoted:
            return []
        sql = f"""
            SELECT DISTINCT t.keyword
            FROM keyword_set_terms t
            JOIN keyword_sets s ON t.keyword_set_id = s.id
            WHERE s.code IN ({quoted}) AND t.active = 1 AND s.active = 1
            ORDER BY t.priority DESC
        """
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
            return [r[0] for r in rows]
