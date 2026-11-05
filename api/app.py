from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import shutil
import os
import tempfile
from pathlib import Path

import config
from database.repositories import RosterRepository, SmartKeywordRepository
from core.vision_engine import VisionEngine
from core.photo_classifier import PhotoClassifier
from core.caption_builder import CaptionBuilder
from core.smart_keywords_builder import SmartKeywordsBuilder
from core.metadata_writer import MetadataWriter

app = FastAPI(
    title="PhotoSender Smart AI Engine",
    description="Local AI Engine for Match Photo Analysis, SmartKeywords IMAGO and IPTC metadata injection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
vision_engine = VisionEngine()

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "device": str(config.DEVICE),
        "device_name": config.DEVICE_NAME,
        "database": config.DB_NAME,
        "database_host": config.DB_HOST
    }

@app.get("/api/matchups")
def get_matchups():
    return RosterRepository.get_matchups(limit=20)

@app.get("/api/teams/{team_id}/players")
def get_team_players(team_id: int):
    return RosterRepository.get_players_by_team(team_id)

@app.post("/api/analyze-photo")
async def analyze_photo(
    file: UploadFile = File(...),
    matchup_id: Optional[int] = Form(None),
    home_team_id: Optional[int] = Form(None),
    away_team_id: Optional[int] = Form(None),
    photographer: str = Form("Marco Canoniero"),
    agency: str = Form("Sportphoto24")
):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"ai_analyze_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Vision & OCR
        vision_res = vision_engine.detect_persons_and_jerseys(temp_path)

        # 2. Matchup & Teams resolution
        team_ids = []
        home_team_name = "Home Team"
        away_team_name = "Away Team"

        if home_team_id and away_team_id:
            team_ids = [home_team_id, away_team_id]
        elif matchup_id:
            matchups = RosterRepository.get_matchups(limit=50)
            target = next((m for m in matchups if m["id"] == matchup_id), None)
            if target:
                home_team_id = target["home_team_id"]
                away_team_id = target["away_team_id"]
                home_team_name = target["home_team_name"] or home_team_name
                away_team_name = target["away_team_name"] or away_team_name
                team_ids = [home_team_id, away_team_id]

        # 3. Player identification via OCR numbers
        matched_players = []
        candidate_numbers = vision_res.get("candidate_jersey_numbers", [])
        for c in candidate_numbers:
            found = RosterRepository.find_player_by_number_and_teams(c["number"], team_ids)
            for f in found:
                if f not in matched_players:
                    matched_players.append(f)

        primary_player = matched_players[0] if matched_players else None
        player_name = f"{primary_player['first_name']} {primary_player['last_name']}" if primary_player else ""
        player_team = primary_player.get("team_name", "") if primary_player else home_team_name

        # 4. Situation Classification
        photo_type = PhotoClassifier.classify(vision_res, context={})
        details = PhotoClassifier.get_details(photo_type)

        # 5. Caption Generation
        caption = CaptionBuilder.build_caption(
            player_name=player_name,
            team_name=player_team,
            action_verb=details["action_verb"],
            competition_name="Serie A",
            home_team=home_team_name,
            away_team=away_team_name,
            venue="Stadio San Siro",
            city="Milan",
            country="Italy",
            event_date=None,
            photographer_credit=photographer,
            agency=agency
        )

        # 6. SmartKeywords IMAGO
        keywords = SmartKeywordsBuilder.generate_keywords(
            player_name=player_name,
            team_name=player_team,
            opponent_team=away_team_name,
            competition="Serie A",
            venue="Stadio San Siro",
            city="Milan",
            country="Italy",
            photo_type_code=photo_type
        )

        return {
            "success": True,
            "vision": vision_res,
            "matched_players": matched_players,
            "primary_player": primary_player,
            "photo_type": photo_type,
            "photo_type_label": details["label"],
            "suggested_caption": caption,
            "suggested_keywords": keywords
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
