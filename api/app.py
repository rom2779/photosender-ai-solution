from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import shutil
import os
import tempfile
import json
import numpy as np
from pathlib import Path

import config
from database.repositories import RosterRepository, SmartKeywordRepository
from core.face_engine import FaceEngine
from core.photo_classifier import PhotoClassifier
from core.caption_builder import CaptionBuilder
from core.smart_keywords_builder import SmartKeywordsBuilder
from core.metadata_writer import MetadataWriter

app = FastAPI(
    title="PhotoSender Smart AI Engine",
    description="Dedicated Face AI & Match Desk for Sports Photojournalism",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PhotoSender AI Desk</h1><p>index.html not found</p>"

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
    return RosterRepository.get_matchups(limit=30)

@app.get("/api/matchups/{matchup_id}/roster")
def get_matchup_roster(matchup_id: int):
    matchups = RosterRepository.get_matchups(limit=50)
    matchup = next((m for m in matchups if m["id"] == matchup_id), None)
    if not matchup:
        raise HTTPException(status_code=404, detail="Matchup not found")

    home_players = RosterRepository.get_players_by_team(matchup["home_team_id"]) if matchup.get("home_team_id") else []
    away_players = RosterRepository.get_players_by_team(matchup["away_team_id"]) if matchup.get("away_team_id") else []

    # Tag team names
    for p in home_players:
        p["team_name"] = matchup.get("home_team_name", "Home Team")
    for p in away_players:
        p["team_name"] = matchup.get("away_team_name", "Away Team")

    return {
        "matchup": matchup,
        "home_players": home_players,
        "away_players": away_players
    }

@app.post("/api/analyze-face")
async def analyze_face(
    file: UploadFile = File(...),
    matchup_id: Optional[int] = Form(None)
):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"face_ingest_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Face detection & 128-d embedding extraction
        faces = FaceEngine.detect_and_encode_faces(temp_path)
        primary_face = faces[0] if faces else None

        # 2. Get active roster players for the match
        all_players = []
        home_team_name = "Home Team"
        away_team_name = "Away Team"

        if matchup_id:
            matchups = RosterRepository.get_matchups(limit=50)
            target = next((m for m in matchups if m["id"] == matchup_id), None)
            if target:
                home_team_name = target.get("home_team_name") or home_team_name
                away_team_name = target.get("away_team_name") or away_team_name
                if target.get("home_team_id"):
                    hp = RosterRepository.get_players_by_team(target["home_team_id"])
                    for p in hp: p["team_name"] = home_team_name
                    all_players.extend(hp)
                if target.get("away_team_id"):
                    ap = RosterRepository.get_players_by_team(target["away_team_id"])
                    for p in ap: p["team_name"] = away_team_name
                    all_players.extend(ap)

        # 3. Match face against active players
        matched_player = None
        face_confidence = 0.0

        if primary_face and all_players:
            face_matches = FaceEngine.match_face_against_roster(primary_face["embedding"], all_players)
            if face_matches:
                best = face_matches[0]
                if best["is_match"]:
                    matched_player = best["player"]
                    face_confidence = best["confidence"]

        player_name = f"{matched_player['first_name']} {matched_player['last_name']}" if matched_player else ""
        player_team = matched_player.get("team_name", home_team_name) if matched_player else home_team_name

        # 4. Situation classification
        photo_type = "PLAYER_PORTRAIT" if (primary_face and primary_face["framing"] == "CLOSE_UP") else "BALL_CHALLENGE"
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
            photographer_credit="Marco Canoniero",
            agency="Sportphoto24"
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

        # Prepare face embedding as JSON list for possible training
        face_emb_list = primary_face["embedding"].tolist() if primary_face else None

        # Clean box without numpy ints
        clean_primary = None
        if primary_face:
            clean_primary = {
                "box": [int(v) for v in primary_face["box"]],
                "framing": primary_face["framing"],
                "sharpness": primary_face["sharpness"]
            }

        return {
            "success": True,
            "vision": {
                "face_count": len(faces),
                "primary_face": clean_primary,
                "overall_sharpness": clean_primary["sharpness"] if clean_primary else 25.0
            },
            "matched_player": matched_player,
            "face_confidence": face_confidence,
            "primary_face_embedding": face_emb_list,
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

@app.post("/api/rebuild-caption")
def rebuild_caption(payload: Dict[str, Any] = Body(...)):
    player_name = payload.get("player_name", "")
    team_name = payload.get("team_name", "Home Team")
    home_team = payload.get("home_team", "Home Team")
    away_team = payload.get("away_team", "Away Team")
    photo_type = payload.get("photo_type", "BALL_CHALLENGE")

    details = PhotoClassifier.get_details(photo_type)

    caption = CaptionBuilder.build_caption(
        player_name=player_name,
        team_name=team_name,
        action_verb=details["action_verb"],
        competition_name="Serie A",
        home_team=home_team,
        away_team=away_team,
        venue="Stadio San Siro",
        city="Milan",
        country="Italy",
        event_date=None,
        photographer_credit="Marco Canoniero",
        agency="Sportphoto24"
    )

    keywords = SmartKeywordsBuilder.generate_keywords(
        player_name=player_name,
        team_name=team_name,
        opponent_team=away_team if team_name == home_team else home_team,
        competition="Serie A",
        venue="Stadio San Siro",
        city="Milan",
        country="Italy",
        photo_type_code=photo_type
    )

    return {"caption": caption, "keywords": keywords}

@app.post("/api/players/{player_id}/train-face")
def train_player_face(player_id: int, payload: Dict[str, Any] = Body(...)):
    embedding = payload.get("embedding")
    if not embedding:
        raise HTTPException(status_code=400, detail="Missing embedding")

    np_emb = np.array(embedding, dtype=np.float64)
    FaceEngine.save_player_face_embedding(player_id, np_emb, update_db_flag=True)
    return {"success": True, "message": f"Face embedding learned for player {player_id}"}

@app.post("/api/approve-and-send")
async def approve_and_send(
    file: UploadFile = File(...),
    caption: str = Form(...),
    player_id: Optional[int] = Form(None),
    matchup_id: Optional[int] = Form(None),
    face_embedding: Optional[str] = Form(None)
):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"send_{file.filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. If player was confirmed and face_embedding provided, auto-learn face
        if player_id and face_embedding:
            try:
                emb_arr = json.loads(face_embedding)
                if emb_arr:
                    FaceEngine.save_player_face_embedding(player_id, np.array(emb_arr, dtype=np.float64))
            except Exception as e:
                logger.warning(f"Failed to auto-train face: {e}")

        # 2. Get IMAGO keywords for the caption
        keywords = SmartKeywordsBuilder.generate_keywords(
            player_name="",
            team_name="",
            opponent_team="",
            competition="Serie A",
            venue="Stadio San Siro",
            city="Milan",
            country="Italy",
            photo_type_code="BALL_CHALLENGE"
        )

        # 3. Write IPTC/XMP with ExifTool (Preserving DateTimeOriginal with -P flag)
        MetadataWriter.write_iptc_xmp(
            image_path=temp_path,
            caption=caption,
            keywords=keywords,
            byline="Marco Canoniero",
            credit="Sportphoto24",
            headline="Serie A - Match Action"
        )

        # 4. Dispatch to PhotoSender's PhotoFtp/send folder
        # This triggers PhotoSender's FileWalker -> Queue -> FileSenderService (FTP TRANSMISSION)
        dispatched_path = MetadataWriter.dispatch_to_photoftp(temp_path)

        return {
            "success": True,
            "message": "Metadata written and dispatched to PhotoSender transmission queue",
            "filename": os.path.basename(dispatched_path)
        }
    except Exception as e:
        logger.error(f"Error in approve_and_send: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
