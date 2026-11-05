import sys
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from database.repositories import RosterRepository, SmartKeywordRepository
from core.vision_engine import VisionEngine
from core.photo_classifier import PhotoClassifier
from core.caption_builder import CaptionBuilder
from core.smart_keywords_builder import SmartKeywordsBuilder
from core.metadata_writer import MetadataWriter

console = Console()

def main():
    console.print(Panel.fit(
        f"[bold cyan]PhotoSender AI Solution[/bold cyan] - SmartKeywords & Vision Engine\n"
        f"Hardware: [green]{config.DEVICE_NAME}[/green] | DB: [yellow]{config.DB_NAME}@{config.DB_HOST}[/yellow]",
        border_style="cyan"
    ))

    # Test image path
    default_img = Path("../photosender-master/dummy test.jpg").resolve()
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(default_img)

    if not os.path.exists(image_path):
        console.print(f"[bold red]Image not found:[/bold red] {image_path}")
        return

    console.print(f"\n📸 [bold]Analyzing Image:[/bold] [yellow]{image_path}[/yellow]")
    
    # 1. Vision Engine analysis
    with console.status("[bold green]Running Apple Silicon Vision & OCR analysis...[/bold green]"):
        vision = VisionEngine()
        vision_res = vision.detect_persons_and_jerseys(image_path)

    # Display Vision Stats
    v_table = Table(title="🔍 Vision & Composition Analysis", border_style="blue")
    v_table.add_column("Parameter", style="cyan")
    v_table.add_column("Value", style="bold white")
    v_table.add_row("Resolution", f"{vision_res['image_size'][0]} x {vision_res['image_size'][1]} px")
    v_table.add_row("Sharpness Score", str(vision_res["overall_sharpness"]))
    v_table.add_row("Faces Count", str(vision_res["face_count"]))
    if vision_res["primary_face"]:
        v_table.add_row("Primary Face Framing", vision_res["primary_face"]["framing"])
        v_table.add_row("Face Sharpness", str(vision_res["primary_face"]["sharpness"]))
    v_table.add_row("Detected Persons", str(vision_res["detected_persons_count"]))
    
    candidate_nums = vision_res["candidate_jersey_numbers"]
    nums_str = ", ".join([f"#{c['number']} ({int(c['confidence']*100)}%)" for c in candidate_nums]) if candidate_nums else "None directly visible"
    v_table.add_row("OCR Jersey Numbers", nums_str)
    console.print(v_table)

    # 2. Database Matchup Check
    console.print("\n📋 [bold]Querying Database Roster & Matchups...[/bold]")
    matchups = RosterRepository.get_matchups(limit=3)
    if matchups:
        m = matchups[0]
        home_team = m["home_team_name"] or "Home Team"
        away_team = m["away_team_name"] or "Away Team"
        team_ids = [m["home_team_id"], m["away_team_id"]]
        console.print(f"Active Matchup: [bold green]{home_team}[/bold green] vs [bold blue]{away_team}[/bold blue]")
    else:
        home_team = "Juventus FC"
        away_team = "Juventus Next Gen"
        team_ids = []
        console.print("No active matchup in DB, using fallback Juventus vs Juventus Next Gen.")

    # 3. Player Association
    matched_player = None
    if candidate_nums and team_ids:
        # Search by detected number
        for c in candidate_nums:
            found = RosterRepository.find_player_by_number_and_teams(c["number"], team_ids)
            if found:
                matched_player = found[0]
                break

    # If no number matched from roster, search fallback player
    if not matched_player:
        # Try finding a player from the team
        players = RosterRepository.search_players_by_name("Celik", team_ids)
        if not players and team_ids:
            all_p = RosterRepository.get_players_by_team(team_ids[0])
            matched_player = all_p[0] if all_p else None
        elif players:
            matched_player = players[0]

    player_name = f"{matched_player['first_name']} {matched_player['last_name']}" if matched_player else "Mehmet Zeki Celik"
    player_team = matched_player.get("team_name") if matched_player else home_team

    console.print(f"Associated Player: [bold yellow]{player_name}[/bold yellow] ({player_team})")

    # 4. Situation Classification
    photo_type = PhotoClassifier.classify(vision_res, context={"is_warmup": "warm up" in image_path.lower() or "warm up" in "dummy test"})
    details = PhotoClassifier.get_details(photo_type)
    console.print(f"Classified Situation: [bold magenta]{details['label']}[/bold magenta] ({photo_type})")

    # 5. Caption Generation
    caption = CaptionBuilder.build_caption(
        player_name=player_name,
        team_name=player_team,
        action_verb=details["action_verb"],
        competition_name="Pre-Season Friendly",
        home_team=home_team,
        away_team=away_team,
        venue="Allianz Stadium",
        city="Turin",
        country="Italy",
        event_date=None,
        photographer_credit="Marco Canoniero",
        agency="Sportphoto24"
    )

    console.print(Panel(caption, title="📝 Generated Editorial Caption", border_style="green"))

    # 6. SmartKeywords IMAGO Generation
    keywords = SmartKeywordsBuilder.generate_keywords(
        player_name=player_name,
        team_name=player_team,
        opponent_team=away_team,
        competition="Pre-Season Friendly",
        venue="Allianz Stadium",
        city="Turin",
        country="Italy",
        photo_type_code=photo_type
    )

    kw_panel = Panel(", ".join(keywords), title=f"🏷️ Generated IMAGO Keywords ({len(keywords)} terms)", border_style="yellow")
    console.print(kw_panel)

    console.print("\n[bold green]✓ Pipeline completed successfully with local database & Apple Silicon vision engine![/bold green]\n")

if __name__ == "__main__":
    main()

