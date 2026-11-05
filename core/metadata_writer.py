import subprocess
import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import config

logger = logging.getLogger(__name__)

class MetadataWriter:
    """
    ExifTool wrapper for writing IPTC/XMP metadata while strictly preserving
    original capture timestamps and adhering to agency standards.
    """

    @staticmethod
    def read_metadata(image_path: str) -> Dict[str, Any]:
        cmd = [
            config.EXIFTOOL_PATH,
            "-json",
            "-DateTimeOriginal",
            "-CreateDate",
            "-Caption-Abstract",
            "-Keywords",
            "-By-line",
            "-Credit",
            "-TransmissionReference",
            "-OriginalTransmissionReference",
            "-Headline",
            image_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            import json
            data = json.loads(res.stdout)
            return data[0] if data else {}
        return {}

    @staticmethod
    def write_iptc_xmp(
        image_path: str,
        caption: str,
        keywords: List[str],
        headline: Optional[str] = None,
        byline: Optional[str] = None,
        credit: Optional[str] = None,
        job_id: Optional[str] = None,
        person_shown: Optional[str] = None
    ) -> bool:
        """
        Writes caption, keywords, headline, credit and IPTC/XMP fields.
        CRITICAL: Uses -P flag to preserve file system modification timestamp,
        and never alters original capture dates (DateTimeOriginal, CreateDate, etc.).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        args = [
            config.EXIFTOOL_PATH,
            "-overwrite_original",
            "-P",  # Preserve filesystem timestamp!
            "-codedcharacterset=utf8",
            f"-IPTC:Caption-Abstract={caption}",
            f"-XMP-dc:Description={caption}",
        ]

        if headline:
            args.extend([f"-IPTC:Headline={headline}", f"-XMP-photoshop:Headline={headline}"])

        if byline:
            args.extend([f"-IPTC:By-line={byline}", f"-XMP-dc:Creator={byline}"])

        if credit:
            args.extend([f"-IPTC:Credit={credit}", f"-XMP-photoshop:Credit={credit}"])

        if job_id:
            args.extend([
                f"-IPTC:OriginalTransmissionReference={job_id}",
                f"-XMP-photoshop:TransmissionReference={job_id}"
            ])

        if person_shown:
            args.extend([
                f"-IPTC:PersonInImage={person_shown}",
                f"-XMP-iptcExt:PersonInImage={person_shown}"
            ])

        # Keywords: clear old and set new to avoid duplicate accumulation
        args.append("-IPTC:Keywords=")
        args.append("-XMP-dc:Subject=")
        for kw in keywords:
            if kw and kw.strip():
                args.append(f"-IPTC:Keywords={kw.strip()}")
                args.append(f"-XMP-dc:Subject={kw.strip()}")

        args.append(image_path)

        logger.info(f"Writing metadata to {image_path} with ExifTool...")
        res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"ExifTool error: {res.stderr}")
            return False
            
        return True

    @staticmethod
    def dispatch_to_photoftp(image_path: str, destination_folder: str = "PhotoFtp/send", target_filename: Optional[str] = None) -> str:
        """
        Copies or moves the metadata-tagged photo into PhotoSender's transmission folder.
        This triggers PhotoSender's FileWalker -> Queue -> FileSenderService (FTP send).
        """
        dest_dir = Path(destination_folder)
        # If relative, check if running on VPS (/photoftp/PhotoFtp/send) or local
        if not dest_dir.is_absolute():
            vps_path = Path("/photoftp") / destination_folder
            if vps_path.parent.exists():
                dest_dir = vps_path
            else:
                dest_dir = Path(__file__).resolve().parent.parent.parent / "photosender-master" / destination_folder

        dest_dir.mkdir(parents=True, exist_ok=True)

        final_name = target_filename if target_filename else os.path.basename(image_path)
        dest_file = dest_dir / final_name

        shutil.copy2(image_path, str(dest_file))
        logger.info(f"Dispatched photo {final_name} to PhotoSender ingest folder: {dest_file}")
        return str(dest_file)
