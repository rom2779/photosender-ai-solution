import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Database configuration (matches PhotoSender application.properties)
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "photoftp")
DB_NAME = os.getenv("DB_NAME", "photoftpnewtesting")

SQLALCHEMY_DATABASE_URL = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Hardware Acceleration: Apple Silicon Metal Performance Shaders (MPS)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = "NVIDIA CUDA"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

# ExifTool binary
EXIFTOOL_PATH = os.getenv("EXIFTOOL_PATH", "/usr/local/bin/exiftool")

# Defaults
DEFAULT_AGENCY = "Sportphoto24"
DEFAULT_SPORT = "FOOTBALL"

