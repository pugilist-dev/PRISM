from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml
import torch

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

tune_config = {
    "wandb_key" : "f8ef98e88768dcd2b7d55361a041652918a62844",
    "tune" : True,
    "count" : 20,
    "augment" : False,
    "seed" : 42,
    "debug" : True,
    "device" : "cuda:0",
}

with open(PROJ_ROOT / "config" / "sweep_config.yml") as f:
    sweep_config = yaml.safe_load(f)