from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

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
    "wandb_key": "f50e7404c274fd0240b3de443ad368d762b16643",
    "tune": True,
    "count" : 20,
    "augment" : False,
    "seed" : 42,
    "debug" : True,
    "device" : "cuda:0",
}

aug_params = {
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "rotation": 10,
    "scale": 0.1,
    "shear": 0.1,
    "hflip": 0.5,
    "vflip": 0.5,
    "normalize": True,

}

with open(PROJ_ROOT / "config" / "sweep_config.yml") as f:
    sweep_config = yaml.safe_load(f)