import logging
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

DATE = datetime.now().strftime("%Y-%m-%d")
log_path = os.path.join(LOG_DIR, f"{DATE}.log")

logger = logging.getLogger("concept_vector")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# output to file
handler = logging.FileHandler(log_path, encoding="utf-8")
handler.setFormatter(formatter)
logger.addHandler(handler)

# output to console
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)