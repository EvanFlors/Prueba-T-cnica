import logging
import sys
import os
from datetime import datetime
from colorlog import ColoredFormatter

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_formatter = logging.Formatter(
    "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

color_formatter = ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(color_formatter)
logger.addHandler(console_handler)