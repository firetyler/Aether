import logging
import os
import sys

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "Endpoint.log")

def get_logger(name=None):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # FILE HANDLER
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # CONSOLE HANDLER (FIX)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
