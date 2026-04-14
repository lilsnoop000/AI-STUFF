import logging
import os
from logging.handlers import RotatingFileHandler
from config import LOG_PATH, MAX_LOG_SIZE_MB, LOG_BACKUP_COUNT

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
                             backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
