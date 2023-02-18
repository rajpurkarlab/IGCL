import logging

logger = logging.getLogger('transfer')
logger.propagate = False

if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_handler.setFormatter(log_formatter)

if not logger.hasHandlers():
    logger.addHandler(log_handler)