import logging
import os


def setup_logging(
    logs_dir: str, log_level: str = "DEBUG", log_file: str = "pipeline.log"
) -> None:
    """
    Setup logging configurations.

    Args:
        logs_dir (str): Directory to store log files.
        log_level (str): Logging level (e.g., 'DEBUG', 'INFO').
        log_file (str): Name of the log file.
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, log_file)

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    level = getattr(logging, log_level.upper(), logging.DEBUG)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
