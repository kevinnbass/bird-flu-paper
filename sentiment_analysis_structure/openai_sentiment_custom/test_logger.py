# test_logger.py

from utils import logger

def test_logging():
    logger.info("Test log message: INFO level.")
    logger.warning("Test log message: WARNING level.")
    logger.error("Test log message: ERROR level.")

if __name__ == "__main__":
    test_logging()
