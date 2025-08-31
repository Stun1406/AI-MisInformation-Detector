from loguru import logger
from pathlib import Path
from app.core.config import settings

def setup_logger():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        log_dir / "app.log",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

setup_logger()