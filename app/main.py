from fastapi import FastAPI
from app.core.config import settings
from app.core.logger import logger

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "version": settings.API_VERSION}