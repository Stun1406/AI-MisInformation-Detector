from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.core.logger import logger
from app.services.text_processor import TextProcessor
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Initialize services
text_processor = TextProcessor()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()

class ClaimRequest(BaseModel):
    claim: str

class ClaimResponse(BaseModel):
    claim: str
    similar_facts: list[dict]

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "version": settings.API_VERSION}

@app.post("/analyze", response_model=ClaimResponse)
async def analyze_claim(request: ClaimRequest):
    """Analyze a claim and retrieve similar facts."""
    logger.info(f"Analyzing claim: {request.claim[:50]}...")
    try:
        # Clean and embed claim
        cleaned_claim = text_processor.clean_text(request.claim)
        # Retrieve similar facts
        similar_facts = retrieval_service.retrieve_similar_facts(cleaned_claim, limit=3)
        return ClaimResponse(
            claim=request.claim,
            similar_facts=similar_facts
        )
    except Exception as e:
        logger.error(f"Error analyzing claim: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing claim: {str(e)}")