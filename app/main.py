from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.core.logger import logger
from app.services.text_processor import TextProcessor
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.classification_service import ClassificationService

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Initialize services
text_processor = TextProcessor()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()
classification_service = ClassificationService()

class ClaimRequest(BaseModel):
    claim: str

class ClaimResponse(BaseModel):
    claim: str
    similar_facts: list[dict]
    classification: str
    confidence: float

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy", "version": settings.API_VERSION}

@app.post("/analyze", response_model=ClaimResponse)
async def analyze_claim(request: ClaimRequest):
    """Analyze a claim and retrieve similar facts with classification."""
    logger.info(f"Analyzing claim: {request.claim[:50]}...")
    try:
        # Clean and embed claim
        cleaned_claim = text_processor.clean_text(request.claim)
        # Retrieve similar facts
        similar_facts = retrieval_service.retrieve_similar_facts(cleaned_claim, limit=3)
        # Classify claim
        classification, confidence = classification_service.classify_claim(request.claim, similar_facts)
        return ClaimResponse(
            claim=request.claim,
            similar_facts=similar_facts,
            classification=classification,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error analyzing claim: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing claim: {str(e)}")