from transformers import pipeline
from loguru import logger
from app.services.text_processor import TextProcessor

class ClassificationService:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.classifier = None
        self.load_classifier()
        logger.info("Initialized ClassificationService with DistilBERT")

    def load_classifier(self):
        """Load DistilBERT model for text classification."""
        try:
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            logger.info("Loaded DistilBERT classifier")
        except Exception as e:
            logger.error(f"Failed to load classifier: {str(e)}")
            raise

    def classify_claim(self, claim: str, similar_facts: list[dict]) -> tuple[str, float]:
        """Classify a claim as true/false based on text and retrieved facts."""
        logger.debug(f"Classifying claim: {claim[:50]}...")
        try:
            cleaned_claim = self.text_processor.clean_text(claim)
            # Combine claim with retrieved facts for context
            context = cleaned_claim
            if similar_facts:
                fact_texts = [fact["text"] for fact in similar_facts]
                context += " " + " ".join(fact_texts)
            # Classify with DistilBERT
            results = self.classifier(context)[0]
            # Map SST-2 labels (POSITIVE/NEGATIVE) to true/false
            label = "true" if results[0]["label"] == "POSITIVE" else "false"
            confidence = results[0]["score"] if results[0]["label"] == "POSITIVE" else results[1]["score"]
            logger.info(f"Classified claim as {label} with confidence {confidence:.3f}")
            return label, confidence
        except Exception as e:
            logger.error(f"Failed to classify claim: {str(e)}")
            return "unknown", 0.0