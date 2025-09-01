from sentence_transformers import SentenceTransformer
from loguru import logger
from app.core.config import settings
from app.services.text_processor import TextProcessor
import torch

class EmbeddingService:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load the Sentence Transformer model."""
        if self.model is None:
            logger.info(f"Loading model: {settings.EMBEDDING_MODEL}")
            try:
                self.model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=self.device
                )
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise
        return self.model

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text after cleaning."""
        logger.debug(f"Generating embedding for text: {text[:50]}...")
        cleaned_text = self.text_processor.clean_text(text)
        model = self.load_model()
        embedding = model.encode([cleaned_text], convert_to_tensor=False)[0]
        return embedding.tolist()

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        logger.debug(f"Generating embeddings for {len(texts)} texts...")
        cleaned_texts = [self.text_processor.clean_text(text) for text in texts]
        model = self.load_model()
        embeddings = model.encode(cleaned_texts, convert_to_tensor=False)
        return embeddings.tolist()