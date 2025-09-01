import re
import spacy
from bs4 import BeautifulSoup
from loguru import logger
from app.core.config import settings

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class TextProcessor:
    def __init__(self):
        self.stop_words = nlp.Defaults.stop_words

    def clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags, URLs, and special characters."""
        logger.debug(f"Cleaning text: {text[:50]}...")
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text, excluding stop words."""
        logger.debug(f"Extracting keywords from: {text[:50]}...")
        cleaned_text = self.clean_text(text)
        doc = nlp(cleaned_text)
        keywords = [
            token.text for token in doc
            if token.text not in self.stop_words and len(token.text) > 2
        ]
        return keywords[:settings.BATCH_SIZE]