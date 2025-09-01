import re
import nltk
from bs4 import BeautifulSoup
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app.core.config import settings

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK data for text processing")
    nltk.download('punkt')
    nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

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
        tokens = word_tokenize(cleaned_text)
        keywords = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return keywords[:settings.BATCH_SIZE]