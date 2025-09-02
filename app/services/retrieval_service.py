from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from loguru import logger
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
import json
import os

class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.initialize_collection()
        logger.info(f"Initialized RetrievalService with collection: {self.collection_name}")

    def initialize_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
                self.load_sample_facts()
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    def load_sample_facts(self):
        """Load sample facts from JSON and store their embeddings in Qdrant."""
        try:
            facts_file = os.path.join("data", "sample_facts.json")
            if not os.path.exists(facts_file):
                logger.warning(f"Sample facts file not found: {facts_file}")
                return
            with open(facts_file, 'r') as f:
                facts = json.load(f)
            points = []
            for fact in facts:
                embedding = self.embedding_service.generate_embedding(fact["text"])
                points.append(PointStruct(
                    id=fact["id"],
                    vector=embedding,
                    payload={"text": fact["text"], "source": fact["source"]}
                ))
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Loaded {len(points)} sample facts into Qdrant")
        except Exception as e:
            logger.error(f"Failed to load sample facts: {str(e)}")
            raise

    def retrieve_similar_facts(self, claim: str, limit: int = 3) -> list[dict]:
        """Retrieve facts similar to the given claim."""
        logger.debug(f"Retrieving similar facts for claim: {claim[:50]}...")
        try:
            embedding = self.embedding_service.generate_embedding(claim)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
                score_threshold=settings.SIMILARITY_THRESHOLD
            )
            results = [
                {
                    "id": result.id,
                    "text": result.payload["text"],
                    "source": result.payload["source"],
                    "similarity": result.score
                }
                for result in search_results
            ]
            logger.info(f"Retrieved {len(results)} similar facts for claim")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve facts: {str(e)}")
            return []