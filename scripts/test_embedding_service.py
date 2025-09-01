import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.embedding_service import EmbeddingService

def test_embedding_service():
    embedder = EmbeddingService()
    text = "COVID vaccines contain microchips"
    batch_texts = [
        "COVID vaccines are safe",
        "Vaccines cause autism"
    ]

    # Test single embedding
    embedding = embedder.generate_embedding(text)
    print(f"Single embedding length: {len(embedding)}")
    print(f"Single embedding sample: {embedding[:5]}...")

    # Test batch embeddings
    embeddings = embedder.generate_batch_embeddings(batch_texts)
    print(f"Batch embeddings count: {len(embeddings)}")
    print(f"First batch embedding length: {len(embeddings[0])}")

if __name__ == "__main__":
    test_embedding_service()