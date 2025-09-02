import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.retrieval_service import RetrievalService

def test_retrieval_service():
    retriever = RetrievalService()
    claim = "COVID vaccines contain microchips"
    
    # Test retrieval
    results = retriever.retrieve_similar_facts(claim, limit=3)
    print(f"Retrieved {len(results)} facts for claim: {claim}")
    for result in results:
        print(f"Fact ID: {result['id']}, Text: {result['text']}, Source: {result['source']}, Similarity: {result['similarity']:.3f}")

if __name__ == "__main__":
    test_retrieval_service()