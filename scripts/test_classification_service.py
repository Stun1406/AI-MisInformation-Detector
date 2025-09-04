import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.classification_service import ClassificationService

def test_classification_service():
    classifier = ClassificationService()
    claim = "COVID vaccines contain microchips"
    sample_facts = [
        {"id": 3, "text": "Microchips are not included in COVID vaccines.", "source": "FDA", "similarity": 0.921}
    ]
    
    # Test classification
    label, confidence = classifier.classify_claim(claim, sample_facts)
    print(f"Claim: {claim}")
    print(f"Classification: {label}, Confidence: {confidence:.3f}")

if __name__ == "__main__":
    test_classification_service()