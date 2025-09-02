import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import requests

def test_analyze_endpoint():
    url = "http://localhost:8000/analyze"
    claim = {"claim": "COVID vaccines contain microchips"}
    
    # Test analyze endpoint
    try:
        response = requests.post(url, json=claim)
        response.raise_for_status()
        result = response.json()
        print(f"Claim: {result['claim']}")
        print(f"Retrieved {len(result['similar_facts'])} similar facts:")
        for fact in result['similar_facts']:
            print(f"Fact ID: {fact['id']}, Text: {fact['text']}, Source: {fact['source']}, Similarity: {fact['similarity']:.3f}")
    except requests.RequestException as e:
        print(f"Error testing endpoint: {str(e)}")

if __name__ == "__main__":
    test_analyze_endpoint()