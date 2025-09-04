import json
import os
from loguru import logger
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_HOST: str = 'localhost'
    QDRANT_PORT: int = 6334
    QDRANT_COLLECTION_NAME: str = 'fact_embeddings'
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'

settings = Settings()

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logger.add('logs/app.log', rotation='1 MB')

def populate_qdrant():
    try:
        # Connect with compatibility check disabled
        client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT,
            prefer_grpc=False,  # Use REST API
            timeout=60
        )
        
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Load facts
        with open('data/sample_facts.json', 'r') as f:
            facts = json.load(f)
        logger.info(f'Loading {len(facts)} facts from data/sample_facts.json')
        
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Handle existing collection more gracefully
        try:
            # Check if collection exists in API
            collections_response = client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            if collection_name in existing_collections:
                logger.info(f"Collection {collection_name} exists and is accessible")
                # Just clear the collection instead of deleting it
                try:
                    # Get collection info first
                    collection_info = client.get_collection(collection_name)
                    if collection_info.points_count > 0:
                        logger.info(f"Clearing {collection_info.points_count} existing points")
                        # Clear all points from collection
                        client.delete(
                            collection_name=collection_name,
                            points_selector=True  # Delete all points
                        )
                        logger.info("Collection cleared successfully")
                    else:
                        logger.info("Collection is already empty")
                except Exception as clear_error:
                    logger.warning(f"Could not clear collection: {clear_error}")
                    
            else:
                # Collection doesn't exist in API but might exist on disk
                logger.info(f"Collection {collection_name} not found in API")
                try:
                    # Try to create it
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "size": 384,
                            "distance": "Cosine"
                        }
                    )
                    logger.info("Collection created successfully")
                except Exception as create_error:
                    if "already exists" in str(create_error):
                        logger.warning("Collection exists on disk but not in API. This might be a permission issue.")
                        logger.info("Try restarting Qdrant or check file permissions")
                        # Try to recover by using a different collection name
                        alternative_name = f"{collection_name}_new"
                        logger.info(f"Attempting to create alternative collection: {alternative_name}")
                        client.create_collection(
                            collection_name=alternative_name,
                            vectors_config={
                                "size": 384,
                                "distance": "Cosine"
                            }
                        )
                        collection_name = alternative_name
                        logger.info(f"Using alternative collection name: {collection_name}")
                    else:
                        raise create_error
                        
        except Exception as e:
            logger.error(f"Collection management failed: {e}")
            raise
        
        # Embed and store facts
        points = []
        for fact in facts:
            embedding = model.encode(fact['text']).tolist()
            points.append({
                "id": fact['id'],
                "vector": embedding,
                "payload": {
                    "text": fact['text'], 
                    "source": fact['source']
                }
            })
        
        # Upload points
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f'Successfully embedded and stored {len(points)} facts in Qdrant')
        
        # Verify
        collection_info = client.get_collection(collection_name)
        logger.info(f'Collection info: {collection_info.points_count} points total')
        
    except Exception as e:
        logger.error(f'Failed to populate Qdrant: {str(e)}')
        raise

if __name__ == '__main__':
    populate_qdrant()