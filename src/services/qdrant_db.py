from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Charger les variables d'environnement (URL de Qdrant, etc.)
load_dotenv()

class QdrantService:
    def __init__(self):
        # On récupère l'URL dans le fichier .env, sinon on utilise localhost par défaut
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )

    def init_collection(self, collection_name="loan_history"):
        """Crée l'espace de stockage si il n'existe pas encore."""
        return self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"size": 384, "distance": "Cosine"} 
        )