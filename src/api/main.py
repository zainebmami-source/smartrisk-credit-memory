import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models
from src.services.embedding import EmbeddingService
from src.core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

app = FastAPI()

# 1. Autoriser les requêtes du navigateur (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Configurer le dossier static
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(script_dir))
static_path = os.path.join(base_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# 3. Services
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = EmbeddingService()

@app.get("/search")
def search_loan(query: str):
    try:
        vector = embedder.generate_vector(query)
        
        # On utilise search() qui est la méthode la plus fiable
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=5
        )
        
        # ON FORMATE ICI POUR LE JAVASCRIPT
        formatted_points = []
        for hit in hits:
            formatted_points.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        return {"results": {"points": formatted_points}}
        
    except Exception as e:
        return {"error": str(e)}