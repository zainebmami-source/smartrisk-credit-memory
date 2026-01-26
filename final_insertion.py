import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.services.embedding import EmbeddingService

# 1. Initialisation des services
client = QdrantClient(host="localhost", port=6333)
embedder = EmbeddingService()
collection_name = "loan_memory"

# 2. Création de la collection (Mémoire de Qdrant)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 3. Lecture du fichier nettoyé
df = pd.read_csv('data/credit_risk_cleaned.csv').head(500) # On commence par 500 lignes

print("Insertion en cours...")
points = []

for index, row in df.iterrows():
    # On prépare le texte pour le transformer en vecteur [cite: 103]
    text_profile = f"Age: {row['person_age']}, Income: {row['person_income']}, Debt: {row['loan_amnt']}"
    vector = embedder.generate_vector(text_profile)
    
    # On crée le point avec les métadonnées (Payload) [cite: 97, 98]
    points.append(PointStruct(
        id=index,
        vector=vector,
        payload=row.to_dict() # Contient le statut du prêt (0 ou 1) [cite: 100]
    ))

# 4. Envoi groupé à Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"Succès ! {len(points)} dossiers insérés dans la mémoire SmartRisk.")