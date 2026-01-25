import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.services.embedding import EmbeddingService

# 1. Connexion à Qdrant (assure-toi que Docker est lancé !)
client = QdrantClient(host="localhost", port=6333)
collection_name = "loan_vectors"

# 2. Initialisation du service d'embedding
embedder = EmbeddingService()

# 3. Création de la "table" (Collection) dans Qdrant
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 4. Chargement de la data nettoyée
df = pd.read_csv('data/credit_risk_cleaned.csv').head(100) # On teste sur 100 lignes

print("Début de la vectorisation...")

points = []
for index, row in df.iterrows():
    # On transforme la ligne en texte pour l'IA
    text_data = f"Age: {row['person_age']}, Income: {row['person_income']}, Intent: {row['loan_intent']}, Status: {row['loan_status']}"
    
    # Génération du vecteur (liste de nombres)
    vector = embedder.generate_vector(text_data)
    
    # Préparation du point pour Qdrant
    points.append(PointStruct(
        id=index,
        vector=vector,
        payload=row.to_dict() # On garde les infos d'origine pour les afficher plus tard
    ))

# 5. Envoi massif vers Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"Succès ! {len(points)} profils sont maintenant dans la mémoire de l'IA.")