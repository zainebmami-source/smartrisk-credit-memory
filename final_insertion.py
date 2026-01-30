import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.services.embedding import EmbeddingService

# ---------------------------------------------------------
# 1. Configuration de la connexion (Version Cloud)
# ---------------------------------------------------------
# REMPLACE les valeurs ci-dessous par tes infos Qdrant Cloud
# Dans final_insertion.py
URL_CLOUD = "https://30185953-4cd2-4aa1-9f6d-c7c0fea9c53f.europe-west3-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.J40g5Fklv2RnF7Zj3t2So4lvPRy3Ngy-3pMmyAaB8Nc"

client = QdrantClient(
    url=URL_CLOUD, 
    api_key=API_KEY
)

embedder = EmbeddingService()
collection_name = "loan_memory"

# ---------------------------------------------------------
# 2. Création de la "mémoire" (Collection)
# ---------------------------------------------------------
print("Initialisation de la collection dans le Cloud...")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# ---------------------------------------------------------
# 3. Préparation et Nettoyage des données
# ---------------------------------------------------------
try:
    # On charge les données nettoyées
    df = pd.read_csv('data/credit_risk_cleaned.csv').head(500)
    print(f"Fichier chargé : {len(df)} lignes prêtes à l'insertion.")
except FileNotFoundError:
    print("Erreur : Le fichier 'data/credit_risk_cleaned.csv' est introuvable !")
    exit()

# ---------------------------------------------------------
# 4. Transformation en Vecteurs et Envoi
# ---------------------------------------------------------
print("Transformation des profils en vecteurs (Embeddings)...")
points = []

for index, row in df.iterrows():
    # Création d'une description textuelle du profil financier
    text_profile = f"Age: {row['person_age']}, Income: {row['person_income']}, Debt: {row['loan_amnt']}"
    
    # Transformation du texte en vecteur numérique par l'IA
    vector = embedder.generate_vector(text_profile)
    
    # Préparation du "Point" pour Qdrant
    points.append(PointStruct(
        id=index,
        vector=vector,
        payload=row.to_dict() # On garde toutes les infos (revenu, statut du prêt, etc.)
    ))

# Envoi groupé pour plus de rapidité
client.upsert(collection_name=collection_name, points=points)

print("---")
print(f"Succès ! {len(points)} dossiers ont été mémorisés dans SmartRisk Cloud.")
print(f"Tu peux vérifier tes données ici : {URL_CLOUD}/dashboard")