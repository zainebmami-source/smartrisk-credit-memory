from src.services.embedding import EmbeddingService
from src.services.qdrant_db import QdrantService

class RiskEngine:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()

    def evaluate_risk(self, applicant_data: dict):
        """
        Calcule le score de risque en comparant le candidat 
        aux dossiers historiques.
        """
        # 1. Transformer les données du candidat en vecteur
        vector = self.embedding_service.generate_vector(applicant_data)

        # 2. Chercher les profils similaires dans Qdrant
        # Note: on cherche les 5 profils les plus proches
        similar_profiles = self.qdrant_service.client.search(
            collection_name="loan_history",
            query_vector=vector,
            limit=5
        )

        # 3. Calculer la moyenne de risque des profils similaires
        if not similar_profiles:
            return {"score": 0.5, "status": "Inconnu (pas d'historique)"}

        # On fait une moyenne simple des scores de risque trouvés
        total_risk = sum([hit.payload.get("risk_score", 0) for hit in similar_profiles])
        final_score = total_risk / len(similar_profiles)

        return {
            "score": round(final_score, 2),
            "status": "Risque Élevé" if final_score > 0.7 else "Risque Faible",
            "matches": len(similar_profiles)
        }