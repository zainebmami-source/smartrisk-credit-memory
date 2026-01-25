from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Ce modèle transforme du texte/chiffres en un vecteur de taille 384
        # Il est téléchargé automatiquement la première fois
        self.model = SentenceTransformer(model_name)

    def generate_vector(self, applicant_data: dict):
        """
        Transforme un dictionnaire de données (ex: {'revenu': 3000, 'dette': 500})
        en un vecteur mathématique.
        """
        # On convertit le dictionnaire en une seule chaîne de texte
        # Exemple : "revenu: 3000, dette: 500"
        text_data = ", ".join([f"{key}: {value}" for key, value in applicant_data.items()])
        
        # On génère le vecteur à partir de ce texte
        return self.model.encode(text_data).tolist()