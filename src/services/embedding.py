import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force Python à ignorer ta carte graphique
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Ce modèle transforme du texte/chiffres en un vecteur de taille 384
        # Il est téléchargé automatiquement la première fois
        self.model = SentenceTransformer(model_name)

    def generate_vector(self, applicant_data):
        # Il faut 4 espaces ici
        if isinstance(applicant_data, str):
            text_data = applicant_data
        else:
            text_data = ", ".join([f"{key}: {value}" for key, value in applicant_data.items()])
        
        return self.model.encode(text_data).tolist()