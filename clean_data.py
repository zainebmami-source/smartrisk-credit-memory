import pandas as pd

# 1. Charger les données
df = pd.read_csv('data/credit_risk_dataset.csv')
print(f"Lignes avant nettoyage : {len(df)}")

# 2. Supprimer les doublons
df = df.drop_duplicates()

# 3. Gérer les valeurs manquantes (NaN)
# On remplace les revenus manquants par la moyenne et le taux d'intérêt par la médiane
df['person_income'] = df['person_income'].fillna(df['person_income'].mean())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# 4. Supprimer les lignes où il manque encore des informations critiques
df = df.dropna()

# 5. Filtrer les valeurs aberrantes (Outliers)
# Exemple : On enlève les gens qui ont plus de 100 ans (probablement une erreur de saisie)
df = df[df['person_age'] < 100]

# 6. Sauvegarder le fichier propre
df.to_csv('data/credit_risk_cleaned.csv', index=False)
print(f"Lignes après nettoyage : {len(df)}")
print("Fichier 'credit_risk_cleaned.csv' créé avec succès !")