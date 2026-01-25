# 🛡️ SmartRisk : Mémoire de Crédit Intelligente

SmartRisk est une application d'évaluation du risque de crédit basée sur la **similarité vectorielle**. Contrairement aux systèmes classiques, il utilise l'IA pour comparer un nouveau candidat à des profils historiques stockés dans une base de données vectorielle.

## 🚀 Fonctionnalités
* **Vectorisation des données** : Transformation des profils financiers en vecteurs mathématiques (via Sentence-Transformers).
* **Mémoire Vectorielle** : Recherche ultra-rapide de profils similaires avec **Qdrant**.
* **Score de Risque** : Calcul automatique du risque basé sur le comportement passé de profils similaires.
* **API Fastly** : Interface moderne pour interroger le moteur de risque en temps réel.

## 📁 Structure du Projet
* `src/api` : Point d'entrée de l'application (FastAPI).
* `src/core` : Moteur de décision (RiskEngine).
* `src/services` : Services d'IA (Embedding) et de base de données (Qdrant).
* `data/` : Stockage des données brutes et traitées.

## 🛠️ Installation
1. Cloner le projet :
   ```bash
   git clone [https://github.com/zainebmami-source/smartrisk-credit-memory.git](https://github.com/zainebmami-source/smartrisk-credit-memory.git)