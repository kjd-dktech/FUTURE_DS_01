# 📊 Superstore Sales Analytics

Analyse des performances commerciales d'un retailer américain (2014–2017).  

## 🎯 Mission

Une grande enseigne de distribution américaine cherche à identifier
ses stratégies les plus performantes sur 2014–2017.

**Objectif** : déterminer quels produits, régions, catégories et segments
cibler ou éviter — et produire des recommandations actionnables pour la direction.

**5 questions directrices** : santé financière globale, performance géographique,
valeur par catégorie, impact des remises, rentabilité par segment client.

## 🛠️ Stack technique

- **Python** — pandas, plotly
- **FastAPI** — API Microservice Machine Learning
- **Streamlit** — Dashboard interactif découplé
- **Données** — Superstore Dataset (Kaggle, 9 994 transactions, 2014–2017)

## 🚀 Déploiement et Lancement

Le projet est entièrement modulaire. Le front-end (Streamlit) utilise une API de Machining Learning (FastAPI) pour l'inférence prédictive continue par unité ou Batch.

1. **Pré-requis & Installation globale** :
```bash
pip install -r requirements.txt
```

2. **Configuration de la Sécurité (.env)** :
Copiez le fichier d'exemple pour configurer vos variables de production :
```bash
cp .env.exemple .env
# Éditez le .env pour y définir STREAMLIT_COOKIE_SECRET, ADMIN_SECRET_KEY, REDIS_URL...
```

3. **Étape 1 : Lancer l'API (Backend)**
Depuis la racine du projet :
```bash
# Note : Pour le Rate-Limiting distribué, assurez-vous qu'un serveur Redis tourne (optionnel).
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
# Accédez à http://127.0.0.1:8000/developer pour créer votre X-API-KEY
# Accédez à http://127.0.0.1:8000/admin/dashboard pour le panneau d'administration sécurisé
```

4. **Étape 2 : Lancer le Dashboard (Frontend)**
Dans un nouveau terminal depuis la racine :
```bash
streamlit run app/main.py
```
*Le tableau de bord apparaîtra et réclamera la clé API que vous avez provisionnée côté backend API dans le menu latéral.*

## 📁 Structure du projet *(contenus clés)*

```text
├── api/                           # Microservice Backend FastAPI (Prédictions ML)
│   ├── main.py                    # Serveur, Routes
│   ├── requirements.txt           # Dépendances Backend spécifiques
│   └── Dockerfile                 # Conteneurisation de l'API
├── app/                           # Application Frontend Streamlit
│   └── main.py                    # Formulaire, Tableau de bord, Requêtes HTTP
├── assets/exports/                # Modèle ML exporté
├── data/
│   ├── processed/                 # Données nettoyées (Dashboard)
│   └── raw/                       # Source Kaggle originale
├── notebook/
│   ├── data_analysis.ipynb        # Analyse et Insights
│   └── ml_modeling.ipynb          # Recherche : EDA, Random Forest, Export Joblib
├── requirements.txt               # Pile de dépendances globale (Frontend/ML)
└── README.md                      # Documentation root
```

## 📈 Résultats clés

- **Cartographie des profits** : Identification immédiate des États "Top Rentables" (générateurs de valeur) face aux États "Flop" (destructeurs de valeur).
- **Seuil de rentabilité (Remises)** : Visualisation claire démontrant au-delà de quel taux de "Discount" la Marge Nette plonge dans le négatif.
- **Valeur Client (LTV)** : Classement des segments de clientèle selon leur profit cumulé ("Profit par Client") et par commande.
- **Simulateur Prédictif (Random Forest)** : Modèle de Machine Learning intégré (importé dynamiquement depuis Hugging Face par le microservice) permettant d'estimer unitairement ou par lots le profit via l'API.

## 👤 Auteur

Kodjo Jean DEGBEVI — [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) — [GitHub](https://github.com/kjd-dktech)
