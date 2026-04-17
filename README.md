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
- **Streamlit** — dashboard interactif
- **Données** — Superstore Dataset (Kaggle, 9 994 transactions, 2014–2017)

## 🚀 Lancer le dashboard

1. **Cloner le projet** et installer les dépendances :
```bash
pip install -r requirements.txt
```

2. **Configurer l'environnement** :
Créer un fichier `.env` à la racine du projet et y insérer votre token de lecture Hugging Face :
```env
HF_TOKEN=votre_token_lecture_ici
```

1. **Lancer l'application** :
```bash
streamlit run app/main.py
```

## 📁 Structure du projet

```text
├── app/
│   └── main.py                 # Application Streamlit
├── assets/exports/             # Modèles ML exportés
├── data/
│   ├── processed/              # Données nettoyées et prétraitées
│   └── raw/                    # Données brutes
├── notebook/
│   └── ml_modeling.ipynb       # Analyse exploratoire et entraînement des modèles
├── requirements.txt            # Dépendances du projet
└── README.md                   # Documentation interactive
```

## 📈 Résultats clés

- **Cartographie des profits** : Identification immédiate des États "Top Rentables" (générateurs de valeur) face aux États "Flop" (destructeurs de valeur).
- **Seuil de rentabilité (Remises)** : Visualisation claire démontrant au-delà de quel taux de "Discount" la Marge Nette plonge dans le négatif.
- **Valeur Client (LTV)** : Classement des segments de clientèle selon leur profit cumulé ("Profit par Client") et par commande.
- **Simulateur Prédictif (Random Forest)** : Modèle de Machine Learning intégré (importé dynamiquement depuis Hugging Face) permettant d'estimer unitairement ou par lots le profit selon les paramètres de la vente.

## 👤 Auteur

Kodjo Jean DEGBEVI — [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) — [GitHub](https://github.com/kjd-dktech)