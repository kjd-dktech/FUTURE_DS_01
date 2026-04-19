# 🚀 Superstore Profit Predictor - Microservice API

Ce répertoire correspond au backend de Machine Learning pour l'application d'analyse Superstore. Ce service est entièrement découplé du dashboard Streamlit.

## 🎯 Architecture
- **Serveur** : [FastAPI](https://fastapi.tiangolo.com)
- **Rate-Limiting** : [SlowAPI](https://slowapi.readthedocs.io/en/latest/) avec prise en charge **Redis** pour un comptage persistant et distribué.
- **Data & Model** : Pandas, NumPy, Scikit-Learn -> Exportés via Joblib et stockés sur [Hugging Face Hub](https://huggingface.co/).
- **Sécurité & DB** : Mots de passe et clés hachés et sécurisés via la librairie standard native bcrypt dans SQLite (`db/api_keys.db`). Cookies Admin protégés par chiffrement symétrique AES (**Fernet**).

## 🛠️ Installation et Lancement Local

1. Configurez la sécurité via un fichier `.env` à la racine du projet (ajoutez impérativement votre `ADMIN_SECRET_KEY` ainsi que `REDIS_URL` si nécessaire).
2. Assurez-vous d'être dans le dossier `api/` puis installez les dépendances :
   ```bash
   cd api
   pip install -r requirements.txt
   ```
3. Démarrez l'API :
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

*(Interfaces web disponibles après lancement :)*
- 🔑 **Portail Développeur** : `http://127.0.0.1:8000/developer` (Création de clés)
- ⚡ **Dashboard Admin** : `http://127.0.0.1:8000/admin/dashboard` (Console de gestion sécurisée)
- 📚 **Swagger UI** : `http://127.0.0.1:8000/docs` (Disponible uniquement en environnement de développement)
- 📚 **Documentation** : `http://127.0.0.1:8000/documentation`

## 🐋 Déploiement via Docker

Pour des raisons de consistance en production (sur un VPS, un cluster Kubernetes ou via le repository Docker officiel de Hugging Face Spaces), vous pouvez isoler l'API à l'aide de Docker.

1. Construire l'image :
   ```bash
   docker build -t superstore-profit-predictor-api .
   ```
2. Lancer le conteneur en associant le port local :
   ```bash
   docker run -p 8000:8000 superstore-profit-predictor-api
   ```
> 💡 *Cette image s'exécute avec les autorisations de l'utilisateur standard (non-root id=1000) et crée logiquement ses sous-dossiers locaux (`db/` et `logs/`) au sein de son scope, la rendant très facilement intégrable sur Kubernetes ou Hugging Face Spaces.*

## 👤 Auteur

Kodjo Jean DEGBEVI — [LinkedIn](https://www.linkedin.com/in/kodjo-jean-degbevi-ba5170369) — [GitHub](https://github.com/kjd-dktech)

