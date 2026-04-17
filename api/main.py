# Copyright (c) 2026 Kodjo Jean DEGBEVI. Tous droits réservés.
#========================================================
# Application FastAPI pour le Superstore Profit Predictor
#========================================================

import os
import sqlite3
import secrets
import contextlib
import logging
import gc
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ==========================================
# CONFIGURATION & CONSTANTES
# ==========================================
load_dotenv()
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
model_path = ROOT_DIR / "assets" / "exports" / "profit_predictor.joblib"
DB_DIR = CURRENT_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "api_keys.db"
LOG_DIR = CURRENT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID", "kjd-dktech/superstore-profit-predictor")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "").strip('"').strip("'")
ENV = os.getenv("ENV", "development").lower()

limiter = Limiter(key_func=get_remote_address)

# ==========================================
# JOURNALISATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api_activity.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_logger")

# ==========================================
# BASE DE DONNÉES
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            first_name TEXT DEFAULT '',
            last_name TEXT DEFAULT '',
            email TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            requests_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            tier TEXT DEFAULT 'free'
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==========================================
# SÉCURITÉ & DÉPENDANCES
# ==========================================
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        logger.warning("Tentative d'accès bloquée : Clé API manquante")
        raise HTTPException(status_code=403, detail="Une clé API est requise (Entête: X-API-KEY).")

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM api_keys WHERE key = ?", (api_key,))
    key_record = c.fetchone()
    
    if not key_record:
        conn.close()
        logger.warning(f"Tentative d'accès bloquée : Clé API invalide utilisée ({api_key[:8]}...)")
        raise HTTPException(status_code=403, detail="Clé API invalide.")
        
    if not key_record["is_active"]:
        conn.close()
        logger.warning(f"Tentative d'accès bloquée : Clé API désactivée ({key_record['email']})")
        raise HTTPException(status_code=403, detail="Cette clé API a été désactivée.")
    
    c.execute("UPDATE api_keys SET requests_count = requests_count + 1 WHERE id = ?", (key_record["id"],))
    conn.commit()
    conn.close()
    
    return key_record

def verify_admin(admin_key: str = Depends(APIKeyHeader(name="X-ADMIN-KEY"))):
    if admin_key != ADMIN_SECRET_KEY:
        logger.error("Tentative d'accès ADMINISTRATEUR refusée.")
        raise HTTPException(status_code=403, detail="Accès administrateur refusé.")
    return True

# ==========================================
# SCHÉMAS DE DONNÉES
# ==========================================
class SaleRecord(BaseModel):
    Sales: float
    Discount: float
    Sub_Category: str
    Region: str
    Segment: str

class PredictionResponse(BaseModel):
    predicted_profit: float

class BatchSaleRecord(BaseModel):
    records: List[SaleRecord]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]

# ==========================================
# LIFESPAN & INITIALISATION
# ==========================================
ml_model = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    logger.info("Démarrage de l'API Superstore...")
    if model_path.exists():
        logger.info("Chargement du modèle local...")
        ml_model = joblib.load(model_path)
    else:
        logger.info("Modèle local absent. Téléchargement depuis HuggingFace...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename="profit_predictor.joblib", 
                token=HF_TOKEN,
                local_dir=CURRENT_DIR / "predictor",
                cache_dir=CURRENT_DIR / "hf_cache"
            )
            ml_model = joblib.load(downloaded_path)
            logger.info("Modèle téléchargé et chargé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            ml_model = None
    yield
    logger.info("Arrêt de l'API...")
    del ml_model
    ml_model = None
    gc.collect()
    logger.info("Mémoire du modèle libérée.")

# ==========================================
# APPLICATION FASTAPI
# ==========================================
app = FastAPI(
    title="Superstore Profit Predictor API", 
    description="Microservice ML.",
    docs_url=None if ENV == "production" else "/docs",
    redoc_url=None if ENV == "production" else "/redoc",
    openapi_url=None if ENV == "production" else "/openapi.json",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================
# ROUTES : DÉVELOPPEURS & DOCS
# ==========================================
def get_developer_portal():
    return f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔑</text></svg>">
        <title>Portail Développeur - Predictor API</title>
        <style>
            body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            label {{ font-weight: 500; display: block; margin-top: 10px; }}
            input[type="email"], input[type="text"] {{ width: 100%; padding: 10px; margin: 5px 0 15px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
            button {{ background-color: #007bff; color: white; border: none; padding: 12px 20px; cursor: pointer; border-radius: 4px; width: 100%; font-size: 16px; transition: 0.2s; }}
            button:hover {{ background-color: #0056b3; }}
            button:disabled {{ background-color: #ccc; cursor: not-allowed; }}
            code {{ background-color: #eee; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .row {{ display: flex; gap: 15px; }}
            .col {{ flex: 1; }}
        </style>
    </head>
    <body>
        <h1>Superstore API - Espace Développeur</h1>
        <p>Obtenez une clé API pour intégrer nos prédictions de profit dans vos propres outils.</p>
        
        <div id="dynamicAlert"></div>

        <div class="card">
            <h3>Générer une clé API</h3>
            <form id="devForm">
                <div class="row">
                    <div class="col">
                        <label for="last_name">Nom :</label>
                        <input type="text" id="last_name" name="last_name" required placeholder="Votre nom">
                    </div>
                    <div class="col">
                        <label for="first_name">Prénoms :</label>
                        <input type="text" id="first_name" name="first_name" required placeholder="Vos prénoms">
                    </div>
                </div>
                <label for="email">Adresse Email :</label>
                <input type="email" id="email" name="email" required placeholder="nom@exemple.com">
                <button type="submit" id="submitBtn">Créer ma clé</button>
            </form>
        </div>
        
        <h3>Ressources</h3>
        <a href="/documentation" style="color: #007bff; text-decoration: none; font-weight: bold;">→ Accéder à la documentation officielle de l'API</a>
        
        <script>
            document.getElementById('devForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                btn.disabled = true;
                btn.innerText = 'Création en cours...';
                
                const formData = new FormData(e.target);
                try {{
                    const response = await fetch('/developer/generate', {{
                        method: 'POST',
                        body: formData
                    }});
                    const data = await response.json();
                    
                    if(response.ok) {{
                        const alertHtml = `
                        <div style="background-color: #f8f9fa; border: 1px solid #ced4da; color: #155724; padding: 15px; border-radius: 5px; margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <strong>Succès !</strong> ${{data.message}}<br><br>
                                Votre clé API : <code id="apiKeyStr" style="font-size: 110%; background: #fff; padding: 5px; border: 1px solid #ccc; user-select: all;">${{data.api_key}}</code>
                                <br><br><em style="font-size:0.9em; opacity:0.8;">(Copiez cette clé maintenant, elle n'est affichée qu'une seule fois)</em>
                            </div>
                            <button id="copyBtn" type="button" onclick="copyToClipboardGen()" style="width: 40px; height: 40px; border-radius: 50%; background-color: #e9ecef; border: 1px solid #ced4da; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s;" title="Copier la clé">
                                <div id="copyIcon" style="display: flex; align-items: center; justify-content: center;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#495057" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                    </svg>
                                </div>
                                <div id="copySuccess" style="display: none; align-items: center; justify-content: center; width: 100%; white-space: nowrap; font-weight: bold; color: #28a745;">
                                    ✅ Copié
                                </div>
                            </button>
                        </div>`;
                        document.getElementById('dynamicAlert').innerHTML = alertHtml;
                        e.target.reset();
                    }} else {{
                        alert("Erreur : " + (data.detail || "Impossible de créer la clé"));
                    }}
                }} catch(err) {{
                    alert("Erreur de connexion.");
                }} finally {{
                    btn.disabled = false;
                    btn.innerText = 'Créer ma clé';
                }}
            }});
            
            function copyToClipboardGen() {{
                var keyText = document.getElementById("apiKeyStr").innerText;
                navigator.clipboard.writeText(keyText).then(function() {{
                    var btn = document.getElementById("copyBtn");
                    var icon = document.getElementById("copyIcon");
                    var success = document.getElementById("copySuccess");
                    
                    btn.style.width = "90px";
                    btn.style.borderRadius = "20px";
                    icon.style.display = "none";
                    success.style.display = "flex";
                    
                    setTimeout(() => {{
                        success.style.display = "none";
                        icon.style.display = "flex";
                        btn.style.width = "40px";
                        btn.style.borderRadius = "50%";
                    }}, 2000);
                }});
            }}
        </script>
    </body>
    </html>
    """

@app.get("/developer", response_class=HTMLResponse)
async def developer_get():
    return HTMLResponse(content=get_developer_portal())

@app.post("/developer/generate")
@limiter.limit("5/minute")
async def generate_key(request: Request, first_name: str = Form(...), last_name: str = Form(...), email: str = Form(...)):
    new_key = "sk_" + secrets.token_hex(16)
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO api_keys (key, first_name, last_name, email) VALUES (?, ?, ?, ?)", 
              (new_key, first_name, last_name, email))
    conn.commit()
    conn.close()
    
    logger.info(f"Nouvelle clé générée via le portail développeur : {first_name} {last_name} ({email}).")
    msg = f"Clé générée pour {first_name} {last_name}."
    
    return {"message": msg, "api_key": new_key}


@app.get("/documentation", response_class=HTMLResponse)
async def api_documentation():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📚</text></svg>">
        <title>Documentation - Predictor API</title>
        <style>
            body { font-family: -apple-system, system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }
            h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            h3 { color: #007bff; margin-top: 25px; }
            code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; color: #d63384; font-size: 0.95em; }
            pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; border: 1px solid #dee2e6; }
            .endpoint { background: #eef2f5; padding: 10px; border-left: 4px solid #007bff; border-radius: 4px; font-weight: bold; margin-top: 15px; display: inline-block; width: 100%; box-sizing: border-box; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95em; }
            th, td { padding: 10px; border: 1px solid #dee2e6; text-align: left; }
            th { background: #e9ecef; }
            .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; background: #6c757d; color: white; }
        </style>
    </head>
    <body>
        <h1>Documentation Officielle - Superstore API</h1>
        <p>Bienvenue dans la documentation de l'API de prédiction de profit. L'API REST vous permet d'estimer les bénéfices commerciaux d'interventions de vente au détail grâce à notre modèle Machine Learning mis très gentiment à votre disposition.</p>
        
        <h2>Authentification et Limites</h2>
        <p>Toutes les requêtes vers les endpoints de Machine Learning doivent inclure votre clé API dans l'en-tête HTTP :</p>
        <pre>X-API-KEY: sk_votre_cle_secrete_ici</pre>
        <p><strong>Limites de requêtes (Rate Limits) :</strong></p>
        <ul>
            <li><strong>/predict :</strong> Limitée à 60 requêtes par minute par utilisateur.</li>
            <li><strong>/predict_batch :</strong> Limitée à 20 requêtes par minute par utilisateur. Le traitement par lot (batch) de plusieurs transactions est plus coûteux en ressources calcul, la limite est donc abaissée pour garantir la stabilité et la disponibilité de l'API pour tous les utilisateurs.</li>
        </ul>

        <h2>Format des données (Features)</h2>
        <p>Le corps JSON envoyé à l'API doit contenir les attributs exacts représentant les conditions de la transaction de vente :</p>
        <table>
            <thead>
                <tr>
                    <th>Attribut (Clé)</th>
                    <th>Type de donnée</th>
                    <th>Description & Valeurs valides</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>Sales</code></td>
                    <td>Numérique (Float)</td>
                    <td>Le montant total des ventes pour la transaction (ex: <code>261.96</code>). Doit être supérieur à 0.</td>
                </tr>
                <tr>
                    <td><code>Discount</code></td>
                    <td>Numérique (Float)</td>
                    <td>Le taux de réduction appliqué (ex: <code>0.15</code> pour 15%). Généralement compris entre <code>0.0</code> et <code>1.0</code>.</td>
                </tr>
                <tr>
                    <td><code>Sub_Category</code></td>
                    <td>Chaîne (String)</td>
                    <td>La sous-catégorie du produit. Valeurs courantes : <br><code>Bookcases</code>, <code>Chairs</code>, <code>Labels</code>, <code>Tables</code>, <code>Storage</code>, <code>Furnishings</code>, <code>Art</code>, <code>Phones</code>, <code>Binders</code>, <code>Appliances</code>, <code>Paper</code>, <code>Accessories</code>, <code>Envelopes</code>, <code>Fasteners</code>, <code>Supplies</code>, <code>Machines</code>, <code>Copiers</code>.</td>
                </tr>
                <tr>
                    <td><code>Region</code></td>
                    <td>Chaîne (String)</td>
                    <td>La région de vente. Valeurs possibles : <br><code>South</code>, <code>West</code>, <code>Central</code>, <code>East</code>.</td>
                </tr>
                <tr>
                    <td><code>Segment</code></td>
                    <td>Chaîne (String)</td>
                    <td>Le type de segment client. Valeurs possibles : <br><code>Consumer</code>, <code>Corporate</code>, <code>Home Office</code>.</td>
                </tr>
            </tbody>
        </table>

        <h2>Endpoints Disponibles</h2>
        
        <div class="endpoint">POST /predict</div>
        <p>Estime le profit pour une seule transaction métier.</p>
        <strong>Corps de la requête (JSON) :</strong>
        <pre>{
  "Sales": 261.96,
  "Discount": 0.0,
  "Sub_Category": "Bookcases",
  "Region": "South",
  "Segment": "Consumer"
}</pre>
        <strong>Réponse :</strong>
        <pre>{
  "predicted_profit": 41.91
}</pre>

        <div class="endpoint">POST /predict_batch</div>
        <p>Estime le profit pour une liste de transactions (Max 20 req/min).</p>
        <strong>Corps de la requête (JSON) :</strong>
        <pre>{
  "records": [
    {
      "Sales": 261.96,
      "Discount": 0.0,
      "Sub_Category": "Bookcases",
      "Region": "South",
      "Segment": "Consumer"
    },
    {
      "Sales": 731.94,
      "Discount": 0.2,
      "Sub_Category": "Chairs",
      "Region": "West",
      "Segment": "Corporate"
    }
  ]
}</pre>
        <strong>Réponse :</strong>
        <pre>{
  "predictions": [41.91, 120.45]
}</pre>
        <br>
        <p><a href="/developer" style="color: #007bff; text-decoration: none;">&larr; Retour au portail développeur</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==========================================
# ROUTES : MACHINE LEARNING
# ==========================================
@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": ml_model is not None}

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute") 
def predict_profit(request: Request, record: SaleRecord, key_info: dict = Depends(verify_api_key)):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    
    input_df = pd.DataFrame([{
        'Sales': record.Sales, 'Discount': record.Discount, 
        'Sub-Category': record.Sub_Category, 'Region': record.Region, 'Segment': record.Segment
    }])
    pred_log = ml_model.predict(input_df)[0]
    pred_profit = np.sign(pred_log) * np.expm1(np.abs(pred_log))
    
    logger.info(f"Prédiction réussie (user: {key_info['email']} | requête: single)")
    return {"predicted_profit": float(pred_profit)}

@app.post("/predict_batch", response_model=BatchPredictionResponse)
@limiter.limit("20/minute") 
def predict_batch_profit(request: Request, batch: BatchSaleRecord, key_info: dict = Depends(verify_api_key)):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    try:
        import pandas as pd
        import numpy as np
        df = pd.DataFrame([r.model_dump() for r in batch.records])
        df = df.rename(columns={'Sub_Category': 'Sub-Category'})
        df = df[['Sales', 'Discount', 'Sub-Category', 'Region', 'Segment']]
        logs_pred = ml_model.predict(df)
        preds = np.sign(logs_pred) * np.expm1(np.abs(logs_pred))
        logger.info(f"Prédiction batch réussie (user: {key_info['email']} | taille: {len(df)})")
        return {"predictions": preds.tolist()}
    except Exception as e:
        logger.error(f"Erreur modèle batch: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction en lot")

def get_admin_login(error_msg=""):
    err_html = f"<p style='color:red;'>{error_msg}</p>" if error_msg else ""
    return f'''
    <!DOCTYPE html>
    <html><head><title>Admin Login - API</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>📊</text></svg>">
    <style>body{{font-family:sans-serif; max-width: 400px; margin: 50px auto; text-align:center;}}</style>
    </head><body>
    <h2>Accès Administrateur</h2>
    {err_html}
    <form action="/admin/dashboard" method="post">
        <input type="password" name="admin_key" placeholder="Clé secrète admin" required style="width:100%; padding:10px; margin-bottom:10px; box-sizing:border-box;">
        <button type="submit" style="width:100%; padding:10px; background:#007bff; color:#fff; border:none; cursor:pointer;">Se connecter</button>
    </form>
    </body></html>
    '''

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard_get():
    return HTMLResponse(content=get_admin_login())

@app.post("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard_post(request: Request, admin_key: str = Form(...)):
    if admin_key != str(ADMIN_SECRET_KEY):
        logger.warning("Echec de connexion à l'interface d'administration.")
        return HTMLResponse(content=get_admin_login("Clé secrète incorrecte."))
    
    html = f'''
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard Admin - API</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>⚡</text></svg>">
        <style>
            body{{font-family: -apple-system, system-ui, sans-serif; margin: 20px; color: #333; background: #f8f9fa;}}
            .header{{display:flex; justify-content:space-between; align-items:center; background:#fff; padding:15px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); margin-bottom:20px;}}
            .panel{{background:#fff; padding:20px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1);}}
            table{{width: 100%; border-collapse: collapse; margin-top:20px; font-size:14px;}}
            th, td{{padding: 12px; border-bottom: 1px solid #dee2e6; text-align: left;}}
            th{{background: #e9ecef; font-weight:600;}}
            code{{background:#f1f3f5; padding:3px 6px; border-radius:4px; font-family:monospace; color:#d63384;}}
            .badge-active{{background:#d4edda; color:#155724; padding:3px 8px; border-radius:12px; font-size:0.85em;}}
            .badge-inactive{{background:#f8d7da; color:#721c24; padding:3px 8px; border-radius:12px; font-size:0.85em;}}
            .btn{{padding:6px 12px; border:none; border-radius:4px; cursor:pointer; font-size:0.9em; transition:0.2s;}}
            .btn-primary{{background:#007bff; color:white;}}
            .btn-primary:hover{{background:#0056b3;}}
            .btn-danger{{background:#dc3545; color:white;}}
            .btn-danger:hover{{background:#c82333;}}
            .btn-warning{{background:#ffc107; color:#212529;}}
            .btn-secondary{{background:#6c757d; color:white;}}
            select, input{{padding:8px; border:1px solid #ced4da; border-radius:4px;}}
            /* MODAL STYLES */
            .modal {{display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); align-items:center; justify-content:center; z-index:1000;}}
            .modal-content {{background:#fff; padding:20px; border-radius:8px; width:400px; max-width:90%; position:relative; box-shadow:0 4px 6px rgba(0,0,0,0.1);}}
            .modal-content h3 {{margin-top:0; border-bottom:1px solid #eee; padding-bottom:10px;}}
            .form-group {{margin-bottom:15px; text-align:left;}}
            .form-group label {{display:block; margin-bottom:5px; font-weight:500;}}
            .form-group input, .form-group select {{width:100%; box-sizing:border-box;}}
            .modal-actions {{display:flex; justify-content:flex-end; gap:10px; margin-top:20px;}}
        </style>
    </head>
    <body>
        <!-- MODAL HTML -->
        <div id="keyModal" class="modal">
            <div class="modal-content">
                <h3 id="modalTitle">Nouvelle Clé</h3>
                <div class="form-group">
                    <label>Prénom</label>
                    <input type="text" id="mFirstName" placeholder="Prénom">
                </div>
                <div class="form-group">
                    <label>Nom</label>
                    <input type="text" id="mLastName" placeholder="Nom">
                </div>
                <div class="form-group">
                    <label>Email</label>
                    <input type="email" id="mEmail" placeholder="Email">
                </div>
                <div class="form-group">
                    <label>Tier</label>
                    <select id="mTier">
                        <option value="free">Free</option>
                        <option value="premium">Premium</option>
                        <option value="enterprise">Enterprise</option>
                    </select>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="closeModal()">Annuler</button>
                    <button class="btn btn-primary" onclick="saveKey()">Enregistrer</button>
                </div>
            </div>
        </div>

        <div class="header">
            <h2 style="margin:0;">⚡ Console d'Administration API</h2>
            <div>
                <button class="btn btn-primary" onclick="openModal(false)">+ Nouvelle Clé</button>
            </div>
        </div>
        
        <div class="panel">
            <div style="display:flex; gap:15px; margin-bottom:15px; align-items:center;">
                <select id="filterColumn" onchange="renderTable()" style="width: auto;">
                    <option value="all">📝 Toutes les colonnes</option>
                    <option value="first_name">Prénom</option>
                    <option value="last_name">Nom</option>
                    <option value="email">Email</option>
                    <option value="key">Clé</option>
                    <option value="tier">Tier</option>
                </select>
                <input type="text" id="filterValue" placeholder="Rechercher (ex: nom, email...)" onkeyup="renderTable()" style="flex:1; max-width:300px;">
                <span id="stats" style="margin-left:auto; font-weight:bold; color:#6c757d;"></span>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Utilisateur</th>
                        <th>Email</th>
                        <th>Clé (Aperçu)</th>
                        <th>Tier</th>
                        <th>Requêtes</th>
                        <th>Statut</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="keysBody">
                    <tr><td colspan="8" style="text-align:center;">Chargement des données...</td></tr>
                </tbody>
            </table>
        </div>

        <script>
            const ADMIN_KEY = "{admin_key}";
            let allKeys = [];

            async function apiCall(endpoint, method = 'GET', body = null) {{
                const options = {{
                    method: method,
                    headers: {{
                        'X-ADMIN-KEY': ADMIN_KEY,
                        'Content-Type': 'application/json'
                    }}
                }};
                if(body) options.body = JSON.stringify(body);
                try {{
                    const res = await fetch(endpoint, options);
                    if(!res.ok) throw new Error(await res.text());
                    return await res.json();
                }} catch(e) {{
                    alert("Erreur: " + e.message);
                    throw e;
                }}
            }}

            async function loadKeys() {{
                const data = await apiCall('/admin/keys');
                allKeys = data.keys || [];
                renderTable();
            }}

            function renderTable() {{
                const col = document.getElementById('filterColumn').value;
                const val = document.getElementById('filterValue').value.toLowerCase();
                
                let keysToRender = allKeys.filter(k => {{
                    if (!val) return true;
                    if (col === 'all') {{
                        return (k.first_name||'').toLowerCase().includes(val) || 
                               (k.last_name||'').toLowerCase().includes(val) || 
                               (k.email||'').toLowerCase().includes(val) || 
                               (k.key||'').toLowerCase().includes(val) ||
                               (k.tier||'').toLowerCase().includes(val);
                    }} else {{
                        return (k[col]||'').toLowerCase().includes(val);
                    }}
                }});

                const tbody = document.getElementById('keysBody');
                if(keysToRender.length === 0) {{
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Aucune clé trouvée.</td></tr>';
                }} else {{
                    tbody.innerHTML = keysToRender.map((k, index) => `
                        <tr>
                            <td>#${{index + 1}}</td>
                            <td><strong>${{k.first_name}} ${{k.last_name}}</strong></td>
                            <td>${{k.email}}</td>
                            <td title="${{k.key}}"><code>${{k.key.substring(0,8)}}...</code></td>
                            <td><span style="text-transform:capitalize">${{k.tier}}</span></td>
                            <td><strong>${{k.requests_count}}</strong></td>
                            <td>${{k.is_active ? '<span class="badge-active">Actif</span>' : '<span class="badge-inactive">Inactif</span>'}}</td>
                            <td>
                                <button class="btn ${{k.is_active ? 'btn-secondary' : 'btn-primary'}}" onclick="toggleKey(${{k.id}}, ${{k.is_active}})">
                                    ${{k.is_active ? 'Désactiver' : 'Activer'}}
                                </button>
                                <button class="btn btn-warning" onclick="openModal(true, ${{k.id}})">✏️ Editer</button>
                                <button class="btn btn-danger" onclick="deleteKey(${{k.id}})">🗑️ Sup.</button>
                            </td>
                        </tr>
                    `).join('');
                }}
                
                document.getElementById('stats').innerText = keysToRender.length + " clés affichées au total (" + allKeys.reduce((acc, k) => acc + k.requests_count, 0) + " requêtes cumulées)";
            }}

            async function toggleKey(id, isActive) {{
                const action = isActive ? 'deactivate' : 'activate';
                await apiCall(`/admin/keys/${{id}}/${{action}}`, 'PATCH');
                loadKeys();
            }}

            async function deleteKey(id) {{
                if(confirm("Êtes-vous sûr de vouloir supprimer définitivement cette clé ? Cette action est irréversible.")) {{
                    await apiCall(`/admin/keys/${{id}}`, 'DELETE');
                    loadKeys();
                }}
            }}

            let currentEditId = null;

            function openModal(isEdit, id = null) {{
                const modal = document.getElementById('keyModal');
                modal.style.display = 'flex';
                
                if (isEdit && id) {{
                    const k = allKeys.find(x => x.id === id);
                    if(!k) return closeModal();
                    document.getElementById('modalTitle').innerText = 'Modifier la clé';
                    document.getElementById('mFirstName').value = k.first_name;
                    document.getElementById('mLastName').value = k.last_name;
                    document.getElementById('mEmail').value = k.email;
                    document.getElementById('mTier').value = k.tier;
                    currentEditId = k.id;
                }} else {{
                    document.getElementById('modalTitle').innerText = 'Nouvelle Clé';
                    document.getElementById('mFirstName').value = '';
                    document.getElementById('mLastName').value = '';
                    document.getElementById('mEmail').value = '';
                    document.getElementById('mTier').value = 'free';
                    currentEditId = null;
                }}
            }}

            function closeModal() {{
                document.getElementById('keyModal').style.display = 'none';
            }}

            async function saveKey() {{
                const newFn = document.getElementById('mFirstName').value.trim();
                const newLn = document.getElementById('mLastName').value.trim();
                const newEm = document.getElementById('mEmail').value.trim();
                const newTi = document.getElementById('mTier').value;

                if (!newFn || !newLn || !newEm) {{
                    alert("Veuillez remplir tous les champs.");
                    return;
                }}
                
                try {{
                    if (currentEditId) {{
                        await apiCall(`/admin/keys/${{currentEditId}}`, 'PUT', {{
                            first_name: newFn, 
                            last_name: newLn, 
                            email: newEm, 
                            tier: newTi
                        }});
                    }} else {{
                        await apiCall(`/admin/keys/create`, 'POST', {{
                            first_name: newFn,
                            last_name: newLn,
                            email: newEm,
                            tier: newTi
                        }});
                    }}
                    closeModal();
                    loadKeys();
                }} catch(e) {{
                    console.error("Erreur lors de l'enregistrement :", e);
                }}
            }}

            window.onload = loadKeys;
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(html)

class AdminKeyInput(BaseModel):
    first_name: str
    last_name: str
    email: str
    tier: str

@app.post("/admin/keys/create")
def admin_create_key(data: AdminKeyInput, is_admin: bool = Depends(verify_admin)):
    new_key = "sk_" + secrets.token_hex(16)
    conn = get_db_connection()
    conn.cursor().execute("INSERT INTO api_keys (key, first_name, last_name, email, tier) VALUES (?, ?, ?, ?, ?)", 
                          (new_key, data.first_name, data.last_name, data.email, data.tier))
    conn.commit()
    conn.close()
    logger.info(f"ADMIN: Nouvelle clé créée manuellement pour {data.email} (Tier: {data.tier})")
    return {"message": "Clé créée", "email": data.email, "key": new_key, "tier": data.tier}

@app.get("/admin/keys")
def admin_list_keys(is_admin: bool = Depends(verify_admin)):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, key, first_name, last_name, email, created_at, requests_count, is_active, tier FROM api_keys ORDER BY requests_count DESC")
    keys = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"total_keys": len(keys), "keys": keys}

@app.put("/admin/keys/{key_id}")
def admin_update_key(key_id: int, data: AdminKeyInput, is_admin: bool = Depends(verify_admin)):
    conn = get_db_connection()
    conn.cursor().execute("UPDATE api_keys SET first_name=?, last_name=?, email=?, tier=? WHERE id=?", 
                          (data.first_name, data.last_name, data.email, data.tier, key_id))
    conn.commit()
    conn.close()
    logger.info(f"ADMIN: Informations de la clé n°{key_id} mises à jour.")
    return {"message": f"Clé {key_id} mise à jour."}

@app.delete("/admin/keys/{key_id}")
def admin_delete_key(key_id: int, is_admin: bool = Depends(verify_admin)):
    conn = get_db_connection()
    conn.cursor().execute("DELETE FROM api_keys WHERE id=?", (key_id,))
    conn.commit()
    conn.close()
    logger.info(f"ADMIN: Clé n°{key_id} SUPPRIMÉE de la base de données.")
    return {"message": f"Clé {key_id} supprimée définitivement."}

@app.patch("/admin/keys/{key_id}/deactivate")
def admin_deactivate_key(key_id: int, is_admin: bool = Depends(verify_admin)):
    conn = get_db_connection()
    conn.cursor().execute("UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,))
    conn.commit()
    conn.close()
    logger.info(f"ADMIN: Clé n°{key_id} désactivée.")
    return {"message": f"Clé {key_id} désactivée avec succès."}

@app.patch("/admin/keys/{key_id}/activate")
def admin_activate_key(key_id: int, is_admin: bool = Depends(verify_admin)):
    conn = get_db_connection()
    conn.cursor().execute("UPDATE api_keys SET is_active = 1 WHERE id = ?", (key_id,))
    conn.commit()
    conn.close()
    logger.info(f"ADMIN: Clé n°{key_id} activée.")
    return {"message": f"Clé {key_id} activée avec succès."}
