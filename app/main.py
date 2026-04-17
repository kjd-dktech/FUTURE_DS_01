# Copyright (c) 2026 Kodjo Jean DEGBEVI. Tous droits réservés.
#=============================================================
# Application Streamlit pour l'analyse du Superstore
#=============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import requests
from pathlib import Path
import io
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CURRENT_FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_FILE_DIR.parent
data_path = ROOT_DIR / "data" / "processed" / "superstore_processed.csv"
sys.path.append(str(ROOT_DIR))

from src.utils import get_us_state_abbrev

st.set_page_config(page_title="Superstore Analytics", page_icon="📊", layout="wide")

# --- HISTORIQUE EN SESSION ---
history_dir = ROOT_DIR / "data" / "logs"
os.makedirs(history_dir, exist_ok=True)
sim_history_file = history_dir / "sim_history.csv"
batch_history_file = history_dir / "batch_history.csv"
api_key_file = history_dir / ".api_key"

if 'sim_history' not in st.session_state:
    if sim_history_file.exists():
        try:
            st.session_state['sim_history'] = pd.read_csv(sim_history_file).to_dict('records')
        except:
            st.session_state['sim_history'] = []
    else:
        st.session_state['sim_history'] = []

if 'batch_history' not in st.session_state:
    if batch_history_file.exists():
        try:
            st.session_state['batch_history'] = pd.read_csv(batch_history_file).to_dict('records')
        except:
            st.session_state['batch_history'] = []
    else:
        st.session_state['batch_history'] = []

if 'api_key_val' not in st.session_state:
    if api_key_file.exists():
        st.session_state['api_key_val'] = api_key_file.read_text().strip()
    else:
        st.session_state['api_key_val'] = ""

def save_api_key():
    val = st.session_state.get('api_input', '')
    st.session_state['api_key_val'] = val
    if val:
        api_key_file.write_text(val)
    else:
        if api_key_file.exists():
            api_key_file.unlink()

def save_history_to_disk(history_type):
    if history_type == 'sim':
        pd.DataFrame(st.session_state['sim_history']).to_csv(sim_history_file, index=False)
    elif history_type == 'batch':
        pd.DataFrame(st.session_state['batch_history']).to_csv(batch_history_file, index=False)

def read_data_robust(file_obj, file_name=""):
    if file_name.endswith('.json'):
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_json(file_obj)
    elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_excel(file_obj)
        
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'latin1', 'utf-16', 'utf-8-sig', 'mac_roman']
    for enc in encodings:
        try:
            if hasattr(file_obj, 'seek'): file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Impossible de décoder le fichier CSV. Encodages tentés: {', '.join(encodings)}")

def check_api_status():
    try:
        res = requests.get(f"{API_URL}/", timeout=3)
        return res.status_code == 200 and res.json().get("model_loaded", False)
    except:
        return False

@st.cache_data
def load_data():
    df = read_data_robust(data_path, "superstore_processed.csv")
        
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Order Year'] = df['Order Date'].dt.year
    return df

df = load_data()
api_ready = check_api_status()

# --- SIDEBAR ---
st.sidebar.header("Configuration API")
api_key = st.sidebar.text_input(
    "Clé API", 
    type="password", 
    key="api_input", 
    value=st.session_state['api_key_val'], 
    on_change=save_api_key,
    help="Appuyez sur Entrée pour valider. Obtenez-la sur http://127.0.0.1:8000/developer"
)

if not api_ready:
    st.sidebar.warning("❌ L'API de prédiction ne semble pas être en ligne.")
else:
    st.sidebar.success("✅ Modèle API en Ligne")
st.sidebar.markdown("---")

st.sidebar.header("Filtres")

selected_years = st.sidebar.multiselect(
    "Années", 
    options=sorted(df['Order Year'].unique()) if 'Order Year' in df.columns else [], 
    default=sorted(df['Order Year'].unique()) if 'Order Year' in df.columns else []
)

selected_regions = st.sidebar.multiselect(
    "Régions", 
    options=sorted(df['Region'].unique()), 
    default=sorted(df['Region'].unique())
)

selected_categories = st.sidebar.multiselect(
    "Catégories", 
    options=sorted(df['Category'].unique()) if 'Category' in df.columns else [], 
    default=sorted(df['Category'].unique()) if 'Category' in df.columns else []
)

if 'Order Year' in df.columns and not selected_years: selected_years = df['Order Year'].unique()
if not selected_regions: selected_regions = df['Region'].unique()
if 'Category' in df.columns and not selected_categories: selected_categories = df['Category'].unique()

mask = df['Region'].isin(selected_regions)
if 'Order Year' in df.columns:
    mask &= df['Order Year'].isin(selected_years)
if 'Category' in df.columns:
    mask &= df['Category'].isin(selected_categories)
    
filtered_df = df[mask]

# --- HEADER ET KPIS ---
st.title("📊 Dashboard - Superstore")

col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df['Sales'].sum() if 'Sales' in filtered_df.columns else 0
total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
marge_globale = total_profit / total_sales if total_sales > 0 else 0
total_customers = filtered_df['Customer ID'].nunique() if 'Customer ID' in filtered_df.columns else 0

# Amplitude temporelle
if not filtered_df.empty and 'Order Date' in filtered_df.columns:
    date_min = filtered_df['Order Date'].min()
    date_max = filtered_df['Order Date'].max()
    days = max(1, (date_max - date_min).days)
    nb_annees = max(1, days / 365.25)
else:
    nb_annees = 1

avg_sales = total_sales / nb_annees
avg_profit = total_profit / nb_annees
avg_customers = total_customers / nb_annees

col1.metric("Chiffre d'Affaires", f"${total_sales:,.0f}", f"{avg_sales:,.0f} $ / an", delta_color="off")
col2.metric("Bénéfice Net", f"${total_profit:,.0f}", f"{avg_profit:,.0f} $ / an", delta_color="off")
col3.metric("Marge Globale", f"{marge_globale:.2%}")
col4.metric("Clients Uniques", f"{total_customers:,}", f"~ {int(avg_customers):,} / an", delta_color="off")

st.markdown("---")

tabs_names = [
    "📍 Géographie", 
    "📦 Rentabilité Produit", 
    "💸 Point Mort", 
    "👥 Valeur Client",
    "🤖 Modélisation",
    "🕒 Historique (Logs)"
]

def update_query_params():
    try:
        st.query_params["tab"] = st.session_state.app_nav_tabs
    except AttributeError:
        st.experimental_set_query_params(tab=st.session_state.app_nav_tabs)

if "app_nav_tabs" not in st.session_state:
    try:
        q_tab = st.query_params.get("tab")
    except AttributeError:
        q_tab = st.experimental_get_query_params().get("tab", [None])[0]
        
    st.session_state.app_nav_tabs = q_tab if q_tab in tabs_names else tabs_names[0]

selected_tab = st.radio(
    "Navigation", 
    tabs_names, 
    horizontal=True, 
    label_visibility="collapsed", 
    key="app_nav_tabs",
    on_change=update_query_params
)

st.markdown("---")

# --- VUES DES ONGLETS ---
if selected_tab == tabs_names[0]:
    st.subheader("Cartographie de la Rentabilité")
    if 'State' in filtered_df.columns and 'Sales' in filtered_df.columns and 'Profit' in filtered_df.columns and 'Discount' in filtered_df.columns:
        us_state_abbrev = get_us_state_abbrev()

        df_state = filtered_df.groupby('State', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'})
        df_state['State Code'] = df_state['State'].map(us_state_abbrev)

        fig_map = px.choropleth(
            df_state, locations='State Code', locationmode="USA-states", color='Profit',
            scope="usa", hover_name='State',
            hover_data={'State Code': False, 'Sales': ':$,.0f', 'Profit': ':$,.0f', 'Discount': ':.1%'},
            color_continuous_scale='RdBu', color_continuous_midpoint=0
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_map, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("États Extrêmes (Top vs Flop)")

        top_10 = df_state.sort_values(by='Profit', ascending=False).head(5)
        flop_10 = df_state.sort_values(by='Profit', ascending=True).head(5)

        fig_tf = make_subplots(rows=1, cols=2, subplot_titles=("Top 5 - Profits Stratégiques", "Flop 5 - Destructeurs de Valeur"))
        fig_tf.add_trace(go.Bar(x=top_10['State'], y=top_10['Profit'], marker=dict(color='seagreen'), name='Profit'), row=1, col=1)
        fig_tf.add_trace(go.Bar(x=flop_10['State'], y=flop_10['Profit'], marker=dict(color='crimson'), name='Pertes'), row=1, col=2)
        fig_tf.update_layout(showlegend=False)
        st.plotly_chart(fig_tf, width='stretch')

elif selected_tab == tabs_names[1]:
    st.subheader("Impact des Catégories sur la Marge")
    if 'Category' in filtered_df.columns and 'Sub-Category' in filtered_df.columns:
        df_subcat = filtered_df.groupby(['Category', 'Sub-Category'], as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'})
        df_subcat = df_subcat.sort_values(by='Profit', ascending=True)

        fig_subcat = px.bar(
            df_subcat, y='Sub-Category', x='Profit', orientation='h',
            color='Discount', color_continuous_scale='Reds', text_auto='$.2s',
            hover_data={'Category': True, 'Sales': ':$,.0f', 'Discount': ':.1%'},
            title="Profit par Sous-Catégorie"
        )
        fig_subcat.update_layout(height=600)
        st.plotly_chart(fig_subcat, width='stretch')

elif selected_tab == tabs_names[2]:
    st.subheader("Analyse du Point Mort")
    if 'Discount' in filtered_df.columns and 'Sales' in filtered_df.columns and 'Profit' in filtered_df.columns:
        df_discount = filtered_df.groupby('Discount', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Order ID': 'count'})
        df_discount['Marge Bénéficiaire'] = df_discount.apply(lambda row: row['Profit'] / row['Sales'] if row['Sales'] > 0 else 0, axis=1)

        fig_be = go.Figure()
        fig_be.add_trace(go.Scatter(
            x=df_discount['Discount'], y=df_discount['Marge Bénéficiaire'],
            mode='lines+markers', name='Marge Nette',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8, color=df_discount['Marge Bénéficiaire'], colorscale='RdYlGn', showscale=False)
        ))
        fig_be.add_hline(y=0, line_dash="dash", line_color="red", annotation_text=" SEUIL DE RENTABILITÉ (0%)")
        fig_be.update_layout(
            title="Évolution de la Marge Nette selon le Taux de Remise",
            xaxis=dict(title='Taux de Remise', tickformat='.0%'),
            yaxis=dict(title='Marge Bénéficiaire Nette', tickformat='.0%')
        )
        st.plotly_chart(fig_be, width='stretch')

elif selected_tab == tabs_names[3]:
    st.subheader("Valeur par Segment Client")
    if 'Segment' in filtered_df.columns and 'Customer ID' in filtered_df.columns:
        df_segment = filtered_df.groupby('Segment', as_index=False).agg({
            'Sales': 'sum', 
            'Profit': 'sum', 
            'Customer ID': 'nunique',
            'Order ID': 'nunique'
        })
        
        df_segment['Profit_par_Client'] = df_segment.apply(lambda r: r['Profit'] / r['Customer ID'] if r['Customer ID'] > 0 else 0, axis=1)
        df_segment['Profit_par_Commande'] = df_segment.apply(lambda r: r['Profit'] / r['Order ID'] if r['Order ID'] > 0 else 0, axis=1)
        
        df_segment = df_segment.sort_values(by='Profit_par_Client', ascending=False)

        col_client, col_order = st.columns(2)
        
        with col_client:
            fig_seg_client = px.bar(
                df_segment, x='Segment', y='Profit_par_Client',
                title="Profit Cumulé / Client",
                text_auto='$.0f', color='Segment', color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_seg_client.update_layout(yaxis_title="Profit / Client ($)", showlegend=False)
            st.plotly_chart(fig_seg_client, width='stretch')

        with col_order:
            fig_seg_order = px.bar(
                df_segment, x='Segment', y='Profit_par_Commande',
                title="Profit / Commande",
                text_auto='$.1f', color='Segment', color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_seg_order.update_layout(yaxis_title="Profit / Commande ($)", showlegend=False)
            st.plotly_chart(fig_seg_order, width='stretch')

elif selected_tab == tabs_names[4]:
    st.subheader("Modélisation de la Rentabilité")
    
    if not api_ready:
        st.warning("⚠️ L'API de prédiction est introuvable. Assurez-vous que le serveur FastAPI est actif.")
    elif not api_key:
        st.info("ℹ️ Veuillez entrer votre clé API dans le menu latéral pour utiliser les fonctionnalités de modélisation.")
    else:
        tab_sim, tab_batch = st.tabs(["🎯 Simulateur What-If", "📂 Traitement par Lot"])

        with tab_sim:
            st.markdown("### 🎯 Simulateur What-If")
            with st.form("whatif_form"):
                sim_sales = st.number_input("Montant de la vente ($)", min_value=0.0, value=500.0, step=50.0)
                sim_discount = st.slider("Taux de remise (%)", min_value=0.0, max_value=0.8, value=0.0, step=0.01)
                
                cats_available = df['Sub-Category'].unique() if 'Sub-Category' in df.columns else ['Chairs', 'Phones', 'Storage']
                sim_subcat = st.selectbox("Sous-Catégorie", cats_available)
                
                regs_available = df['Region'].unique() if 'Region' in df.columns else ['West', 'East', 'Central', 'South']
                sim_region = st.selectbox("Région", regs_available)
                
                segs_available = df['Segment'].unique() if 'Segment' in df.columns else ['Consumer', 'Corporate', 'Home Office']
                sim_segment = st.selectbox("Segment Client", segs_available)
                
                submit_sim = st.form_submit_button("Calculer le Profit Estimé")
                
            if submit_sim:
                payload = {
                    'Sales': sim_sales, 'Discount': sim_discount, 
                    'Sub_Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment
                }
                
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload, headers={"X-API-KEY": api_key})
                    
                    if res.status_code == 200:
                        pred_profit = res.json()["predicted_profit"]
                        marge = pred_profit / sim_sales if sim_sales > 0 else 0
                        
                        st.success("✅ **Prédiction Réalisée avec Succès !**")
                        
                        res_col1, res_col2 = st.columns(2)
                        res_col1.metric(label="Profit Net Estimé", value=f"{pred_profit:,.2f} $", delta=f"{marge:.2%} de marge", delta_color="normal")
                        res_col2.metric(label="Chiffre d'Affaires Simulé", value=f"{sim_sales:,.2f} $", delta=f"Remise: {sim_discount:.0%}", delta_color="inverse")

                        st.markdown("#### Variation du profit estimé selon la remise")
                        d_range = np.linspace(0, 0.8, 20)
                        
                        batch_payload = {"records": [
                            {'Sales': sim_sales, 'Discount': float(d), 'Sub_Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment}
                            for d in d_range
                        ]}
                        
                        res_batch = requests.post(f"{API_URL}/predict_batch", json=batch_payload, headers={"X-API-KEY": api_key})
                        
                        if res_batch.status_code == 200:
                            sim_df = pd.DataFrame([{
                                'Sales': sim_sales, 'Discount': d,
                                'Sub-Category': sim_subcat, 'Region': sim_region, 'Segment': sim_segment
                            } for d in d_range])
                            sim_df['Profit Estimé'] = res_batch.json()["predictions"]
                            fig_sim = px.line(sim_df, x='Discount', y='Profit Estimé', markers=True, labels={'Discount': 'Taux de Remise', 'Profit Estimé': 'Profit Estimé ($)'})
                            fig_sim.add_hline(y=0, line_dash="dash", line_color="red")
                            fig_sim.add_vline(x=sim_discount, line_dash="dot", line_color="green", annotation_text="Remise Actuelle")
                            fig_sim.update_layout(xaxis=dict(tickformat='.0%'))
                            st.plotly_chart(fig_sim, width='stretch')
                        else:
                            st.warning(f"Erreur API pour la variation ({res_batch.status_code}): {res_batch.text}")

                        # Ajout à l'historique Unitaire
                        st.session_state['sim_history'].append({
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Sales ($)": sim_sales,
                            "Discount (%)": sim_discount,
                            "Sub-Category": sim_subcat,
                            "Region": sim_region,
                            "Profit Estimé ($)": round(pred_profit, 2)
                        })
                        save_history_to_disk('sim')
                    else:
                        st.error(f"Accès Refusé / Erreur API ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Erreur de communication avec l'API : {e}")

        with tab_batch:
            st.markdown("### 📂 Traitement par Lot (Batch CSV, XLS, JSON)")
            uploaded_file = st.file_uploader("Chargez vos données transactionnelles", type=["csv", "xls", "xlsx", "json"])
            
            if uploaded_file is not None:
                try:
                    # Lecture : CSV multiprocess, JSON, ou Excel
                    batch_df = read_data_robust(uploaded_file, uploaded_file.name)
                        
                    st.success(f"{len(batch_df)} lignes chargées avec succès.")
                    
                    required_cols = ['Sales', 'Discount', 'Sub-Category', 'Region', 'Segment']
                    if all(c in batch_df.columns for c in required_cols):
                        
                        api_records = batch_df[required_cols].rename(columns={"Sub-Category": "Sub_Category"}).to_dict(orient='records')
                        
                        try:
                            res_batch_file = requests.post(f"{API_URL}/predict_batch", json={"records": api_records}, headers={"X-API-KEY": api_key})
                            
                            if res_batch_file.status_code == 200:
                                batch_df['Predicted_Profit'] = res_batch_file.json()["predictions"]
                                
                                st.dataframe(batch_df[['Sub-Category', 'Sales', 'Discount', 'Predicted_Profit']].head())

                                # Ajout à l'historique Batch
                                st.session_state['batch_history'].append({
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Fichier": uploaded_file.name,
                                    "Lignes": len(batch_df),
                                    "Statut": "Succès"
                                })
                                save_history_to_disk('batch')

                                st.markdown("📥 **Télécharger les résultats**")
                                
                                # Export JSON
                                json_data = batch_df.to_json(orient='records').encode('utf-8')
                                st.download_button("Télécharger JSON", data=json_data, file_name="predictions_batch.json", mime="application/json")
                                
                                # Export CSV
                                csv_data = batch_df.to_csv(index=False, encoding='utf-8').encode('utf-8')
                                st.download_button("Télécharger CSV", data=csv_data, file_name="predictions_batch.csv", mime="text/csv")
                                
                                # Export Excel
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    batch_df.to_excel(writer, index=False, sheet_name='Prédictions')
                                st.download_button("Télécharger Excel", data=buffer.getvalue(), file_name="predictions_batch.xlsx", mime="application/vnd.ms-excel")
                            else:
                                st.error(f"Erreur API ({res_batch_file.status_code}) : {res_batch_file.text}")
                        except Exception as api_err:
                            st.error(f"Erreur de communication avec l'API : {api_err}")
                            
                    else:
                        st.error(f"Le fichier doit impérativement contenir les colonnes : {', '.join(required_cols)}")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du fichier : {e}")

elif selected_tab == tabs_names[5]:
    st.subheader("Historique de Session")
    
    st.markdown("### 📈 Simulations Unitaires")
    if st.session_state['sim_history']:
        st.dataframe(pd.DataFrame(st.session_state['sim_history']), width='stretch')
    else:
        st.info("Aucune simulation unitaire réalisée dans cette session.")
        
    st.markdown("### 📂 Imports")
    if st.session_state['batch_history']:
        st.dataframe(pd.DataFrame(st.session_state['batch_history']), width='stretch')
    else:
        st.info("Aucun import par lot validé dans cette session.")
        
    if st.button("🗑️ Vider l'historique"):
        st.session_state['sim_history'] = []
        st.session_state['batch_history'] = []
        save_history_to_disk('sim')
        save_history_to_disk('batch')
        st.rerun()
