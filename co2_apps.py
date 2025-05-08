# Accueil
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Chargement des données avec cache
@st.cache_data
def load_data():
    df_fr = pd.read_csv("data/mars-2014-complete.csv", encoding="latin1", sep=";")
    df_eu = pd.read_csv("data/CO2_passenger_cars_v10.csv", sep="\t", encoding='utf-8', index_col='id', on_bad_lines='warn')
    return df_fr, df_eu

# Prétraitement des données France
def preprocess_fr(df):
    df = df.copy()

    # Remplacement des virgules par des points
    cols_to_replace = ['co2', 'conso_urb', 'conso_exurb', 'conso_mixte',
                       'co_typ_1', 'nox', 'hcnox', 'ptcl', 'masse_ordma_min', 'masse_ordma_max']
    for col in cols_to_replace:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

    # Conversion en float
    for col in cols_to_replace:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Suppression des lignes sans CO2
    df = df.dropna(subset=['co2'])

    # Suppression des doublons
    df = df.drop_duplicates()

    # Remplacement des NaN par la médiane pour certaines colonnes
    median_cols = ['conso_urb', 'conso_exurb', 'conso_mixte', 'co_typ_1', 'nox', 'hcnox', 'ptcl']
    for col in median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Remplacement des NaN par la moyenne pour les masses
    mean_cols = ['masse_ordma_min', 'masse_ordma_max']
    for col in mean_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Suppression des colonnes inutiles
    df = df.drop(columns=[col for col in ['hc', 'date_maj'] if col in df.columns])

    return df

# Prétraitement des données Europe
def preprocess_eu(df):
    df = df.copy()

    # Harmonisation des types de carburants
    fuel_mapping = {
        'Petrol': 'Essence',
        'Diesel': 'Diesel',
        'Gasoline': 'Essence',
        'Electric': 'Electrique',
        'Hybrid': 'Hybride'
    }
    if 'fuelType' in df.columns:
        df['fuelType'] = df['fuelType'].replace(fuel_mapping)

    # Suppression des doublons
    df = df.drop_duplicates()

    # Suppression des colonnes inutiles
    cols_to_drop = ['z (Wh/km)', 'IT', 'Er (g/km)']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Suppression des lignes avec CO2 manquant ou égal à 0
    if 'CO2' in df.columns:
        df = df[df['CO2'].notna() & (df['CO2'] != 0)]

    # Suppression des lignes avec carburant manquant
    if 'fuelType' in df.columns:
        df = df[df['fuelType'].notna()]

    return df

df_france, df_europe = load_data()

# Prétraitement des données
df_fr_clean = preprocess_fr(df_france)
df_eu_clean = preprocess_eu(df_europe)

# Envoi des données nettoyées à la fonction d'affichage
#modelisation.show(df_fr_clean, df_eu_clean)

st.title("Projet Emission de CO2")

#st.sidebar.title('Sommaire')

# Barre de navigation
#page = st.sidebar.selectbox(
#    "Navigation",
#    ["Accueil","Exploration", "Data Visualisation", "Modelisation"]
#)

# Page Accueil (Introduction)
#if page == "Accueil":
st.markdown(""" 
### Introduction

Le secteur du transport, responsable d'environ 25% des émissions mondiales de dioxyde de carbone (CO2), représente un défi majeur dans la lutte contre le réchauffement climatique. En réponse à l'urgence climatique et aux réglementations de plus en plus strictes, l'industrie automobile doit repenser ses modèles pour réduire l'impact environnemental des véhicules.

L'analyse de données joue un rôle crucial dans ce processus. En explorant les caractéristiques techniques des véhicules, nous pouvons non seulement comprendre les facteurs influençant les émissions de CO2, mais aussi anticiper les émissions des futurs modèles. Grâce à des approches statistiques avancées et à la modélisation prédictive, nous pouvons identifier des leviers de réduction des émissions et proposer des solutions adaptées pour une mobilité plus durable.

### Objectifs

L'objectif de ce projet est d'appliquer des techniques d'analyse de données pour étudier les émissions de CO2 des véhicules. En utilisant des jeux de données détaillant les caractéristiques techniques de nombreux modèles, nous allons :

    - Mener une analyse exploratoire pour identifier les relations entre les caractéristiques des véhicules et leurs émissions.
    
    - Utiliser des modèles prédictifs pour estimer les émissions des véhicules futurs en fonction de leurs spécifications techniques.

    - Proposer des recommandations pratiques pour réduire les émissions de CO2, en s'appuyant sur les résultats de nos analyses.

À travers ce projet, nous visons à mieux comprendre les dynamiques sous-jacentes aux émissions de CO2 et à apporter des contributions significatives pour une mobilité plus verte et durable.
""")

# Page Exploration
#if page == "Exploration":
#    import exploration as exploration
#    exploration.show()

# Page Data Visualisation
#elif page == "Data Visualisation":
#    import dataviz
#    dataviz.show()

# Page Modélisation
#elif page == "Modélisation":
#    import modelisation
#    modelisation.show(df_fr_clean, df_eu_clean)
