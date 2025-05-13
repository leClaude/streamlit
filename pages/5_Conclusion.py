# Modélisation
import streamlit as st
import pandas as pd
import openpyxl

df_results = pd.read_excel("data/modélisation.xlsx", sheet_name='ademe2014', header=0)
st.title("Conclusion")

st.subheader("Résultats des modélisation")

st.dataframe(df_results)

st.subheader("Conclusion")

st.markdown("""
    L’objectif de notre projet était d’identifier les facteurs influençant les émissions de CO2 des
    véhicules en utilisant différents modèles de prédiction sur deux jeux de données: un issu de
    l’ADEME et l’autre issu de l’Agence Européenne de l’Environnement. Chacun offrait des
    caractéristiques propres tant en termes de variables disponibles que de qualité des
    données. Ils montraient toutefois une certaine complémentarité qui nous a poussé à réaliser
    notre analyse sur les deux jeux de données en parallèle.
    Parmi les modèles testés, le Gradient Boosting Regressor offre les meilleures performances
    globales avec un score allant jusqu’à 0.965 sur le jeu de données de l’ADEME et 0.925 sur
    le jeu de l’Agence Européenne. Le Random Forest Regressor a également montré de très
    bonnes performances avec des temps de calcul moins longs.
    L’analyse de l’importance des variables a montré que pour les deux jeux de données se sont
    la masse, la puissance, la cylindrée ou encore le type de carburant des véhicules qui
    prédominent.
    """)
