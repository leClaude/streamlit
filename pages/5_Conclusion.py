# Modélisation
import streamlit as st
import pandas as pd

st.title("Conclusion")

st.subheader("Conclusion")

st.markdown("""
    Notre objectif était d’identifier les principaux facteurs influençant les émissions de CO₂ des véhicules, à l’aide de modèles de machine learning appliqués à deux jeux de données : celui de l’ADEME pour le marché français, et celui de l’Agence Européenne de l’Environnement pour une vision à l’échelle européenne.
Côté modélisation, le Gradient Boosting Regressor a donné les meilleurs résultats, avec un score R² allant jusqu’à 0.965 sur les données ADEME et 0.925 sur le dataset européen avec des variables clés comme la masse, la puissance, la cylindrée et le type de carburant.
#Limites
-Fort déséquilibre du dataset : >80 % de minibus.
-Surreprésentation de certains constructeurs.
-Difficulté à généraliser les résultats à l’ensemble du parc automobile
#Pistes à explorer
-Rééquilibrer ou enrichir le dataset ADEME avec des véhicules plus variés.
-Explorer la fusion de datasets pour améliorer la représentativité.
-Alléger le modèle en supprimant les variables les moins influentes, pour gagner en simplicité et en performance.
    """)

st.subheader("Perspectives")