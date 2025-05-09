# Exploration
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Chargement des datasets
@st.cache_data
def load_data():
    df_fr = pd.read_csv("data/mars-2014-complete.csv", encoding="latin1", sep=";")
    df_eu = pd.read_csv("data/CO2_passenger_cars_v10.csv", sep ="\t", encoding='utf-8', index_col='id', on_bad_lines='warn')
    return df_fr, df_eu

df_france, df_europe = load_data()

# Interface utilisateur
st.title("🔍 Exploration des données")

dataset_choice = st.selectbox("Sélectionnez un dataset :", ["Dataset français", "Dataset européen"])

# Choix du dataset
if dataset_choice == "Dataset français":
    df = df_france.copy()
else:
    df = df_europe.copy()

st.subheader("Aperçu des données")
st.dataframe(df.head())

#Présentation de la source
    
st.subheader("Source")
if dataset_choice == "Dataset français":
    st.markdown("""
    Ces données sont collectées par l’ADEME auprès de l'Union Technique de l’Automobile du motocycle et du Cycle – UTAC (en charge de l’homologation des véhicules avant leur mise en vente) et sont disponibles librement sur le site data.gouv.fr. Il comprend pour chaque véhicule des caractéristiques telles que : le type de carburant, la consommation, le poids du véhicule , sa puissance, sa cylindrée et ses émissions de CO2. 
                    
                    
    Il est composé de 55 044 lignes et 26 colonnes.
                """)
else:
    st.markdown("""
            Ce jeu de données reprend tous les enregistrements des voitures immatriculées dans l’Union Européenne. Il est mis à disposition par l’Agence Européenne de l’Environnement. Les informations sont enregistrées par chaque Etat membre. Il comprend notamment: le nom du constructeur, les poids des et dimensions des véhicules, la cylindrée, la puissance du moteur, le type de carburant.

                
            Il est composé de 417 938 lignes et 26 colonnes.
            """)
    
if dataset_choice == "Dataset français" and st.checkbox("Afficher la description des colonnes"):
    meta_data = {
        "nom-colonne": [
            "lib_mrq_utac", "lib_mod_doss", "lib_mod", "dscom", "cnit", "tvv", "cod_cbr", "hybride",
            "puiss_admin_98", "puiss_max", "typ_boite_nb_rapp", "conso_urb", "conso_exurb",
            "conso_mixte", "co2", "co_typ_1", "hc", "nox", "hcnox", "ptcl", "masse_ordma_min",
            "masse_ordma_max", "champ_v9", "date_maj", "Carrosserie", "gamme"
        ],
        "typerubrique": [
            "varchar", "varchar", "varchar", "varchar", "varchar", "varchar", "varchar", "varchar",
            "varnb", "varnb", "varchar", "varnb", "varnb", "varnb", "varnb", "varnb", "varnb", "varnb",
            "varnb", "varnb", "varnb", "varnb", "varchar", "varchar", "varchar", "varchar"
        ],
        "longueur": [
            12, 20, 20, 91, 15, 36, 5, 3, 2, 11, 3, 11, 11, 11, 3, 11, 11, 11, 11, 5, 4, 4, 23, 6, 19, 14
        ],
        "légende": [
            "la marque", "le modele du dossier", "le modèle commercial", "la désignation commerciale",
            "le Code National d'Identification du Type (CNIT)", "le Type-Variante-Version (TVV) ou le type Mines",
            "le type de carburant", "véhicule hybride (O/N)", "puissance administrative",
            "puissance maximale (en kW)", "type boîte de vitesse et nombre de rapports",
            "conso urbaine (l/100km)", "conso extra-urbaine (l/100km)", "conso mixte (l/100km)",
            "émission CO2 (g/km)", "CO type I", "HC", "NOx", "HC+NOX", "particules", "masse mini (kg)",
            "masse maxi (kg)", "norme EURO (champ V9)", "date de mise à jour", "Carrosserie", "gamme"
        ],
        "unité": [
            "", "", "", "", "", "", "", "", "", "kW", "", "l/100 km", "l/100 km", "l/100 km",
            "g/km", "g/km", "g/km", "g/km", "g/km", "g/km", "kg", "kg", "", "", "", ""
        ]
    }
    meta_df = pd.DataFrame(meta_data)
    st.subheader("📄 Description des colonnes")
    st.dataframe(meta_df)    
    st.subheader("Exploration des données")
    st.markdown("""
                Langage utilisé : Python 
                Librairies utilisées : Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn
                """)
    
# Analyse des valeurs manquantes
st.subheader("🔧 Valeurs manquantes")
missing = df.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]
st.write(f"Colonnes avec valeurs manquantes : {len(missing)}")
st.bar_chart(missing)

if st.checkbox("Afficher les statistiques descriptives"):
    st.subheader("📈 Statistiques descriptives")
    st.dataframe(df.describe())

st.subheader("Les données pertinentes")
st.markdown("""
Dans le cadre de notre étude, plusieurs variables se révèlent particulièrement pertinentes pour expliquer les émissions de CO₂, notre variable cible (exprimée en grammes par kilomètre – g/km) :

- La puissance du moteur, exprimée en kilowatts (kW), indique la capacité du moteur à produire de l’énergie.

- La cylindrée, mesurée en centimètres cubes (cm³), correspond au volume total des cylindres du moteur.

- Le type de carburant utilisé par le véhicule : essence, diesel, électrique, hybride, GPL (gaz de pétrole liquéfié) ou GNV (gaz naturel pour véhicule).

- La consommation de carburant, exprimée en litres pour 100 kilomètres (l/100 km), permet d’évaluer l’efficacité énergétique du véhicule.

- La masse du véhicule, en kilogrammes (kg), représente son poids à vide.

- Le type de carrosserie, qui renvoie à la structure et au design du véhicule (berline, break, coupé, etc.).

- Le constructeur, autrement dit la marque qui fabrique le véhicule (ex. Renault, Peugeot, Volkswagen, Mercedes…). Certains constructeurs sont en effet plus avancés que d’autres en matière d’innovation écologique et de réduction des émissions.
            """)