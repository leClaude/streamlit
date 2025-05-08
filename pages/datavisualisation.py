# Data Visualisation
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Chargement des datasets
@st.cache_data
def load_data():
    df_fr = pd.read_csv("data/mars-2014-complete.csv", encoding="latin1", sep=";")
    df_eu = pd.read_csv("data/CO2_passenger_cars_v10.csv", sep ="\t", encoding='utf-8', index_col='id', on_bad_lines='warn')
    return df_fr, df_eu

# Fonction pour générer le boxplot
@st.cache_data
def generate_boxplot(df_plot):
    fig = px.box(df_plot, x='Dataset', y='CO2', color='Dataset',
                 points='outliers',  # Limiter les points à afficher aux valeurs aberrantes
                 labels={'CO2': 'Émissions de CO2 (g/km)'},
                 title='Distribution interactive des émissions de CO2')
    return fig

# Fonction pour générer l'histogramme
@st.cache_data
def generate_histogram(df_fr, df_eu):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Histogramme pour Europe 2014
    sns.histplot(df_eu['e (g/km)'], ax=axes[0], bins=30, kde=False, color="#21918c", alpha=0.6, label="Europe 2014")
    
    # Histogramme pour ADEME 2014
    sns.histplot(df_fr['co2'], ax=axes[0], bins=30, kde=False, color="#fde725", alpha=0.6, label="Ademe 2014")
    
    # Titre et labels
    axes[0].set_title("Histogramme des émissions de CO2")
    axes[0].set_xlabel("Émissions de CO2 (g/km)")
    axes[0].set_ylabel("Nombre de véhicules")
    axes[0].legend()

    return fig

# Fonction pour générer le graphique KDE (densité)
@st.cache_data
def generate_kde_plot(df_fr, df_eu):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # KDE pour Europe 2014
    sns.kdeplot(df_eu['e (g/km)'], ax=axes[1], fill=True, alpha=0.5, color="#21918c", label="Europe 2014")
    
    # KDE pour ADEME 2014
    sns.kdeplot(df_fr['co2'], ax=axes[1], fill=True, alpha=0.5, color="#fde725", label="ADEME 2014")
    
    # Titre et labels
    axes[1].set_title("Densité des émissions de CO2 (KDE)")
    axes[1].set_xlabel("Émissions de CO2 (g/km)")
    axes[1].legend()

    return fig

def generate_distribution_plots(df_fr):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    plt.subplots_adjust(wspace=1)

    # Répartition des constructeurs dans le dataset ADEME
    constructeurs = df_fr["lib_mrq"].value_counts()
    constructeurs_autres = constructeurs.copy()
    constructeurs_autres["Autres"] = constructeurs_autres[~constructeurs_autres.index.isin(['MERCEDES', 'VOLKSWAGEN'])].sum()
    constructeurs_autres = constructeurs_autres.loc[['MERCEDES', 'VOLKSWAGEN', "Autres"]]

    # Création du camembert pour les constructeurs
    colors = sns.color_palette('Set2')
    axes[0].pie(constructeurs_autres, labels=constructeurs_autres.index, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title("Répartition des constructeurs (ADEME 2014)")

    # Répartition des carrosseries (Ademe)
    carrosserie = df_fr['Carrosserie'].value_counts()
    carrosserie_autres = carrosserie.copy()
    carrosserie_autres["Autres"] = carrosserie_autres[~carrosserie_autres.index.isin(['BERLINE', 'MINIBUS', 'BREAK', 'TS TERRAINS/CHEMINS'])].sum()
    carrosserie_autres = carrosserie_autres.loc[['BERLINE', 'MINIBUS', 'BREAK', 'TS TERRAINS/CHEMINS', 'Autres']]

    # Création du camembert pour les carrosseries
    colors2 = sns.color_palette('Set3')
    axes[1].pie(carrosserie_autres, labels=carrosserie_autres.index, autopct='%1.1f%%', pctdistance=0.85, colors=('#fc8d62', '#66c2a5', '#e78ac3', '#a6d854', '#8da0cb'), startangle=45)
    axes[1].set_title('Répartition des différentes carrosseries (ADEME 2014)')

    return fig

def show():
    st.title("📊 Datavizualisation")
    st.subheader("🎯 Distribution de la variable cible")

    # Chargement des données
    df_fr, df_eu = load_data()

    # Préparation des données pour la comparaison
    df_fr_plot = df_fr[['co2']].copy()
    df_fr_plot = df_fr_plot.rename(columns={'co2': 'CO2'})
    df_fr_plot['Dataset'] = 'France'

    df_eu_plot = df_eu[['e (g/km)']].copy()
    df_eu_plot = df_eu_plot.rename(columns={'e (g/km)': 'CO2'})
    df_eu_plot['Dataset'] = 'Europe'

    df_plot = pd.concat([df_fr_plot, df_eu_plot], ignore_index=True)
    df_plot = df_plot.dropna(subset=['CO2'])

    # Générer le boxplot en utilisant la fonction mise en cache
    fig_boxplot = generate_boxplot(df_plot)

    # Affichage du graphique
    st.plotly_chart(fig_boxplot, use_container_width=True)

    # Affichage du texte explicatif
    st.markdown("""
                Le graphique ci-dessus illustre la répartition des émissions de CO2 dans les deux jeux de données. On constate des différences légères entre les distributions des données de l'ADEME 2014 et de l'Europe 2014. 
                
                Dans le jeu de données de l'ADEME, la médiane des émissions de CO2 est de 205 g/km, avec une majorité des valeurs regroupées autour de cette médiane. Plusieurs valeurs aberrantes sont observées, en dessous de 150 g/km et au-dessus de 250 g/km, ce qui pourrait correspondre à des véhicules très économes en carburant ou très polluants. En revanche, dans le jeu de données européen, la médiane des émissions de CO2 est plus faible, se situant autour de 132 g/km. La boîte est également plus étroite, suggérant une faible variabilité des émissions. Bien que les deux jeux de données présentent des valeurs extrêmes, celles-ci sont plus fréquentes dans le jeu de données européen.
                """)

    # Générer l'histogramme en utilisant la fonction mise en cache
    fig_histogram = generate_histogram(df_fr, df_eu)

    # Affichage de l'histogramme
    st.pyplot(fig_histogram)

    st.markdown("""
                L'histogramme ci-dessus illustre la répartition des émissions de CO2 dans les deux jeux de données : Ademe 2014 (en jaune) et Europe 2014 (en bleu-vert). On observe que, de manière générale, les émissions sont plus élevées dans le jeu de données de l'Ademe, avec un pic autour de 200 g/km. Les véhicules de cette base de données semblent ainsi être, en moyenne, plus polluants que ceux présents dans la base Europe 2014.
                """)
    
    # Générer le graphique KDE en utilisant la fonction mise en cache
    fig_kde = generate_kde_plot(df_fr, df_eu)

    # Affichage du graphique KDE
    st.pyplot(fig_kde)

    st.markdown("""
                
                Le graphique ci-dessus représente la densité des émissions de CO2 pour chaque dataset. Il permet de mieux visualiser la distribution sans être affecté par les tailles d’échantillons différentes. Ces deux courbes confirment ce que nous avons pu observer précédemment. La distribution des émissions de CO2 dans le jeux de données de l’Ademe est plus concentrée et possède un pic très marqué autour de  200g/km. Cela suggère que les véhicules de cette base sont plus homogènes et ont, en moyenne, des émissions plus élevées.

                """)
    
    st.subheader("Répartition des entrées par constructeur et carrosserie")

    st.markdown("""
                - Répartition des constructeurs
                Lorsque l'on regarde les constructeurs présents dans le jeu de données de l'Ademe, on remarque une très forte représentation des véhicules Mercedes (65.8%) ainsi que Volkswagen(24.9%). Les autres constructeurs sont regroupés dans la catégorie “Autres” et ne représentent que 9,3%.
                Cette répartition des constructeurs peut potentiellement influencer les résultats si l’on cherche à généraliser l’analyse à l’ensemble du marché automobile.
                - Répartition des carrosseries
                Les minibus dominent avec 83.8%. Les berlines représentent 8% du total. Les autres carrosseries sont très peu représentées.
                Ce déséquilibre peut également influencer les prédictions d'émissions des autres véhicules.

                """)
    # Générer les graphiques de répartition
    fig_distribution = generate_distribution_plots(df_fr)

     # Affichage des graphiques dans Streamlit
    st.pyplot(fig_distribution)