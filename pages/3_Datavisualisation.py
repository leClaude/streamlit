# Data Visualisation
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Chargement des datasets
@st.cache_data
def load_data():
    ademe_2014 = pd.read_csv("data/ademe_2014_clean.csv", encoding="utf-8", sep=",")
    europe_2014 = pd.read_csv("data/europe_2014_clean.csv", sep =",", encoding='utf-8', index_col='id', on_bad_lines='warn')
    
    # Renommer les colonnes ademe
    ademe_2014=ademe_2014.rename(columns={'puiss_admin_98': 'Puissance administrative',
                                  'puiss_max': 'Puissance maximale (kW)',
                                  'conso_urb': 'Consommation urbaine (l/100km)',
                                  'conso_exurb': 'Consommation extra urbaine (l/100km)',
                                  'conso_mixte': 'Consommation mixte (l/100km)',
                                  'co2': 'Co2 (g/km)',
                                  'co_typ_1': 'CO type I (g/km)',
                                  'ptcl': 'Particules (g/km)',
                                  'nox':'NOx (g/km)',
                                  'masse_ordma_max': 'masse en ordre de marche max',
                                  'masse_ordma_min': 'masse en ordre de marche min'})
                                  
    # Renommer les colonnes Europe 2014
    europe_2014=europe_2014.rename(columns={'Mh':'Fabricant',
                                        'T':'Type',
                                        'Va': 'Variante',
                                        'Ve': 'Version',
                                        'Mk' : 'Marque',
                                        'Cn': 'Nom Commercial',
                                        'e (g/km)': 'Co2 (g/km)',
                                        'm (kg)': 'Masse en ordre de marche',
                                        'Ft': 'Carburant',
                                        'ec (cm3)': 'Cylindrée (cm3)',
                                        'ep (KW)': 'Puissance (KW)',
                                        'w(mm)': 'Empattement (mm)',
                                        'at1 (mm)': 'Largeur essieu directeur (mm)',
                                        'at2 (mm)': 'Largeur autre essieu (mm)',
                                        'w (mm)': 'Empattement (mm)',
                                        'z (Wh/km)' :'Consommation électrique (Wh/km)'})
    return ademe_2014, europe_2014
    


# Fonction pour générer le boxplot
@st.cache_data
def generate_boxplot(df_plot):
    fig = px.box(df_plot, x='Dataset', y='Co2 (g/km)', color='Dataset',
                 points='outliers',  # Limiter les points à afficher aux valeurs aberrantes
                 labels={'CO2': 'Émissions de CO2 (g/km)'},
                 title='Distribution interactive des émissions de CO2')
    return fig


# Fonction pour générer l'histogramme
@st.cache_data
def generate_histogram(df_fr, df_eu):
    fig = plt.figure(figsize=(18, 5))

    # Histogramme pour Europe 2014
    sns.histplot(df_eu['Co2 (g/km)'], bins=30, kde=False, color="#21918c", alpha=0.6, label="Europe 2014")
    
    # Histogramme pour ADEME 2014
    sns.histplot(df_fr['Co2 (g/km)'], bins=30, kde=False, color="#fde725", alpha=0.6, label="Ademe 2014")
    plt.show()
    # Titre et labels
    #axes[0].set_title("Histogramme des émissions de CO2")
    #axes[0].set_xlabel("Émissions de CO2 (g/km)")
    #axes[0].set_ylabel("Nombre de véhicules")
    #axes[0].legend()

    return fig

# Fonction pour générer la matrice de corrélation ademe 2014
def generate_heatmap(ademe_2014):
    ademe_numerique=ademe_2014.select_dtypes(include=['number'])
    heatmap=ademe_numerique.corr()
    fig = px.imshow(heatmap,
              labels=dict(x="colonnes",y="lignes", color="valeur"),
              x=heatmap.columns,
              y=heatmap.index,
              color_continuous_scale="viridis",
              text_auto=".2f")
    fig.update_layout(title= "Matrice de corrélation ADEME 2014",
                  coloraxis_colorbar=dict(title="Valeur", len=0.6),
                  width=800, height=800)
    return fig

# Création du graphique plot
def generate_plot(ademe_2014):
    # Renommer les valeurs de la colonne cod_cbr (type de carburant)
    mapping={'EE':'Essence/Electrique',
         'EH': 'Essence/Electrique',
         'EL': 'Electrique',
         'GH':'Gazole / Electrique',
         'GL':'Gazole / Electrique',
         'GO':'Gazole',
         'ES/GP' : "Essence / GPL",
         'GP/ES':"Essence / GPL",
         'ES':'Essence',
         'ES/GN':'Essence /  GNV',
         'GN/ES':'Essence /  GNV',
         'FE' : 'Véhicule E85',
         'GN' : 'GNV',
         'H2' : 'Hydrogène'}

    ademe_2014['cod_cbr_grouped'] = ademe_2014['cod_cbr'].replace(mapping)
    
    fig = px.scatter(
    ademe_2014,
    x="Consommation mixte (l/100km)",
    y="Co2 (g/km)",
    color="cod_cbr_grouped",  # Utilisation de la variable regroupée
    title="Relation entre consommation et émissions de CO2",
    labels={"Consommation mixte (l/100km)": "Consommation mixte (l/100km)", "Co2 (g/km)": "Émissions de CO2 (g/km)"},
    color_discrete_sequence=px.colors.qualitative.Set2,  # Palette de couleurs plus lisible
    hover_data=["cod_cbr"])

    fig.update_layout(
    xaxis_title="Consommation mixte de carburant (en l/100km)",
    yaxis_title="Émission de CO2 (en g/km)",
    legend_title="Type de carburant",
    template="plotly_white")
    
    return fig

def generate_distribution_plots(df_fr):
    # Répartition des constructeurs dans le dataset ADEME
    constructeurs = df_fr["lib_mrq"].value_counts()
    constructeurs_autres = constructeurs.copy()
    constructeurs_autres["Autres"] = constructeurs_autres[~constructeurs_autres.index.isin(['MERCEDES', 'VOLKSWAGEN'])].sum()
    constructeurs_autres = constructeurs_autres.loc[['MERCEDES', 'VOLKSWAGEN',"Autres"]]
    
    # Répartition des carrosseries (Ademe)
    carrosserie=df_fr['Carrosserie'].value_counts()
    carrosserie_autres=carrosserie.copy()
    carrosserie_autres["Autres"]=carrosserie_autres[~carrosserie_autres.index.isin(['BERLINE', 'MINIBUS','BREAK','TS TERRAINS/CHEMINS'])].sum()
    carrosserie_autres=carrosserie_autres.loc[['BERLINE', 'MINIBUS','BREAK','TS TERRAINS/CHEMINS','Autres']]

    fig=make_subplots(rows=1, cols=2, subplot_titles=["Répartition des constructeurs (ADEME 2014)", "Répartition des différentes carrosseries (ADEME 2014)"],specs=[[{"type":"domain"},{"type":"domain"}]])

    constructeurs_df = pd.DataFrame({'Constructeur': constructeurs_autres.index, 'Nombre': constructeurs_autres.values})

    # Ajout du premier camembert (Constructeurs)
    fig.add_trace(go.Pie(labels=constructeurs_df['Constructeur'], values=constructeurs_df['Nombre'],
                     name="Constructeurs", marker=dict(colors=px.colors.qualitative.Set2, line=dict(color='white', width=1)),
                     textinfo='label+percent'), row=1, col=1)

    # Création du DataFrame
    carrosserie_df = pd.DataFrame({'Carrosserie': carrosserie_autres.index, 'Nombre': carrosserie_autres.values})

    # Ajout du deuxième camembert (Carrosseries)
    fig.add_trace(go.Pie(labels=carrosserie_df['Carrosserie'], values=carrosserie_df['Nombre'],
                     name="Carrosseries", marker=dict(colors=['#fc8d62','#66c2a5','#e78ac3','#a6d854','#8da0cb'], line=dict(color='white', width=1)),
                     textinfo='label+percent'), row=1, col=2)

    return fig

def show():
    st.title("Datavisualisation")
    st.subheader("Distribution de la variable cible")

    # Chargement des données
    df_fr, df_eu = load_data()

    # Préparation des données pour la comparaison
    df_fr_plot = df_fr[['Co2 (g/km)']].copy()
    df_fr_plot['Dataset'] = 'France'

    df_eu_plot = df_eu[['Co2 (g/km)']].copy()
    df_eu_plot['Dataset'] = 'Europe'

    df_plot = pd.concat([df_fr_plot, df_eu_plot], ignore_index=True)
    df_plot = df_plot.dropna(subset=['Co2 (g/km)'])

    # Générer le boxplot en utilisant la fonction mise en cache
    fig_boxplot = generate_boxplot(df_plot)

    # Affichage du graphique
    st.plotly_chart(fig_boxplot, use_container_width=True)

    # Affichage du texte explicatif
    st.markdown("""
                Le graphique ci-dessus illustre la répartition des émissions de CO2 dans les deux jeux de données. On constate des différences légères entre les distributions des données de l'ADEME 2014 et de l'Europe 2014. 
                
                Dans le jeu de données de l'ADEME, la médiane des émissions de CO2 est de 205 g/km, avec une majorité des valeurs regroupées autour de cette médiane. Plusieurs valeurs aberrantes sont observées, en dessous de 150 g/km et au-dessus de 250 g/km, ce qui pourrait correspondre à des véhicules très économes en carburant ou très polluants. En revanche, dans le jeu de données européen, la médiane des émissions de CO2 est plus faible, se situant autour de 132 g/km. La boîte est également plus étroite, suggérant une faible variabilité des émissions. Bien que les deux jeux de données présentent des valeurs extrêmes, celles-ci sont plus fréquentes dans le jeu de données européen.
                """)

    st.subheader("Distribution des émissions de CO2 dans les deux datasets")
    # Générer l'histogramme en utilisant la fonction mise en cache
    fig_histogram = generate_histogram(df_fr, df_eu)

    # Affichage de l'histogramme
    st.pyplot(fig_histogram)

    st.markdown("""
                L'histogramme ci-dessus illustre la répartition des émissions de CO2 dans les deux jeux de données : Ademe 2014 (en jaune) et Europe 2014 (en bleu-vert). On observe que, de manière générale, les émissions sont plus élevées dans le jeu de données de l'Ademe, avec un pic autour de 200 g/km. Les véhicules de cette base de données semblent ainsi être, en moyenne, plus polluants que ceux présents dans la base Europe 2014.
                """)
    
    st.subheader("Matrice de correlation du dataset ADEME")
    #Générer la heatmap
    fig_heatmap = generate_heatmap(df_fr)
    
    #Affichage de la heatmap
    st.plotly_chart(fig_heatmap)
    
    st.markdown("""
    On remarque une forte corrélation entre les émissions de CO2 et la consommation des véhicules (<0.97). Plus un véhicule consomme de carburant, plus celui-ci émet de CO2. 
    On note également, un lien entre la masse du véhicule et les émissions de CO2(entre 0.54 et 0.64). Plus un véhicule est lourd, plus il consomme et donc émet de CO2.
                """)

    st.subheader("Emissions de CO2 en fonction de la consommation et du type de carburant")
    #Générer le graphique
    fig_plot = generate_plot(df_fr)
    
    #Affichage du graphe CO2 vs conso
    st.plotly_chart(fig_plot)
    
    st.markdown("""
    On remarque une corrélation linéaire forte entre la consommation de carburant et les émissions de CO2. 
    Cette corrélation est attendue puisque la combustion du carburant entraîne directement la production de CO2, et que chaque carburant n'a ni la même concentration en carbone ni la même densité.
                """)
                
    st.subheader("Répartition des entrées par constructeur et carrosserie du dataset ADEME")
    # Générer les graphiques de répartition
    fig_distribution = generate_distribution_plots(df_fr)

     # Affichage des graphiques dans Streamlit
    st.plotly_chart(fig_distribution)
    st.markdown("""
                ###### Répartition des constructeurs
                Lorsque l'on regarde les constructeurs présents dans le jeu de données de l'Ademe, on remarque une très forte représentation des véhicules Mercedes (65.8%) ainsi que Volkswagen(24.9%). Les autres constructeurs sont regroupés dans la catégorie “Autres” et ne représentent que 9,3%.
                Cette répartition des constructeurs peut potentiellement influencer les résultats si l’on cherche à généraliser l’analyse à l’ensemble du marché automobile.
                ###### Répartition des carrosseries
                Les minibus dominent avec 83.8%. Les berlines représentent 8% du total. Les autres carrosseries sont très peu représentées.
                Ce déséquilibre peut également influencer les prédictions d'émissions des autres véhicules.
                """)
                
    
    
    

show()